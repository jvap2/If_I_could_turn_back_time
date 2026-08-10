"""
Composed W4A4 (+KV4) sweep across models. Auto-selects block-sequential offload
for models too big for the 16GB GPU (opt-6.7b / 7b). Appends to kv4_sweep.csv.

Arms per model: A16 (sanity), Adaptive GF4, Adaptive GF4+KV4,
                Residual GF4, Residual GF4+KV4.
KV4 = GF4-adaptive on k_proj/v_proj outputs (block=head_dim), no rotation.

Run:  python3 kv4_sweep.py facebook/opt-125m facebook/opt-2.7b ...
"""
import sys, os, csv, random, gc, datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from FP_Quantization_Experiments import (
    quantize_model_fp, enable_fast_kernels, act_quant_mode,
    quantize_activations_gf4_adaptive,
)
from FP_Quantization_Experiments.bit_split import evaluate_ppl_offload

DEV, SEQLEN, CALIB_SEQLEN, NCALIB = "cuda", 2048, 2048, 16
VRAM_FRAC = 0.70          # offload once fp16 weights exceed this fraction of GPU memory
CSV = "kv4_sweep.csv"


def _should_offload(nparams):
    """Block-sequential offload only when fp16 weights won't comfortably fit the GPU.
    On a 48GB card: 13B (26GB) < 0.7*48 -> in-GPU fast path; 30B/66B -> offload."""
    total = torch.cuda.get_device_properties(0).total_memory
    return nparams * 2 > VRAM_FRAC * total


@torch.no_grad()
def eval_ppl_gpu(model, ids, seqlen=SEQLEN):
    model.eval()
    n = ids.numel() // seqlen
    nlls = []
    for i in range(n):
        batch = ids[:, i*seqlen:(i+1)*seqlen].to(DEV)
        nlls.append(model(batch, labels=batch).loss.float() * seqlen)
    return torch.exp(torch.stack(nlls).sum() / (n * seqlen)).item()


def add_kv4_hooks(model, head_dim, passes=1):
    """GF4-adaptive KV4 on k_proj/v_proj outputs, block=head_dim (per-head, per-token).
    passes=1: single-term (the original KV4). passes=2: residual 2-term KV — each pass
    re-quantizes the leftover residual (mirrors npass in validate_multipass.py), giving a
    ~2-bit-effective-richer KV cache to test against the single-term ceiling."""
    def hook(_m, _i, out):
        if passes == 1:
            return quantize_activations_gf4_adaptive(out, head_dim)
        q = torch.zeros_like(out, dtype=torch.float32)
        for _ in range(passes):
            q = q + quantize_activations_gf4_adaptive(out.float() - q, head_dim).float()
        return q.to(out.dtype)
    handles = []
    for name, m in model.named_modules():
        if name.split(".")[-1] in ("k_proj", "v_proj"):
            handles.append(m.register_forward_hook(hook))
    return handles


def _mlp_skip(mn):
    mn = mn.lower()
    if "llama" in mn or "mistral" in mn: return ("down_proj",)
    if "opt" in mn or "gpt" in mn:       return ("fc2",)
    return ("down_proj", "fc2", "dense_4h_to_h")


def run_model(MODEL):
    print(f"\n{'='*66}\n### {MODEL}\n{'='*66}", flush=True)
    random.seed(0); torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                                 low_cpu_mem_usage=True)
    nparams = sum(p.numel() for p in model.parameters())
    offload = _should_offload(nparams)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"  params={nparams/1e9:.2f}B  GPU={total_gb:.0f}GB  offload={offload}", flush=True)
    if not offload:
        model = model.to(DEV)
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    print(f"  head_dim={head_dim}", flush=True)

    train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    test  = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(train["text"]), return_tensors="pt",
              add_special_tokens=False).input_ids.squeeze(0)
    test_ids = tok("\n\n".join(test["text"]), return_tensors="pt",
                   add_special_tokens=False).input_ids
    calib = [ids[(s:=random.randint(0, ids.size(0)-CALIB_SEQLEN-1)):s+CALIB_SEQLEN].unsqueeze(0)
             for _ in range(NCALIB)]

    model = quantize_model_fp(model, calib, block_size=16, e_bits=2, m_bits=1,
        e_bits_scale=4, m_bits_scale=3, device=DEV,
        use_HG=False, use_Hessian=False, use_adap=False, use_forward=False,
        Hadamard=True, joint=False, preshift=False, decompose=False,
        had_block_size="auto", use_gf4=True, extra_skip_patterns=_mlp_skip(MODEL),
        lean=True, offload=offload)
    enable_fast_kernels(model, enable=True)

    def EVAL():
        return evaluate_ppl_offload(model, test_ids, DEV, SEQLEN) if offload \
               else eval_ppl_gpu(model, test_ids)

    def run(label, mode, kv4):
        handles = add_kv4_hooks(model, head_dim, passes=kv4) if kv4 else []
        with act_quant_mode(model, mode=mode):
            ppl = EVAL()
        for h in handles: h.remove()
        print(f"  {label:34s} {ppl:9.4f}", flush=True)
        return ppl

    a16  = run("A16 (W4A16) [sanity]",          None,           0)
    g    = run("Adaptive GF4 (W4A4)",           "gf4_adaptive", 0)
    gkv  = run("Adaptive GF4 + KV4",            "gf4_adaptive", 1)
    r    = run("Residual GF4 (W4A4)",           "gf4_residual", 0)
    rkv  = run("Residual GF4 + KV4 (1-term)",   "gf4_residual", 1)
    rkv2 = run("Residual GF4 + KV4 (2-term)",   "gf4_residual", 2)
    print(f"  KV4 adds:  Adaptive +{gkv-g:.4f}   "
          f"Residual 1t +{rkv-r:.4f}   Residual 2t +{rkv2-r:.4f}", flush=True)

    row = {"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "model": MODEL, "offload": int(offload), "A16": a16,
           "adaptive_gf4": g, "adaptive_kv4": gkv,
           "residual_gf4": r, "residual_kv4": rkv, "residual_kv4_2t": rkv2,
           "kv4_add_adaptive": gkv-g, "kv4_add_residual": rkv-r,
           "kv4_add_residual_2t": rkv2-r}
    newf = not os.path.exists(CSV)
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if newf: w.writeheader()
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"  [saved] -> {CSV}", flush=True)

    del model; gc.collect(); torch.cuda.empty_cache()


def main():
    models = sys.argv[1:] or ["facebook/opt-125m", "facebook/opt-2.7b"]
    for m in models:
        try:
            run_model(m)
        except Exception:
            import traceback; traceback.print_exc()
            print(f"[FAILED] {m} — continuing", flush=True)


if __name__ == "__main__":
    main()
