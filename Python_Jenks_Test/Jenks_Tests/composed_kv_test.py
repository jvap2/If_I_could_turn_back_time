"""
Composed KV4 test: full W4A4 build (E2M1 weights + GF4 acts, kappa=100),
then eval each W4A4 arm WITH vs WITHOUT KV4 (GF4-adaptive, no rotation) hooks.
Gives the real +KV4 delta comparable to QuaRot/SpinQuant W4A4KV4.

Run from Jenks_Tests/:  python3 composed_kv_test.py [model]
"""
import sys, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from FP_Quantization_Experiments import (
    quantize_model_fp, enable_fast_kernels, act_quant_mode,
    quantize_activations_gf4_adaptive,
)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "facebook/opt-1.3b"
DEV, SEQLEN, CALIB_SEQLEN, NCALIB = "cuda", 2048, 2048, 16
random.seed(0); torch.manual_seed(0)


@torch.no_grad()
def eval_ppl(model, ids, seqlen=SEQLEN):
    model.eval()
    n = ids.numel() // seqlen
    nlls = []
    for i in range(n):
        batch = ids[:, i*seqlen:(i+1)*seqlen].to(DEV)
        loss = model(batch, labels=batch).loss
        nlls.append(loss.float() * seqlen)
    return torch.exp(torch.stack(nlls).sum() / (n * seqlen)).item()


def add_kv4_hooks(model, head_dim, passes=1):
    """GF4-adaptive KV4, no rotation — the recipe from the isolation sweep.
    passes=1: single-term KV4. passes=2: residual 2-term KV (each pass re-quantizes
    the leftover residual), still native-basis / no rotation."""
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


def main():
    print(f"loading {MODEL} ...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV)
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    print(f"  head_dim={head_dim}")

    train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    test  = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(train["text"]), return_tensors="pt",
              add_special_tokens=False).input_ids.squeeze(0)
    test_ids = tok("\n\n".join(test["text"]), return_tensors="pt",
                   add_special_tokens=False).input_ids
    calib = [ids[(s:=random.randint(0, ids.size(0)-CALIB_SEQLEN-1)):s+CALIB_SEQLEN].unsqueeze(0)
             for _ in range(NCALIB)]

    print("building W4A4 (E2M1 weights, GF4 acts, kappa=100) ...")
    model = quantize_model_fp(model, calib, block_size=16, e_bits=2, m_bits=1,
        e_bits_scale=4, m_bits_scale=3, device=DEV,
        use_HG=False, use_Hessian=False, use_adap=False, use_forward=False,
        Hadamard=True, joint=False, preshift=False, decompose=False,
        had_block_size="auto", use_gf4=True, extra_skip_patterns=_mlp_skip(MODEL),
        lean=True, offload=False)
    enable_fast_kernels(model, enable=True)

    def run(label, mode, kv4):
        handles = add_kv4_hooks(model, head_dim, passes=kv4) if kv4 else []
        with act_quant_mode(model, mode=mode):
            ppl = eval_ppl(model, test_ids)
        for h in handles: h.remove()
        print(f"  {label:34s} {ppl:9.4f}")
        return ppl

    print(f"\n===== {MODEL}  composed W4A4 (+KV4) =====")
    a16  = run("A16 (W4A16, weights only)   [sanity]", None,           0)
    g    = run("Adaptive GF4 (W4A4)",                  "gf4_adaptive", 0)
    gkv  = run("Adaptive GF4 + KV4 (W4A4KV4)",         "gf4_adaptive", 1)
    r    = run("Residual GF4 (W4A4)",                  "gf4_residual", 0)
    rkv  = run("Residual GF4 + KV4 (1-term)",          "gf4_residual", 1)
    rkv2 = run("Residual GF4 + KV4 (2-term)",          "gf4_residual", 2)
    print(f"\n  KV4 adds:  Adaptive {gkv-g:+.4f}   Residual 1t {rkv-r:+.4f}   "
          f"Residual 2t {rkv2-r:+.4f}")


if __name__ == "__main__":
    main()
