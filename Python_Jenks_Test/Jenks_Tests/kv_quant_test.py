"""
Isolated KV-cache quantization test (no full pipeline).

Hooks k_proj/v_proj outputs and GF4-quantizes them at block_size=head_dim
=> per-head, per-token 4-bit KV. Optional head_dim Hadamard rotation
(QuaRot-style): rotate Q+K so scores QK^T are preserved, rotate V then
rotate back so o_proj sees de-rotated V. This isolates "how much does KV4
cost, and does the rotation help" before wiring it into the full W4A4 path.

Run from Jenks_Tests/:  python3 kv_quant_test.py [model]
"""
import sys, math, random
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from FP_Quantization_Experiments import (
    quantize_activations_gf4,
    quantize_activations_gf4_adaptive,
    fwht_blockwise,
)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "facebook/opt-125m"
DEV, SEQLEN = "cuda", 2048
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


def make_hook(role, mode, head_dim, rotate):
    """role in {q,k,v}; mode in {fixed,adaptive}."""
    def hook(_module, _inp, out):
        x = out
        if role == "q":                          # rotate only (no quant)
            return fwht_blockwise(x, head_dim) if rotate else x
        if rotate:
            x = fwht_blockwise(x, head_dim)
        xq = (quantize_activations_gf4_adaptive(x, head_dim) if mode == "adaptive"
              else quantize_activations_gf4(x, head_dim))
        if role == "v" and rotate:               # bring V back to original basis
            xq = fwht_blockwise(xq, head_dim)
        return xq
    return hook


def add_kv_hooks(model, mode, head_dim, rotate):
    handles = []
    for name, m in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf == "k_proj":
            handles.append(m.register_forward_hook(make_hook("k", mode, head_dim, rotate)))
        elif leaf == "v_proj":
            handles.append(m.register_forward_hook(make_hook("v", mode, head_dim, rotate)))
        elif leaf == "q_proj" and rotate:
            handles.append(m.register_forward_hook(make_hook("q", mode, head_dim, rotate)))
    return handles


def main():
    print(f"loading {MODEL} ...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV)
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    print(f"  hidden={cfg.hidden_size}  heads={cfg.num_attention_heads}  head_dim={head_dim}"
          f"  (pow2={(head_dim & (head_dim-1)) == 0})")

    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(test["text"]), return_tensors="pt",
              add_special_tokens=False).input_ids
    print(f"  eval tokens={ids.numel()}  windows={ids.numel()//SEQLEN}")

    pow2 = (head_dim & (head_dim - 1)) == 0
    arms = [
        ("baseline fp16 (no KV quant)", None,       False),
        ("KV4 GF4 fixed",              "fixed",     False),
        ("KV4 GF4 adaptive",           "adaptive",  False),
    ]
    if pow2:
        arms.append(("KV4 GF4 adaptive + rot", "adaptive", True))
    else:
        print(f"  [note] head_dim={head_dim} not power-of-2 -> skipping rotation arm")
    base = None
    print(f"\n===== {MODEL}  WikiText-2 KV-quant isolation =====")
    for label, mode, rotate in arms:
        handles = [] if mode is None else add_kv_hooks(model, mode, head_dim, rotate)
        ppl = eval_ppl(model, ids)
        for h in handles: h.remove()
        if base is None: base = ppl
        print(f"  {label:30s} {ppl:9.4f}   (+{ppl-base:6.4f})")


if __name__ == "__main__":
    main()
