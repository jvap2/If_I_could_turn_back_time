"""
Localize the opt-13b W4A16 collapse to the v5 weight solver.

We already proved (diag_hadamard_outliers.py) that a SIMPLE fixed-scale 4-bit
quantization of opt-13b's rotated weights is clean (~0.07 output relerr, either
codebook). Yet the pipeline collapses opt-13b to A16=304. The only remaining
difference is the actual solver, reconstruct_layer_fp_blockdiag_scaled_v5
(per-row exponent-bias search + Hessian-weighted alpha, alpha rounded to E4M3).

This script runs the REAL v5 on captured (W_had, Hessian) for a few layers and
compares its output relerr to the simple quantizer, with alpha/bias diagnostics:

  v5 out relerr  >>  simple out relerr   on opt-13b (but not opt-6.7b)
      => v5 is the bug; the alpha/bias stats say which step.

v5 minimizes output error (min (w-q)^T H (w-q)), so it SHOULD beat the simple
quantizer. If it is far WORSE, a numerical step inside it is failing.

Usage:  python3 diag_v5_solver.py facebook/opt-6.7b facebook/opt-13b
Env: NSEQ(2) SEQLEN(512) WTOK(256).  TARGET layer indices/suffixes below.
"""
import os, sys, math, hashlib, gc
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# The ONE thing we must use from the real pipeline — the weight solver itself:
from FP_Quantization_Experiments.bit_split import reconstruct_layer_fp_blockdiag_scaled_v5

NSEQ   = int(os.environ.get("NSEQ", 2))
SEQLEN = int(os.environ.get("SEQLEN", 512))
WTOK   = int(os.environ.get("WTOK", 256))
BS     = 16
E_B, M_B, E_S, M_S = 2, 1, 4, 3          # E2M1 weights, E4M3 scale (pipeline defaults)
DEV    = "cuda"
TARGET_IDX = [0, 10, 20, 30]
SUFFIXES   = ("q_proj", "fc1")
E4M3_MIN_NORMAL = 2.0 ** -6              # 0.015625 — below this, E4M3 goes subnormal

E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.]) / 6.0


def _next_pow2(n):
    return 1 << (n - 1).bit_length()


def _signs(name, P, device):
    seed = int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randint(0, 2, (P,), generator=g).float() * 2 - 1).to(device)


def fwht(a):
    P = a.shape[-1]
    flat = a.clone().view(-1, P)
    h = 1
    while h < P:
        flat = flat.view(flat.shape[0], -1, 2 * h)
        x = flat[:, :, :h].clone(); y = flat[:, :, h:2 * h].clone()
        flat[:, :, :h] = x + y; flat[:, :, h:2 * h] = x - y
        flat = flat.view(flat.shape[0], P)
        h *= 2
    return (flat / math.sqrt(P)).view(*a.shape)


def rotate(z, D, P):
    if z.shape[-1] < P:
        z = F.pad(z, (0, P - z.shape[-1]))
    return fwht(z * D)


def hessian_blocks(xr, bs):
    n = xr.shape[0]
    H = (xr.T @ xr) / n                  # same as compute_hessian_blocks
    P = xr.shape[1]
    return [H[i:i + bs, i:i + bs].clone() for i in range(0, P, bs)]


def simple_e2m1_q(W, bs=16, clip=2.5):
    N, P = W.shape
    pad = (bs - P % bs) % bs
    if pad:
        W = F.pad(W, (0, pad))
    Wb    = W.view(N, -1, bs)
    scale = (Wb.pow(2).mean(-1, keepdim=True).sqrt() * clip).clamp_min(1e-8)
    x     = (Wb / scale).clamp(-1, 1)
    lv    = E2M1.to(W.device)
    idx   = (x.abs().unsqueeze(-1) - lv).abs().argmin(-1)
    return (lv[idx] * x.sign() * scale).view(N, -1)[:, :P]


def is_target(name):
    parts = name.split(".")
    if parts[-1] not in SUFFIXES:
        return False
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1]) in TARGET_IDX
    return False


def run(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    # Cap the model's GPU footprint so the v5 analysis has headroom on the L4 —
    # device_map's `del model` doesn't reliably free GPU (accelerate holds refs),
    # so we leave ~10GB free regardless instead of relying on the free.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True,
        device_map="auto", max_memory={0: "12GiB", "cpu": "60GiB"}).eval()
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt",
              add_special_tokens=False).input_ids[0]

    cap = {}           # name -> {"x": [chunks on cpu], "W": tensor on cpu}
    handles = []

    def mk(name):
        def hook(mod, args):
            W = mod.weight
            if W.device.type == "meta":
                return
            x = args[0].detach().reshape(-1, args[0].shape[-1]).float().cpu()
            d = cap.setdefault(name, {"x": [], "W": None})
            d["x"].append(x)
            if d["W"] is None:
                d["W"] = W.detach().float().cpu()
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and is_target(name):
            handles.append(mod.register_forward_pre_hook(mk(name)))

    dev0 = model.get_input_embeddings().weight.device
    n = ids.size(0) // SEQLEN
    step = max(n // max(NSEQ, 1), 1)
    with torch.inference_mode():
        for i in range(NSEQ):
            s = (i * step) * SEQLEN
            model(ids[s:s + SEQLEN].unsqueeze(0).to(dev0), use_cache=False)
    for h in handles:
        h.remove()
    hid = getattr(model.config, "hidden_size", None)
    P = _next_pow2(hid)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n===== {model_id}  hidden={hid} -> pad {P} "
          f"({(1 - hid / P) * 100:.0f}% zero-pad) =====")
    print(f"  {'layer':16} {'v5 relerr':>10} {'simple relerr':>14} "
          f"{'a_min':>10} {'a_max':>10} {'a<subnrm%':>10} {'bias set':>14}")
    for name in sorted(cap):
        d = cap[name]
        if d["W"] is None or not d["x"]:
            continue
        D  = _signs(name, P, DEV)
        Wr = rotate(d["W"].to(DEV), D, P)                       # [N, P]
        xr = rotate(torch.cat(d["x"], 0)[:WTOK].to(DEV), D, P)  # [tok, P]
        Hb = hessian_blocks(xr, BS)

        tmp = nn.Linear(P, Wr.shape[0], bias=False).to(DEV)
        tmp.weight.data = Wr.clone()
        res = reconstruct_layer_fp_blockdiag_scaled_v5(
            tmp, Hb, BS, E_B, M_B, E_S, M_S, DEV)
        Wq_v5 = res["weight_q"].to(DEV)
        Wq_s  = simple_e2m1_q(Wr)

        ref = xr @ Wr.T
        rn  = ref.norm().clamp_min(1e-9)
        e_v5 = ((xr @ Wq_v5.T - ref).norm() / rn).item()
        e_s  = ((xr @ Wq_s.T  - ref).norm() / rn).item()

        a = res["alpha"].float().flatten()
        sub = (a < E4M3_MIN_NORMAL).float().mean().item() * 100
        bset = sorted(set(int(v) for v in res["bias"].flatten().tolist()))
        short = ".".join(name.split(".")[-3:])[:16]
        print(f"  {short:16} {e_v5:10.3f} {e_s:14.3f} "
              f"{a.min():10.2e} {a.max():10.2e} {sub:10.1f} {str(bset):>14}")

        del D, Wr, xr, Hb, tmp, res, Wq_v5, Wq_s, ref
        torch.cuda.empty_cache()


if __name__ == "__main__":
    for m in (sys.argv[1:] or ["facebook/opt-6.7b", "facebook/opt-13b"]):
        try:
            run(m)
        except Exception:
            import traceback
            traceback.print_exc()
