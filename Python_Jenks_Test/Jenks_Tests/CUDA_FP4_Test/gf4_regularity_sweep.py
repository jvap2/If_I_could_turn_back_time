"""
gf4_regularity_sweep.py -- Accuracy vs. codebook-regularity for GF4.

The GF4 hardware decode costs ~1.5% of the PE precisely because its
Gaussian-quantile magnitudes are scattered-bit constants; a more "regular"
(power-of-two / low-fractional-bit) codebook would synthesize to a cheaper,
more GPU/shift-friendly decode (cf. E2M1's ~0.6%). This script measures what
that regularity COSTS in accuracy, so the paper can plot PPL degradation vs.
decode-area savings and defend the choice of the accuracy-optimal codebook.

Method (faithful GF4 regime): each targeted nn.Linear's input is rotated by a
fixed randomized blockwise Hadamard (block=32, random signs) -- the same
rotation applied once to the weight columns, so the layer output is preserved
-- then the rotated activation is GF4-quantized with the codebook under test.
Weights stay fp16 (A-only), isolating the ACTIVATION codebook's effect, which
is exactly the knob the regularity dial turns. WikiText-2 perplexity is the
metric. The rotation is codebook-independent, so we rotate once and sweep
codebooks cheaply.

Codebook family (all keep 0 and 1.0 as endpoints):
  exact  : the real GF4 levels (Q1.~ full precision).
  q1b<B> : GF4 levels snapped to a Q1.B grid (round to nearest k/2^B), B=8..1.
  pow2   : each non-zero level snapped to the nearest power of two (decode = shift).

Usage:
  python3 gf4_regularity_sweep.py --model facebook/opt-125m --eval-windows 20
"""
import argparse, csv, math, time, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import hadamard as scipy_hadamard

# ---- the real GF4 activation codebook -------------------------------------
GF4_LEVEL = np.array([0.0, 0.0796082, 0.1737177, 0.2828685,
                      0.3952704, 0.5250730, 0.6961928, 1.0], dtype=np.float64)
BLOCK = 32
CLIP_RATIO = 2.5

# ---------------------------------------------------------------------------
# Codebook family
# ---------------------------------------------------------------------------
def _thresholds(levels):
    """Midpoints between adjacent (sorted) levels -> 7 interior boundaries."""
    lv = np.sort(np.unique(levels))
    # if snapping collapsed levels, pad back to 8 by keeping duplicates ordered
    if lv.shape[0] < 8:
        lv = np.sort(levels)   # keep 8 slots even with ties
    return np.array([(lv[i] + lv[i + 1]) / 2 for i in range(len(lv) - 1)], dtype=np.float32), lv


def snap_q1b(levels, B):
    """Round each level to the nearest k/2^B; clamp endpoints 0 and 1."""
    g = np.round(levels * (2 ** B)) / (2 ** B)
    g[0] = 0.0
    g[-1] = 1.0
    return g.astype(np.float64)


def snap_pow2(levels):
    """Snap each non-zero level to the nearest power of two (decode -> shift)."""
    out = levels.copy()
    for i, v in enumerate(levels):
        if v > 0:
            out[i] = 2.0 ** round(math.log2(v))
    out[-1] = 1.0
    return out.astype(np.float64)


def build_codebooks():
    books = {}
    books["exact"] = GF4_LEVEL.copy()
    for B in (8, 7, 6, 5, 4, 3, 2, 1):
        books[f"q1b{B}"] = snap_q1b(GF4_LEVEL, B)
    books["pow2"] = snap_pow2(GF4_LEVEL)
    return books


# active codebook, set by the sweep loop (torch tensors on device)
_LEVELS = None        # [8] fp32
_THR = None           # [<=7] fp32, sorted


def set_codebook(levels_np, device):
    global _LEVELS, _THR
    lv = np.sort(levels_np).astype(np.float32)          # levels used for reconstruction (sorted)
    thr = np.array([(lv[i] + lv[i + 1]) / 2 for i in range(len(lv) - 1)], dtype=np.float32)
    _LEVELS = torch.tensor(lv, device=device)
    _THR = torch.tensor(thr, device=device)


# ---------------------------------------------------------------------------
# Blockwise randomized Hadamard (orthonormal, block=32)
# ---------------------------------------------------------------------------
_H32 = None
def _hmat(device, dtype):
    global _H32
    if _H32 is None or _H32.device != device:
        H = scipy_hadamard(BLOCK).astype(np.float32) / math.sqrt(BLOCK)   # orthonormal, symmetric
        _H32 = torch.tensor(H, device=device, dtype=torch.float32)
    return _H32.to(dtype)


def rotate(x, signs):
    """Apply the fixed randomized blockwise Hadamard along the last dim.
    x: [..., F], signs: [F] +-1. Returns same shape."""
    F_ = x.shape[-1]
    H = _hmat(x.device, x.dtype)
    xr = (x * signs).reshape(*x.shape[:-1], F_ // BLOCK, BLOCK)
    xr = xr @ H                                         # per-block Hadamard
    return xr.reshape(*x.shape[:-1], F_)


# ---------------------------------------------------------------------------
# GF4 activation quantization with the active codebook
# ---------------------------------------------------------------------------
def gf4_quant(x):
    shp = x.shape
    xb = x.reshape(-1, BLOCK).float()
    rms = xb.pow(2).mean(-1, keepdim=True).add(1e-12).sqrt()
    scale = rms * CLIP_RATIO
    xn = xb / scale
    mag = xn.abs().clamp(0.0, 1.0)
    idx = torch.bucketize(mag, _THR)                    # 0..len(levels)-1
    deq = _LEVELS[idx] * torch.sign(xn) * scale
    return deq.reshape(shp).to(x.dtype)


# ---------------------------------------------------------------------------
# Hooking: rotate weight columns once, quantize rotated input per forward
# ---------------------------------------------------------------------------
def install_hooks(model, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    targets = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and m.in_features % BLOCK == 0 and "lm_head" not in name:
            targets.append((name, m))
    for i, (name, m) in enumerate(targets):
        dev = m.weight.device
        signs = (torch.randint(0, 2, (m.in_features,), generator=g).float() * 2 - 1).to(dev)
        m.register_buffer("_gf4_signs", signs, persistent=False)
        with torch.no_grad():
            Wr = rotate(m.weight.data.float(), signs).to(m.weight.dtype)   # W' = W H (columns)
            m.weight.data.copy_(Wr)

        def pre_hook(mod, inp, _name=name):
            x = inp[0]
            xr = rotate(x.float(), mod._gf4_signs)
            xq = gf4_quant(xr).to(x.dtype)
            return (xq,) + inp[1:]

        m.register_forward_pre_hook(pre_hook)
    return [n for n, _ in targets]


# ---------------------------------------------------------------------------
# WikiText-2 perplexity
# ---------------------------------------------------------------------------
def load_windows(tok, seqlen, n_windows):
    from datasets import load_dataset
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    n = min(n_windows, ids.shape[0] // seqlen)
    return [ids[i * seqlen:(i + 1) * seqlen] for i in range(n)]


@torch.no_grad()
def perplexity(model, windows, device):
    nll, ntok = 0.0, 0
    for w in windows:
        w = w.to(device)
        out = model(w.unsqueeze(0), labels=w.unsqueeze(0))
        nll += out.loss.item() * (w.numel() - 1)
        ntok += (w.numel() - 1)
    return math.exp(nll / ntok)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--eval-windows", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="gf4_regularity_results.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(device).eval()
    windows = load_windows(tok, args.seqlen, args.eval_windows)
    print(f"{len(windows)} eval windows of {args.seqlen} tokens.\n")

    # fp16 baseline (no hooks, unrotated)
    ppl_fp16 = perplexity(model, windows, device)
    print(f"baseline fp16 PPL: {ppl_fp16:.4f}\n")

    # install rotation + A-quant hooks (rotates weights in place, once)
    targets = install_hooks(model, seed=args.seed)
    print(f"hooked {len(targets)} linear layers (A-only GF4, blockwise Hadamard)\n")

    books = build_codebooks()
    rows = []
    # rotation-only sanity: a near-lossless codebook check is 'exact' itself;
    # we report each codebook's PPL and delta vs fp16 and vs exact-GF4.
    ppl_exact = None
    for name, levels in books.items():
        set_codebook(levels, device)
        t0 = time.time()
        ppl = perplexity(model, windows, device)
        if name == "exact":
            ppl_exact = ppl
        rows.append((name, levels, ppl))
        print(f"  {name:7s}  PPL {ppl:8.4f}   dPPL(fp16) {ppl-ppl_fp16:+7.4f}   "
              f"levels=[{', '.join(f'{v:.4f}' for v in np.sort(levels))}]   ({time.time()-t0:.1f}s)")

    # write CSV
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["codebook", "ppl", "dppl_fp16", "dppl_exact", "levels"])
        for name, levels, ppl in rows:
            w.writerow([name, f"{ppl:.4f}", f"{ppl-ppl_fp16:.4f}",
                        f"{ppl-ppl_exact:.4f}", ";".join(f"{v:.6f}" for v in np.sort(levels))])
    print(f"\nbaseline fp16 {ppl_fp16:.4f}; exact-GF4 {ppl_exact:.4f}. wrote {args.out}")


if __name__ == "__main__":
    main()
