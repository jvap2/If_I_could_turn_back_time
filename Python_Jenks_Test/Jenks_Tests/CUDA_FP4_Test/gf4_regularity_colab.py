"""
gf4_regularity_colab.py -- GF4 codebook-regularity + scale study for LARGER
models, Colab-ready.

Extends gf4_regularity_sweep.py in two ways:
  (1) Scales to bigger models (device_map="auto", so a 7B fits across GPU+CPU on
      a Colab T4/A100).
  (2) Adds the SCALE axis, tying the study to GF4's actual operand form.

WHY THE SCALE MATTERS (the GF4 connection). A GF4 operand is
    x_i = scale_b * level(code_i),   scale_b = block_rms * clip_ratio,
so the codebook is applied AFTER per-block RMS normalization: it quantizes the
*shape* of the distribution within a block, while the per-block scale carries the
*dynamic range*. This is exactly why the codebook can be coarsely regularized
(Q1.4, ~1/16 relative resolution) with almost no accuracy loss -- the scale, not
the codebook, is responsible for magnitude. The real deployment stores that scale
at low precision too (E4M3 for weights, block-RMS for activations), so this
script sweeps codebook precision B *and* scale precision {fp16, e4m3} jointly, to
confirm the Q1.4 result holds under GF4's genuine low-bit block scale.

Colab setup:
    !pip install -q transformers datasets accelerate scipy
    !python gf4_regularity_colab.py --model meta-llama/Llama-2-7b-hf --eval-windows 10
(or --model facebook/opt-1.3b for a quick pass). Use --load-8bit only to FIT a
model; note it quantizes the base weights, a separate axis from this study.
"""
import argparse, csv, math, os
import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import hadamard as scipy_hadamard

GF4_LEVEL = np.array([0.0, 0.0796082, 0.1737177, 0.2828685,
                      0.3952704, 0.5250730, 0.6961928, 1.0], dtype=np.float64)
BLOCK = 32
CLIP_RATIO = 2.5

# ---------------------------------------------------------------------------
# codebook family
# ---------------------------------------------------------------------------
def snap_q1b(levels, B):
    g = np.round(levels * (2 ** B)) / (2 ** B)
    g[0], g[-1] = 0.0, 1.0
    return g

def snap_pow2(levels):
    out = levels.copy()
    for i, v in enumerate(levels):
        out[i] = (2.0 ** round(math.log2(v))) if v > 0 else 0.0
    out[-1] = 1.0
    return out

def build_codebooks(which):
    b = {"exact": GF4_LEVEL.copy()}
    for B in (6, 5, 4, 3):
        b[f"q1b{B}"] = snap_q1b(GF4_LEVEL, B)
    b["pow2"] = snap_pow2(GF4_LEVEL)
    if which != "all":
        b = {k: v for k, v in b.items() if k in which.split(",")}
    return b

# ---------------------------------------------------------------------------
# scale precision (the GF4 block scale)
# ---------------------------------------------------------------------------
def quant_e4m3(scale):
    """Fake-quantize a positive per-block scale to E4M3 (4 exp, 3 mantissa,
    max 448), GF4's actual weight-scale format."""
    s = scale.clamp(min=2.0 ** -9, max=448.0)
    e = torch.floor(torch.log2(s))
    m = torch.round((s / torch.exp2(e) - 1.0) * 8.0) / 8.0     # 3 mantissa bits
    return (1.0 + m) * torch.exp2(e)

_SCALE_MODE = "fp16"     # or "e4m3"

# ---- NVFP4 weight quantization (E2M1 elements + our E4M3 block scale) --------
# The REAL scheme is W4A4: weights in NVFP4 (native E2M1 grid), activations in
# our codebook. Weights are quantized per-block with our microscaling method
# (per-block absmax -> E4M3 scale), so the codebook comparison is measured on
# top of quantized weights, not fp16 weights.
E2M1_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
WBLOCK = 16                 # NVFP4-native microscaling block
QUANT_WEIGHTS_NVFP4 = True  # W4A4 when True; A-only (fp16 weights) when False
_E2M1_LV = _E2M1_THR = None

def _ensure_e2m1(device):
    global _E2M1_LV, _E2M1_THR
    if _E2M1_LV is None or _E2M1_LV.device != device:
        _E2M1_LV = torch.tensor(E2M1_MAG, device=device)
        _E2M1_THR = torch.tensor((E2M1_MAG[:-1] + E2M1_MAG[1:]) / 2.0, device=device)

def nvfp4_weight_quant(W, block=WBLOCK):
    """Fake-quantize weights to NVFP4: per-block absmax -> E4M3 scale (our scale
    method), elements to the nearest E2M1 magnitude. W: [out, in]."""
    dev = W.device; _ensure_e2m1(dev)
    shp = W.shape
    if shp[1] % block != 0:
        block = 32 if shp[1] % 32 == 0 else BLOCK
    wb = W.reshape(-1, block).float()
    amax = wb.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = quant_e4m3(amax / 6.0)     # top E2M1 magnitude is 6
    wn = wb / scale
    idx = torch.bucketize(wn.abs().clamp(0.0, 6.0), _E2M1_THR)
    q = _E2M1_LV[idx] * torch.sign(wn) * scale
    return q.reshape(shp).to(W.dtype)

# ---------------------------------------------------------------------------
# active codebook (torch tensors)
# ---------------------------------------------------------------------------
_LEVELS = _THR = None
def set_codebook(levels_np, device):
    global _LEVELS, _THR
    lv = np.sort(levels_np).astype(np.float32)
    thr = np.array([(lv[i] + lv[i + 1]) / 2 for i in range(len(lv) - 1)], dtype=np.float32)
    _LEVELS = torch.tensor(lv, device=device)
    _THR = torch.tensor(thr, device=device)

# ---------------------------------------------------------------------------
# blockwise randomized Hadamard
# ---------------------------------------------------------------------------
_H = None
def hmat(device, dtype):
    global _H
    if _H is None or _H.device != device:
        _H = torch.tensor(scipy_hadamard(BLOCK).astype(np.float32) / math.sqrt(BLOCK), device=device)
    return _H.to(dtype)

def rotate(x, signs):
    F_ = x.shape[-1]
    H = hmat(x.device, x.dtype)
    xr = (x * signs).reshape(*x.shape[:-1], F_ // BLOCK, BLOCK) @ H
    return xr.reshape(*x.shape[:-1], F_)

# ---------------------------------------------------------------------------
# GF4 activation quant with active codebook + scale mode
# ---------------------------------------------------------------------------
def gf4_quant(x):
    shp = x.shape
    xb = x.reshape(-1, BLOCK).float()
    rms = xb.pow(2).mean(-1, keepdim=True).add(1e-12).sqrt()
    scale = rms * CLIP_RATIO
    if _SCALE_MODE == "e4m3":
        scale = quant_e4m3(scale)
    xn = xb / scale
    mag = xn.abs().clamp(0.0, 1.0)
    thr = _THR.to(xb.device); lv = _LEVELS.to(xb.device)   # device-safe (multi-GPU)
    idx = torch.bucketize(mag, thr)
    deq = lv[idx] * torch.sign(xn) * scale
    return deq.reshape(shp).to(x.dtype)

# ---------------------------------------------------------------------------
# Constrained-optimal codebook design ON THE MODEL'S OWN activations.
# _CALIB: when set to a list, the pre-hook records normalized activation
# magnitudes (m = clip(|Hx|/(rms*clip), 0, 1)) instead of quantizing, so we can
# SOLVE the 1/16-lattice optimum on real data rather than assuming N(0,1).
# ---------------------------------------------------------------------------
import itertools
_CALIB = None

def solve_constrained_codebook(mags, grid=16, popcount=None):
    """Exact 1/16-lattice MSE optimum on an empirical magnitude sample. Endpoints
    pinned (0 and grid); interior 6 levels enumerated. Returns (levels, ks)."""
    m = np.sort(np.asarray(mags, dtype=np.float64))
    w = np.full(m.shape, 1.0 / len(m))
    cP = np.concatenate([[0.0], np.cumsum(w)])
    cM = np.concatenate([[0.0], np.cumsum(w * m)])
    cM2 = np.concatenate([[0.0], np.cumsum(w * m * m)])
    def cum(x):
        i = np.searchsorted(m, x, side="right"); return cP[i], cM[i], cM2[i]
    def dist(ks):
        lv = np.array(ks, dtype=np.float64) / grid
        b = np.concatenate([[0.0], (lv[:-1] + lv[1:]) / 2, [1.0 + 1e-9]])
        d = 0.0
        for t, l in enumerate(lv):
            P0, M0, M20 = cum(b[t]); P1, M1, M21 = cum(b[t + 1])
            d += (M21 - M20) - 2 * l * (M1 - M0) + l * l * (P1 - P0)
        return d
    pool = [k for k in range(1, grid) if popcount is None or bin(k).count("1") <= popcount]
    best, bd = None, float("inf")
    for c in itertools.combinations(pool, 6):
        ks = (0,) + c + (grid,); d = dist(ks)
        if d < bd:
            bd, best = d, ks
    return np.array(best, dtype=np.float64) / grid, best

def collect_mags(model, windows, n_windows=2, max_samples=1_500_000):
    """Collect this model's post-Hadamard normalized activation magnitudes over a
    few calibration windows (pre-hook records them, forward is otherwise clean)."""
    global _CALIB
    _CALIB = []
    perplexity(model, windows[:n_windows])          # forward only; pre-hook records mags
    mags = np.concatenate(_CALIB); _CALIB = None
    if mags.shape[0] > max_samples:
        mags = np.random.default_rng(0).choice(mags, max_samples, replace=False)
    return mags

def calibrate_and_solve(model, windows, popcount=None, **kw):
    return solve_constrained_codebook(collect_mags(model, windows, **kw), popcount=popcount)

# ---------------------------------------------------------------------------
# hooks: rotate weights once, quantize rotated input per forward
# ---------------------------------------------------------------------------
def install_hooks(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and m.in_features % BLOCK == 0 and "lm_head" not in name:
            dev = m.weight.device
            signs = (torch.randint(0, 2, (m.in_features,), generator=g).float() * 2 - 1).to(dev)
            m.register_buffer("_gf4_signs", signs, persistent=False)
            with torch.no_grad():
                Wr = rotate(m.weight.data.float(), signs)
                if QUANT_WEIGHTS_NVFP4:          # W4A4: weights -> NVFP4/E2M1
                    Wr = nvfp4_weight_quant(Wr)
                m.weight.data.copy_(Wr.to(m.weight.dtype))
            def pre_hook(mod, inp):
                x = inp[0]
                signs = mod._gf4_signs.to(x.device)        # device-safe (multi-GPU)
                xr = rotate(x.float(), signs)
                if _CALIB is not None:                     # calibration: record mags, pass through
                    xb = xr.reshape(-1, BLOCK)
                    rms = xb.pow(2).mean(-1, keepdim=True).add(1e-12).sqrt()
                    mg = (xb / (rms * CLIP_RATIO)).abs().clamp(0.0, 1.0).reshape(-1)
                    k = min(4096, mg.numel())
                    sel = torch.randint(0, mg.numel(), (k,), device=mg.device)
                    _CALIB.append(mg[sel].detach().float().cpu().numpy())
                    return inp
                return (gf4_quant(xr).to(x.dtype),) + inp[1:]
            m.register_forward_pre_hook(pre_hook)
            n += 1
    return n

# ---------------------------------------------------------------------------
def load_windows(tok, seqlen, n):
    from datasets import load_dataset
    ds = None                                    # parquet-hosted mirror -> no loading script
    for repo in ("Salesforce/wikitext", "wikitext", "EleutherAI/wikitext_document_level"):
        try:
            ds = load_dataset(repo, "wikitext-2-raw-v1", split="test"); break
        except Exception:
            ds = None
    if ds is None:
        raise RuntimeError("could not load wikitext-2-raw-v1 (try: pip install -U datasets)")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    k = min(n, ids.shape[0] // seqlen)
    return [ids[i * seqlen:(i + 1) * seqlen] for i in range(k)]

@torch.no_grad()
def perplexity(model, windows):
    dev = next(model.parameters()).device
    nll = ntok = 0
    for w in windows:
        w = w.to(dev)
        out = model(w.unsqueeze(0), labels=w.unsqueeze(0))
        nll += out.loss.item() * (w.numel() - 1); ntok += w.numel() - 1
    return math.exp(nll / ntok)

def main():
    global _SCALE_MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-1.3b")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--eval-windows", type=int, default=10)
    ap.add_argument("--codebooks", default="all", help="comma list or 'all'")
    ap.add_argument("--scale-modes", default="fp16,e4m3")
    ap.add_argument("--load-8bit", action="store_true", help="only to FIT big models (quantizes base weights)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="gf4_regularity_colab.csv")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    kw = dict(device_map="auto", torch_dtype=torch.float16)
    if args.load_8bit:
        kw.update(load_in_8bit=True); kw.pop("torch_dtype")
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).eval()
    windows = load_windows(tok, args.seqlen, args.eval_windows)
    print(f"{len(windows)} windows x {args.seqlen} tok\n")

    ppl_fp16 = perplexity(model, windows)
    print(f"baseline fp16 PPL: {ppl_fp16:.4f}\n")
    nh = install_hooks(model, args.seed)
    print(f"hooked {nh} linears (A-only GF4, blockwise Hadamard)\n")

    books = build_codebooks(args.codebooks)
    dev = next(model.parameters()).device
    rows = []; exact = {}
    for sm in args.scale_modes.split(","):
        _SCALE_MODE = sm
        for name, levels in books.items():
            set_codebook(levels, dev)
            ppl = perplexity(model, windows)
            if name == "exact":
                exact[sm] = ppl
            rows.append((sm, name, ppl))
            de = ppl - exact.get(sm, float("nan"))
            print(f"  scale={sm:5s} {name:6s} PPL {ppl:8.4f}  dPPL(fp16) {ppl-ppl_fp16:+7.3f}  dPPL(exact) {de:+7.3f}")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scale_mode", "codebook", "ppl", "dppl_fp16", "dppl_exact"])
        for sm, name, ppl in rows:
            w.writerow([sm, name, f"{ppl:.4f}", f"{ppl-ppl_fp16:.4f}", f"{ppl-exact.get(sm,float('nan')):.4f}"])
    print(f"\nbaseline fp16 {ppl_fp16:.4f}; wrote {args.out}")

if __name__ == "__main__":
    main()
