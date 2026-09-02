"""
codebook_per_layer.py -- is the constrained-optimal codebook UNIVERSAL, or does
each layer want its own? (Answers the GANQ per-layer-LUT concern.)

GANQ optimizes an arbitrary LUT PER LAYER/CHANNEL. Our pitch is a SINGLE
fixed-point-lattice codebook reused everywhere (no per-layer LUT storage). That
only holds if the per-layer optima cluster near a global one. This script tests
it directly on WEIGHTS (W4A16 -- the regime the codebook-quant baselines live
in): for each nn.Linear it Hadamard-rotates and per-block-absmax-normalizes the
weights, solves the exact 1/16-grid-constrained optimum for that layer's
magnitude distribution, and compares the per-layer optima against the global
optimum (solved on the pooled distribution).

Objective is normalized-domain MSE (fast, exact by enumeration). This is the
"narrow with MSE" stage; the PPL confirmation on the shortlisted codebooks is a
separate, cheaper step (see codebook_suite.py).

Usage: python3 codebook_per_layer.py --model facebook/opt-1.3b
"""
import argparse, itertools, math
from collections import Counter
import numpy as np
import torch, torch.nn as nn
from scipy.linalg import hadamard as scipy_hadamard

BLOCK = 32
GRID = 16                      # 1/16 lattice (Q1.4)
GF4_ROUND = (0, 1, 3, 5, 6, 8, 11, 16)
GLOBAL_OPT = (0, 2, 4, 6, 8, 10, 13, 16)   # the global activation optimum, for reference


# ---- fast exact solver over an empirical magnitude sample --------------------
class Solver:
    def __init__(self, mags):
        m = np.sort(np.asarray(mags, dtype=np.float64))
        w = np.full(m.shape, 1.0 / m.shape[0])
        self.ms = m
        self.cumP = np.concatenate([[0.0], np.cumsum(w)])
        self.cumM = np.concatenate([[0.0], np.cumsum(w * m)])
        self.cumM2 = np.concatenate([[0.0], np.cumsum(w * m * m)])

    def _cum(self, x):
        i = np.searchsorted(self.ms, x, side="right")
        return self.cumP[i], self.cumM[i], self.cumM2[i]

    def distortion(self, ks):
        lv = np.array(ks, dtype=np.float64) / GRID
        mids = (lv[:-1] + lv[1:]) / 2.0
        b = np.concatenate([[0.0], mids, [1.0 + 1e-9]])
        d = 0.0
        for t, l in enumerate(lv):
            P0, M0, M20 = self._cum(b[t]); P1, M1, M21 = self._cum(b[t + 1])
            d += (M21 - M20) - 2 * l * (M1 - M0) + l * l * (P1 - P0)
        return d

    def solve(self):
        best, bd = None, math.inf
        for combo in itertools.combinations(range(1, GRID), 6):
            ks = (0,) + combo + (GRID,)
            d = self.distortion(ks)
            if d < bd:
                bd, best = d, ks
        return best, bd


# ---- blockwise randomized Hadamard on the last dim ---------------------------
def rotate(x, signs, H):
    F_ = x.shape[-1]
    xr = (x * signs).reshape(*x.shape[:-1], F_ // BLOCK, BLOCK) @ H
    return xr.reshape(*x.shape[:-1], F_)


def layer_mags(W, gen, H, cap=200000):
    """Hadamard-rotate columns, per-block absmax-normalize, return |w_norm|."""
    n_in = W.shape[1]
    signs = (torch.randint(0, 2, (n_in,), generator=gen).float() * 2 - 1)
    Wr = rotate(W.float(), signs, H)                      # [out, in]
    b = Wr.reshape(-1, BLOCK)
    amax = b.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    mags = (b / amax).abs().reshape(-1).numpy()           # in [0,1]
    if mags.shape[0] > cap:
        mags = np.random.default_rng(0).choice(mags, cap, replace=False)
    return mags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-1.3b")
    ap.add_argument("--max-layers", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM
    print(f"Loading {args.model} (weights only, CPU) ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    H = torch.tensor(scipy_hadamard(BLOCK).astype(np.float32) / math.sqrt(BLOCK))
    gen = torch.Generator().manual_seed(0)

    layers = [(n, m) for n, m in model.named_modules()
              if isinstance(m, nn.Linear) and m.in_features % BLOCK == 0 and "lm_head" not in n]
    if args.max_layers:
        layers = layers[:args.max_layers]
    print(f"{len(layers)} linear layers\n")

    pooled = []
    per_layer_opt = []
    for n, m in layers:
        mags = layer_mags(m.weight.data, gen, H)
        pooled.append(mags[::4])                          # thinned for the global pool
        opt, d = Solver(mags).solve()
        per_layer_opt.append(opt)

    pooled = np.concatenate(pooled)
    g_opt, _ = Solver(pooled).solve()
    S = Solver(pooled)
    d_global = S.distortion(g_opt)
    d_gf4r = S.distortion(GF4_ROUND)
    d_actopt = S.distortion(GLOBAL_OPT)

    print("=== per-layer constrained optima (1/16 grid), interior levels ===")
    cnt = Counter(per_layer_opt)
    for ks, c in cnt.most_common():
        star = "  <-- == weight-global" if ks == g_opt else ""
        print(f"  {c:3d}/{len(layers)} layers: {list(ks)}{star}")

    n_eq = sum(1 for ks in per_layer_opt if ks == g_opt)
    # per-interior-level spread
    inter = np.array([ks[1:7] for ks in per_layer_opt])
    spread = [f"{inter[:,i].min()}-{inter[:,i].max()} (mode {Counter(inter[:,i]).most_common(1)[0][0]})"
              for i in range(6)]
    print(f"\nweight-GLOBAL optimum:      {list(g_opt)}   distortion {d_global:.3e}")
    print(f"activation-global optimum:  {list(GLOBAL_OPT)}   distortion {d_actopt:.3e} on weights")
    print(f"GF4 rounded {list(GF4_ROUND)}: distortion {d_gf4r:.3e} on weights")
    print(f"\nlayers whose own optimum == weight-global: {n_eq}/{len(layers)} "
          f"({100*n_eq/len(layers):.0f}%)")
    print(f"interior-level spread across layers: {spread}")
    print("\nInterpretation: tight clustering => one fixed-point lattice codebook "
          "suffices (no per-layer LUT, unlike GANQ).")


if __name__ == "__main__":
    main()
