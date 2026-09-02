"""
constrained_codebook.py -- hardware-constrained distribution-aware codebook design.

Formalizes GF4 (and its Q1.4 fixed-point form) as solutions of

    C* = argmin_C  E_{w~p}[ (w - Q_C(w))^2 ]
    s.t.  C subset (1/2^m) Z,           (representable on an m-bit fractional grid)
          popcount(k_i) <= r,           (few set bits -> shift-and-add decode)
          |k_i| <= K = 2^m,             (bounded magnitude, top level = 1.0)

i.e. we do NOT round a Gaussian codebook; we minimize quantization distortion for
the operating distribution p subject to hardware-friendliness constraints. The
symmetric 8-magnitude codebook {0 = l_0 < ... < l_7 = 1} is applied to the
per-block-RMS-normalized magnitude m = clip(|z|/alpha, 0, 1), z ~ N(0,1) (alpha =
clip ratio), matching GF4's operand form (the scale carries range, the codebook
shape). With the grid fixed at 1/2^m and endpoints pinned (l_0 = 0, l_7 = 1), the
interior 6 levels are integers in {1..2^m-1}: the search is small enough to solve
EXACTLY by enumeration, so the reported codebook is the true constrained optimum,
not a heuristic rounding.

Outputs, for each constraint set: the optimal integer codebook, its distortion,
and the distortion of (a) the unconstrained Gaussian-quantile codebook (GF4) and
(b) GF4 naively rounded to the grid -- so the paper can show the optimizer
matches or beats rounding, and that GF4/Q1.4 are points in one framework.
"""
import itertools, math
import numpy as np
from scipy.stats import norm

# GF4's own levels: equal-mass |N(0,1)| quantiles normalized to max 1 (the
# unconstrained, continuous distribution-aware codebook).
GF4_LEVEL = np.array([0.0, 0.0796082, 0.1737177, 0.2828685,
                      0.3952704, 0.5250730, 0.6961928, 1.0])
ALPHA = 2.5   # clip ratio; source magnitude m = clip(|z|/alpha, 0, 1)

# --- operating distribution p on the normalized magnitude m = clip(|z|/a,0,1) --
# Build it once, sorted, with cumulative sums of {weight, w*m, w*m^2} so any
# interval's distortion for a level l is O(1): sum_[x0,x1] (m-l)^2 w
#   = (M2) - 2 l (M) + l^2 (P),  each term a cumulative-sum difference.
_Z = np.linspace(-8, 8, 80001)
_W = norm.pdf(_Z); _W /= _W.sum()
_M = np.clip(np.abs(_Z) / ALPHA, 0.0, 1.0)
_order = np.argsort(_M, kind="stable")
_MS = _M[_order]
_WS = _W[_order]
_cumP = np.concatenate([[0.0], np.cumsum(_WS)])
_cumM = np.concatenate([[0.0], np.cumsum(_WS * _MS)])
_cumM2 = np.concatenate([[0.0], np.cumsum(_WS * _MS * _MS)])


def _cum_at(x):
    i = np.searchsorted(_MS, x, side="right")
    return _cumP[i], _cumM[i], _cumM2[i]


def distortion(levels):
    """E[(m - Q(m))^2], nearest-level (midpoint) reconstruction. O(#levels)."""
    lv = np.sort(np.asarray(levels, dtype=np.float64))
    mids = (lv[:-1] + lv[1:]) / 2.0
    bounds = np.concatenate([[0.0], mids, [1.0 + 1e-9]])
    d = 0.0
    for t, l in enumerate(lv):
        P0, M0, M20 = _cum_at(bounds[t]); P1, M1, M21 = _cum_at(bounds[t + 1])
        P, M, M2 = P1 - P0, M1 - M0, M21 - M20
        d += M2 - 2.0 * l * M + l * l * P
    return float(d)


def popcount(n):
    return bin(int(n)).count("1")


def solve(m, r=None, verbose=False):
    """Exact constrained optimum: interior levels are integers in [1, 2^m-1] on
    the 1/2^m grid; endpoints pinned to 0 and 2^m. Optional popcount(k)<=r."""
    K = 2 ** m
    pool = [k for k in range(1, K) if (r is None or popcount(k) <= r)]
    best, best_d = None, math.inf
    for combo in itertools.combinations(pool, 6):
        ks = (0,) + combo + (K,)
        d = distortion(np.array(ks) / K)
        if d < best_d:
            best_d, best = d, ks
    return best, best_d, K


def main():
    print(f"source: m = clip(|z|/{ALPHA}, 0, 1), z ~ N(0,1)\n")
    d_gf4 = distortion(GF4_LEVEL)
    print(f"unconstrained GF4 (float quantiles):   distortion {d_gf4:.6e}   "
          f"levels={[f'{v:.4f}' for v in GF4_LEVEL]}")

    for m in (5, 4, 3):
        K = 2 ** m
        gf4_round = np.round(GF4_LEVEL * K).astype(int)
        gf4_round[0], gf4_round[-1] = 0, K
        d_round = distortion(gf4_round / K)
        opt, d_opt, _ = solve(m)
        tag = "  <-- Q1.4 (paper codebook)" if m == 4 else ""
        print(f"\n--- grid 1/{K} (Q1.{m}) ---")
        print(f"  GF4 rounded to grid:  distortion {d_round:.6e}   k={list(gf4_round)}")
        print(f"  CONSTRAINED OPTIMUM:  distortion {d_opt:.6e}   k={list(opt)}"
              f"   (1/{K}){list(opt)}{tag}")
        print(f"  optimum vs rounding:  {100*(d_round-d_opt)/d_round:+.2f}% distortion")
        # popcount-constrained (shift-and-add friendly): <=2 set bits
        opt2, d2, _ = solve(m, r=2)
        if opt2:
            print(f"  + popcount<=2 (shift-add): distortion {d2:.6e}   k={list(opt2)}"
                  f"   (+{100*(d2-d_opt)/d_opt:.1f}% vs unconstrained-grid optimum)")


if __name__ == "__main__":
    main()
