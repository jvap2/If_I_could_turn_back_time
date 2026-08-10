"""
GF4_equal_mass_proof.py
=======================================================================
Shows *exactly* in what sense the GF4 levels carry equal probability mass,
and reconciles with GF4_test.py (the "refutation") by identifying the two
DISTINCT quantities that script conflates.

Run:  python GF4_equal_mass_proof.py
Deps: numpy, scipy  (a pure-stdlib Monte-Carlo cross-check is also included
                     so it can be run in the same environment as GF4_test.py)

-----------------------------------------------------------------------
THE TWO CLAIMS (only ONE of them is what "equal probability mass" means)
-----------------------------------------------------------------------
  (A) CONSTRUCTION claim  [TRUE]:
      The GF4 levels are Gaussian quantiles at equally-spaced cumulative
      probability. Equivalently: equal probability mass lies BETWEEN
      consecutive levels. The equal-mass interval boundaries ARE THE
      LEVELS THEMSELVES.

  (B) OCCUPANCY claim      [FALSE, and nobody should claim it]:
      Each code is *used* equally often. This would require the MIDPOINTS
      between levels (the nearest-rounding / Voronoi boundaries) to be the
      equal-mass dividers. They are not.

GF4_test.py measures (B) -- it builds midpoint boundaries via center() and
counts occupancy -- then uses the (expected) inequality to refute (A).
Those are different quantities. This file proves (A) and reproduces (B)
side-by-side so the distinction is explicit.
"""

import numpy as np
from scipy.stats import norm

# --- the GF4 levels (positive half; sign bit handles the negatives) ----------
GF4 = np.array([0., 0.0796082, 0.1737177, 0.2828685,
                0.3952704, 0.5250730, 0.6961928, 1.0])

# The codebook is defined on a NORMALIZED [0,1] axis. To speak of probability
# mass you map it onto a Gaussian. The construction fixes the scale: the top
# level 1.0 is the OFFSET quantile used to build the code (NF4/GF4 offset).
OFFSET = 0.9677083                 # cumulative prob that the top level represents
S      = norm.ppf(OFFSET)          # => 1.0 maps to this many sigma (= 1.848)
print(f"Construction scale: level 1.0  ==  {OFFSET} quantile  ==  {S:.4f} sigma")
print("(This is NOT cherry-picked -- it is the definition of 'top level = 1'.)\n")


# =====================================================================
# PART 1 -- ANALYTIC PROOF of claim (A): equal mass BETWEEN the levels
# =====================================================================
# Map each level to sigma units, take the standard-normal CDF, and look at
# the mass in each interval [level_k, level_{k+1}).
cdf = norm.cdf(GF4 * S)                       # cumulative prob at each level
mass_between = np.diff(cdf)                   # mass in each inter-level band

print("PART 1  ANALYTIC (scipy) -- mass BETWEEN consecutive GF4 levels")
print("  CDF at levels      :", np.round(cdf, 4))
print("  mass between levels :", np.round(mass_between, 4))
print(f"  ideal equal share   : {(OFFSET-0.5)/7:.4f}   (= (0.9677-0.5)/7)")
print(f"  CV of the bands     : {mass_between.std()/mass_between.mean():.3f}"
      "   (~0 => equal probability mass)\n")

# The clinching identity: the levels are ppf() of an equally-spaced CDF grid.
grid = np.linspace(0.5, OFFSET, 8)            # equally spaced cumulative prob
recon = norm.ppf(grid); recon = recon / recon[-1]
print("  GF4 reconstructed as ppf(linspace(0.5, 0.9677, 8)), normalized:")
print("    reconstructed:", np.round(recon, 4))
print("    actual GF4   :", np.round(GF4, 4))
print(f"    max abs diff : {np.abs(recon-GF4).max():.4f}"
      "  (levels 3-7 exact; low levels differ by ~0.01, NF4-style zero region)\n")


# =====================================================================
# PART 2 -- MONTE CARLO in GF4_test.py's own style, but with the
#           CORRECT boundaries (the levels), confirming equal counts.
# =====================================================================
import random
random.seed(0)

def counts_between(boundaries, sd, n):
    """Count |x| falling in each band [boundaries[k], boundaries[k+1])."""
    edges = list(boundaries) + [1e9]
    c = [0] * (len(boundaries))
    for _ in range(n):
        x = abs(random.gauss(0.0, sd))
        for k in range(len(boundaries)):
            if edges[k] <= x < edges[k + 1]:
                c[k] += 1
                break
    return c

# sd chosen so the codebook's top level (1.0) sits at the construction scale:
# 1.0 == S sigma  =>  sd = 1/S.
SD = 1.0 / S
N  = 800_000
mc = counts_between(GF4, SD, N)                       # 7 inter-level bands + clip tail

# Analytic prediction for each band. The MC folds BOTH signs (abs), so a band
# holds 2x its one-sided normal mass. Bands 0..6 are [level_k, level_{k+1});
# band 7 is the clipped tail [1.0, inf).
pred = list(2 * mass_between * N) + [2 * (1 - OFFSET) * N]
print("PART 2  MONTE CARLO (random.gauss) -- counts in [level_k, level_{k+1})")
print(f"  sd = 1/{S:.3f} = {SD:.4f}  so top level 1.0 sits at the 0.9677 quantile")
print("  band            :  " + "  ".join(f"{k}" if k < 7 else "tail" for k in range(8)))
print("  MC counts       :", [f"{c:6d}" for c in mc])
print("  analytic predict:", [f"{int(p):6d}" for p in pred])
print(f"  the 7 inter-level bands ideal to ~{int(2*(OFFSET-0.5)/7*N):,} each"
      " (equal mass => equal counts);")
print("  bands 0-2 differ only by the ~0.01 codebook offset in the zero region,")
print("  and 'tail' is the clipped mass beyond level 1.0 (last ~6.5% folded).\n")


# =====================================================================
# PART 3 -- REPRODUCE GF4_test.py's result and NAME it correctly.
#           center() midpoints measure OCCUPANCY (claim B), not (A).
# =====================================================================
def center(values):
    b = [0.0]
    for i in range(len(values) - 1):
        b.append((values[i] + values[i + 1]) / 2)
    return b

def occupancy(levels, sd, n):
    """Nearest-level rounding: which code does each sample snap to."""
    mids = center(levels)                     # rounding boundaries = midpoints
    edges = mids + [1e9]
    c = [0] * len(levels)
    for _ in range(n):
        x = abs(random.gauss(0.0, sd))
        for k in range(len(levels)):
            if edges[k] <= x < edges[k + 1]:
                c[k] += 1
                break
    return c

occ = occupancy(GF4, SD, N)
occ = np.array(occ, float)
print("PART 3  OCCUPANCY via midpoint boundaries -- this is what GF4_test.py measures")
print("  midpoint boundaries :", np.round(center(GF4), 4))
print("  occupancy counts    :", occ.astype(int).tolist())
print(f"  CV of occupancy     : {occ.std()/occ.mean():.3f}   (LARGE -> codes NOT used equally)")
print("  --> This inequality is EXPECTED and does not refute claim (A).")
print("      Nearest-rounding cells (midpoint-bounded) are a different partition")
print("      than the equal-mass bands (level-bounded).\n")


# =====================================================================
# PART 4 -- Recover the equal-mass boundaries empirically (his find_edge
#           idea) and show they land ON the GF4 levels, given the right
#           convention: equally-spaced FULL-normal CDF up to the 0.9677
#           offset, then absmax-normalized -- NOT k/8 half-normal steps.
# =====================================================================
# Analytic equal-mass boundaries at the construction convention:
emb = norm.ppf(np.linspace(0.5, OFFSET, 8)) / norm.ppf(OFFSET)
print("PART 4  Equal-mass interval boundaries (analytic) vs GF4 levels")
print("  equal-mass boundaries:", np.round(emb, 4))
print("  GF4 levels           :", np.round(GF4, 4))
print("  => The boundaries of the equal-probability-mass intervals ARE the")
print("     GF4 levels (to ~0.01). GF4_test.py compared them against k/8")
print("     half-normal quantiles at sd=0.25 (top level -> 4 sigma), which is")
print("     a different convention, so of course they did not match.\n")

print("=" * 63)
print("SUMMARY")
print("  * GF4 levels ARE equal-probability-mass quantiles: equal mass sits")
print("    between consecutive LEVELS (Parts 1,2,4).  Claim (A) holds.")
print("  * Unequal code OCCUPANCY (Part 3) is a separate fact about nearest-")
print("    rounding cells and was never the claim. GF4_test.py measured that.")
print("  * Scale matters: the property is defined at absmax->0.9677-quantile")
print("    (1.848 sigma), not at the arbitrary sd=0.25 (4 sigma) used there.")
print("=" * 63)
