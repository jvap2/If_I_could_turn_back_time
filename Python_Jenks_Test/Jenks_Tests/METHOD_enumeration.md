# Method: n-bit Quantile Codebook for Activations & KV Cache

Enumerated process, dual-purpose:
- **Paper (§Method):** steps 0–13 are the algorithm; §4 is the two deployment modes.
- **Disclosure:** same enumeration + the **Novelty Map** at the end (what is clean-air
  vs. anticipated), consistent with `PATENT_datapath_novelty.md`.

The construct: a **custom n-bit floating-point LUT (codebook)** whose levels are the
**quantiles of a target distribution**. For LLMs the post-rotation activation
distribution is ~Gaussian, so the 4-bit instance ("GF4") uses Gaussian-quantile levels
(8 magnitudes × sign). It applies uniformly to **linear activations** and the **KV
cache**, and supports **two compute modes** from one stored representation.

---

## 0. Codebook construction (offline, once, shared network-wide)
1. Pick bit-width `n` (n=4 → 16 codes: 8 magnitudes × sign) and a target distribution
   `D` (Gaussian for post-rotation LLM activations).
2. Build codebook `C` = the **equal-mass quantile levels of `D`** on the normalized range
   `[0,1]` (positive half; sign carried separately). For n=4 this is `GF4_POS` (8 levels).
   Store `C` (tiny; reused across the whole network).
3. *(optional)* Refine `C` by Lloyd–Max / MSE minimization on calibration activations,
   per-network or per-layer.
   - **Generalization:** any `n` (codebook size `2^n`), any `D` (Gaussian, Laplace,
     Student-t, per-layer empirical) — the levels are just that distribution's quantiles.

## 1. Per-tensor preparation (inference time)
4. **Rotation (linear activations only):** apply an orthogonal rotation `R` (randomized
   Hadamard, block size `b`) so the activation distribution becomes ~Gaussian and matches
   `C`. `R` is **fused into the weight offline**, so it is free at runtime.
   - **KV cache: skip rotation.** The per-block scale (step 6) already absorbs KV
     outliers; on OPT-1.3B rotation *hurt* KV quality (+1.53 vs +1.20 without). This is a
     KV-specific simplification vs. INT-rotation methods (QuaRot/SpinQuant).
5. **Blocking:** partition the tensor into contiguous blocks of size `B` along the last
   dimension.
   - Linear activations: `B` = quantization block (e.g., 16).
   - KV cache: `B` = `head_dim`, so each block is **one head's K (or V) vector for one
     token** → per-head, per-token scaling. (Applied via a forward hook on `k_proj`/`v_proj`.)

## 2. Scale + quantize (per block)
6. **Scale:** `s = RMS(block) × clip_ratio`.
7. *(optional, "adaptive"):* search a small set of `clip_ratio ∈ {1.5,2.0,2.5,3.0,4.0}`
   and keep the `s` that **minimizes block reconstruction MSE**. (This is the per-block
   online clip search.)
8. **Quantize:** `x̂ = clamp(|block|/s, 0, 1)`; index `i = argmin_k |x̂ − C[k]|`; keep sign.
9. **Store:** per element → `n`-bit index `i` (+ sign bit); per block → FP scale `s`.
   This `{indices, scales}` pair is the compact representation.

## 3. (optional) Residual / multi-pass — higher effective precision
10. Quantize the residual `(block − dequant)` with a 2nd pass of the same codebook →
    **sum of two `n`-bit codes ≈ 2n-bit effective**. Both codes stored; **summed in full
    precision before the GEMM**, so throughput stays at the single-code (W4A16) level.

## 4. Two compute modes (the deployment fork — one stored representation)

### Mode A — decode-to-FP16 (memory-saver; runs on today's GPUs)
11a. Keep activations / KV in the cache and on the bus as `{indices, scales}` →
     **4-bit storage/bandwidth** instead of 16-bit (the "plug in to save memory" path).
12a. At GEMM time: **LUT-decode** `index → C[i]`, multiply by block scale `s` → FP16
     operand; run the **standard FP16 / W4A16 tensor-core matmul**. Only added hardware is
     the small **decode LUT** (a ROM). Memory saved in transit; compute unchanged.

### Mode B — codebook×codebook product-LUT MAC ("Microsoft LUT" / T-MAC style)
11b. Keep **both** operands as indices: weights in codebook `C_w` (FP4/E2M1), activations
     or KV in codebook `C_a` (this codebook).
12b. Precompute **once** a `|C_w| × |C_a|` product table `T[i,j] = C_w[i]·C_a[j]`
     (e.g., 16×16). It is **static, reused network-wide**.
13b. MAC = look up `T[i_w, i_a]` and accumulate; apply `(s_w · s_a)` per block. **No
     per-element multiplies, no dequantization.**

## 5. Generalization summary
- **Bit-width:** `n` arbitrary; product table is `2^{n_w} × 2^{n_a}`.
- **Distribution:** levels = quantiles of any `D` (per-layer empirical, Gaussian, Laplace…).
- **Operand:** identical procedure for linear activations and KV cache (K and V).

---

## Novelty Map (for the disclosure — read with `PATENT_datapath_novelty.md`)

| Step | Element | Status |
|---|---|---|
| 0–2 | quantile/Gaussian codebook | **Anticipated** — NF4, Lloyd–Max, companding |
| 5–6 | per-block RMS × clip scale | **Anticipated** — MX/NVFP4 block scaling |
| 4 | rotation → Gaussian activations | **Anticipated** — QuaRot / SpinQuant / InfoQuant |
| 11a–12a | decode-to-FP16 memory-saver | **Standard** — any FP4-with-scale format |
| 11b–13b | both-operand product-LUT MAC | **Anticipated** — Cartesian Product LUT / T-MAC / DeepGEMM / OASIS |
| 7 | **adaptive per-block clip search** | *Distinct-but-narrow* — online per-block clip selection by MSE |
| 4 (KV) | **per-block-scaled GF4 needs no KV rotation** | *Distinct-but-narrow, empirical* — simpler KV path than QuaRot |
| — | **condition-number Hessian damping** (weight-side, separate) | *Distinct* — the one element absent from every codebook/LUT/rotation reference |

**Filing guidance (unchanged):** do **not** file on the codebook or the product-LUT
datapath — both are thoroughly anticipated. If filing at all, anchor on the
**condition-number damping**; the adaptive clip and the KV-no-rotation result are
paper contributions, not strong independent claims.

**Paper guidance:** use §0–4 verbatim as the Method; present **Mode A** as the drop-in on
today's FP4 GPUs and **Mode B** as the accelerator datapath (cite the product-LUT prior
art openly — the contribution is the codebook choice + the PPA/design-space analysis, not
a new datapath).
