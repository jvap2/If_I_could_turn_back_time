# GF4 Patent — Prior-Art Landscape & Reframed Claims

**Purpose.** The general idea "a lookup table with levels at the quantiles of the
data distribution, at n bits, for any distribution" is *foundational, decades-old
prior art*. Claiming it broadly will very likely be rejected. This document maps the
prior art and reframes the claims onto the parts that are actually clean air — the
**native n-bit LUT-MAC hardware** and the **specific rotation-induced-distribution +
matched-LUT + damped-calibration pipeline**.

_Technical analysis only — not legal advice. Confirm with counsel and a formal
prior-art search._

## 1. Prior-art landscape

| Reference | What it teaches | Target | Rotation | Compute | Why it does NOT reach GF4's anchor |
|---|---|---|---|---|---|
| **Lloyd–Max** (1957/60) | Distortion-optimal reconstruction levels for a given source pdf — the canonical "fit the quantizer to the distribution" | any signal | — | theory | No rotation, no NN, no programmable digital LUT-MAC datapath |
| **Companding** (μ-law/A-law, 1950s–60s) | CDF-based non-uniform quantizer → levels at the distribution's **quantiles** | signals | — | analog/telephony | The quantile-LUT idea itself, but no NN, no native digital MAC-in-codebook |
| **NF4 / NormalFloat** (2023) | 4-bit codebook at the **quantiles of a normal distribution** | weights (storage) | — | **dequantizes to FP16** | Weight *storage* only; dequantizes to compute; no rotation-induced distribution; no native MAC |
| **NVFP4 / MXFP4** | E2M1 block floating-point + shared microscale | acts+weights | optional | fixed-format MAC | Log-spaced (not distribution-matched) levels; hardwired format, not a programmable quantile LUT |
| **QuaRot / SpinQuant** | Hadamard / learned rotation + INT4 | acts+weights | yes | INT4 MAC | Uniform INT4 — not a distribution-matched codebook |
| **InfoQuant** (2026) | Learned rotation → Gaussian acts → Gaussian-matched activation codebook | W4A4 | learned | **VERIFY** | Closest concept-overlap. Distinctions to confirm: learned vs Hadamard rotation; **whether it does native LUT arithmetic or dequantizes**; no condition-number damping |
| **AAAC** (2026) | Learned per-layer *weight* codebooks from activation stats | weights | — | dequantizes | Weight-only, learned (not quantile-fixed), no rotation, dequantizes |
| **GF4 (this invention)** | Programmable **quantile LUT matched to a rotation-INDUCED distribution**, executed by a **native n-bit LUT-MAC** (multiply-accumulate performed in the encoded domain), with condition-number-damped calibration | acts (+weights) | Hadamard | **native LUT-MAC** | — |

## 1b. LUT-MAC HARDWARE PRIOR ART (critical — the hardware LUT is NOT clean air)

A verification search (2026-07) shows "a lookup table realized in hardware to do the
MAC for quantized/low-bit NN inference" is a **crowded, established field**:

| Reference | What it teaches | Overlap with GF4 hardware anchor |
|---|---|---|
| **LUT Tensor Core** (Microsoft, 2024) | HW tensor core doing low-bit LLM MAC by table look-up of precomputed partial products; **format-agnostic / programmable** | HIGH — a general programmable LUT-MAC that would run a GF4 codebook as one configuration |
| **T-MAC / T-MAN** (Microsoft) | Table-lookup mpGEMM replacing multiplies with LUT reads; end-to-end low-bit LLM on NPUs via unified table lookup | HIGH — native table-lookup MAC, no dequantize |
| **Platinum / MxGLUT / LUT-GEMM** | LUT-based low-bit weight matmul accelerators (reconfigurable, mixed-precision) | HIGH |
| **"Multipurpose DL Accelerator for Arbitrary Quantization"** | Reconfigurable **codebook register file** + a MAC operating on **arbitrarily-quantized** vectors | VERY HIGH — a programmable-codebook LUT-MAC |
| **EIE / Deep Compression** (Han 2016) | Codebook (LUT) weight-sharing accelerator | MED — decodes to 16-bit to compute (not native) |

**Refined conclusion (after reading the papers, 2026-07):** these LUT-MACs are LESS
overlapping than their names imply:
- **LUT Tensor Core** & **T-MAN**: **weight-only** (low-bit weights × HIGH-precision
  activations, W4A16-style mpGEMM), **uniform integer / BCQ** (NOT distribution-matched),
  their LUT stores **activation×weight partial products** (a compute trick, not a value
  codebook), and **no rotation**.
- **Arbitrary-quant accelerator** (Moon 2024): handles nonlinear quantization
  *generically* via bit-serial + reconfigurable LUT, but does **not** prescribe a
  distribution-matched (quantile) codebook, does **not** use a rotation, and does not
  link a codebook to induced statistics.
- **InfoQuant**: rotation + Gaussian activation codebook + W4A4, but **dequantizes**
  (no native codebook-domain MAC) and no damping.
- **NF4**: Gaussian value codebook, but weight-**storage** only, **dequantizes**, relies
  on a *natural* (not induced) Gaussian, no rotation.

So NO single reference teaches GF4's specific combination — that combination is the
candidate carve-out.

## 2. The carve-out (plausible, narrow — validate with a professional search)

The *broad* claims are still gone (each pillar exists somewhere). But no reference
combines all of the following, which is GF4's candidate unique space:

**(C-1) The system:** an incoherence rotation that **induces** an approximately-Gaussian
distribution in the operands; a shared codebook **derived from the quantiles of that
induced distribution**; **both** weights and activations quantized to that same codebook
at the same low bit-width; the layer MAC computed **natively in the codebook domain**
(e.g., via a fixed codebook×codebook product table reused network-wide — structurally
different from LUT Tensor Core's per-activation act×weight product tables); stabilized by
condition-number-damped calibration.

**(C-2) Two specific, defensible anchors within it:**
1. **Condition-number-targeted Hessian damping** — the cleanest standalone novelty; did
   not appear in any codebook / LUT / hardware reference. Consider filing separately.
2. **A native codebook-domain product-LUT MAC in which BOTH operands are encoded in a
   shared *distribution-matched* codebook derived from a rotation-induced distribution** —
   distinct from (a) weight-only uniform product-LUTs (LUT Tensor Core/T-MAN), (b)
   dequantizing codebook methods (NF4/InfoQuant), and (c) generic nonlinear-quant
   datapaths (Moon 2024), none of which tie the codebook to a rotation-induced Gaussian.

Remaining risk (obviousness): an examiner may argue "InfoQuant (rotation+Gaussian
codebook) + LUT Tensor Core (LUT-MAC) = obvious." The rebuttal must lean on the specific
non-obvious integration in (C-1)/(C-2#2) and the demonstrated iso-cost benefit — a real
prior-art search is needed to confirm the gap holds. Other survivors to consider:

1. **Condition-number-targeted Hessian damping** — the clearest remaining novelty. It is
   a *calibration* method, not a codebook or a datapath, and did not surface in any of
   the LUT/codebook prior art. Consider filing this **separately**.
2. **A specific, non-obvious combination** — rotation that *induces* the target
   distribution + a quantile-matched programmable codebook + native LUT-MAC + the
   damped calibration, *as one system with a demonstrated benefit* (iso-cost accuracy
   gain). Combination claims can survive in crowded fields if the combination is
   non-obvious and synergistic — but each element here is individually known, so this is
   a hard argument and depends on drafting + a real search.
3. **A specific technical detail not yet identified** — e.g., a particular
   codebook-derivation or LUT-programming step tied to the rotation statistics that no
   reference teaches. Requires digging into the exact mechanisms of LUT Tensor Core /
   InfoQuant / the arbitrary-quant accelerator.

**Honest bottom line:** the space is dense on every axis. A **professional patentability
search is now clearly warranted before investing further** — give the attorney: Lloyd–Max
/ companding, NF4, InfoQuant, AAAC, LUT Tensor Core, T-MAC/T-MAN, the arbitrary-quant
codebook accelerator, and EIE. The engineering result (GF4 works, beats baselines,
scale-robust) is unaffected — this is purely about what is patentable and how narrowly.

## 3. Reframed claims (anchored on hardware + pipeline, not the general codebook)

**Claim 1 (apparatus — primary anchor).** A processing element for a neural-network
accelerator, comprising: a programmable look-up table configured to store *n*
reconstruction levels; an encoder that maps an operand to an *m*-bit index of the
table; and a multiply-accumulate unit configured to compute a dot product from the
*m*-bit indices **by table look-up, directly in the encoded domain, without
dequantizing the operands to a higher-precision floating-point representation**;
wherein the *n* reconstruction levels are programmed to approximate the quantiles of a
target distribution of the operands.

**Claim 2.** The processing element of claim 1, wherein the look-up table is
reprogrammable between at least a distribution-matched (e.g. Gaussian-quantile) level
set and a floating-point (E2M1) level set, at the same index width and block size.

**Claim 3 (method — the pipeline).** A method of executing a neural-network layer,
comprising: applying an orthogonal incoherence transform to a tensor of the layer to
induce an approximately Gaussian distribution; programming a look-up table with *n*
levels positioned at the quantiles of said induced distribution; encoding the
transformed tensor to *m*-bit indices of the table; and computing the layer's
multiply-accumulate directly on the *m*-bit indices via the look-up table, without
dequantizing to a higher-precision floating-point format.

**Claim 4.** The method of claim 3, wherein the target distribution and hence the *n*
levels are selected per tensor or per layer from the transformed statistics, such that
the same apparatus supports non-Gaussian distributions for other data.

**Claim 5 (calibration).** The method of claim 3, further comprising quantizing the
layer's weights by a Hessian-weighted reconstruction that is stabilized by adding, to
each block Hessian, a damping term sized to cap the block's condition number at a
target value.

**Dependent embodiments to add:** *n*=16 with 8 magnitudes + sign; the specific
Gaussian-quantile levels; per-block RMS×clip scale; two-term residual encoding;
adaptive per-block clip; application to weights as well as activations.

## 4. Action items

1. **Verify InfoQuant's arithmetic** (native LUT vs dequantize) — decides whether
   claim 1's "without dequantizing" cleanly separates GF4 from it. *(Highest priority.)*
2. **Formal prior-art search** on Lloyd–Max, companding/μ-law, NF4, and vector-quantized
   codebook design — before filing anything reciting "quantiles of a distribution."
3. **Anchor independent claims on the apparatus (Claim 1)** and the pipeline (Claim 3),
   keeping the bare quantile-LUT idea only in dependent claims, if at all.
4. Decide with counsel whether the **condition-number damping** is a separate filing.
