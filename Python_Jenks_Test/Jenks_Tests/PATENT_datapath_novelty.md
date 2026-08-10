# The "Both-Operands Product-LUT MAC" Datapath — Analysis & Novelty Verdict

_Technical prior-art analysis, not legal advice. A professional patentability search
is strongly recommended; this documents what a targeted search found._

## 1. The datapath angle we were testing

The proposed hardware anchor (not the codebook contents, which are NF4/Lloyd–Max):

> A multiply-accumulate unit that computes a dot product of two operands **both
> low-bit codebook-indexed** — a *weight* codebook and an *activation* codebook — via a
> **fixed product table** of size `|weight-codebook| × |activation-codebook|`
> (e.g. 16×16) that is **computed once and reused across the whole network**, replacing
> multiplications with lookups, **without dequantizing**.

Key points established:
- It implies **W4A4** (both operands quantized). GF4's current experiments are E2M1
  weights + GF4 activations — a *two-codebook* case, not "both GF4."
- The A16 / W4A16 arms cannot use it (FP16 activations can't be codebook indices).
- The intended distinction from LUT Tensor Core / T-MAC / T-MAN: those keep the
  **activation in high precision** and rebuild an activation×weight table *per
  activation tile*; here **both operands are low-bit**, so the product table is a
  single small **static** table reused network-wide.

## 2. Novelty verdict: NOT NOVEL — this exact datapath is prior art

A targeted search (2026-07) found the "both-operands-quantized → precomputed product
lookup table" mechanism described directly and repeatedly:

| Reference | What it teaches | Overlap |
|---|---|---|
| **Cartesian Product LUT (dual-side / WAQ)** | "A **Cartesian Product LUT is constructed offline to store all possible multiplication results between the weight and activation centroids**. At inference, activation and weight indices are concatenated and GEMM multiplications are replaced with a weighted sum of the stored products." | **EXACT** — this *is* the proposed datapath |
| **DeepGEMM** | "For quantized CNNs where **both weights and activations are quantized** (4/2/1-bit), DeepGEMM **precomputes all possible products of weights and activations, stores them in a lookup table**." | **EXACT** |
| **OASIS** (2507.23035) | "Outlier-Aware **LUT-Based GEMM with Dual-Side Quantization** for LLM inference" — both operands quantized, LUT-GEMM | **EXACT (LLM, W4A4)** |
| **MADDNESS** | Approximate matmul via **product quantization** + codebook LUTs, multiplication-free | HIGH |
| **Residual-VQ multiplication-free CNN accelerator** (FPGA) | Precomputed codebook dot products in SRAM, retrieved by index | HIGH |
| **LUT-DLA / LUTIN / VPTQ** | Vector-quantized weights (+acts) → LUT-based inference | HIGH |

**Conclusion:** the "fixed product table over two quantized-operand codebooks, reused
at inference, multiplication-free" datapath is an established technique with a named
form ("Cartesian Product LUT" / dual-side LUT-GEMM). It is **not** a novel carve-out.
Neither the codebook contents (NF4/Lloyd–Max) nor this datapath is clean air.

## 3. Where that leaves the patent

Every axis is now occupied:
- Quantile codebook → Lloyd–Max / companding / NF4
- Rotation → Gaussian activations → QuaRot / SpinQuant / InfoQuant
- Weight-only LUT-MAC → LUT Tensor Core / T-MAC / T-MAN
- **Both-operands product-LUT MAC → Cartesian Product LUT / DeepGEMM / OASIS / MADDNESS**

**The one piece that did NOT appear in any codebook / LUT / datapath / hardware
reference is the condition-number-targeted Hessian damping** — a *calibration* method
(cap each block Hessian's condition number before the weight solve). That is the only
element with a plausible independent novelty story, and it is where I would focus a
filing, if any.

## 4. Recommendation

1. **Do not file on the lookup table or the product-LUT datapath.** Both are
   thoroughly anticipated; a professional search will confirm it quickly.
2. **If filing, anchor on the condition-number damping** — verify it against the
   quantization-calibration literature (GPTQ/OBQ damping uses a *fixed* diagonal jitter;
   the novelty here is the *adaptive, condition-number-targeted, per-block* damping and
   its scale-robustness result). That comparison is the next thing to check.
3. **Regardless of patentability, the work is a strong paper** — iso-cost GF4 accuracy,
   the 304→10.5 fix, and scale-robustness across a 240× range stand on their own.

## References
- Cartesian Product LUT / LUT-quantized LLM GEMM — https://aclanthology.org/2024.findings-emnlp.724.pdf
- OASIS (dual-side quantization LUT-GEMM) — https://arxiv.org/html/2507.23035
- T-MAC — https://arxiv.org/html/2407.00088v2
- LUT-DLA — https://arxiv.org/pdf/2501.10658
- VPTQ (vector PTQ) — https://arxiv.org/pdf/2409.17066
- Residual-VQ multiplication-free CNN accelerator — https://ieeexplore.ieee.org/abstract/document/10608107
