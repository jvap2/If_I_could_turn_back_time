# Paper Outline — Scale-Robust Low-Bit LLM Calibration (GF4 + condition-number damping)

**Framing decision (from the prior-art sweep):** lead with the **calibration diagnosis +
fix** (the genuinely novel, defensible piece), and use the **iso-cost GF4 codebook study**
as the empirical vehicle. Do NOT frame it as "a new codebook" — NF4/InfoQuant/QuaRot cover
that and reviewers will find them. Target: **MLSys / an efficient-ML venue / strong
workshop**, not a top-tier "novel method" track.

## Candidate titles (lead with the damping)
- "Why Low-Bit LLM Calibration Collapses at Scale — and a Condition-Number Fix"
- "Condition-Number-Stabilized Calibration for Scale-Robust 4-bit LLM Quantization"
- "Stabilizing Hessian-Weighted PTQ Across Model Scale"

## Abstract (skeleton)
Hessian-weighted post-training quantization (GPTQ/OBQ-style) can **collapse at scale**:
on opt-13b, W4A16 perplexity jumps to ~300 while smaller OPTs are fine. We show the cause
is **ill-conditioned Hessians in the early (post-LayerNorm) decoder blocks**, driven by
emergent activation outliers, which the standard fixed diagonal damping does not address.
We introduce **adaptive, per-block, condition-number-targeted Hessian damping** (cap each
block's κ via its eigenvalues), which removes the collapse (opt-13b 304→10.5) and is
**scale-robust across a 240× parameter range with a single setting** (opt-125m→30b),
neutral-to-helpful on the models that already worked. Within the same, iso-cost pipeline
we study the **activation codebook** in isolation and show a Gaussian-quantile codebook
(GF4) meets or beats E2M1/NVFP4 and MXFP4 at identical hardware cost, with a two-term
variant matching the weight-only ceiling.

## Contributions (ordered by strength — honest)
1. **[Primary] Diagnosis + fix.** Identify ill-conditioned early-block Hessians as the
   cause of scale-dependent W4A16 collapse; introduce adaptive condition-number damping
   that fixes it and generalizes across scale (one κ, no re-tuning). Distinct from GPTQ's
   fixed jitter, ADMM-Q's fixed λI + ADMM penalty, and HAS-VQ/HAWQ's eigenvalue *sensitivity*.
2. **[Secondary] Controlled iso-cost codebook study.** One shared Hadamard build; only the
   activation codebook varies → the PPL delta is attributable to the codebook. Gaussian
   (GF4) ≥ E2M1/NVFP4/MXFP4 at every OPT scale; residual-GF4 ≈ weight ceiling. Advantage is
   U-shaped, largest in the outlier regime (13B/30B).
3. **[Supporting] Hardware iso-cost grounding.** Timeloop/Accelergy energy+area model
   confirming the codebook is a drop-in at equal width/block/scale.

## Section plan
1. **Introduction** — the scale-dependent collapse (motivating plot: opt-125m→30b W4A16,
   with/without the fix); state contributions honestly.
2. **Background** — Hadamard incoherence (QuaRot/SpinQuant); Hessian-weighted PTQ (OBQ/GPTQ)
   and its **fixed** damping; FP4 codebooks (NVFP4/MXFP4) and Gaussian codebooks (NF4).
3. **The collapse, diagnosed** — per-layer weight-error map (early blocks 1–4 spike to ~0.48,
   decay with depth; residual-stream readers affected, out_proj not); tie to Hessian
   conditioning; show fixed damping (WQDAMP=0.01) only partially fixes it (304→27.5).
4. **Method: adaptive condition-number damping** — per 16×16 block, `λ = max(0, λmax/κ − λmin)`;
   caps κ; heavy on ill-conditioned early blocks, ~0 elsewhere. Complexity (cheap 16×16
   eigvals). Ablation: fixed-λ vs adaptive-κ (27.5 vs 10.5 on opt-13b).
5. **Iso-cost GF4 codebook (the vehicle)** — the Gaussian-quantile levels; per-block RMS×clip
   scale; the "one build, only codebook varies" protocol; residual (2-term) variant.
6. **Experiments** — full OPT family table (125m–30b) at κ=100; scale-robustness (one κ);
   no-regression on small models; before/after (WQKAPPA=0 vs 100); GF4 vs E2M1/MXFP4 deltas;
   (add: Llama-2-7B/13B, Llama-3-8B for cross-architecture; Qwen2.5-32B for a recent point).
7. **Related work (honest positioning)** — codebooks: NF4, InfoQuant, AAAC; rotation: QuaRot,
   SpinQuant, InfoQuant; Hessian PTQ + damping: GPTQ, OBQ, ADMM-Q, HAS-VQ, HAWQ; LUT compute:
   LUT Tensor Core, T-MAC, DeepGEMM/OASIS. **Explicitly state what is and isn't new** (the
   damping/diagnosis is; the codebook and rotation are shared with prior work).
8. **Limitations** — weights are E2M1 (not GF4) in the current implementation; condition-number
   regularization is a classical numerical technique (novelty is the diagnosis + scale-robust
   result, not the formula); no 66B/70B yet (RAM); activation-side entropy coding out of scope.
9. **Conclusion.**

## Experiments status (what we have vs need)
- ✅ Full OPT family at κ=100 (125m/1.3b/2.7b/6.7b/13b/30b) — `iso_results_kappa100.csv`.
- ✅ Before/after (undamped collapse) — `iso_results_predamp.csv`; ablation fixed vs adaptive.
- ✅ Per-layer weight-error diagnostic (`diag_v5_solver.py`, WQDEBUG map).
- ✅ Scale-robustness (one κ across 240×).
- ⏳ **Cross-architecture**: Llama-2-7B/13B, Llama-3-8B at κ=100 (running/queued).
- ⏳ **Recent point**: Qwen2.5-32B (fits Colab; directly comparable to AAAC/APEX4/InfoQuant).
- ⏳ **Optional marquee**: Llama-2-70B / OPT-66B (needs ~256 GB box).
- ⏳ Timeloop energy+area numbers finalized.

## Reviewer-proofing checklist
- Cite InfoQuant/NF4/QuaRot up front; own the overlap.
- Report **both** the fixed-λ and adaptive-κ results (shows why adaptive matters).
- Include a WQKAPPA=0 "before" column so the collapse and fix are unambiguous.
- Add C4 perplexity (papers report WikiText-2 **and** C4) and a couple of zero-shot tasks
  (lm-eval: PIQA/ARC/HellaSwag) — reviewers expect more than WikiText-2 alone.
- State the damping is orthogonal to the codebook (helps all arms) — it is.
