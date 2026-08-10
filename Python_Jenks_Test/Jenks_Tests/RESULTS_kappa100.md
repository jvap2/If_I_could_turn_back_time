# GF4 Iso-Energy Results — Adaptive Hessian Damping (κ = 100)

WikiText-2 perplexity, non-overlapping 2048-token windows. All arms share **one**
Hadamard build (identical rotation, per-16 block, E4M3 scale) — only the activation
codebook differs, so every comparison is at **identical hardware cost** (4.5 bpv,
0.08× FP16 energy). Weights: 4-bit, calibrated once (codebook-agnostic). `RETAIN=1`
(MLP-output layer kept FP16). Data: `iso_results_kappa100.csv`.

## Headline

- **A single adaptive knob (condition-number-targeted Hessian damping, κ=100) fixes
  the OPT W4A16 collapse and generalizes across a 240× size range with zero
  re-tuning** — from opt-125m to opt-30b.
- **It generalizes across architecture too**: it *rescues* the ill-conditioned OPT
  models and is a *no-op* on the already-healthy Llama-2-13B (4.946→4.947) and
  Llama-3-8B — firing only where the Hessian is ill-conditioned, never perturbing a
  healthy model.
- **residual-GF4 (2-term) ≈ A16 weight ceiling at every scale and architecture** — the
  ~8-bit-effective codebook is near-lossless against the 4-bit weight-only ceiling.
- **GF4 beats E2M1/NVFP4 and MXFP4 everywhere**, and the advantage **tracks quantization
  *difficulty*** — largest where the source is hardest to represent (emergent outliers
  *or* training-dense weights), not simply where the model is biggest.

## Full table (κ = 100)

| model | A16 | NVFP4 fixed | NVFP4 adap | MXFP4 | GF4 1-term | GF4 adap | residual-GF4 |
|---|---|---|---|---|---|---|---|
| opt-125m | 30.970 | 36.144 | 34.427 | 34.730 | 34.023 | 33.333 | 31.011 |
| opt-1.3b | 16.806 | 18.885 | 18.356 | 18.673 | 18.681 | 18.302 | 16.859 |
| opt-2.7b | 13.292 | 13.588 | 13.467 | 13.486 | 13.560 | 13.436 | 13.297 |
| opt-6.7b | 12.112 | 12.538 | 12.370 | 12.439 | 12.453 | 12.351 | 12.115 |
| opt-13b  | 10.513 | **532.016** | 11.157 | 11.251 | 12.218 | 10.788 | 10.514 |
| opt-30b  |  9.869 | 12.297 | 11.140 | 11.172 | 12.484 | 10.552 |  9.874 |
| Llama-2-13B | 4.947 | 5.129 | 5.041 | 5.074 | 5.094 | 5.039 | 4.949 |
| Llama-3-8B  | 6.502 | 6.977 | 6.800 | 6.845 | 6.898 | 6.752 | 6.508 |

## Key findings

### 1. residual-GF4 ≈ A16 at every scale
Two GF4 terms (~8-bit effective) reconstruct the activations to within noise of the
weight-only ceiling:

| model | 125m | 1.3b | 2.7b | 6.7b | 13b | 30b |
|---|---|---|---|---|---|---|
| residual − A16 | +0.04 | +0.05 | +0.01 | +0.00 | +0.00 | +0.01 |

### 2. GF4 vs E2M1/NVFP4 (adaptive clip, iso-cost) — the advantage tracks *difficulty*, not size
GF4's Gaussian levels beat E2M1's floating-point levels on **every** model tested; the
margin is smallest for easy models and **grows where quantization is hard**:

| model | opt-125m | opt-1.3b | opt-2.7b | opt-6.7b | Llama-2-13B | **opt-13b** | Llama-3-8B | **opt-30b** |
|---|---|---|---|---|---|---|---|---|
| GF4 adap advantage (PPL) | +1.09 | +0.05 | +0.03 | +0.02 | +0.002 | **+0.37** | +0.048 | **+0.59** |

The driver is **quantization difficulty**, not parameter count. Within OPT it looks
U-shaped in size (outliers emerge past ~6.7B). But the Llama points make the real story
clear: **Llama-3-8B — harder to quantize (~4× more training tokens, denser weights) —
shows a *larger* GF4 margin (+0.048) than the bigger but easier Llama-2-13B (+0.002).**
So GF4 pulls ahead precisely where the source is hardest to represent, whether the
hardness comes from emergent outliers (opt-13b/30b) or training density (Llama-3) — a
cleaner claim than "bigger = bigger win."

### 3. Fixed-clip regime — GF4 dominates the hardest case
With a fixed clip (no per-block search), E2M1 **collapses on the heavily-padded
opt-13b** (38% zero-pad after 5120→8192 Hadamard), while GF4 holds:

- opt-13b: **GF4 12.22 vs E2M1 532.0 → GF4 +519.8**

On the less-padded opt-30b (12.5% pad) E2M1-fixed survives (12.30) and slightly edges
GF4-fixed (12.48, −0.19). So the fixed-clip GF4 win is specific to the hardest /
most-padded cases, not universal.

### 4. Scale-robustness (the methods contribution)
Same κ=100, no per-model tuning:

| model | A16 (undamped) | A16 (κ=100) |
|---|---|---|
| opt-13b | 304  (collapse) | **10.513** |
| opt-30b |  31.8 (degraded) | **9.869** |
| opt-6.7b | ~12.28 | 12.112 (no regression) |
| opt-1.3b | ~17.76 | 16.806 (no regression) |

The damping fires only where the Hessian is ill-conditioned (dramatic on 13b/30b,
a no-op-to-mild improvement on the small models) and never regresses.

## The fix

**Root cause.** The weight solver (`reconstruct_layer_fp_blockdiag_scaled_v5`)
minimizes a Hessian-weighted error `(w−q)ᵀ H (w−q)`, where `H = E[x_had x_hadᵀ]` is the
rotated-activation second moment. In OPT's early decoder blocks the post-LayerNorm
residual stream carries **massive-activation / emergent-outlier** directions, so `H`
is severely **ill-conditioned**. Un-damped, the solve over-commits precision to the
high-energy directions and wrecks the rest — up to **48% per-layer weight error in
blocks 1–4**, collapsing W4A16 to ~300 PPL. Diagnostic isolation (WNOQUANT gate:
returning exact weights recovers A16 to 10.13) confirmed the weights, not the forward
path, were the failure.

**Fix — adaptive condition-number damping.** For each 16×16 block Hessian, add just
enough `λI` to cap its condition number at κ:

```
λ_damp = max(0, λ_max/κ − λ_min);   H ← H + λ_damp·I
```

Heavy on the ill-conditioned early blocks, ≈0 on well-conditioned ones (out_proj, deep
layers). This targets numerical *stability* directly rather than a fixed relative
amount, which is what makes it **scale-agnostic**: a κ that fixes 13b also fixes 30b
without change. It cleanly beat fixed-λ GPTQ-style damping (κ=100 → A16 10.5 vs
WQDAMP=0.01 → 27.5).

## Reproduction

`WQKAPPA` (env) controls κ in `reconstruct_layer_fp_blockdiag_scaled_v5`; **default is
100 (on)**. `WQKAPPA=0` disables it and reproduces the undamped/collapsed numbers.

```bash
RETAIN=1 OFFLOAD=1 EVAL_OFFLOAD=1 HAD_BS=auto \
  python3 -u iso_energy_125m.py facebook/opt-13b        # κ=100 by default -> A16 ~10.5
WQKAPPA=0 RETAIN=1 OFFLOAD=1 EVAL_OFFLOAD=1 HAD_BS=auto \
  python3 -u iso_energy_125m.py facebook/opt-13b        # disabled -> A16 ~304 (before)
```

## Status

- ✅ Full OPT family (125m, 1.3b, 2.7b, 6.7b, 13b, 30b) at κ=100.
- ✅ **Cross-architecture**: Llama-2-13B (RMSNorm / SwiGLU / no-bias) and Llama-3-8B
  (GQA, 128k vocab) at κ=100 — both clean; damping a no-op (neither ever collapsed).
- ⏳ Llama-2-7B, **Qwen2.5-32B** (recent-model comparison vs AAAC/APEX4/InfoQuant),
  optional Llama-2-70B / OPT-66B (needs ~256 GB RAM).
- Pre-damping numbers archived on the VM (`iso_results_predamp.csv`).

_Note: opt-30b's raw CSV was lost with a Colab runtime disconnect; its row here is
reconstructed from the run output (values exact, timestamp approximate). opt-13b's
authoritative row is in the VM's `iso_results.csv`._
