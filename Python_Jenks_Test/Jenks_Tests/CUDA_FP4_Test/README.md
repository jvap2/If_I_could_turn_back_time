# GF4 CUDA Kernel Pipeline

CUDA kernels built for the technical interview presentation, covering the
**full quantization pipeline** used by the actual GF4 research (traced from
the real research notebook, `FP_Quant_Colab.ipynb`, not invented from the
deck alone):

| Stage | File | What it does |
|---|---|---|
| 1. Randomized blockwise Hadamard | `hadamard_kernel.cu` | Rotates activations (and, offline, weights) with a per-channel random-sign Hadamard transform — Gaussianizes the distribution and spreads structured outliers before quantization |
| 2. GF4 activation encoder | `gf4_encode_kernel.cu` | Encodes **activations** to 4-bit GF4 codes; per-block scale = `block_rms * clip_ratio`; optionally mean-centers (subtracts a per-channel `mu`) before quantizing, matching the real calibration hook |
| 3. Hessian-weighted E2M1 weight quantization | `hessian_weight_quant_kernel.cu` | Quantizes **weights** to 4-bit E2M1 codes via a Hessian-damped alpha/bias solve — this is `reconstruct_layer_fp_blockdiag_scaled_v5`, the function the notebook's call graph actually routes to (`quantize_model_fp -> calibrate_model_stochastic_fp4 -> calibrate_layer_stochastic_fp4 -> reconstruct_layer_fp_blockdiag_scaled_v5`) |
| 4. Fused dequant + GEMV | `e2m1_fused_gemv_kernel.cu` | Removes the "4-bit codes reconstructed to FP16 before the GEMM" step the deck's own evaluation protocol pays for, by decoding E2M1 weight codes in registers during the dot product; optionally adds a precomputed `bias_correction` term per output row to restore the mean that stage 2 subtracted |

Each of kernels 1, 2, and 4 ships with a naive/baseline counterpart so the
"reduces a computational bottleneck" claim is a measured comparison. Kernel 3
is a calibration-time (run-once-per-layer) kernel, so its baseline comparison
is correctness against an exact NumPy reference, not a fused-vs-naive speed
race — see "Why kernel 3 doesn't get a naive/fused pair" below.

## Weights and activations use *different* codebooks — the correction that mattered most

The first version of this pipeline assumed weights and activations both used
the GF4 (Gaussian) codebook, because that's what the deck's slides describe
for activations and nothing in the deck says otherwise. Tracing the actual
notebook call graph showed that's wrong: **activations use GF4 (Gaussian,
equal-mass quantile codebook); weights use E2M1 (a standard exponent+mantissa
float format), fit per (row, block) with a Hessian-damped least-squares
solve.** Two separate codebook headers exist for exactly this reason:

- `gf4_common.cuh` — GF4 codebook (`GF4_LEVEL`/`GF4_THRESHOLD`), activations
  only.
- `e2m1_common.cuh` — E2M1 codebook (3 exponent-bias variants x 8 magnitude
  levels), weights only.

`gf4_fused_gemv_kernel.cu` (the original, GF4-based weight-decode kernel) is
kept on disk only as a deprecation stub pointing to its replacement,
`e2m1_fused_gemv_kernel.cu` — don't build the old file.

## Mean-centering, bias-correction, and clip-ratio calibration search

Three pieces added in a later pass, after being asked directly whether
smoothing/clipping matched the notebook and going back to re-read the actual
source rather than relying on an earlier (incomplete) call-graph trace. All
three are confirmed **active, currently-used parts of the real pipeline** —
not optional extras or "future work":

- **Mean-centering (`mu`).** `calibrate_model_stochastic_fp4` computes
  `mu = X_had_all.mean(dim=0)` — the per-input-channel mean of post-Hadamard
  calibration activations — and every downstream activation gets
  `x_h - mu` subtracted before GF4 quantization (`gf4_encode_kernel.cu` now
  takes an optional, nullable `mu` pointer; same nullable-pointer + modulo-
  indexing convention `hadamard_kernel.cu` already used for its random sign
  vector).
- **Bias-correction.** Since `y = W @ (x_centered + mu) = W @ x_centered +
  W @ mu`, the real pipeline adds `bias_correction = W_had_q @ mu` back after
  the GEMM to compensate for the mean that got subtracted before
  quantization, rather than re-adding `mu` to every activation element.
  `e2m1_fused_gemv_kernel.cu`'s fused and naive GEMV kernels both take an
  optional, nullable `bias_correction` pointer (one fp32 scalar per output
  row) added right before the final write.
- **Clip-ratio calibration search.** `calibrate_model_gf4` does a per-layer
  grid search over `clip_ratio` candidates `(1.5, 2.0, 2.5, 3.0, 4.0)`,
  picking whichever minimizes encode-then-decode round-trip MSE on the
  (mean-centered) calibration activations. This needed a GF4 **decode**
  kernel that didn't exist before (`gf4_decode_kernel` in
  `gf4_encode_kernel.cu` — GF4 previously had encode-only, unlike E2M1 which
  already had a dequant kernel for the GEMV path). `reference.py`'s
  `clip_ratio_search_ref` mirrors the exact same grid-search algorithm, and
  `bench.py` section 5 confirms the CUDA-kernel-driven search and the NumPy
  reference always pick the same candidate.

**Scope note (updated):** `llm_quant_eval.py` now runs two quantized
configurations on the real model, not one. Its original W4A16 (weight-only)
pass still never touches mu/bias-correction/clip-search — there's no
activation quantization for them to hook into there. A new W4A4 pass was
added on top of it specifically to close that gap: it rotates both weights
and calibration activations with the same real `hadamard_fwht` kernel, redoes
the Hessian weight-quant in that rotated basis, computes mu and
`bias_correction = W_had_q @ mu` from the real rotated calibration
activations, runs the real per-layer clip-ratio search, and patches each
target layer's forward pass to fake-quantize its activations with the real
`gf4_encode(mu=...)`/`gf4_decode` kernels on every call. Once this pass is
actually run, mu/bias-correction/clip-search will be validated end-to-end on
OPT-125M, not just against the NumPy reference in `bench.py` section 5 — both
are still worth keeping, since `bench.py` isolates the kernels from
model/data noise and runs far faster. What's still NOT in the W4A4 pass: GF4
residual quantization passes and the outlier-retention policy the deck's
Table 1 numbers use, so a remaining PPL gap vs. Table 1 is expected, not a
bug.

## Honest status — read this before the interview

I do not have access to a CUDA GPU in this environment (no `nvcc`, no
`nvidia-smi`), so **none of this has been compiled or run by me** — every
number below came from Jay running `bench.py`/`llm_quant_eval.py` on his own
GPU machine. Current state, updated after the latest run:

- **`bench.py`, all 5 sections (kernels 1/2/4 fused-vs-naive timing, kernel 3
  Hessian-weight-quant correctness, and section 5's mean-centering/
  bias-correction/clip-ratio-search additions) — passed, real hardware.**
  All correctness assertions passed; real timing numbers are in the deck's
  results slide (kernels 1/2/4) and printed by section 5 for the newest
  additions.
- `llm_quant_eval.py`'s W4A16 (weight-only) pass — real numbers obtained
  (FP16 baseline PPL 29.5207, Hessian-weighted PPL 50.9507, naive-rounding
  ablation added); the gap vs. the deck's Table 1 is expected given this
  pass's deliberately narrow scope (see its own docstring).
- `llm_quant_eval.py`'s new **W4A4 pass — not yet run on real hardware.**
  I validated its orchestration logic (reshapes, broadcasting, the
  `gf4_encode(mu=...)` equivalence to manual mean-centering) against
  `reference.py`'s NumPy functions on tiny synthetic layers, but that's not
  a substitute for an actual GPU run. Treat the first run as a debug session
  the same way the Hessian kernels were.

What I did to de-risk kernel 3 before it was ever run on real hardware (all
since confirmed correct by the actual `bench.py` run):
- Traced the notebook's actual call graph to confirm
  `reconstruct_layer_fp_blockdiag_scaled_v5` is the production path (not one
  of the ~270 other experimental variants in the notebook), including the
  comment in the notebook noting a fancier alternative was tried and
  rejected.
- Ported the per-(row,block) alpha/bias fixed-point solve faithfully:
  3 exponent-bias candidates x 5 fixed-point iterations each, picking the
  candidate with lowest Hessian-weighted residual loss, matching the
  reference function line for line.
- Approximated the reference's exact eigenvalue-based condition-number
  damping with **shifted power iteration** (no cuSOLVER dependency), and
  built `reference.py` to validate that approximation two independent ways
  (see "Why kernel 3 doesn't get a naive/fused pair" below) rather than just
  asserting it's close enough.

What I could **not** do, and still can't: catch nvcc compiler errors or torch
extension ABI mismatches directly — every compile/link error so far (a stale
build cache, a truncated file from a manual transfer, a mismatched kernel
signature) got caught by Jay running it on `ricky-bobby` and pasting the
error back. The W4A4 pass is the next thing that needs that same treatment.

## Setup (on your CUDA machine)

```bash
pip install torch numpy scipy   # if not already installed
cd <this folder>
python bench.py
```

Requirements: an SM70+ NVIDIA GPU (needs `__half` + warp shuffle intrinsics —
anything from Volta onward), a CUDA toolkit whose version matches your
installed PyTorch build, and enough shared memory per block for your largest
test size (see "Shared memory ceiling" below).

`bench.py` will:
1. JIT-compile the extension with `torch.utils.cpp_extension.load` (~30-60s
   the first time, cached after).
2. Check every kernel against a from-scratch NumPy/SciPy reference
   (`reference.py`): kernels 1, 2, 4 also get checked against their own
   naive-baseline counterpart; kernel 3 gets checked against an exact
   (`eigvalsh`-damped) reference at two levels (algorithm and kernel).
3. Time fused vs. naive on several sizes and print a results table (kernels
   1, 2, 4); print correctness-only diagnostics for kernel 3.

**Copy the printed tables into the deck's CUDA results slides** — they
currently show the last real run's numbers for kernels 1/2/4; the kernel 3
numbers still need a real run.

## The core story, in one paragraph

The deck's own PTQ pipeline slide says perplexity is measured "under
fake-quantization — 4-bit codes reconstructed to FP16 before the GEMM." That
reconstruction is a real kernel with a real cost: written as two passes (a
dequant kernel that writes a full fp16 weight tensor to HBM, then a normal
GEMM that reads it back), the *reconstruction itself* moves several times
more bytes through HBM than the packed 4-bit weights it started from —
memory traffic that a fused kernel (decode in registers, during the GEMM's
tile load, never writing the dequantized tensor to HBM) simply never
generates. That's the one-sentence version of `e2m1_fused_gemv_kernel.cu`,
and it's the natural "CUDA kernel that reduces a computational bottleneck"
answer to the interviewer's prompt. Kernels 1-3 are what feed that GEMV
kernel its correctly-quantized inputs in the first place — together they're
the whole PTQ pipeline as one cohesive story, not just the flashiest single
kernel in isolation.

## Why kernel 3 doesn't get a naive/fused pair

Kernels 1, 2, and 4 run in the hot path — once per token, at inference time —
so a naive-vs-fused wall-clock comparison is the right way to demonstrate
"reduces a bottleneck." Kernel 3 (Hessian weight quantization) runs once per
layer, offline, during calibration — it is never on the inference critical
path, so timing it against a naive baseline would be a misleading demo.
What *does* matter for kernel 3 is numerical correctness of an approximation
made for a real engineering reason: computing exact eigenvalues of each
Hessian block would need `cuSOLVER`, an extra dependency for a kernel that
runs a handful of times per model. Shifted power iteration gets the top and
bottom eigenvalues to within the damping step's tolerance in ~30 iterations
without it. `bench.py` checks this two ways: a pure-NumPy mirror of the same
power-iteration algorithm against `eigvalsh` (is the *algorithm* sound?), and
the actual CUDA kernel's output against `eigvalsh` (does the *kernel*
implement that algorithm correctly?). That split is deliberate — it's a
measured tradeoff, not a hidden one, and a good answer if asked "why not just
call cuSOLVER."

## Why the "LUTxLUT" trick doesn't help on GPU (a Q&A land mine, defused)

The RTL side of this research replaces decode-then-multiply with a
product-table lookup + sign XOR because on an ASIC, a multiplier costs
silicon area and energy that a small ROM read doesn't. **On a GPU this
doesn't apply** — an FP32 FMA is already full-throughput on every CUDA core,
and a LUT read from constant/shared memory is not obviously cheaper than a
multiply, especially with the bank conflicts an indexed lookup can trigger
across a warp. If asked "did you port the multiplier-free hardware trick to
CUDA," the honest answer is *no, deliberately* — the bottleneck GPUs actually
have is HBM bandwidth, not multiplier area, so the software kernels target
that instead. This is a better answer than pretending the hardware trick
ports 1:1 to a different kind of hardware constraint.

## Design decisions worth being able to defend

- **Weight-only (W4A16), not W4A4, for the GEMV kernel.** At batch-size-1
  autoregressive decode, the activation vector is tiny and cache-resident, so
  the entire bottleneck is streaming the weight matrix. A W4A4 fused GEMM is
  a reasonable "future work" answer if asked, but weight-only is the correct,
  defensible scope for a memory-bandwidth-bound kernel demo.
- **Different codebooks for weights vs. activations** (E2M1 vs. GF4) — see
  above. Getting this right required tracing the real call graph rather than
  assuming symmetry.
- **Power iteration over exact eigendecomposition for the Hessian damping
  step** — avoids a cuSOLVER dependency for a calibration-time kernel;
  explicitly validated against `eigvalsh`, not just asserted safe.
- **Zero-kernel-body-change randomized/blockwise Hadamard.** Adding
  block-wise application and a per-channel random sign flip required no
  change to the FWHT kernel's math — flattening `(row, block_within_row)`
  into an "effective row" index works because the memory is already
  contiguous; only `block_within_row = effective_row % n_blocks_per_row` is
  needed to index into the shared sign vector. Low-risk way to add real
  QuaRot-style randomization.
- **One warp per output row** in the fused GEMV kernel, lanes strided across
  `K`, warp-shuffle reduction — coalesced reads of packed weight bytes, no
  shared-memory bank conflicts, no cross-warp synchronization needed.
- **`x` staged into shared memory once per thread block** in the fused GEMV
  kernel, reused by every warp (every output row) in that block.
- **Bias-correction as one add per output row, not one add per activation
  element.** Restoring the mean subtracted during activation quantization
  could be done either by re-adding `mu` to every element of `x` before the
  GEMV, or by adding `W @ mu` once per output row after it. The second is
  strictly cheaper (M adds vs. K adds per token, and M ≪ K·(number of rows)
  in aggregate) and is what the real pipeline does — `gf4_decode_kernel`
  deliberately does *not* re-add `mu` for this reason.

## Shared memory ceiling

Both the fused and naive GEMV kernels stage the full `x` vector (`K` floats)
into dynamic shared memory. At `K=11008` that's ~43KB — under the default
48KB/block limit, so the provided sizes run without extra setup. For larger
`K`, you'll need:

```cpp
cudaFuncSetAttribute(e2m1_fused_wonly_gemv_kernel,
                      cudaFuncAttributeMaxDynamicSharedMemorySize, desired_bytes);
```

before the launch, and to check `desired_bytes` against your GPU's actual
opt-in maximum (`cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlockOptin)`).
The Hessian kernels similarly stage one damped Hessian block (`block_size^2`
floats, e.g. 32x32 = 4KB) in shared memory per CUDA block — well under any
limit at the block sizes used here.

## Files

- `gf4_common.cuh` — GF4 codebook constants + encode/decode/pack helpers,
  **activations only**.
- `e2m1_common.cuh` — E2M1 codebook constants (3 exponent-bias variants) +
  encode/decode/scale-quantize helpers, **weights only**.
- `hadamard_kernel.cu` — fused single-launch randomized blockwise FWHT +
  naive O(N²) baseline.
- `gf4_encode_kernel.cu` — fused RMS+quantize warp kernel (activations) +
  naive 3-kernel baseline, both with an optional mean-centering (`mu`) input;
  plus `gf4_decode_kernel`, a new decode-only kernel used by the clip-ratio
  calibration search.
- `hessian_weight_quant_kernel.cu` — the three-kernel Hessian weight-quant
  pipeline: `hessian_accumulate` (H = X^T X / n), `hessian_damp_blocks`
  (condition-number damping via power iteration), `hessian_weight_solve`
  (per-row-block alpha/bias/code solve).
- `e2m1_fused_gemv_kernel.cu` — fused E2M1-dequant+GEMV + naive
  dequant-then-GEMV baseline (the inference-time centerpiece), both with an
  optional per-row `bias_correction` input to restore the mean subtracted
  during activation quantization.
- `gf4_fused_gemv_kernel.cu` — deprecated stub; superseded by
  `e2m1_fused_gemv_kernel.cu` once GF4-for-weights was found to be wrong.
- `bindings.cpp` — pybind11/torch glue for all of the above, no kernel logic.
- `reference.py` — NumPy/SciPy ground truth for every kernel, including an
  exact (`eigvalsh`-based) Hessian-damping reference used to validate the
  CUDA kernel's power-iteration approximation, plus `gf4_decode_vector_ref`
  and `clip_ratio_search_ref` for the mean-centering/clip-search additions.
- `bench.py` — compiles everything, runs correctness checks on synthetic
  matrices, prints timing tables. **Run this and paste the output into the
  deck's results slides.** Its printed "HBM(W) bytes fused/naive" figure is
  a CALCULATED estimate (bytes moved, from each kernel's known access
  pattern) - explicitly labeled as such in its own print statement - not a
  profiler measurement, and it comes out nearly flat across every tested
  size (~6.8x), unlike the real Nsight-measured ratio (see `profile_hbm.py`
  below), which grows with size (~4.0x at M=K=4096 to ~6.3x at M=K=11008).
  Use `profile_hbm.py`'s numbers for the deck's HBM traffic claim, not this
  script's.
- `profile_hbm.py` — minimal single-call harness for Nsight Compute (`ncu`)
  to measure REAL HBM traffic (`dram__bytes_read.sum`/`dram__bytes_write.sum`)
  for the fused vs. naive E2M1 GEMV kernels, at a configurable `--M`/`--K`.
  Requires `ncu` (ships with most CUDA toolkit installs) and, on most
  non-datacenter GPUs, elevated permissions to read GPU performance counters
  (`ERR_NVGPUCTRPERM` otherwise - run under `sudo env "PATH=$PATH"
  "HOME=$HOME" ncu ...`, preserving both so root's `python3` still finds
  your user-installed `torch`). The naive path is two kernel launches
  (`e2m1_dequantize_to_fp16_kernel` + `fp16_dense_gemv_kernel`) - sum both
  kernels' read+write bytes for the naive total; the fused path
  (`e2m1_fused_wonly_gemv_kernel`) is one launch. Real measured result on
  ricky-bobby: 3.96x less HBM traffic at M=K=4096, 6.29x at M=K=11008 - the
  deck's HBM claim now cites these numbers, not `bench.py`'s calculated
  estimate.
- `llm_quant_eval.py` — runs the real CUDA kernels on a real pretrained model
  (`facebook/opt-125m` by default, any causal LM via `--model`) with real
  calibration activations from a real forward pass, and reports WikiText-2
  perplexity for four configurations in one run: FP16 baseline; naive
  round-to-nearest (no Hessian, control); Hessian-weighted W4A16 in the raw,
  un-rotated basis (weight-only —
  `hessian_accumulate`/`hessian_damp_blocks`/`hessian_weight_solve`);
  Hessian-weighted W4A16 in the **Hadamard-rotated** basis with activations
  left FP16 (isolation ablation — same rotated/quantized weight as W4A4
  below, but activations only get the real `hadamard_fwht` rotation, no
  GF4 quantization); and W4A4, weight + activation (adds mean-centering (mu)
  and `bias_correction` from the rotated calibration activations, a real
  per-layer clip-ratio search, and a patched forward pass that fake-quantizes
  every layer's activations with the real `gf4_encode(mu=...)`/`gf4_decode`
  kernels). The Hadamard-rotated-vs-raw-basis pair exists specifically to
  separate "did rotating before quantizing weights help" from "did
  quantizing activations cost anything" — real run (OPT-125M): raw-basis
  W4A16 (50.95 PPL) was worse than naive rounding (45.58), but the same
  weights re-quantized in the rotated basis with FP16 activations came in
  much lower (35.52), showing the earlier regression was mostly a
  missing-rotation problem, not a Hessian-math problem; W4A4 added back
  +2.92 PPL on top of that for real activation quantization (38.44). OPT-1.3B
  and Llama-3.2-1B runs show the same qualitative pattern (rotation recovers
  most of the regression, activation quant costs a smaller amount on top),
  though on OPT-1.3B the raw-basis Hessian result actually beat naive
  rounding outright, unlike OPT-125M — a reminder the naive-vs-Hessian
  comparison isn't uniform across model sizes. None of the four quantized
  configs route the forward pass through the custom fused GEMV kernel — see
  the file's own docstring for that scope boundary (real future work, not
  something to bolt on right before an interview). Requires
  `pip install transformers datasets accelerate` and internet access to
  download the model + dataset on first run.

  **Larger-model support.** Two additions target models too big for the
  original single-GPU, hold-everything-in-RAM structure: (1) each target
  layer's calibration activations are now quantized in both bases (raw and
  Hadamard-rotated) immediately after capture and freed right away, instead
  of being kept in CPU RAM for the whole script — the real scaling
  bottleneck, since it previously meant O(all layers') calibration data
  resident at once (rough estimate: tens of GB at 7B+ scale) instead of
  O(one layer's); (2) `--device-map` loads the model with `accelerate`'s
  `device_map="auto"`, and `--max-memory "0=20GiB,cpu=64GiB"` / `--offload-folder`
  cap and split that placement across GPU(s), CPU RAM, and disk. Every CUDA
  kernel call stages its own inputs on a fixed `KERNEL_DEVICE` and writes
  results back to wherever that layer's weight actually lives, so this works
  regardless of how `accelerate` placed a given layer.

  **Disk offload interacts with target (to-be-quantized) layers in a way
  that needed a real fix, not just a warning.** accelerate attaches its
  dispatch hooks at the "no-split" block granularity (a whole decoder layer,
  not per-`Linear`), and by default those hooks reload a block's original
  weights from disk fresh before every `forward()` call and discard whatever
  was written into `module.weight.data` afterward — exactly what this script
  does to fake-quantize a layer, so left alone the quantized weight would
  silently get overwritten back to FP16 on the very next forward pass. The
  script now detects every target layer's hooked ancestor right after
  loading, materializes it for real, and disables *only* that hook's
  weight-offload behavior — deliberately leaving the hook itself attached,
  since its other job (moving activations across the GPU/CPU boundary on
  every call) is what makes a split model work at all; removing the hook
  outright would break the model at every device boundary instead. Net
  effect: `--offload-folder` now works even for quantized layers, at the
  cost of pulling any disk-tier block that contains one fully into CPU RAM
  for the rest of the run — trading away disk's RAM savings for that block
  in exchange for correctness. **Not** addressed either way: `original_weights`,
  a full clone of every target layer kept to restore between the four
  configs, is still one extra full-size in-memory copy of most of the model
  (in CPU RAM) — fine at 7-13B, a real limitation somewhere past that.

  **W4A16 (both bases), the naive control, and W4A4 have real numbers on
  three models (OPT-125M, OPT-1.3B, Llama-3.2-1B). The `--device-map`/
  `--max-memory`/`--offload-folder` large-model path is brand new and has
  NOT been run successfully on real hardware yet** — an initial attempt at
  Mistral-7B surfaced real bugs since fixed (an on-GPU weight clone that
  OOM'd against a tight `--max-memory` budget, and meta-tensor crashes from
  accelerate's offload hooks that needed the fix described above, not just
  a bigger memory budget) — I don't have GPU access in
  this environment, so treat the next attempt at a bigger model (something
  in the 6-8B range, e.g. Llama-3.1-8B / Mistral-7B / OPT-6.7B, is a
  reasonable next step) as a debug session, the same way every other
  kernel/pipeline change in this project has been.

## If something doesn't compile

Most likely culprits, roughly in order of likelihood: (1) a PyTorch/CUDA
version mismatch between your installed `nvcc` and the one PyTorch was built
against — `torch.utils.cpp_extension` prints this clearly if so; (2) an ABI
mismatch on the `__half` cast in `bindings.cpp` on older PyTorch versions
(fix: swap `at::Half` for whatever your version's half-precision scalar type
is called); (3) shared-memory-too-large errors for the largest GEMV test case
on older/smaller GPUs (see "Shared memory ceiling" above); (4) for the new
Hessian kernels specifically, a `HESSIAN_MAX_BLOCK` mismatch if you use a
`block_size` in `hessian_accumulate`/`hessian_weight_solve` larger than the
`#define HESSIAN_MAX_BLOCK 32` compiled into `hessian_weight_quant_kernel.cu`
— raise that define and recompile if your calibration uses wider Hessian
blocks. Send me the actual error text and I'll patch the source directly
rather than guessing.