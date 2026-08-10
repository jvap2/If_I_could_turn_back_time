"""
bench.py - build the CUDA extension, check correctness against reference.py,
then time fused vs. naive/baseline kernels. Run this on a CUDA machine with
PyTorch installed:

    python bench.py

First run takes ~30-60s (nvcc compiling the extension); subsequent runs use
the cached build in ./build/. Requires an SM70+ GPU (fp16 + warp shuffle) and
PyTorch >= 1.10ish with a CUDA build matching your installed nvcc/driver.

Covers the full pipeline: randomized blockwise Hadamard -> GF4 activation
encoder -> Hessian-damped E2M1 weight quantization -> fused dequant+GEMV ->
mean-centering (mu) + bias-correction + per-layer clip-ratio calibration
search (section 5 - added to match the real notebook's
calibrate_model_stochastic_fp4/calibrate_model_gf4, confirmed as active,
currently-used parts of the production pipeline, not optional extras).
Copy the printed numbers into the "CUDA KERNELS - RESULTS" slide.
"""
import time
import numpy as np
import torch
from torch.utils.cpp_extension import load

import reference as ref

print("Compiling CUDA extension (nvcc) - first run may take a minute...")
ext = load(
    name="gf4_kernels",
    sources=[
        "bindings.cpp",
        "hadamard_kernel.cu",
        "gf4_encode_kernel.cu",
        "e2m1_fused_gemv_kernel.cu",
        "hessian_weight_quant_kernel.cu",
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)
print("Build OK.\n")

device = torch.device("cuda")
torch.manual_seed(0)


def cuda_time(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms/iter


# ============================================================================
# 1. Randomized blockwise Hadamard transform: correctness + timing
# ============================================================================
print("=" * 72)
print("1. RANDOMIZED BLOCKWISE HADAMARD TRANSFORM  (fused kernel vs naive O(N^2))")
print("=" * 72)
HAD_BLOCK = 128
for N in [128, 256, 512, 1024]:
    rows = 64
    x_np = np.random.randn(rows, N).astype(np.float32)
    x = torch.from_numpy(x_np).to(device)

    d_sign_np = ref.generate_random_signs_ref(N, seed=0)
    d_sign = torch.from_numpy(d_sign_np).to(device)

    y_fused = ext.hadamard_fwht(x, HAD_BLOCK, d_sign).cpu().numpy()
    y_naive = ext.hadamard_naive(x, HAD_BLOCK, d_sign).cpu().numpy()
    y_ref = ref.hadamard_ref(x_np, block_size=min(HAD_BLOCK, N), d_sign=d_sign_np)

    err_fused = np.abs(y_fused - y_ref).max()
    err_naive = np.abs(y_naive - y_ref).max()
    assert err_fused < 1e-2, f"fused Hadamard mismatch, max err {err_fused}"
    assert err_naive < 1e-2, f"naive Hadamard mismatch, max err {err_naive}"

    t_fused = cuda_time(lambda: ext.hadamard_fwht(x, HAD_BLOCK, d_sign))
    t_naive = cuda_time(lambda: ext.hadamard_naive(x, HAD_BLOCK, d_sign))
    print(f"  N={N:5d}  fused={t_fused*1000:8.2f} us   naive={t_naive*1000:8.2f} us   "
          f"speedup={t_naive/t_fused:5.2f}x   max_err(fused)={err_fused:.2e}")

# ============================================================================
# 2. GF4 activation encoder: correctness + timing
# ============================================================================
print()
print("=" * 72)
print("2. GF4 ACTIVATION ENCODER  (fused RMS+quantize vs naive 3-kernel path)")
print("=" * 72)
CLIP_RATIO = 2.5
for n_elems in [32 * 1024, 32 * 65536, 32 * 262144]:
    x_np = np.random.randn(n_elems).astype(np.float32) * 3.0  # exercise a few outlier codes
    x = torch.from_numpy(x_np).to(device)

    codes_f, scales_f = ext.gf4_encode(x, CLIP_RATIO, True)
    codes_n, scales_n = ext.gf4_encode(x, CLIP_RATIO, False)
    codes_ref, scales_ref = ref.gf4_quantize_vector_ref(x_np, clip_ratio=CLIP_RATIO)
    packed_ref = ref.pack_codes_ref(codes_ref)

    # Compare unpacked per-element codes with a small tolerance for mismatch
    # rate rather than requiring bit-exact equality. Two independent sources
    # of sub-ULP noise vs. the exact NumPy reference: (1) GPU warp-shuffle
    # tree reduction sums the same 32 values in a different order than
    # NumPy's reduction, and (2) this extension is compiled with
    # --use_fast_math, which swaps IEEE division for the approximate
    # __fdividef (~1-2 ULP looser). Both are large enough to flip a sample
    # that lands almost exactly on a quantization threshold, but nowhere
    # near large enough to move a typical sample. Empirically this runs
    # ~0.01-0.05% mismatch (a few codes in 10,000) - real boundary noise,
    # not a bug. A genuine logic error would show up as a mismatch rate
    # many orders of magnitude larger (wrong codebook, wrong block size,
    # wrong pack order, etc.), so 5e-3 (0.5%) is a safe ceiling that still
    # catches real bugs while not chasing floating-point ghosts.
    def mismatch_rate(a_packed, b_packed):
        a = ref.unpack_codes_ref(a_packed)
        b = ref.unpack_codes_ref(b_packed)
        return (a != b).mean()

    mr_f = mismatch_rate(codes_f.cpu().numpy(), packed_ref)
    mr_n = mismatch_rate(codes_n.cpu().numpy(), packed_ref)
    assert mr_f < 5e-3, f"fused encoder mismatch rate too high vs reference: {mr_f:.2e}"
    assert mr_n < 5e-3, f"naive encoder mismatch rate too high vs reference: {mr_n:.2e}"
    assert np.allclose(scales_f.cpu().float().numpy(), scales_ref, atol=1e-2)

    t_fused = cuda_time(lambda: ext.gf4_encode(x, CLIP_RATIO, True))
    t_naive = cuda_time(lambda: ext.gf4_encode(x, CLIP_RATIO, False))
    print(f"  n={n_elems:9d}  fused={t_fused*1000:8.2f} us   naive={t_naive*1000:8.2f} us   "
          f"speedup={t_naive/t_fused:5.2f}x   (code mismatch vs ref: {mr_f:.1e})")

# ============================================================================
# 3. Hessian-damped E2M1 weight quantization: correctness (3 sub-kernels)
# ============================================================================
print()
print("=" * 72)
print("3. HESSIAN-WEIGHTED E2M1 WEIGHT QUANTIZATION  (reconstruct_layer_..._v5)")
print("=" * 72)
HESS_BLOCK = 32
KAPPA = 100.0
POWER_ITERS = 30

M_rows, K_cols, n_tokens = 256, HESS_BLOCK * 4, 512
n_blocks = K_cols // HESS_BLOCK

X_np = np.random.randn(n_tokens, K_cols).astype(np.float32)
X = torch.from_numpy(X_np).to(device)

# --- 3a. Hessian accumulation: H = X^T X / n_tokens, block-diagonal ---
H_cuda = ext.hessian_accumulate(X, HESS_BLOCK).cpu().numpy()
H_ref = np.zeros((n_blocks, HESS_BLOCK, HESS_BLOCK), dtype=np.float32)
for b in range(n_blocks):
    Xb = X_np[:, b * HESS_BLOCK:(b + 1) * HESS_BLOCK]
    H_ref[b] = (Xb.T @ Xb) / n_tokens
err_h = np.abs(H_cuda - H_ref).max() / (np.abs(H_ref).max() + 1e-8)
assert err_h < 1e-2, f"hessian_accumulate mismatch: rel err {err_h:.2e}"
print(f"  3a. hessian_accumulate:        rel_err={err_h:.2e}  (n_tokens={n_tokens}, "
      f"n_blocks={n_blocks}, block={HESS_BLOCK})")

# --- 3b. Condition-number damping: power iteration (CUDA) vs exact eigvalsh (NumPy) ---
# This is the measured tradeoff, not a hidden one: the CUDA kernel avoids
# cuSOLVER by approximating lambda_max/lambda_min with shifted power
# iteration. We validate that approximation two ways: (i) a pure-NumPy
# mirror of the same algorithm (damp_hessian_block_poweriter) to confirm the
# ALGORITHM converges to the right answer, and (ii) the actual CUDA kernel
# output vs. the exact eigvalsh-damped reference, to confirm the KERNEL
# implements that algorithm correctly.
H_damped_cuda = ext.hessian_damp_blocks(torch.from_numpy(H_ref).to(device), KAPPA, POWER_ITERS).cpu().numpy()
H_damped_exact = np.stack([ref.damp_hessian_block_exact(H_ref[b], KAPPA) for b in range(n_blocks)])
H_damped_poweriter_np = np.stack([ref.damp_hessian_block_poweriter(H_ref[b], KAPPA, POWER_ITERS) for b in range(n_blocks)])

err_algo = np.abs(H_damped_poweriter_np - H_damped_exact).max() / (np.abs(H_damped_exact).max() + 1e-8)
err_kernel = np.abs(H_damped_cuda - H_damped_exact).max() / (np.abs(H_damped_exact).max() + 1e-8)
print(f"  3b. hessian_damp_blocks:       power-iter algorithm vs exact eigvalsh: rel_err={err_algo:.2e}")
print(f"                                 CUDA kernel vs exact eigvalsh:         rel_err={err_kernel:.2e}")
assert err_kernel < 5e-2, f"hessian_damp_blocks kernel too far from eigvalsh-exact reference: {err_kernel:.2e}"

# --- 3c. Per-(row,block) alpha/bias/codes solve ---
W_np = np.random.randn(M_rows, K_cols).astype(np.float32) * 0.02
W_codes_cuda, W_alpha_cuda, W_bias_cuda = ext.hessian_weight_solve(
    torch.from_numpy(W_np).to(device), torch.from_numpy(H_damped_exact).to(device))
W_codes_cuda = W_codes_cuda.cpu().numpy()
W_alpha_cuda = W_alpha_cuda.cpu().float().numpy()
W_bias_cuda = W_bias_cuda.cpu().numpy()

packed_ref, alpha_ref, bias_ref = ref.quantize_weight_hessian_ref(W_np, H_ref, HESS_BLOCK, KAPPA)

y_cuda = ref.dequant_hessian_gemv_ref(W_codes_cuda, W_alpha_cuda, W_bias_cuda,
                                       np.ones(K_cols, dtype=np.float32), K_cols, HESS_BLOCK)
y_ref = ref.dequant_hessian_gemv_ref(packed_ref, alpha_ref, bias_ref,
                                      np.ones(K_cols, dtype=np.float32), K_cols, HESS_BLOCK)
rel_err_solve = np.abs(y_cuda - y_ref).max() / (np.abs(y_ref).max() + 1e-6)
assert rel_err_solve < 5e-2, f"hessian_weight_solve reconstruction too far off: {rel_err_solve:.2e}"
print(f"  3c. hessian_weight_solve:      dequant(ones) rel_err vs CPU reference={rel_err_solve:.2e}  "
      f"(M={M_rows}, K={K_cols})")

t_accum = cuda_time(lambda: ext.hessian_accumulate(X, HESS_BLOCK))
t_damp = cuda_time(lambda: ext.hessian_damp_blocks(torch.from_numpy(H_ref).to(device), KAPPA, POWER_ITERS))
t_solve = cuda_time(lambda: ext.hessian_weight_solve(
    torch.from_numpy(W_np).to(device), torch.from_numpy(H_damped_exact).to(device)))
print(f"  timing: accumulate={t_accum*1000:8.2f} us   damp={t_damp*1000:8.2f} us   "
      f"solve={t_solve*1000:8.2f} us   (calibration-time kernels, run once per layer offline)")

# ============================================================================
# 4. Fused E2M1-dequant + GEMV: correctness + timing (the centerpiece kernel)
# ============================================================================
print()
print("=" * 72)
print("4. FUSED E2M1-DEQUANT + GEMV  (weight-only W4A16, single-token decode)")
print("=" * 72)
E2M1_BLOCK = 16
for (M, K) in [(4096, 4096), (11008, 4096), (11008, 11008)]:
    n_blocks_gemv = K // E2M1_BLOCK
    W_np = (np.random.randn(M, K).astype(np.float32)) * 0.02
    x_np = np.random.randn(K).astype(np.float32)

    # This section tests the GEMV kernel's decode+matmul, not the Hessian
    # solve (already validated against the slow, exact
    # quantize_weight_hessian_ref in section 3 at a deliberately small size).
    # Using that same per-(row,block) Python solve here at full production
    # size (up to 11008x11008 / 16-elem blocks = ~7.5M pure-Python
    # iterations) is what made this section look "hung" - it wasn't a bug,
    # just a reference implementation applied at the wrong scale. The fast
    # vectorized quantizer below produces valid (if not variance-optimal)
    # E2M1 codes in a fraction of a second at any size.
    packed_np, alpha_np, bias_np = ref.quantize_weight_e2m1_fast_ref(W_np, E2M1_BLOCK)
    y_ref = ref.dequant_hessian_gemv_ref(packed_np, alpha_np, bias_np, x_np, K, E2M1_BLOCK)

    W_codes = torch.from_numpy(packed_np).to(device)
    W_alpha = torch.from_numpy(alpha_np).to(device).half()
    W_bias = torch.from_numpy(bias_np).to(device)
    x = torch.from_numpy(x_np).to(device).half()

    y_fused = ext.e2m1_fused_gemv(W_codes, W_alpha, W_bias, x, K).float().cpu().numpy()
    y_naive = ext.e2m1_naive_gemv(W_codes, W_alpha, W_bias, x, K).float().cpu().numpy()

    rel_err_fused = np.abs(y_fused - y_ref).max() / (np.abs(y_ref).max() + 1e-6)
    rel_err_cross = np.abs(y_fused - y_naive).max() / (np.abs(y_naive).max() + 1e-6)
    assert rel_err_fused < 5e-2, f"fused GEMV vs fp32 reference too far off: {rel_err_fused}"
    assert rel_err_cross < 1e-2, f"fused vs naive CUDA paths disagree: {rel_err_cross}"

    t_fused = cuda_time(lambda: ext.e2m1_fused_gemv(W_codes, W_alpha, W_bias, x, K))
    t_naive = cuda_time(lambda: ext.e2m1_naive_gemv(W_codes, W_alpha, W_bias, x, K))

    # NOTE: bytes_fused/bytes_naive below are a CALCULATED estimate (bytes
    # moved, from each kernel's known access pattern given M/K) - not a
    # profiler measurement. This estimate comes out nearly IDENTICAL across
    # all three sizes tested here (~6.8x), i.e. it does NOT predict the
    # ratio growing with matrix size, unlike the wall-clock speedup above,
    # which does. Real Nsight Compute profiling (see profile_hbm.py,
    # dram__bytes_read/write.sum) tells a different, more interesting story:
    # the MEASURED ratio actually goes from ~4.0x at M=K=4096 to ~6.3x at
    # M=K=11008 - it DOES grow with size, just not for the reason this rough
    # byte-counting model would suggest. Use profile_hbm.py's real numbers
    # for the deck, not this print's calculated ones.
    bytes_fused = M * K // 2 + M * n_blocks_gemv * 2 + M * n_blocks_gemv + K * 2
    bytes_naive = bytes_fused - (K * 2) + (M * K * 2) + (M * K * 2 + K * 2)
    print(f"  M={M:6d} K={K:6d}  fused={t_fused*1000:8.2f} us   naive={t_naive*1000:8.2f} us   "
          f"speedup={t_naive/t_fused:5.2f}x   "
          f"HBM(W) bytes fused/naive (CALCULATED, not measured) = "
          f"{bytes_fused/1e6:7.2f}MB / {bytes_naive/1e6:7.2f}MB "
          f"({bytes_naive/bytes_fused:.1f}x less - see profile_hbm.py for the real, Nsight-measured number)")

# ============================================================================
# 5. Mean-centering + bias-correction + clip-ratio calibration search.
#
# These three additions were NOT in the original kernel pass - they were
# added after re-reading the real notebook source and confirming all three
# are active, currently-used parts of the production pipeline (not optional
# extras): calibrate_model_stochastic_fp4 computes mu = per-channel mean of
# post-Hadamard activations and subtracts it before quantizing (mu is later
# compensated for via bias_correction = W_had_q @ mu, added after the GEMM),
# and calibrate_model_gf4 does a per-layer grid search over clip_ratio
# candidates to minimize encode+decode round-trip MSE. See gf4_encode_kernel.cu
# (mu param + new gf4_decode kernel) and e2m1_fused_gemv_kernel.cu
# (bias_correction param) for the kernel-level implementation.
# ============================================================================
print()
print("=" * 72)
print("5. MEAN-CENTERING + BIAS-CORRECTION + CLIP-RATIO CALIBRATION SEARCH")
print("=" * 72)

# --- 5a. Mean-centered GF4 encode vs reference ---
ROW_WIDTH = 4096  # must be a multiple of 32
N_TOKENS_5 = 64

mu_np = (np.random.randn(ROW_WIDTH).astype(np.float32) * 0.5)
X5_np = np.random.randn(N_TOKENS_5, ROW_WIDTH).astype(np.float32) * 3.0
X5_flat_np = X5_np.reshape(-1)

mu = torch.from_numpy(mu_np).to(device)
X5 = torch.from_numpy(X5_flat_np).to(device)

codes_mu_f, scales_mu_f = ext.gf4_encode(X5, CLIP_RATIO, True, mu)
codes_mu_n, scales_mu_n = ext.gf4_encode(X5, CLIP_RATIO, False, mu)
codes_mu_ref, scales_mu_ref = ref.gf4_quantize_vector_ref(X5_flat_np, clip_ratio=CLIP_RATIO, mu=mu_np)
packed_mu_ref = ref.pack_codes_ref(codes_mu_ref)

mr_mu_f = mismatch_rate(codes_mu_f.cpu().numpy(), packed_mu_ref)
mr_mu_n = mismatch_rate(codes_mu_n.cpu().numpy(), packed_mu_ref)
assert mr_mu_f < 5e-3, f"mean-centered fused encoder mismatch rate too high: {mr_mu_f:.2e}"
assert mr_mu_n < 5e-3, f"mean-centered naive encoder mismatch rate too high: {mr_mu_n:.2e}"
assert np.allclose(scales_mu_f.cpu().float().numpy(), scales_mu_ref, atol=1e-2)
print(f"  5a. gf4_encode(mu=...):        mismatch vs ref={mr_mu_f:.1e}   "
      f"(row_width={ROW_WIDTH}, tokens={N_TOKENS_5})")

# --- 5b. GF4 decode round trip (new kernel - GF4 previously had no decode at all) ---
n_blocks_5 = X5_flat_np.shape[0] // 32
x_decoded_cuda = ext.gf4_decode(codes_mu_f, scales_mu_f, n_blocks_5).cpu().numpy()
x_decoded_ref = ref.gf4_decode_vector_ref(codes_mu_ref, scales_mu_ref)
err_decode = np.abs(x_decoded_cuda - x_decoded_ref).max()
assert err_decode < 1e-2, f"gf4_decode mismatch vs reference: {err_decode:.2e}"
print(f"  5b. gf4_decode round trip:     max_err vs ref={err_decode:.2e}")

# --- 5c. Bias-corrected fused/naive GEMV vs reference ---
M5, K5 = 4096, 4096
W5_np = (np.random.randn(M5, K5).astype(np.float32)) * 0.02
x5_np = np.random.randn(K5).astype(np.float32)
bias_corr_np = np.random.randn(M5).astype(np.float32) * 0.01  # stand-in for W_had_q @ mu

packed5_np, alpha5_np, bias5_np = ref.quantize_weight_e2m1_fast_ref(W5_np, E2M1_BLOCK)
y_ref_bc = ref.dequant_hessian_gemv_ref(packed5_np, alpha5_np, bias5_np, x5_np, K5, E2M1_BLOCK,
                                         bias_correction=bias_corr_np)

W5_codes = torch.from_numpy(packed5_np).to(device)
W5_alpha = torch.from_numpy(alpha5_np).to(device).half()
W5_bias = torch.from_numpy(bias5_np).to(device)
x5 = torch.from_numpy(x5_np).to(device).half()
bias_corr = torch.from_numpy(bias_corr_np).to(device)

y_fused_bc = ext.e2m1_fused_gemv(W5_codes, W5_alpha, W5_bias, x5, K5, bias_corr).float().cpu().numpy()
y_naive_bc = ext.e2m1_naive_gemv(W5_codes, W5_alpha, W5_bias, x5, K5, bias_corr).float().cpu().numpy()

rel_err_fused_bc = np.abs(y_fused_bc - y_ref_bc).max() / (np.abs(y_ref_bc).max() + 1e-6)
rel_err_naive_bc = np.abs(y_naive_bc - y_ref_bc).max() / (np.abs(y_ref_bc).max() + 1e-6)
assert rel_err_fused_bc < 5e-2, f"bias-corrected fused GEMV vs reference too far off: {rel_err_fused_bc}"
assert rel_err_naive_bc < 5e-2, f"bias-corrected naive GEMV vs reference too far off: {rel_err_naive_bc}"
print(f"  5c. bias-corrected GEMV:       fused rel_err={rel_err_fused_bc:.2e}   "
      f"naive rel_err={rel_err_naive_bc:.2e}  (M={M5}, K={K5})")

# --- 5d. Clip-ratio calibration search: CUDA-kernel-driven vs reference ---
# Mirrors calibrate_model_gf4's per-layer grid search over candidate
# clip_ratios, each evaluated by a full encode+decode round trip, keeping
# whichever minimizes reconstruction MSE. X_cal here stands in for the
# already-mean-centered, post-Hadamard calibration activations the real
# search operates on.
CANDIDATES = (1.5, 2.0, 2.5, 3.0, 4.0)
X_cal_np = (np.random.randn(2048, 1024).astype(np.float32) * 1.5).reshape(-1)
X_cal = torch.from_numpy(X_cal_np).to(device)
n_blocks_cal = X_cal_np.shape[0] // 32

best_alpha_cuda, best_mse_cuda = CANDIDATES[0], float("inf")
for alpha in CANDIDATES:
    codes_c, scales_c = ext.gf4_encode(X_cal, alpha, True)
    x_dec_c = ext.gf4_decode(codes_c, scales_c, n_blocks_cal).cpu().numpy()
    mse_c = float(np.mean((X_cal_np - x_dec_c) ** 2))
    if mse_c < best_mse_cuda:
        best_mse_cuda, best_alpha_cuda = mse_c, alpha

best_alpha_ref = ref.clip_ratio_search_ref(X_cal_np, candidates=CANDIDATES)
assert best_alpha_cuda == best_alpha_ref, (
    f"clip-ratio search disagreement: CUDA picked {best_alpha_cuda}, reference picked {best_alpha_ref}")
print(f"  5d. clip-ratio search:         CUDA picked alpha*={best_alpha_cuda}   "
      f"reference picked alpha*={best_alpha_ref}   (candidates={CANDIDATES})")

print()
print("All correctness checks passed. Paste the tables above into the")
print("'CUDA KERNELS - RESULTS' slide.")
