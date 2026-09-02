"""
reference.py - pure NumPy/SciPy reference implementations used to check the
CUDA kernels for correctness. Nothing here runs on GPU; bench.py imports this
module and diffs its outputs against the compiled extension's outputs.

Ported from FP_Quant_Colab.ipynb (the user-provided research notebook),
following the ACTUAL call graph (quantize_model_fp -> calibrate_model_gf4 /
calibrate_model_stochastic_fp4 -> reconstruct_layer_fp_blockdiag_scaled_v5),
not a generic approximation of GPTQ-style quantization.
"""
import numpy as np
from scipy.linalg import hadamard as scipy_hadamard

# =============================================================================
# GF4 activation codebook (real GF4_POS from the notebook) - ACTIVATIONS ONLY.
# Weights use the separate E2M1 codebook further down.
# =============================================================================
GF4_LEVEL = np.array(
    [0.0, 0.0796082, 0.1737177, 0.2828685,
     0.3952704, 0.5250730, 0.6961928, 1.0], dtype=np.float32)

GF4_THRESHOLD = np.array(
    [(GF4_LEVEL[i] + GF4_LEVEL[i + 1]) / 2 for i in range(7)], dtype=np.float32)

GF4_BLOCK = 32


def gf4_encode_one(xn: np.ndarray) -> np.ndarray:
    """Vectorized version of gf4_encode_one() in gf4_common.cuh. xn is
    already normalized by scale = block_rms * clip_ratio."""
    sign = (xn < 0).astype(np.uint8) << 3
    mag = np.clip(np.abs(xn).astype(np.float32), 0.0, 1.0)  # reference clamps to [0,1]
    idx = np.zeros_like(mag, dtype=np.uint8)
    for t in GF4_THRESHOLD:
        idx += (mag > t).astype(np.uint8)
    return sign | idx


def gf4_decode_one(code: np.ndarray) -> np.ndarray:
    lvl = GF4_LEVEL[code & 0x7]
    sign = np.where((code & 0x8) != 0, -1.0, 1.0).astype(np.float32)
    return lvl * sign


def gf4_quantize_vector_ref(x: np.ndarray, block: int = GF4_BLOCK, clip_ratio: float = 2.5,
                             mu: np.ndarray = None):
    """x: 1D fp32 array, len(x) % block == 0. scale = block_rms * clip_ratio,
    matching quantize_activations_gf4 in the notebook (NOT rms alone).

    mu: optional 1D fp32 array, real pipeline's per-input-channel mean of
    post-Hadamard calibration activations (calibrate_model_stochastic_fp4:
    `x_h = x_h - mu`, applied BEFORE the RMS/quantize step). len(mu) must be
    a multiple of `block` and must evenly divide x's block count - mu is
    tiled across x's blocks the same way gf4_encode_kernel.cu's
    `block_within_row = block_id % n_blocks_per_row` indexing works, so a
    single per-channel mu can be reused across many flattened tokens."""
    assert x.shape[0] % block == 0
    n_blocks = x.shape[0] // block
    xb = x.reshape(n_blocks, block)
    if mu is not None:
        assert mu.shape[0] % block == 0, "mu length must be a multiple of block"
        n_blocks_per_row = mu.shape[0] // block
        assert n_blocks % n_blocks_per_row == 0, "n_blocks must be a multiple of mu's n_blocks_per_row"
        mu_b = mu.reshape(n_blocks_per_row, block)
        mu_tiled = np.tile(mu_b, (n_blocks // n_blocks_per_row, 1))
        xb = xb - mu_tiled
    rms = np.sqrt((xb ** 2).mean(axis=1) + 1e-12).astype(np.float32)
    scale = rms * clip_ratio
    xn = xb / scale[:, None]
    codes = gf4_encode_one(xn).reshape(-1)
    return codes, scale


def gf4_decode_vector_ref(codes: np.ndarray, scale: np.ndarray, block: int = GF4_BLOCK) -> np.ndarray:
    """Inverse of gf4_quantize_vector_ref - matches gf4_decode_kernel.cu.
    Deliberately does NOT re-add mu: the real pipeline compensates for
    mean-centering via bias_correction added after the GEMM (see
    dequant_hessian_gemv_ref's bias_correction argument), not by re-adding mu
    to every decoded activation element."""
    n_blocks = scale.shape[0]
    codes_b = codes.reshape(n_blocks, block)
    vals = gf4_decode_one(codes_b)
    return (vals * scale[:, None]).reshape(-1)


def clip_ratio_search_ref(X_cal: np.ndarray, block: int = GF4_BLOCK,
                           candidates=(1.5, 2.0, 2.5, 3.0, 4.0)) -> float:
    """Matches calibrate_model_gf4's per-layer clip-ratio (alpha*) grid
    search: for each candidate clip_ratio, do a full GF4 encode-then-decode
    round trip on the calibration activations and measure reconstruction
    MSE; return the candidate with the lowest MSE.

    X_cal: flat 1D fp32 array (caller flattens [n_tokens, hidden] calibration
    activations first), already mean-centered (mu subtracted) and
    post-Hadamard, matching what the real notebook's search operates on -
    this function does NOT subtract mu itself, since by the time the search
    runs in the real pipeline mu has already been applied via the hook."""
    best_alpha, best_mse = candidates[0], np.inf
    for alpha in candidates:
        codes, scale = gf4_quantize_vector_ref(X_cal, block=block, clip_ratio=alpha)
        X_q = gf4_decode_vector_ref(codes, scale, block=block)
        mse = float(np.mean((X_cal - X_q) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
    return best_alpha


def gf4_quantize_adaptive_vector_ref(x: np.ndarray, block: int = GF4_BLOCK,
                                      candidates=(1.5, 2.0, 2.5, 3.0, 4.0),
                                      mu: np.ndarray = None):
    """Per-block ONLINE clip selection - the CPU reference for
    gf4_encode_perblock_adaptive_kernel and a faithful port of
    quantize_activations_gf4_adaptive() in bit_split.py.

    For EACH block independently, evaluate every candidate clip_ratio (full
    encode->decode round trip) and keep the one minimizing that block's
    reconstruction MSE. No calibration state - the clip is chosen from the
    block's own data. Returns (codes, scale) with the per-block-optimal clip
    already baked into `scale`, so gf4_decode_vector_ref inverts it unchanged.

    x, mu: same conventions as gf4_quantize_vector_ref (x flat, mu optional
    per-channel and tiled across blocks). mu subtracted before RMS/quantize."""
    assert x.shape[0] % block == 0
    n_blocks = x.shape[0] // block
    xb = x.reshape(n_blocks, block).astype(np.float32)
    if mu is not None:
        assert mu.shape[0] % block == 0, "mu length must be a multiple of block"
        n_blocks_per_row = mu.shape[0] // block
        assert n_blocks % n_blocks_per_row == 0, "n_blocks must be a multiple of mu's n_blocks_per_row"
        mu_b = mu.reshape(n_blocks_per_row, block)
        xb = xb - np.tile(mu_b, (n_blocks // n_blocks_per_row, 1))
    rms = np.sqrt((xb ** 2).mean(axis=1) + 1e-12).astype(np.float32)   # [n_blocks]

    best_mse   = np.full(n_blocks, np.inf, dtype=np.float32)
    best_codes = np.zeros((n_blocks, block), dtype=np.uint8)
    best_scale = (rms * candidates[0]).astype(np.float32)
    for alpha in candidates:
        scale = (rms * alpha).astype(np.float32)                       # [n_blocks]
        codes = gf4_encode_one(xb / scale[:, None])                    # [n_blocks, block]
        recon = gf4_decode_one(codes) * scale[:, None]
        mse   = ((xb - recon) ** 2).mean(axis=1)                       # [n_blocks]
        better = mse < best_mse
        best_mse   = np.where(better, mse, best_mse)
        best_codes = np.where(better[:, None], codes, best_codes)
        best_scale = np.where(better, scale, best_scale)
    return best_codes.reshape(-1), best_scale


def pack_codes_ref(codes: np.ndarray) -> np.ndarray:
    """2 codes per byte, low nibble first - matches gf4_pack()."""
    assert codes.shape[0] % 2 == 0
    lo = codes[0::2].astype(np.uint8)
    hi = codes[1::2].astype(np.uint8)
    return (lo & 0xF) | ((hi & 0xF) << 4)


def unpack_codes_ref(packed: np.ndarray) -> np.ndarray:
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    out = np.empty(packed.shape[0] * 2, dtype=np.uint8)
    out[0::2] = lo
    out[1::2] = hi
    return out


# =============================================================================
# Randomized blockwise Hadamard transform - matches fwht_blockwise / _rotate.
# =============================================================================
def hadamard_ref(x: np.ndarray, block_size: int = None, d_sign: np.ndarray = None) -> np.ndarray:
    """x: [rows, M]. If block_size < M, applies the SAME orthonormal
    transform independently to each block_size-wide chunk (blockwise, like
    the reference's had_block_size). d_sign: optional [M] +-1 array,
    multiplied into x BEFORE the transform (randomized Hadamard / QuaRot-
    style) - the same segment of d_sign is used for every row, and for every
    block within a row it uses d_sign's slice at that block's absolute
    column offset (matching _rotate: D has length M, sliced by fwht_blockwise
    internally per block).
    """
    rows, M = x.shape
    bs = block_size or M
    n_blocks = M // bs
    H = scipy_hadamard(bs).astype(np.float32) / np.sqrt(bs)

    xw = x.copy()
    if d_sign is not None:
        xw = xw * d_sign[None, :]

    out = np.zeros_like(xw)
    for b in range(n_blocks):
        seg = xw[:, b * bs:(b + 1) * bs]
        out[:, b * bs:(b + 1) * bs] = seg @ H.T
    return out


def generate_random_signs_ref(M: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return (rng.randint(0, 2, size=M).astype(np.float32) * 2 - 1)


# =============================================================================
# E2M1 weight codebook - matches assign_fp4_dynamic / e2m1_common.cuh.
# WEIGHTS ONLY. bias in {0, 1, 2} for E_bits=2 (default=1, radius=1).
# =============================================================================
def _e2m1_codebook(bias: int) -> np.ndarray:
    e = np.arange(4)
    m = np.arange(2)
    base = 2.0 ** (e[:, None] - bias)
    mant = 1.0 + m[None, :] / 2.0
    return (base * mant).reshape(-1).astype(np.float32)  # [8]


E2M1_CODEBOOKS = {b: _e2m1_codebook(b) for b in (0, 1, 2)}


def e2m1_assign_magnitude(x: np.ndarray, bias: int) -> np.ndarray:
    cb = E2M1_CODEBOOKS[bias]
    dist = np.abs(x[..., None] - cb[None, :])
    return dist.argmin(axis=-1).astype(np.uint8)


def e2m1_decode_magnitude(code: np.ndarray, bias: int) -> np.ndarray:
    return E2M1_CODEBOOKS[bias][code & 0x7]


def quantize_scale_e4m3_ref(alpha: np.ndarray, e_bits_scale: int = 4, m_bits_scale: int = 3) -> np.ndarray:
    """Matches quantize_scale_batched in the notebook."""
    e_min = -(2 ** (e_bits_scale - 1))
    e_max = (2 ** (e_bits_scale - 1)) - 1
    a = np.clip(alpha, 1e-8, None)
    e = np.floor(np.log2(a))
    e = np.clip(e, e_min, e_max)
    base = 2.0 ** e
    levels = 2 ** m_bits_scale
    frac = a / base - 1.0
    frac_q = np.round(frac * levels) / levels
    return (base * (1.0 + frac_q)).astype(np.float32)


# =============================================================================
# Hessian-weighted weight quantization (v5) - the real, currently-shipping
# algorithm (reconstruct_layer_fp_blockdiag_scaled_v5), used here with EXACT
# eigvalsh so bench.py can measure how close the CUDA kernel's power-iteration
# approximation gets.
# =============================================================================
def damp_hessian_block_exact(H_block: np.ndarray, kappa: float = 100.0) -> np.ndarray:
    """Exact version of the adaptive condition-number damping in v5, using
    np.linalg.eigvalsh instead of power iteration - the ground truth that
    the CUDA kernel's power-iteration approximation is checked against."""
    if kappa <= 1.0:
        return H_block
    ev = np.linalg.eigvalsh(H_block)
    lam_max, lam_min = ev[-1], ev[0]
    lam_add = max(lam_max / kappa - lam_min, 0.0)
    return H_block + lam_add * np.eye(H_block.shape[0], dtype=H_block.dtype)


def power_iteration_top_eig_ref(H: np.ndarray, n_iters: int = 30) -> float:
    """NumPy mirror of the CUDA kernel's power_iteration_top_eig, used to
    validate the CUDA implementation's numerics independent of any CUDA-side
    bug (i.e. checks the ALGORITHM, not just the kernel)."""
    bs = H.shape[0]
    v = np.ones(bs, dtype=np.float64)
    lam = 0.0
    for _ in range(n_iters):
        w = H @ v
        norm = np.sqrt(max((w ** 2).sum(), 1e-20))
        v = w / norm
        lam = norm
    return lam


def damp_hessian_block_poweriter(H_block: np.ndarray, kappa: float = 100.0, n_iters: int = 30) -> np.ndarray:
    """NumPy mirror of hessian_damp_blocks_kernel's approximation, for
    validating the shifted-power-iteration approach before trusting the
    CUDA version."""
    if kappa <= 1.0:
        return H_block
    lam_max = power_iteration_top_eig_ref(H_block, n_iters)
    A = lam_max * np.eye(H_block.shape[0]) - H_block
    mu = power_iteration_top_eig_ref(A, n_iters)
    lam_min = lam_max - mu
    lam_add = max(lam_max / kappa - lam_min, 0.0)
    return H_block + lam_add * np.eye(H_block.shape[0], dtype=H_block.dtype)


def hessian_weight_quantize_row_block_ref(w_block: np.ndarray, H_damped: np.ndarray):
    """Single (row, block)'s worth of reconstruct_layer_fp_blockdiag_scaled_v5:
    bias-candidate grid search x 5-iteration alpha/code fixed point, keeping
    the (alpha, bias, codes) that minimizes Hessian-weighted residual loss.
    Returns (alpha_q, bias, codes) - codes are unsigned magnitude indices
    (0-7); sign is handled by the caller.
    """
    w_abs = np.abs(w_block).astype(np.float64)
    mean_abs = w_abs.mean()
    mean_sq = (w_abs ** 2).mean()
    alpha_init = max(np.sqrt(max(mean_sq, 1e-8)), 1e-4)
    alpha_min = 0.05 * max(mean_abs, 1e-8)
    alpha_max = 20.0 * max(mean_abs, 1e-8)

    Hw = w_abs @ H_damped.T

    best_loss = np.inf
    best_alpha, best_bias, best_codes = alpha_init, 1, None

    for bias in (0, 1, 2):
        alpha = alpha_init
        codes = None
        b_eff = None
        for _ in range(5):
            codes = e2m1_assign_magnitude(w_abs / max(alpha, 1e-8), bias)
            b_eff = e2m1_decode_magnitude(codes, bias)
            Hb = b_eff @ H_damped.T
            num = (b_eff * Hw).sum()
            den = (b_eff * Hb).sum() + 1e-8
            alpha = np.clip(num / den, alpha_min, alpha_max)

        residual = w_abs - alpha * b_eff
        Hr = residual @ H_damped.T
        loss = (residual * Hr).sum()

        if loss < best_loss:
            best_loss = loss
            best_alpha = alpha
            best_bias = bias
            best_codes = codes

    alpha_q = quantize_scale_e4m3_ref(np.array([best_alpha]))[0]
    final_codes = e2m1_assign_magnitude(w_abs / max(alpha_q, 1e-8), best_bias)
    return float(alpha_q), int(best_bias), final_codes


def quantize_weight_hessian_ref(W: np.ndarray, H_blocks: np.ndarray, block_size: int, kappa: float = 100.0):
    """W: [N, M] fp32. H_blocks: [n_blocks, block_size, block_size]. Returns
    (packed_codes [N, M/2], alpha [N, n_blocks], bias [N, n_blocks] uint8) -
    exact (eigvalsh-damped) reference for the full v5 pipeline, one row/block
    at a time (slow, correctness-only - this is a reference, not a kernel)."""
    N, M = W.shape
    n_blocks = M // block_size
    H_damped = np.stack([damp_hessian_block_exact(H_blocks[b], kappa) for b in range(n_blocks)])

    codes_full = np.zeros((N, M), dtype=np.uint8)
    alpha_out = np.zeros((N, n_blocks), dtype=np.float32)
    bias_out = np.zeros((N, n_blocks), dtype=np.uint8)

    for row in range(N):
        for b in range(n_blocks):
            w_block = W[row, b * block_size:(b + 1) * block_size]
            sign = (w_block < 0).astype(np.uint8) << 3
            alpha_q, bias, mag_codes = hessian_weight_quantize_row_block_ref(
                w_block, H_damped[b])
            codes_full[row, b * block_size:(b + 1) * block_size] = sign | mag_codes
            alpha_out[row, b] = alpha_q
            bias_out[row, b] = bias

    packed = np.stack([pack_codes_ref(codes_full[r]) for r in range(N)], axis=0)
    return packed, alpha_out, bias_out


def dequant_hessian_gemv_ref(packed: np.ndarray, alpha: np.ndarray, bias: np.ndarray,
                              x: np.ndarray, K: int, block_size: int,
                              bias_correction: np.ndarray = None):
    """Full-precision reference for y = W_e2m1 @ x (+ bias_correction).

    Fully vectorized (no per-block Python loop) - the only Python-level loop
    is the M-row unpack, which is cheap even at M~11000. An earlier version
    of this function had a second, inner loop over every block of every row
    (up to ~7.5M pure-Python iterations at M=11008/K=11008/block=16), which
    is what actually made bench.py section 4 "hang" - not a bug in the CUDA
    kernel, just an unusably slow NumPy reference for full-size matrices.

    bias_correction: optional [M] fp32 array, W_had_q @ mu precomputed on the
    calibration side. Matches e2m1_fused_gemv_kernel.cu's nullable
    bias_correction argument: since the real pipeline mean-centers
    activations (x_centered = x_had - mu) before quantizing, the GEMM
    computes W @ x_centered, and bias_correction restores the dropped mu
    term (y = W@x_centered + W@mu = W@(x_centered + mu)).
    """
    M = packed.shape[0]
    n_blocks = K // block_size
    # Only loop is over rows (cheap): unpack each row's codes.
    codes = np.stack([unpack_codes_ref(packed[m]) for m in range(M)], axis=0)  # [M, K]
    codes = codes.reshape(M, n_blocks, block_size)
    sign = np.where((codes & 0x8) != 0, -1.0, 1.0).astype(np.float32)         # [M, n_blocks, block_size]

    # Vectorized codebook gather: mag_table[m, b, :] = E2M1_CODEBOOKS[bias[m,b]]
    all_codebooks = np.stack([E2M1_CODEBOOKS[0], E2M1_CODEBOOKS[1], E2M1_CODEBOOKS[2]], axis=0)  # [3, 8]
    mag_table = all_codebooks[bias]                                           # [M, n_blocks, 8]
    row_idx = np.arange(M)[:, None, None]
    blk_idx = np.arange(n_blocks)[None, :, None]
    mag = mag_table[row_idx, blk_idx, (codes & 0x7)]                          # [M, n_blocks, block_size]

    W = (sign * mag * alpha[:, :, None]).reshape(M, K).astype(np.float32)
    y = W @ x.astype(np.float32)
    if bias_correction is not None:
        y = y + bias_correction.astype(np.float32)
    return y


def quantize_weight_e2m1_fast_ref(W: np.ndarray, block_size: int, bias: int = 1, row_chunk: int = 256):
    """Fast, fully-vectorized (non-Hessian-optimal) E2M1 quantization, used
    only to generate a VALID test weight matrix for bench.py section 4 (the
    fused-GEMV kernel test). That section is testing the GEMV kernel's
    decode-and-matmul, not the Hessian solve (already validated against
    quantize_weight_hessian_ref in section 3 at a deliberately small size) -
    so it doesn't need the slow, per-(row,block) optimal solve, just codes
    that round-trip correctly. Per-block scale = max|w| in the block, scaled
    to the codebook's max representable magnitude; single nearest-codebook
    assignment, no iteration. Chunked over rows to bound peak memory (the
    naive fully-vectorized version materializes an
    [rows, n_blocks, block_size, 8] distance tensor, which is ~4GB at
    M=K=11008 if done in one shot)."""
    N, M = W.shape
    n_blocks = M // block_size
    cb = E2M1_CODEBOOKS[bias]
    cb_max = cb.max()

    packed_chunks, alpha_chunks = [], []
    for r0 in range(0, N, row_chunk):
        r1 = min(r0 + row_chunk, N)
        Wc = W[r0:r1].reshape(r1 - r0, n_blocks, block_size)
        w_abs = np.abs(Wc)
        alpha = np.maximum(w_abs.max(axis=2) / cb_max, 1e-8).astype(np.float32)  # [rows, n_blocks]
        xn = w_abs / alpha[:, :, None]
        dist = np.abs(xn[..., None] - cb[None, None, None, :])                  # [rows, n_blocks, block_size, 8]
        codes_mag = dist.argmin(axis=-1).astype(np.uint8)
        sign = (Wc < 0).astype(np.uint8) << 3
        codes_full = (sign | codes_mag).reshape(r1 - r0, M)
        packed = np.stack([pack_codes_ref(codes_full[i]) for i in range(r1 - r0)], axis=0)
        packed_chunks.append(packed)
        alpha_chunks.append(alpha)

    packed_all = np.concatenate(packed_chunks, axis=0)
    alpha_all = np.concatenate(alpha_chunks, axis=0)
    alpha_q = quantize_scale_e4m3_ref(alpha_all.reshape(-1)).reshape(N, n_blocks).astype(np.float32)
    bias_arr = np.full((N, n_blocks), bias, dtype=np.uint8)
    return packed_all, alpha_q, bias_arr
