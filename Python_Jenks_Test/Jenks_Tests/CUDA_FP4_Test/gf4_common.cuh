// gf4_common.cuh
//
// Shared constants for the GF4 (Gaussian FP4) activation-quantization CUDA
// kernels:
//   1. hadamard_kernel.cu       - randomized blockwise Hadamard (Gaussianization)
//   2. gf4_encode_kernel.cu     - fused RMS + quantile encoder (activation -> 4-bit code)
//   3. gf4_fused_gemv_kernel.cu - fused dequant + GEMV (the bottleneck-reduction kernel)
//
// IMPORTANT — this codebook is for ACTIVATIONS ONLY. Weights in the real
// pipeline are quantized separately with a standard E2M1 (exponent+mantissa)
// format via a Hessian-damped closed-form solve, not this table — see
// e2m1_common.cuh and hessian_weight_quant_kernel.cu. Corrected against the
// reference implementation (quantize_activations_gf4 / GF4_POS in the
// provided research notebook) on 2024 — an earlier version of this file
// used a self-derived codebook (bin-conditional means, un-normalized) that
// did NOT match; this one does.
//
// Real scheme: GF4_LEVEL are 8 positive quantile levels of a standard normal,
// normalized so the largest = 1.0 (not bin-conditional means). The per-block
// scale is block_rms * clip_ratio, where clip_ratio is a real calibrated
// per-layer hyperparameter (searched over candidates in [1.5, 4.0] at
// calibration time, then held fixed at inference) — NOT block_rms alone.
// |x|/scale is clamped to [0,1] and nearest-level-assigned; GF4_THRESHOLD
// below are just the midpoints between adjacent GF4_LEVEL entries, so a
// comparator ladder against them is exactly equivalent to the reference's
// argmin-over-8-levels lookup.
#pragma once
#include <cuda_fp16.h>
#include <stdint.h>

#define GF4_LEVELS 8
#define GF4_BITS 4

// Positive quantile levels of N(0,1), normalized to max = 1.0 (real GF4_POS
// from the reference implementation — used directly as reconstruction
// points, not as bin-conditional means).
__device__ __constant__ float GF4_LEVEL[GF4_LEVELS] = {
    0.0f, 0.0796082f, 0.1737177f, 0.2828685f,
    0.3952704f, 0.5250730f, 0.6961928f, 1.0f
};

// Midpoints between adjacent GF4_LEVEL entries (7 interior boundaries) —
// a comparator ladder against these reproduces nearest-level assignment.
__device__ __constant__ float GF4_THRESHOLD[GF4_LEVELS - 1] = {
    0.0398041f, 0.1266630f, 0.2282931f, 0.3390695f,
    0.4601717f, 0.6106329f, 0.8480964f
};

// (No RTL cross-check product table here anymore. The earlier version of
// this file included a 64-entry "LUTxLUT" product table pitched as matching
// the RTL's joint weight-code x activation-code MAC — but that table was
// built from the same self-derived, wrong codebook above, and we don't have
// the actual RTL codebook values to check it against. Removed rather than
// leave a fabricated cross-check in place. Also worth noting for Q&A: the
// RTL hardware slides model a symmetric GF4 x GF4 product (both operands
// GF4-coded), whereas this real software pipeline is asymmetric — weights
// are E2M1 (see e2m1_common.cuh), only activations are GF4. That's a
// legitimate difference between a hardware design-space exploration and
// the specific software recipe that produced the reported numbers, not a
// bug in either one.)

// Per-block quantization granularity. 32 matches a common attention sub-head chunk;
// pass -DGF4_BLOCK=64 etc. at compile time to sweep this.
#ifndef GF4_BLOCK
#define GF4_BLOCK 32
#endif

// Encode one fp32 value already normalized by the block scale (block_rms *
// clip_ratio) into a 4-bit GF4 code: bit 3 = sign, bits 2:0 = level index.
// Matches the reference: |x|/scale clamped to [0,1], nearest GF4_LEVEL.
__device__ __forceinline__ uint8_t gf4_encode_one(float xn) {
    uint8_t sign = xn < 0.0f ? 0x8 : 0x0;
    float m = fminf(fabsf(xn), 1.0f);   // reference clamps to [0,1] before lookup
    uint8_t idx = 0;
#pragma unroll
    for (int i = 0; i < GF4_LEVELS - 1; ++i) {
        idx += (m > GF4_THRESHOLD[i]) ? 1 : 0;
    }
    return sign | idx;
}

// Decode a 4-bit GF4 code back to a signed fp32 level in [-1, 1] (pre-scale;
// multiply by the stored block scale to recover the real magnitude).
__device__ __forceinline__ float gf4_decode_one(uint8_t code) {
    float lvl = GF4_LEVEL[code & 0x7];
    return (code & 0x8) ? -lvl : lvl;
}

// Two codes packed per byte, low nibble first.
__device__ __forceinline__ uint8_t gf4_pack(uint8_t lo, uint8_t hi) {
    return (lo & 0xF) | ((hi & 0xF) << 4);
}
__device__ __forceinline__ uint8_t gf4_unpack_lo(uint8_t packed) { return packed & 0xF; }
__device__ __forceinline__ uint8_t gf4_unpack_hi(uint8_t packed) { return (packed >> 4) & 0xF; }