// e2m1_common.cuh
//
// Standard E2M1 floating-point codebook used for WEIGHTS in the real
// pipeline (assign_fp4_dynamic in the reference implementation) — distinct
// from gf4_common.cuh's Gaussian codebook, which is for ACTIVATIONS only.
// 2 exponent bits + 1 mantissa bit = 8 magnitude levels (+1 sign bit = 4
// bits/value total, same container as NVFP4/MXFP4), reconstructed as
//     value = 2^(exponent - bias) * (1 + mantissa / 2)
//
// `bias` is chosen PER (ROW, BLOCK) by the Hessian-weighted solve in
// hessian_weight_quant_kernel.cu, not fixed globally: the reference searches
// bias in {default-1, default, default+1} = {0, 1, 2} for E_bits=2
// (default = 2^(E_bits-1)-1 = 1, bias_radius = max(1, 2^(E_bits-2)) = 1).
// Precomputing all 3 candidate codebooks up front turns per-element encode
// into an 8-way argmin against a small constant-memory table indexed by
// [bias][code] - no exponentiation needed at encode or decode time.
#pragma once
#include <stdint.h>

#define E2M1_LEVELS 8          // 2^E_bits * 2^M_bits = 4 * 2
#define E2M1_NUM_BIAS 3        // bias in {0, 1, 2}

// CODEBOOK[bias][code], code = exponent*2 + mantissa, magnitude only (unsigned).
// bias=0: 2^e       * (1+m/2), e=0..3 -> base 1,2,4,8
// bias=1: 2^(e-1)   * (1+m/2), e=0..3 -> base 0.5,1,2,4   (this is the E_bits=2 default)
// bias=2: 2^(e-2)   * (1+m/2), e=0..3 -> base 0.25,0.5,1,2
__device__ __constant__ float E2M1_CODEBOOK[E2M1_NUM_BIAS][E2M1_LEVELS] = {
    {1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f},     // bias=0
    {0.5f, 0.75f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f},     // bias=1 (default)
    {0.25f, 0.375f, 0.5f, 0.75f, 1.0f, 1.5f, 2.0f, 3.0f},  // bias=2
};

// Nearest-magnitude assignment: given |w|/alpha and a bias candidate, return
// the 3-bit code (0-7) whose E2M1_CODEBOOK[bias][code] is closest. Brute-force
// 8-way argmin — cheap, branchless-enough, and exactly matches the
// reference's argmin over the same 8-entry codebook (assign_fp4_dynamic).
__device__ __forceinline__ uint8_t e2m1_assign_magnitude(float x, int bias) {
    const float* cb = E2M1_CODEBOOK[bias];
    uint8_t best = 0;
    float best_d = fabsf(x - cb[0]);
#pragma unroll
    for (int k = 1; k < E2M1_LEVELS; ++k) {
        float d = fabsf(x - cb[k]);
        if (d < best_d) { best_d = d; best = (uint8_t)k; }
    }
    return best;
}

__device__ __forceinline__ float e2m1_decode_magnitude(uint8_t code, int bias) {
    return E2M1_CODEBOOK[bias][code & 0x7];
}

// Full signed encode/decode, 4-bit code = sign(bit3) | magnitude(bits2:0).
__device__ __forceinline__ uint8_t e2m1_encode_one(float w, float alpha, int bias) {
    uint8_t sign = w < 0.0f ? 0x8 : 0x0;
    float x = fabsf(w) / fmaxf(alpha, 1e-8f);
    uint8_t mag = e2m1_assign_magnitude(x, bias);
    return sign | mag;
}

__device__ __forceinline__ float e2m1_decode_one(uint8_t code, float alpha, int bias) {
    float mag = e2m1_decode_magnitude(code, bias);
    float signed_mag = (code & 0x8) ? -mag : mag;
    return signed_mag * alpha;
}

// quantize_scale: pack a real-valued alpha into an e_bits_scale/m_bits_scale
// float format (E4M3 by default: e_bits_scale=4, m_bits_scale=3), matching
// quantize_scale_batched in the reference. Returns the quantized VALUE
// (already-decoded float), not the raw bit pattern - this is what alpha
// becomes after being "packed" into the scale format and read back.
__device__ __forceinline__ float quantize_scale_e4m3(float alpha, int e_bits_scale, int m_bits_scale) {
    int e_min = -(1 << (e_bits_scale - 1));
    int e_max = (1 << (e_bits_scale - 1)) - 1;
    float a = fmaxf(alpha, 1e-8f);
    float e = floorf(log2f(a));
    e = fminf(fmaxf(e, (float)e_min), (float)e_max);
    float base = exp2f(e);
    if (m_bits_scale > 0) {
        float levels = (float)(1 << m_bits_scale);
        float frac = a / base - 1.0f;
        float frac_q = roundf(frac * levels) / levels;
        return base * (1.0f + frac_q);
    }
    return base;
}