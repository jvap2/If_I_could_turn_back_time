// gf4_encode_kernel.cu
//
// Activation encoder: fp32/fp16 activations -> packed 4-bit GF4 codes + one
// fp16 scale per block. Mirrors the RTL "Live activation encoder" slide
// (hardware RMS -> comparator ladder -> 4-bit code) but fused into one CUDA
// kernel instead of the three separate passes a naive implementation would
// use (reduce for RMS -> write scale -> re-read x to quantize).
//
// Matches quantize_activations_gf4() in the reference implementation:
// scale = block_rms * clip_ratio, where clip_ratio is a real per-layer
// hyperparameter fixed by offline calibration (searched over candidates in
// [1.5, 4.0]), not 1.0. It's passed in here as a runtime scalar rather than
// hardcoded, since it's a per-layer constant, not something this kernel
// derives itself.
//
// Mean-centering (mu): the real pipeline's calibration
// (calibrate_model_stochastic_fp4) computes mu = per-channel mean of
// post-Hadamard activations, and every downstream activation has mu
// subtracted BEFORE the RMS/quantize step (the hook code is literally
// `x_h = x_h - mu`). This is optional here (pass a null pointer for the
// original, uncentered behavior) via the same nullable-pointer pattern
// hadamard_kernel.cu already uses for its randomized sign vector. mu has one
// value per input CHANNEL (length = row width M), so for an activation
// tensor flattened across multiple tokens, block_within_row =
// block_id % n_blocks_per_row picks out the right slice of mu - the same
// "effective row" indexing trick documented in hadamard_kernel.cu.
//
// Bottleneck being addressed: a naive encoder is 3 kernel launches touching
// global memory 2x on the activation tensor (once for the sum-of-squares
// reduction, once again to quantize using the now-known scale) plus a round
// trip through HBM for the scale itself. This kernel reads each activation
// element from HBM exactly once and writes exactly one packed byte per 2
// elements - the reduction happens in-register via warp shuffles, never
// touching global memory.
//
// One warp (32 lanes) handles exactly one GF4_BLOCK-sized chunk (GF4_BLOCK
// must be 32 for this kernel - see gf4_encode_kernel_generic below for other
// block sizes).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "gf4_common.cuh"

// ---------------------------------------------------------------------------
// Fused kernel, GF4_BLOCK == 32 (one warp per block of 32 activations).
// d_mu: optional (nullable) per-channel mean, length n_blocks_per_row*32.
// Ignored (no centering) when null.
// ---------------------------------------------------------------------------
__global__ void gf4_encode_warp32_kernel(const float* __restrict__ x,
                                          uint8_t* __restrict__ codes_packed,  // [n_blocks*16] bytes (2 codes/byte)
                                          __half* __restrict__ scales,          // [n_blocks]
                                          int n_blocks,
                                          float clip_ratio,
                                          const float* __restrict__ mu,
                                          int n_blocks_per_row) {
    const int block_id = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32); // figure out the the block_id for this warp
    if (block_id >= n_blocks) return;
    const int lane = threadIdx.x & 31; // lane within the warp (0..31)

    float xi = x[(size_t)block_id * 32 + lane]; // lane loads its own element from HBM
    if (mu != nullptr) {
        const int block_within_row = block_id % n_blocks_per_row;
        xi -= mu[block_within_row * 32 + lane];   // real pipeline: x_h = x_h - mu, before RMS/quantize
    }

    // Warp-level sum-of-squares reduction (in registers, no shared memory,
    // no global memory) - this is the "decode-at-ingress"-style amortization:
    // one reduction, reused by all 32 lanes' quantization decisions below.
    float sumsq = xi * xi;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumsq += __shfl_xor_sync(0xFFFFFFFF, sumsq, offset); // warp shuffle XOR reduction, from 16,8,4,2,1
    }
    const float rms = sqrtf(sumsq / 32.0f + 1e-12f);
    const float scale = rms * clip_ratio;           // real reference scale, not rms alone
    const float inv_scale = 1.0f / scale;

    const float xn = xi * inv_scale;                // normalize into [-1/clip, 1/clip], clamped in gf4_encode_one
    const uint8_t code = gf4_encode_one(xn);         // 4-bit code, this lane's (centered) value

    // Pack two adjacent lanes' codes into one byte. Even lane supplies the
    // low nibble, odd lane (its shfl_down neighbor) supplies the high nibble.
    uint8_t other_code = __shfl_down_sync(0xFFFFFFFF, code, 1);
    if ((lane & 1) == 0) {
        uint8_t packed = gf4_pack(code, other_code);
        codes_packed[(size_t)block_id * 16 + (lane / 2)] = packed;
    }
    if (lane == 0) {
        scales[block_id] = __float2half(scale);      // store the FULL scale (rms*clip_ratio)
    }
}

void launch_gf4_encode_warp32(const float* d_x, uint8_t* d_codes, __half* d_scales,
                               int n_blocks, float clip_ratio, const float* d_mu,
                               int n_blocks_per_row, cudaStream_t stream) {
    const int warps_per_block = 8;            // 8 warps = 256 threads/block
    const int threads = warps_per_block * 32;
    const int blocks = (n_blocks + warps_per_block - 1) / warps_per_block;
    gf4_encode_warp32_kernel<<<blocks, threads, 0, stream>>>(
        d_x, d_codes, d_scales, n_blocks, clip_ratio, d_mu, n_blocks_per_row);
}

// ---------------------------------------------------------------------------
// Baseline (unfused) 3-kernel path, for benchmarking the fusion win. Same
// math, split into the passes a first-draft implementation would naturally
// write: (1) reduce for RMS, (2) materialize the scale, (3) re-read x and
// quantize. Included so bench.py can time "naive 3-kernel" vs "fused" on
// identical inputs. Same optional mu mean-centering as the fused path above.
// ---------------------------------------------------------------------------
__global__ void gf4_reduce_rms_kernel(const float* __restrict__ x, __half* __restrict__ scales,
                                       int n_blocks, float clip_ratio,
                                       const float* __restrict__ mu, int n_blocks_per_row) {
    const int block_id = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (block_id >= n_blocks) return;
    const int lane = threadIdx.x & 31;
    float xi = x[(size_t)block_id * 32 + lane];
    if (mu != nullptr) {
        const int block_within_row = block_id % n_blocks_per_row;
        xi -= mu[block_within_row * 32 + lane];
    }
    float sumsq = xi * xi;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumsq += __shfl_xor_sync(0xFFFFFFFF, sumsq, offset);
    }
    if (lane == 0) {
        float rms = sqrtf(sumsq / 32.0f + 1e-12f);
        scales[block_id] = __float2half(rms * clip_ratio);
    }
}

__global__ void gf4_quantize_with_scale_kernel(const float* __restrict__ x,
                                                const __half* __restrict__ scales,
                                                uint8_t* __restrict__ codes_packed,
                                                int n_blocks,
                                                const float* __restrict__ mu, int n_blocks_per_row) {
    const int block_id = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (block_id >= n_blocks) return;
    const int lane = threadIdx.x & 31;
    const float scale = __half2float(scales[block_id]);  // re-read scale from HBM
    float xi = x[(size_t)block_id * 32 + lane];           // re-read x from HBM
    if (mu != nullptr) {
        const int block_within_row = block_id % n_blocks_per_row;
        xi -= mu[block_within_row * 32 + lane];           // re-read mu too (naive path, no fusion)
    }
    const float xn = xi / scale;
    const uint8_t code = gf4_encode_one(xn);
    uint8_t other_code = __shfl_down_sync(0xFFFFFFFF, code, 1);
    if ((lane & 1) == 0) {
        codes_packed[(size_t)block_id * 16 + (lane / 2)] = gf4_pack(code, other_code);
    }
}

void launch_gf4_encode_naive(const float* d_x, uint8_t* d_codes, __half* d_scales,
                              int n_blocks, float clip_ratio, const float* d_mu,
                              int n_blocks_per_row, cudaStream_t stream) {
    const int warps_per_block = 8;
    const int threads = warps_per_block * 32;
    const int blocks = (n_blocks + warps_per_block - 1) / warps_per_block;
    gf4_reduce_rms_kernel<<<blocks, threads, 0, stream>>>(
        d_x, d_scales, n_blocks, clip_ratio, d_mu, n_blocks_per_row);
    gf4_quantize_with_scale_kernel<<<blocks, threads, 0, stream>>>(
        d_x, d_scales, d_codes, n_blocks, d_mu, n_blocks_per_row);
}

// ---------------------------------------------------------------------------
// Decode: packed 4-bit GF4 codes + per-block scale -> dense fp32 activations.
// New addition - the encode kernels above previously had no CUDA-side
// counterpart at all. Needed for (a) completing the encode/decode kernel
// symmetry E2M1 already has (e2m1_fused_gemv_kernel.cu's dequantize kernel),
// and (b) the clip-ratio calibration search below, which needs a real
// encode-then-decode round trip to measure reconstruction MSE per candidate
// clip_ratio - exactly what quantize_activations_gf4() does in the reference
// (it's an encode+decode helper, not encode-only). Does NOT add mu back in -
// the real pipeline corrects for mu via bias_correction added after the
// GEMM (see e2m1_fused_gemv_kernel.cu), not by re-adding mu to the decoded
// activation itself, which is the cheaper, single-add-per-output-row way to
// do it instead of adding it to every one of the (many more) activation
// elements.
// ---------------------------------------------------------------------------
__global__ void gf4_decode_kernel(const uint8_t* __restrict__ codes_packed,
                                   const __half* __restrict__ scales,
                                   float* __restrict__ x_out,
                                   int n_blocks) {
    const int block_id = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (block_id >= n_blocks) return;
    const int lane = threadIdx.x & 31;

    const uint8_t byte = codes_packed[(size_t)block_id * 16 + (lane / 2)];
    // Matches the encode kernels' packing: even lane -> low nibble, odd lane -> high nibble.
    const uint8_t code = (lane & 1) ? gf4_unpack_hi(byte) : gf4_unpack_lo(byte);
    const float scale = __half2float(scales[block_id]);
    x_out[(size_t)block_id * 32 + lane] = gf4_decode_one(code) * scale;
}

void launch_gf4_decode(const uint8_t* d_codes, const __half* d_scales, float* d_x_out,
                        int n_blocks, cudaStream_t stream) {
    const int warps_per_block = 8;
    const int threads = warps_per_block * 32;
    const int blocks = (n_blocks + warps_per_block - 1) / warps_per_block;
    gf4_decode_kernel<<<blocks, threads, 0, stream>>>(d_codes, d_scales, d_x_out, n_blocks);
}

// ---------------------------------------------------------------------------
// Per-block ADAPTIVE-clip encoder (GF4_BLOCK == 32, one warp per block).
//
// Matches quantize_activations_gf4_adaptive() in
// FP_Quantization_Experiments/bit_split.py - the per-block ONLINE clip
// selection the original FP_Quant experiments ran. Instead of one per-layer
// clip_ratio, every block independently searches a small candidate grid and
// keeps the clip minimizing THAT block's reconstruction MSE. There is no
// calibration state: the clip is (re)chosen from the block's own data on every
// forward, so a block's alpha travels with its data, not with a layer-wide
// constant. This is the "alpha per block" iso-configuration.
//
// The candidate loop reuses the block's single RMS reduction and reduces each
// candidate's MSE in-register via warp shuffles - same amortization as the
// single-clip fused encoder. best_mse is warp-uniform after the reduction, so
// the "is this candidate better?" decision is identical on all 32 lanes (no
// divergence); each lane simply keeps its own winning code.
// d_clip_candidates: length n_candidates, device pointer (e.g. {1.5,2,2.5,3,4}).
// d_mu: optional per-channel mean (same nullable/tiling convention as above).
// ---------------------------------------------------------------------------
__global__ void gf4_encode_perblock_adaptive_kernel(const float* __restrict__ x,
                                                     uint8_t* __restrict__ codes_packed,
                                                     __half* __restrict__ scales,
                                                     int n_blocks,
                                                     const float* __restrict__ clip_candidates,
                                                     int n_candidates,
                                                     const float* __restrict__ mu,
                                                     int n_blocks_per_row) {
    const int block_id = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (block_id >= n_blocks) return;
    const int lane = threadIdx.x & 31;

    float xi = x[(size_t)block_id * 32 + lane];
    if (mu != nullptr) {
        const int block_within_row = block_id % n_blocks_per_row;
        xi -= mu[block_within_row * 32 + lane];   // real pipeline: x_h = x_h - mu, before RMS/quantize
    }

    // Per-block RMS (shared across every candidate clip) - one warp reduction.
    float sumsq = xi * xi;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumsq += __shfl_xor_sync(0xFFFFFFFF, sumsq, offset);
    }
    const float rms = sqrtf(sumsq / 32.0f + 1e-12f);

    // Online per-block clip search: each candidate is a full encode->decode
    // round trip whose block MSE is warp-reduced; keep the smallest.
    float   best_mse   = 3.4e38f;                       // ~FLT_MAX; candidate 0 always wins first
    uint8_t best_code  = 0;
    float   best_scale = rms * clip_candidates[0];
    for (int c = 0; c < n_candidates; ++c) {
        const float   scale = rms * clip_candidates[c];
        const uint8_t code  = gf4_encode_one(xi / scale);      // this lane's code for this clip
        const float   recon = gf4_decode_one(code) * scale;    // signed reconstruction
        const float   e     = xi - recon;
        float mse = e * e;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            mse += __shfl_xor_sync(0xFFFFFFFF, mse, offset);   // block sum of squared error (uniform on all lanes)
        }
        if (mse < best_mse) {          // warp-uniform decision -> no divergence
            best_mse   = mse;
            best_code  = code;         // each lane keeps its own winning code
            best_scale = scale;
        }
    }

    // Pack two adjacent lanes' winning codes, same layout as the other encoders.
    uint8_t other_code = __shfl_down_sync(0xFFFFFFFF, best_code, 1);
    if ((lane & 1) == 0) {
        codes_packed[(size_t)block_id * 16 + (lane / 2)] = gf4_pack(best_code, other_code);
    }
    if (lane == 0) {
        scales[block_id] = __float2half(best_scale);   // store the FULL per-block scale (rms*alpha*)
    }
}

void launch_gf4_encode_perblock_adaptive(const float* d_x, uint8_t* d_codes, __half* d_scales,
                                          int n_blocks, const float* d_clip_candidates,
                                          int n_candidates, const float* d_mu,
                                          int n_blocks_per_row, cudaStream_t stream) {
    const int warps_per_block = 8;
    const int threads = warps_per_block * 32;
    const int blocks = (n_blocks + warps_per_block - 1) / warps_per_block;
    gf4_encode_perblock_adaptive_kernel<<<blocks, threads, 0, stream>>>(
        d_x, d_codes, d_scales, n_blocks, d_clip_candidates, n_candidates, d_mu, n_blocks_per_row);
}
