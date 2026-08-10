// e2m1_fused_gemv_kernel.cu  (formerly gf4_fused_gemv_kernel.cu)
//
// The centerpiece kernel: fused E2M1-dequant + GEMV, y[M] = W[M,K] @ x[K].
//
// Renamed and corrected from an earlier version that decoded weights with
// the GF4 (Gaussian activation) codebook. Tracing the reference
// implementation's actual call graph showed weights are quantized
// separately, with a standard E2M1 (exponent+mantissa) format via the
// Hessian-damped solve in hessian_weight_quant_kernel.cu - GF4 is for
// ACTIVATIONS only (see gf4_common.cuh). Same 4-bit container, same
// bottleneck-reduction story below, different decode formula and an extra
// per-block bias input the GF4 path didn't need.
//
// Weight-only 4-bit quantization (W4A16 - the same regime the deck uses for
// outlier layers like down_proj/fc2/LM head). This is the realistic bottleneck
// for autoregressive LLM decode at batch size 1: the activation vector x is
// tiny (K half-precision values) and stays resident in cache/shared memory,
// while the weight matrix W is huge and must stream from HBM exactly once
// per token - so the entire kernel's runtime is set by how many bytes of W
// it reads, not by FLOPs.
//
// The deck's own PTQ pipeline says it plainly on the "SOFTWARE - PTQ PIPELINE"
// slide: perplexity numbers are measured under fake-quantization, i.e. "4-bit
// codes reconstructed to FP16 before the GEMM." That reconstruction, done as
// its own pass, is exactly the bottleneck this kernel removes:
//
//   Baseline (fake-quantization protocol, as a real kernel):
//     kernel 1: read  M*K/2 bytes (codes) + M*(K/bs)*2 (alpha) + M*(K/bs) (bias)
//               write M*K*2 bytes (dequantized fp16 W)          <- new, avoidable traffic
//     kernel 2: read  M*K*2 bytes (dequantized W) + K*2 bytes (x)
//               write M*2 bytes (y)
//
//   Fused kernel (this file):
//     read  M*K/2 bytes (codes) + M*(K/bs)*2 (alpha) + M*(K/bs) (bias) + K*2 (x)
//     write M*2 bytes (y)
//     -> the dequantized W never touches HBM at all.
//
// One warp per output row: lanes stride across K, decode+FMA in registers,
// warp-shuffle reduce to the row's dot product. x is staged into shared
// memory once per block and reused by every warp (every row) in that block.
//
// Bias-correction: the real pipeline mean-centers activations before GF4
// quantization (x_had_centered = x_had - mu, see gf4_encode_kernel.cu) and
// compensates for the dropped mu term by adding bias_correction = W_had_q @ mu
// back onto the GEMM output, since y = W@(x_centered + mu) = W@x_centered +
// W@mu. bias_correction is precomputed once per layer on the host/calibration
// side (it only depends on W and mu, not on the per-token activation), so
// this kernel just adds one extra fp32 scalar per output row - far cheaper
// than adding mu back to every element of x before the dot product. Nullable:
// pass nullptr to recover the original (uncentered) behavior.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "e2m1_common.cuh"

#ifndef GEMV_ROWS_PER_BLOCK
#define GEMV_ROWS_PER_BLOCK 8   // 8 warps per block -> 8 output rows per block
#endif

#ifndef E2M1_BLOCK
#define E2M1_BLOCK 16   // matches BS=16 used for the weight-quant sweep in the reference notebook
#endif

// ---------------------------------------------------------------------------
// Fused kernel: decode-on-the-fly, never materializes dequantized W.
// W_codes: [M, K/2] packed bytes (2 codes/byte, low nibble = even column).
// W_alpha: [M, K/E2M1_BLOCK] fp16 per-block scale.
// W_bias:  [M, K/E2M1_BLOCK] uint8 per-block E2M1 exponent bias (0,1,2).
// x: [K] fp16 dense activation vector.
// y: [M] fp16 output.
// ---------------------------------------------------------------------------
__global__ void e2m1_fused_wonly_gemv_kernel(const uint8_t* __restrict__ W_codes,
                                              const __half* __restrict__ W_alpha,
                                              const uint8_t* __restrict__ W_bias,
                                              const __half* __restrict__ x,
                                              __half* __restrict__ y,
                                              int M, int K,
                                              const float* __restrict__ bias_correction) {
    extern __shared__ float x_sh[];   // K floats, staged once per block
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    for (int k = tid; k < K; k += threads) x_sh[k] = __half2float(x[k]);
    __syncthreads();

    const int warp = tid / 32;
    const int lane = tid & 31;
    const int row = blockIdx.x * GEMV_ROWS_PER_BLOCK + warp; // one warp per output row
    if (row >= M) return;

    const int blocks_per_row = K / E2M1_BLOCK;
    const uint8_t* row_codes = W_codes + (size_t)row * (K / 2);
    const __half* row_alpha = W_alpha + (size_t)row * blocks_per_row;
    const uint8_t* row_bias = W_bias + (size_t)row * blocks_per_row;

    float acc = 0.0f;
    // Each lane owns every 32nd packed byte (= 64 K-columns apart), so 32
    // lanes together cover all K/2 bytes of the row with coalesced reads.
    const int bytes_per_row = K / 2;
    for (int b = lane; b < bytes_per_row; b += 32) {
        uint8_t packed = row_codes[b];
        int k0 = b * 2;
        int k1 = k0 + 1;
        int blk0 = k0 / E2M1_BLOCK;
        int blk1 = k1 / E2M1_BLOCK;
        float a0 = __half2float(row_alpha[blk0]);
        float a1 = __half2float(row_alpha[blk1]);
        int bias0 = row_bias[blk0];
        int bias1 = row_bias[blk1];
        float w0 = e2m1_decode_one(packed & 0xF, a0, bias0);
        float w1 = e2m1_decode_one((packed >> 4) & 0xF, a1, bias1);
        acc += w0 * x_sh[k0] + w1 * x_sh[k1];
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);
    }
    if (lane == 0) {
        const float bc = (bias_correction != nullptr) ? bias_correction[row] : 0.0f;
        y[row] = __float2half(acc + bc);
    }
}

void launch_e2m1_fused_gemv(const uint8_t* d_Wc, const __half* d_Wa, const uint8_t* d_Wb,
                             const __half* d_x, __half* d_y, int M, int K,
                             const float* d_bias_correction, cudaStream_t stream) {
    dim3 block(GEMV_ROWS_PER_BLOCK * 32);
    dim3 grid((M + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK);
    size_t smem = (size_t)K * sizeof(float);
    e2m1_fused_wonly_gemv_kernel<<<grid, block, smem, stream>>>(d_Wc, d_Wa, d_Wb, d_x, d_y, M, K, d_bias_correction);
}

// ---------------------------------------------------------------------------
// Baseline path: materialize dequantized fp16 W (kernel 1), then a plain
// dense fp16 GEMV reads it back (kernel 2). This is what "reconstruct 4-bit
// codes to FP16 before the GEMM" looks like as an actual two-kernel pipeline,
// matching the deck's own stated evaluation protocol - included so bench.py
// can measure the HBM-traffic difference directly as wall-clock time.
// ---------------------------------------------------------------------------
__global__ void e2m1_dequantize_to_fp16_kernel(const uint8_t* __restrict__ W_codes,
                                                const __half* __restrict__ W_alpha,
                                                const uint8_t* __restrict__ W_bias,
                                                __half* __restrict__ W_dequant,
                                                int M, int K) {
    const int row = blockIdx.y;
    const int col_pair = blockIdx.x * blockDim.x + threadIdx.x;  // one packed byte
    const int bytes_per_row = K / 2;
    if (col_pair >= bytes_per_row) return;

    const int blocks_per_row = K / E2M1_BLOCK;
    uint8_t packed = W_codes[(size_t)row * bytes_per_row + col_pair];
    int k0 = col_pair * 2, k1 = k0 + 1;
    int blk0 = k0 / E2M1_BLOCK, blk1 = k1 / E2M1_BLOCK;
    float a0 = __half2float(W_alpha[(size_t)row * blocks_per_row + blk0]);
    float a1 = __half2float(W_alpha[(size_t)row * blocks_per_row + blk1]);
    int bias0 = W_bias[(size_t)row * blocks_per_row + blk0];
    int bias1 = W_bias[(size_t)row * blocks_per_row + blk1];
    W_dequant[(size_t)row * K + k0] = __float2half(e2m1_decode_one(packed & 0xF, a0, bias0));
    W_dequant[(size_t)row * K + k1] = __float2half(e2m1_decode_one((packed >> 4) & 0xF, a1, bias1));
}

__global__ void fp16_dense_gemv_kernel(const __half* __restrict__ W, const __half* __restrict__ x,
                                        __half* __restrict__ y, int M, int K,
                                        const float* __restrict__ bias_correction) {
    extern __shared__ float x_sh2[];
    for (int k = threadIdx.x; k < K; k += blockDim.x) x_sh2[k] = __half2float(x[k]);
    __syncthreads();

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x / 32) + warp;
    if (row >= M) return;

    const __half* row_w = W + (size_t)row * K;
    float acc = 0.0f;
    for (int k = lane; k < K; k += 32) acc += __half2float(row_w[k]) * x_sh2[k];
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);
    if (lane == 0) {
        const float bc = (bias_correction != nullptr) ? bias_correction[row] : 0.0f;
        y[row] = __float2half(acc + bc);
    }
}

void launch_e2m1_naive_gemv(const uint8_t* d_Wc, const __half* d_Wa, const uint8_t* d_Wb,
                             __half* d_Wdequant_scratch, const __half* d_x, __half* d_y,
                             int M, int K, const float* d_bias_correction, cudaStream_t stream) {
    // Kernel 1: materialize the dequantized fp16 weight matrix in HBM.
    {
        dim3 block(256);
        dim3 grid(((K / 2) + block.x - 1) / block.x, M);
        e2m1_dequantize_to_fp16_kernel<<<grid, block, 0, stream>>>(d_Wc, d_Wa, d_Wb, d_Wdequant_scratch, M, K);
    }
    // Kernel 2: standard dense fp16 GEMV reads that buffer back from HBM.
    {
        const int rows_per_block = 8;
        dim3 block(rows_per_block * 32);
        dim3 grid((M + rows_per_block - 1) / rows_per_block);
        size_t smem = (size_t)K * sizeof(float);
        fp16_dense_gemv_kernel<<<grid, block, smem, stream>>>(d_Wdequant_scratch, d_x, d_y, M, K, d_bias_correction);
    }
}
