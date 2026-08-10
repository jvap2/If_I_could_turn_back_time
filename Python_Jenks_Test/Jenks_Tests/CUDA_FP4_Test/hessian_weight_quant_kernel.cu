// hessian_weight_quant_kernel.cu
//
// Port of reconstruct_layer_fp_blockdiag_scaled_v5 from the reference
// implementation: the real, currently-shipping weight-quantization algorithm
// (confirmed by tracing quantize_model_fp -> calibrate_model_stochastic_fp4
// -> calibrate_layer_stochastic_fp4, where a "fancier" stochastic-rounding
// refinement on top of this is explicitly short-circuited out because it
// made every layer of every model tested worse by ~64%. This IS the
// production path, not an ablation.
//
// Three kernels, run in this order per layer:
//   1. hessian_accumulate_kernel   - H = X^T X / n_tokens, block-diagonal,
//                                    from calibration activations.
//   2. hessian_damp_blocks_kernel  - per-block condition-number damping:
//                                    cap kappa = lambda_max/lambda_min via
//                                    lambda_add * I, matching the reference's
//                                    WQKAPPA=100 default (fixed an OPT-13B
//                                    W4A16 collapse, 304 -> 10.5 PPL).
//   3. hessian_weight_solve_kernel - per (row, block): bias-candidate grid
//                                    search (E2M1's 3 valid biases) x 5
//                                    fixed-point alpha/code iterations,
//                                    keeping whichever bias minimizes the
//                                    Hessian-weighted reconstruction loss.
//
// Deliberate approximation, flagged rather than hidden: the reference
// dampens using EXACT eigenvalues (torch.linalg.eigvalsh). A 16x16 (or
// larger) symmetric eigendecomposition has no simple closed form and doing
// it exactly on-GPU means either a Jacobi eigenvalue solver or a cuSOLVER
// call. This kernel instead estimates lambda_max via plain power iteration
// (H is always PSD, X^T X, so no sign ambiguity) and lambda_min via a
// SHIFTED power iteration on (lambda_max*I - H), which is also PSD and
// converges to lambda_max - lambda_min. bench.py checks both estimates
// against numpy's exact eigvalsh so this is a measured tradeoff, not a
// hidden one.
//
// All three kernels favor obviously-correct, simple code over maximal
// throughput: this is a calibration-time cost (runs once per model, not
// once per token), so unlike the GEMV/encoder kernels there is no
// bottleneck-reduction story here to optimize for - the interesting part is
// "can this nontrivial numerical algorithm be parallelized correctly across
// blocks and rows," not raw speed.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "e2m1_common.cuh"

#ifndef HESSIAN_MAX_BLOCK
#define HESSIAN_MAX_BLOCK 32   // max block_size this kernel supports (BS=16 in the reference sweep)
#endif

// ---------------------------------------------------------------------------
// Kernel 1: H_blocks[block_idx] = X_block^T @ X_block / n_tokens
// X: [n_tokens, M] calibration activations (already Hadamard-rotated, if
// applicable, upstream - this kernel doesn't care, it just forms a Gram
// matrix over whatever columns it's given).
// H_out: [n_blocks, block_size, block_size], row-major per block.
// One CUDA thread block per (block_idx); block_size*block_size threads,
// each accumulating one output entry across all n_tokens - a small, thin
// GEMM (K=n_tokens large, output block_size x block_size small).
// ---------------------------------------------------------------------------
__global__ void hessian_accumulate_kernel(const float* __restrict__ X,
                                           float* __restrict__ H_out,
                                           int n_tokens, int M, int block_size) {
    const int block_idx = blockIdx.x;
    const int col0 = block_idx * block_size;
    const int local_i = threadIdx.x / block_size;
    const int local_j = threadIdx.x % block_size;
    if (local_i >= block_size) return;

    const int gi = col0 + local_i;
    const int gj = col0 + local_j;

    float acc = 0.0f;
    for (int t = 0; t < n_tokens; ++t) {
        float xi = X[(size_t)t * M + gi];
        float xj = X[(size_t)t * M + gj];
        acc += xi * xj;
    }
    H_out[(size_t)block_idx * block_size * block_size + local_i * block_size + local_j] =
        acc / (float)n_tokens;
}

void launch_hessian_accumulate(const float* d_X, float* d_H_out, int n_tokens, int M,
                                int block_size, cudaStream_t stream) {
    int n_blocks = M / block_size;
    dim3 grid(n_blocks);
    dim3 block(block_size * block_size);   // e.g. 16*16 = 256 threads
    hessian_accumulate_kernel<<<grid, block, 0, stream>>>(d_X, d_H_out, n_tokens, M, block_size);
}

// ---------------------------------------------------------------------------
// Kernel 2: adaptive condition-number damping via shifted power iteration.
// One CUDA thread block per Hessian block; a single thread does the (tiny,
// 16x16-ish) power iteration serially - block_size is small enough that
// this is cheap, and serial code is much easier to get right blind (no GPU
// here to catch a parallel-reduction bug) than a parallel eigensolver.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void matvec_sym(const float* H, const float* v, float* out, int bs) {
    for (int i = 0; i < bs; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < bs; ++j) acc += H[i * bs + j] * v[j];
        out[i] = acc;
    }
}

__device__ __forceinline__ float power_iteration_top_eig(const float* H, int bs, int n_iters) {
    float v[HESSIAN_MAX_BLOCK], w[HESSIAN_MAX_BLOCK];
    for (int i = 0; i < bs; ++i) v[i] = 1.0f;  // deterministic init - H is PSD, no sign ambiguity
    float lambda = 0.0f;
    for (int it = 0; it < n_iters; ++it) {
        matvec_sym(H, v, w, bs);
        float norm = 0.0f;
        for (int i = 0; i < bs; ++i) norm += w[i] * w[i];
        norm = sqrtf(fmaxf(norm, 1e-20f));
        for (int i = 0; i < bs; ++i) v[i] = w[i] / norm;
        lambda = norm;  // ||H v|| at the fixed point == the top eigenvalue
    }
    return lambda;
}

__global__ void hessian_damp_blocks_kernel(const float* __restrict__ H_in,
                                            float* __restrict__ H_out,   // may alias H_in
                                            int n_blocks, int block_size, float kappa,
                                            int power_iters) {
    const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= n_blocks) return;

    const float* H = H_in + (size_t)block_idx * block_size * block_size;
    float* Hd = H_out + (size_t)block_idx * block_size * block_size;

    float lambda_max = power_iteration_top_eig(H, block_size, power_iters);

    float lambda_min = 0.0f;
    if (kappa > 1.0f) {
        // Shifted operator A = lambda_max*I - H is PSD with top eigenvalue
        // lambda_max - lambda_min. Build it into a local scratch buffer.
        float A[HESSIAN_MAX_BLOCK * HESSIAN_MAX_BLOCK];
        for (int i = 0; i < block_size; ++i)
            for (int j = 0; j < block_size; ++j)
                A[i * block_size + j] = (i == j ? lambda_max : 0.0f) - H[i * block_size + j];
        float mu = power_iteration_top_eig(A, block_size, power_iters);
        lambda_min = lambda_max - mu;

        float lambda_add = fmaxf(lambda_max / kappa - lambda_min, 0.0f);
        for (int i = 0; i < block_size; ++i)
            for (int j = 0; j < block_size; ++j)
                Hd[i * block_size + j] = H[i * block_size + j] + (i == j ? lambda_add : 0.0f);
    } else {
        for (int i = 0; i < block_size * block_size; ++i) Hd[i] = H[i];
    }
}

void launch_hessian_damp_blocks(const float* d_H_in, float* d_H_out, int n_blocks,
                                 int block_size, float kappa, int power_iters,
                                 cudaStream_t stream) {
    int threads = 64;
    int blocks = (n_blocks + threads - 1) / threads;
    hessian_damp_blocks_kernel<<<blocks, threads, 0, stream>>>(
        d_H_in, d_H_out, n_blocks, block_size, kappa, power_iters);
}

// ---------------------------------------------------------------------------
// Kernel 3: per-(row, block) Hessian-weighted alpha/bias solve.
// H_damped is shared across all N rows of a given block (it's a property of
// the input distribution, not the output neuron), so it's staged into
// shared memory once per CUDA thread block and reused by every row that
// thread block processes (grid-stride over rows if N > blockDim.x).
//
// W: [N, M] fp32 (row-major). W_codes: [N, M/2] packed bytes (2 codes/byte,
// same low-nibble-first convention as gf4_pack). W_alpha: [N, M/block_size]
// fp16. W_bias: [N, M/block_size] uint8, value in {0,1,2}.
// ---------------------------------------------------------------------------
__global__ void hessian_weight_solve_kernel(const float* __restrict__ W,
                                             const float* __restrict__ H_damped,  // [n_blocks, bs, bs]
                                             uint8_t* __restrict__ W_codes,
                                             __half* __restrict__ W_alpha,
                                             uint8_t* __restrict__ W_bias,
                                             int N, int M, int block_size) {
    extern __shared__ float H_sh[];  // block_size*block_size floats
    const int block_idx = blockIdx.x;
    const int n_blocks = M / block_size;
    const int col0 = block_idx * block_size;

    for (int i = threadIdx.x; i < block_size * block_size; i += blockDim.x) {
        H_sh[i] = H_damped[(size_t)block_idx * block_size * block_size + i];
    }
    __syncthreads();

    for (int row = threadIdx.x; row < N; row += blockDim.x) {
        float w[HESSIAN_MAX_BLOCK], w_abs[HESSIAN_MAX_BLOCK], sign[HESSIAN_MAX_BLOCK];
        float Hw[HESSIAN_MAX_BLOCK];
        float mean_abs = 0.0f, mean_sq = 0.0f;

        for (int k = 0; k < block_size; ++k) {
            float wv = W[(size_t)row * M + col0 + k];
            w[k] = wv;
            w_abs[k] = fabsf(wv);
            sign[k] = wv < 0.0f ? -1.0f : 1.0f;
            mean_abs += w_abs[k];
            mean_sq += wv * wv;
        }
        mean_abs /= block_size;
        mean_sq /= block_size;

        float alpha_init = fmaxf(sqrtf(fmaxf(mean_sq, 1e-8f)), 1e-4f);
        float alpha_min = 0.05f * fmaxf(mean_abs, 1e-8f);
        float alpha_max = 20.0f * fmaxf(mean_abs, 1e-8f);

        matvec_sym(H_sh, w_abs, Hw, block_size);  // Hw = w_eff @ H_damped  (H symmetric)

        float best_loss = 3.4e38f;
        float best_alpha = alpha_init;
        int best_bias = 1;
        uint8_t best_codes[HESSIAN_MAX_BLOCK];

        for (int bias = 0; bias < E2M1_NUM_BIAS; ++bias) {
            float alpha = alpha_init;
            uint8_t codes[HESSIAN_MAX_BLOCK];
            float b_eff[HESSIAN_MAX_BLOCK];

            for (int iter = 0; iter < 5; ++iter) {
                for (int k = 0; k < block_size; ++k) {
                    codes[k] = e2m1_assign_magnitude(w_abs[k] / fmaxf(alpha, 1e-8f), bias);
                    b_eff[k] = e2m1_decode_magnitude(codes[k], bias);
                }
                float Hb[HESSIAN_MAX_BLOCK];
                matvec_sym(H_sh, b_eff, Hb, block_size);
                float num = 0.0f, den = 0.0f;
                for (int k = 0; k < block_size; ++k) {
                    num += b_eff[k] * Hw[k];
                    den += b_eff[k] * Hb[k];
                }
                float alpha_new = num / (den + 1e-8f);
                alpha = fminf(fmaxf(alpha_new, alpha_min), alpha_max);
            }

            float residual[HESSIAN_MAX_BLOCK], Hr[HESSIAN_MAX_BLOCK];
            for (int k = 0; k < block_size; ++k) residual[k] = w_abs[k] - alpha * b_eff[k];
            matvec_sym(H_sh, residual, Hr, block_size);
            float loss = 0.0f;
            for (int k = 0; k < block_size; ++k) loss += residual[k] * Hr[k];

            if (loss < best_loss) {
                best_loss = loss;
                best_alpha = alpha;
                best_bias = bias;
                for (int k = 0; k < block_size; ++k) best_codes[k] = codes[k];
            }
        }

        float alpha_q = quantize_scale_e4m3(best_alpha, 4, 3);   // e_bits_scale=4, m_bits_scale=3 (E4M3)
        // Re-assign final codes with the quantized alpha (matches the
        // reference's final assign_fp4_dynamic_batched call using alpha_q).
        uint8_t final_codes[HESSIAN_MAX_BLOCK];
        for (int k = 0; k < block_size; ++k) {
            uint8_t mag = e2m1_assign_magnitude(w_abs[k] / fmaxf(alpha_q, 1e-8f), best_bias);
            final_codes[k] = (sign[k] < 0.0f ? 0x8 : 0x0) | mag;
        }

        const int bytes_per_block = block_size / 2;
        uint8_t* row_codes = W_codes + (size_t)row * (M / 2) + (size_t)block_idx * bytes_per_block;
        for (int b = 0; b < bytes_per_block; ++b) {
            row_codes[b] = (final_codes[2 * b] & 0xF) | ((final_codes[2 * b + 1] & 0xF) << 4);
        }
        W_alpha[(size_t)row * n_blocks + block_idx] = __float2half(alpha_q);
        W_bias[(size_t)row * n_blocks + block_idx] = (uint8_t)best_bias;
    }
}

void launch_hessian_weight_solve(const float* d_W, const float* d_H_damped,
                                  uint8_t* d_W_codes, __half* d_W_alpha, uint8_t* d_W_bias,
                                  int N, int M, int block_size, cudaStream_t stream) {
    int n_blocks = M / block_size;
    dim3 grid(n_blocks);
    dim3 block(256);
    size_t smem = (size_t)block_size * block_size * sizeof(float);
    hessian_weight_solve_kernel<<<grid, block, smem, stream>>>(
        d_W, d_H_damped, d_W_codes, d_W_alpha, d_W_bias, N, M, block_size);
}