// hadamard_kernel.cu
//
// Randomized blockwise Fast Walsh-Hadamard Transform (FWHT) - the
// "Gaussianize activations, spread structured outliers" step in the GF4 PTQ
// pipeline (slide: SOFTWARE - PTQ PIPELINE).
//
// Matches fwht_blockwise / _rotate in the reference implementation, which
// this earlier version of the file did NOT: the real transform is
// H(x * D), a per-channel random +-1 sign flip D applied BEFORE the
// butterfly (QuaRot-style randomized Hadamard, not a plain one), and it's
// applied per `had_block_size`-wide chunk of a row, not necessarily the
// whole row. D is generated once per layer (seeded by layer name), folded
// into the weights offline, and applied online to activations every forward
// pass - this kernel covers both uses; d_sign may be nullptr to fall back
// to a plain (unsigned) transform.
//
// Blockwise application needs no extra kernel logic: since x is row-major
// contiguous [rows, M] with M = n_blocks_per_row * block_size, treating
// every (row, block-within-row) pair as its own "effective row" of length
// block_size is just a reinterpretation of the launch grid - row_ptr =
// x + effective_row * block_size addresses the correct, contiguous slice
// with no change to the per-row kernel body. block_within_row = effective_row
// % n_blocks_per_row recovers which slice of D this block should read (D is
// the same across all rows - it's a property of the channel/column, not the
// token/row - hence the modulo, not a plain per-effective-row index).
//
// Bottleneck being addressed: a Hadamard rotation is an NxN orthogonal matrix
// multiply. Written the obvious way (dense matmul, or one kernel launch per
// butterfly stage) it costs O(N^2) FLOPs and/or log2(N) round trips through
// HBM per row. The FWHT computes the identical result in O(N log N) FLOPs
// entirely inside shared memory / registers with a SINGLE kernel launch per
// row-tile - one read from HBM, one write back, no intermediate global-memory
// traffic between stages.

#include <cuda_runtime.h>

#ifndef HADAMARD_MAX_N
#define HADAMARD_MAX_N 2048   // shared-memory budget guard; see README for larger N
#endif

// ---------------------------------------------------------------------------
// Fast kernel: in-place butterfly network in shared memory, one launch total.
// blockIdx.x is an "effective row" = orig_row * n_blocks_per_row + block_idx.
// d_sign may be nullptr (plain Hadamard, no randomization).
// ---------------------------------------------------------------------------
__global__ void hadamard_fwht_kernel(float* __restrict__ x,
                                      const float* __restrict__ d_sign,  // length block_size*n_blocks_per_row, or nullptr
                                      int n_blocks_per_row, int N) {
    extern __shared__ float s[];  // N floats, this effective row's block
    const int eff_row = blockIdx.x;
    const int tid = threadIdx.x;      // 0 .. N/2 - 1
    const int half = N >> 1;
    float* row_in = x + (size_t)eff_row * N;

    if (d_sign != nullptr) {
        const int block_within_row = eff_row % n_blocks_per_row;
        const float* d_seg = d_sign + (size_t)block_within_row * N;
        s[tid]        = row_in[tid]        * d_seg[tid];
        s[tid + half] = row_in[tid + half] * d_seg[tid + half];
    } else {
        s[tid]        = row_in[tid];
        s[tid + half] = row_in[tid + half];
    }
    __syncthreads();

    // log2(N) butterfly stages. Stage with "span" groups pairs (i, i+span)
    // within blocks of size 2*span; this is the standard in-place Hadamard
    // butterfly (same structure as an FFT butterfly, but +/- only, no twiddle).
    for (int span = 1; span < N; span <<= 1) {
        int block_size2 = span << 1;
        int group = tid / span;
        int offset = tid % span;
        int i = group * block_size2 + offset;
        int j = i + span;

        float a = s[i];
        float b = s[j];
        s[i] = a + b;
        s[j] = a - b;
        __syncthreads();
    }

    // Orthonormal scaling (1/sqrt(N)) so the transform is its own inverse
    // (up to the D sign flip, which is applied forward-only, matching the
    // reference: D is baked into the rotated domain, never undone here).
    const float inv_sqrt_n = rsqrtf((float)N);
    row_in[tid]        = s[tid]        * inv_sqrt_n;
    row_in[tid + half] = s[tid + half] * inv_sqrt_n;
}

// rows: number of actual [.., M] rows. M: full row width. block_size: Hadamard
// block width (had_block_size in the reference; pass block_size == M for a
// single full-row transform). d_sign: length M (or nullptr).
void launch_hadamard_fwht(float* d_x, int rows, int M, int block_size,
                           const float* d_sign, cudaStream_t stream) {
    const int n_blocks_per_row = M / block_size;
    dim3 grid(rows * n_blocks_per_row);
    dim3 block(block_size / 2);
    size_t smem = (size_t)block_size * sizeof(float);
    hadamard_fwht_kernel<<<grid, block, smem, stream>>>(d_x, d_sign, n_blocks_per_row, block_size);
}

// ---------------------------------------------------------------------------
// Baseline kernel: the "didn't think about it yet" version. Recomputes each
// output element as a full O(N) dot product against the implicit Sylvester
// Hadamard matrix (H[i][j] = (-1)^popcount(i & j) / sqrt(N)), so the whole
// row costs O(N^2). Same D-then-H semantics as the fused kernel above, for
// a fair, correct comparison.
// ---------------------------------------------------------------------------
__global__ void hadamard_naive_dense_kernel(const float* __restrict__ x_in,
                                             float* __restrict__ x_out,
                                             const float* __restrict__ d_sign,
                                             int n_blocks_per_row, int N) {
    const int eff_row = blockIdx.x;
    const int i = blockIdx.y * blockDim.x + threadIdx.x;  // output index within block
    if (i >= N) return;

    const float* row_in = x_in + (size_t)eff_row * N;
    const float* d_seg = nullptr;
    if (d_sign != nullptr) {
        const int block_within_row = eff_row % n_blocks_per_row;
        d_seg = d_sign + (size_t)block_within_row * N;
    }

    float acc = 0.0f;
#pragma unroll 4
    for (int j = 0; j < N; ++j) {
        int parity = __popc((unsigned int)(i & j)) & 1;
        float h_ij = parity ? -1.0f : 1.0f;
        float xj = row_in[j] * (d_seg ? d_seg[j] : 1.0f);
        acc += h_ij * xj;
    }
    x_out[(size_t)eff_row * N + i] = acc * rsqrtf((float)N);
}

void launch_hadamard_naive(const float* d_x_in, float* d_x_out, int rows, int M, int block_size,
                            const float* d_sign, cudaStream_t stream) {
    const int n_blocks_per_row = M / block_size;
    const int eff_rows = rows * n_blocks_per_row;
    dim3 block(256);
    dim3 grid(eff_rows, (block_size + block.x - 1) / block.x);
    hadamard_naive_dense_kernel<<<grid, block, 0, stream>>>(d_x_in, d_x_out, d_sign, n_blocks_per_row, block_size);
}