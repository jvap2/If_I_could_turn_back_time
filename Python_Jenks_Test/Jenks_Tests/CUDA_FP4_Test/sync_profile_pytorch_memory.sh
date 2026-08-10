#!/bin/bash
# Deploys profile_pytorch_memory.py to ricky-bobby via heredoc (avoids scp
# paste-truncation issues). Run this FROM ricky-bobby's shell, in the same
# directory as bindings.cpp / hadamard_kernel.cu / etc.
#
# Usage after running this to write the file:
#   python3 profile_pytorch_memory.py
#   python3 profile_pytorch_memory.py --M 4096 --K 4096
#
# No ncu/sudo needed this time - torch.cuda.max_memory_allocated() is a
# plain PyTorch call, no GPU perf-counter permissions required.

cat > profile_pytorch_memory.py << 'PYEOF'
"""
profile_pytorch_memory.py - measures the REAL GPU memory cost of doing
fake-quantized inference the "obvious" way in plain PyTorch (dequantize
the packed 4-bit weight to a dense FP16 tensor, then call an ordinary
matmul) versus the fused CUDA kernel (e2m1_fused_gemv), which never
materializes that dense tensor at all.

This is the direct answer to "why did this need a custom CUDA kernel
instead of just calling torch.dequantize()-equivalent + torch.matmul()":
the PyTorch-only path's peak memory is measured here via
torch.cuda.max_memory_allocated() - PyTorch's own, real allocator
statistics, the same kind of built-in measurement already used elsewhere
in this project (e.g. the OOM tracebacks in llm_quant_eval.py). This is
NOT a calculated/estimated number - it's what the allocator actually
reports after really running both paths.

Run directly (no ncu/profiler needed - torch.cuda's own memory stats are
enough here):

    python3 profile_pytorch_memory.py
    python3 profile_pytorch_memory.py --M 4096 --K 4096   # smaller layer, for comparison
"""
import argparse

import torch
from torch.utils.cpp_extension import load

_cli = argparse.ArgumentParser()
_cli.add_argument("--M", type=int, default=11008)
_cli.add_argument("--K", type=int, default=11008)
_args = _cli.parse_args()

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

device = torch.device("cuda")
torch.manual_seed(0)

M, K = _args.M, _args.K
E2M1_BLOCK = 16
n_blocks = K // E2M1_BLOCK

W_codes = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=device)
W_alpha = torch.rand(M, n_blocks, device=device).half()
W_bias = torch.randint(0, 3, (M, n_blocks), dtype=torch.uint8, device=device)
x = torch.randn(K, device=device).half()
torch.cuda.synchronize()

packed_bytes = W_codes.numel() * 1 + W_alpha.numel() * 2 + W_bias.numel() * 1
print(f"M={M}, K={K}. Packed 4-bit weight representation (always resident, both paths): "
      f"{packed_bytes/1e6:.1f} MB\n")

# ============================================================================
# Path 1: plain PyTorch - dequantize the packed codes into a dense FP16
# weight tensor (the "obvious" thing to write without a custom kernel),
# then call an ordinary matmul. This mirrors dequantize_e2m1() in
# llm_quant_eval.py - the exact function this project had to optimize
# (int64->uint8, in-place ops) to avoid OOMing during real-model
# evaluation, which is the point: this cost is not hypothetical, it's what
# this project actually hit.
# ============================================================================
E2M1_CODEBOOKS_T = torch.tensor([
    [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0],
    [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    [0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
], dtype=torch.float32, device=device)


def unpack_codes_torch(packed):
    lo = (packed & 0xF).to(torch.uint8)
    hi = ((packed >> 4) & 0xF).to(torch.uint8)
    out = torch.empty(packed.shape[0], packed.shape[1] * 2, dtype=torch.uint8, device=packed.device)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    return out


def dequantize_e2m1_pytorch(W_codes, W_alpha, W_bias, block_size):
    M = W_codes.shape[0]
    codes_u8 = unpack_codes_torch(W_codes)
    n_blocks = codes_u8.shape[1] // block_size
    codes_u8 = codes_u8.view(M, n_blocks, block_size)
    sign = torch.where((codes_u8 & 0x8) != 0, -1.0, 1.0)
    mag_table = E2M1_CODEBOOKS_T[W_bias.long()]
    idx = (codes_u8 & 0x7).long()
    mag = torch.gather(mag_table, 2, idx)
    W_hat = sign
    W_hat.mul_(mag).mul_(W_alpha.float().unsqueeze(-1))
    return W_hat.reshape(M, n_blocks * block_size).half()


torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
mem_before = torch.cuda.memory_allocated()

W_hat = dequantize_e2m1_pytorch(W_codes, W_alpha, W_bias, E2M1_BLOCK)
y_pytorch = torch.matmul(W_hat, x)
torch.cuda.synchronize()

mem_peak_pytorch = torch.cuda.max_memory_allocated()
pytorch_extra_mb = (mem_peak_pytorch - mem_before) / 1e6
dense_weight_mb = W_hat.numel() * 2 / 1e6
print(f"[Plain PyTorch: dequantize-to-dense + matmul]")
print(f"  Peak extra GPU memory beyond the packed weights: {pytorch_extra_mb:.1f} MB")
print(f"  (of which the materialized dense FP16 weight tensor alone is {dense_weight_mb:.1f} MB)")
print(f"  That's {pytorch_extra_mb / (packed_bytes/1e6):.1f}x the size of the packed weights "
      f"this layer was supposed to be stored as.\n")

del W_hat, y_pytorch
torch.cuda.empty_cache()

# ============================================================================
# Path 2: the fused CUDA kernel - decodes in registers during the GEMV,
# never materializes a dense weight tensor at all.
# ============================================================================
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
mem_before2 = torch.cuda.memory_allocated()

y_fused = ext.e2m1_fused_gemv(W_codes, W_alpha, W_bias, x, K)
torch.cuda.synchronize()

mem_peak_fused = torch.cuda.max_memory_allocated()
fused_extra_mb = (mem_peak_fused - mem_before2) / 1e6
print(f"[Fused CUDA kernel: e2m1_fused_gemv]")
print(f"  Peak extra GPU memory beyond the packed weights: {fused_extra_mb:.2f} MB "
      f"(just the output vector, {M*2/1e6:.3f} MB)")
print()
print(f"Plain PyTorch needs {pytorch_extra_mb/max(fused_extra_mb,1e-6):.0f}x more scratch memory "
      f"than the fused kernel for this single layer, at M={M}, K={K}.")
PYEOF

echo "Wrote profile_pytorch_memory.py"