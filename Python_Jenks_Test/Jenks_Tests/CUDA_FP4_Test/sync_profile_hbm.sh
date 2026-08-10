#!/usr/bin/env bash
set -e

cat > profile_hbm.py << 'PYFILE_EOF_MARKER_hbm7c1'
"""
profile_hbm.py - a minimal, single-call harness for Nsight Compute (ncu) to
measure REAL HBM traffic for the fused vs. naive E2M1 GEMV kernels, replacing
the calculated bytes-moved estimate in bench.py/the deck with an actual
profiler-measured number.

This does NOT time anything itself (bench.py's CUDA-event wall-clock timing
already covers that, and is genuinely measured). This script exists only to
give ncu two clean, single-shot kernel launches to attach to - no warmup
loop, no repeated calls, since ncu counts every launch it sees and repeated
warmup calls would just add duplicate rows to sum up by hand for no reason.

Run it under ncu, NOT directly with python3 - run directly, it just does one
fused and one naive GEMV call and exits (nothing to look at without a
profiler attached).

    ncu --kernel-name regex:"e2m1_fused_wonly_gemv_kernel|e2m1_dequantize_to_fp16_kernel|fp16_dense_gemv_kernel" \
        --metrics dram__bytes_read.sum,dram__bytes_write.sum \
        --csv python3 profile_hbm.py > ncu_hbm.csv
    cat ncu_hbm.csv

If ncu reports ERR_NVGPUCTRPERM (permission denied reading GPU performance
counters - common on non-datacenter GPUs since NVIDIA locked this down by
default around driver 418+), prefix the command with sudo:

    sudo env "PATH=$PATH" ncu --kernel-name regex:"..." --metrics ... --csv python3 profile_hbm.py > ncu_hbm.csv

(sudo alone sometimes isn't enough if `ncu` isn't on root's PATH - the `env
"PATH=$PATH"` keeps your current PATH under sudo). If that still fails, the
permanent fix is a kernel module parameter
(NVreg_RestrictProfilingToAdminUsers=0) requiring a reboot - not worth it
just for this one measurement; sudo is simpler for a one-off.

Interpreting the CSV output: e2m1_fused_wonly_gemv_kernel's row is the FUSED
path's total HBM traffic (one kernel, one row). The NAIVE path is TWO
kernels - e2m1_dequantize_to_fp16_kernel (writes the dequantized fp16
weight matrix to HBM) and fp16_dense_gemv_kernel (reads it back for the
GEMV) - sum dram__bytes_read.sum + dram__bytes_write.sum across BOTH of
those rows to get the naive path's total. The ratio of
(naive total) / (fused total) is the real, Nsight-measured version of the
"8.1x less HBM traffic" claim - send me both totals and I'll update the
slide with the measured number instead of the calculated one.
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

# Defaults to the largest case from bench.py's own sweep; pass --M/--K to
# check other sizes (bench.py itself tests (4096,4096), (11008,4096), and
# (11008,11008) - worth profiling more than one of these, since the
# calculated bytes-moved formula in bench.py gives an almost IDENTICAL
# ratio at all three sizes (~6.8x), not a ratio that grows with size - so
# checking whether the MEASURED ratio is also flat, rather than assuming it
# from the formula, is the actual point of running this more than once.
M, K = _args.M, _args.K
E2M1_BLOCK = 16
n_blocks = K // E2M1_BLOCK

W_codes = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=device)
W_alpha = torch.rand(M, n_blocks, device=device).half()
W_bias = torch.randint(0, 3, (M, n_blocks), dtype=torch.uint8, device=device)
x = torch.randn(K, device=device).half()

# One clean call each - exactly what ncu should see and measure.
y_fused = ext.e2m1_fused_gemv(W_codes, W_alpha, W_bias, x, K)
torch.cuda.synchronize()

y_naive = ext.e2m1_naive_gemv(W_codes, W_alpha, W_bias, x, K)
torch.cuda.synchronize()

print(f"Profiled one fused_gemv call and one naive_gemv call at M={M}, K={K}. "
      f"Run this under ncu (see this file's docstring) to get real HBM traffic numbers.")
PYFILE_EOF_MARKER_hbm7c1

echo "profile_hbm.py written."