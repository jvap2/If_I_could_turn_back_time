# Quantization Throughput Benchmarks

## OPT Series — SSH Server (batch size 1, decode latency)

FP16 is the baseline. Ratios >1 = faster than FP16.

| Model    | FP16 (ms) | W4A16  | W_lattice16 | W4A4  |
|----------|-----------|--------|-------------|-------|
| OPT-125M | ~22       | 1.44×  | 1.46×       | 0.88× |
| OPT-1.3B | ~43       | 1.49×  | 1.51×       | 0.88× |
| OPT-2.7B | ~56       | 1.46×  | 1.48×       | 0.86× |
| OPT-6.7B | ~55       | 1.48×  | 1.49×       | 0.87× |
| OPT-13B  | OOM       | —      | —           | —     |

- OOM also confirmed at ~14B scale on Colab (2026-09-03)

---

## Mistral-7B-v0.1 — SSH Server (all batch sizes)

Calibration: 4 windows × 2048 tokens, quantization time 17.3s.

| Batch | FP16 (ms) | W4A16 (×)    | W_lattice16 (×) | W4A4 (×)      |
|-------|-----------|--------------|-----------------|---------------|
| 1     | 60.72     | 47.16 (1.29) | 46.16 (1.32)    | 81.70 (0.74)  |
| 2     | 61.37     | 66.67 (0.92) | 65.75 (0.93)    | 108.51 (0.57) |
| 4     | 61.08     | 65.61 (0.93) | 65.55 (0.93)    | 108.58 (0.56) |
| 8     | 61.52     | 65.46 (0.94) | 65.60 (0.94)    | 108.32 (0.57) |
| 16    | 61.44     | 65.55 (0.94) | 65.45 (0.94)    | 109.50 (0.56) |
| 32    | 61.16     | 65.83 (0.93) | 64.98 (0.94)    | 108.81 (0.56) |
| 64    | 61.37     | 65.05 (0.94) | 64.91 (0.95)    | 108.87 (0.56) |

---

## OPT-1.3B Extended — Larger Batch Sizes

FP16 baseline ~42-44 ms flat across all batch sizes (memory-bandwidth-bound even at bs=256).

| Batch | FP16 (ms) | W4A16 (×)    | W_lattice16 (×) |
|-------|-----------|--------------|-----------------|
| 1     | 42.73     | 29.33 (1.46) | 29.50 (1.45)    |
| 2     | 44.44     | 41.52 (1.07) | 41.51 (1.07)    |
| 4     | 44.54     | 41.50 (1.07) | 41.58 (1.07)    |
| 8     | 44.45     | 41.95 (1.06) | 41.60 (1.07)    |
| 16    | 44.23     | 40.91 (1.08) | 40.86 (1.08)    |
| 32    | 44.40     | 40.41 (1.10) | 40.52 (1.10)    |
| 64    | 43.24     | 40.39 (1.07) | 40.06 (1.08)    |
| 128   | 42.96     | 40.18 (1.07) | 39.81 (1.08)    |
| 256   | 42.65     | 40.14 (1.06) | 40.16 (1.06)    |

Note: W4A4 not shown — see earlier Mistral-7B table for W4A4 pattern (consistently slower).

---

## Key Findings

- **Speedup is decode-only (bs=1):** W4A16 and W_lattice16 only beat FP16 in the memory-bandwidth-bound regime. At bs≥2 all quantized variants are slower.
- **W_lattice16 edges W4A16 by ~2–3%** at bs=1 consistently across both model families.
- **W4A4 is slower at all batch sizes** — activation quantization overhead (Hadamard + GF4) outweighs weight savings.
- **FP16 latency is flat across batch sizes** for Mistral-7B (~61ms), while quantized variants jump ~40% from bs=1 to bs=2.
- **Hardware ceiling:** OOM at ~13–14B on available GPUs (SSH server + Colab).
