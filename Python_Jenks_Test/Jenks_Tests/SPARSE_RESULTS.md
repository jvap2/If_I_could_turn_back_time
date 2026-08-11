# Sparse-Network Compression Results

Two complementary methods applied to the 14 magnitude-pruned checkpoints in
`Best_Results_HPO/`. All pruning is **unstructured**, which gives **zero** throughput/
energy benefit on a GPU/CPU (the zeros are still multiplied). We recover real gains two
ways:

- **(#1) Structured channel harvest** — physically delete the *emergently dead* channels
  a high unstructured sparsity produces, yielding a smaller **dense** model that runs
  faster on *any* hardware, **losslessly** (float64 output match, no retraining).
- **(#3) Sparsity-aware accelerator (Sparseloop)** — model a zero-skipping accelerator
  (the 256-PE weight-stationary "GF4-Engine") that exploits the residual unstructured
  sparsity a GPU cannot.

---

## Table 1 — Structured channel harvest (#1)

Compact model built by removing dead output channels (VGG/LeNet: exact harvest with
BN/bias constant-folding; ResNet/DenseNet: Torch-Pruning dependency graph). "Lossless"
= max |output difference| vs. the original on random input, float64. GPU throughput
measured on an RTX 4080 at batch 256.

| Model | Sparsity | Dead ch. | Params (full → compact) | Shrink | GPU throughput (fp16 / fp32) | Lossless |
|---|---:|---:|---|---:|---:|---:|
| **VGG-19 CIFAR-100 98%** | 98.9% | 69.9% | 20.08M → 1.66M | **12.1×** | **3.07× / 4.24×** | 2e-13 |
| **VGG-19 CIFAR-10 99%** | 99.5% | 78.1% | 20.04M → 1.87M | **10.7×** | **2.09× / 2.77×** | 1e-14 |
| **LeNet-5** | 99.5% | 79.7% | 0.43M → 0.046M | **9.35×** | 1.15× / 1.14× † | 7e-15 |
| **VGG-19 TinyImageNet** | 97.9% | 58.9% | 20.13M → 5.09M | **3.95×** | **2.04× / 2.34×** | 9e-15 |
| **LeNet-300** | 97.9% | 61.0% | 0.27M → 0.097M | **2.76×** | 1.09× / 1.07× † | 7e-15 |
| VGG-19 CIFAR-10 90% | 90.9% | 4.2% | 20.04M → 18.20M | 1.10× | ~1× | 4e-16 |
| VGG-19 CIFAR-100 90% | 89.4% | 0.5% | 20.08M → 20.06M | 1.00× | ~1× | 2e-15 |
| VGG-19 (Test, TinyIN) | 86.8% | 0.0% | 20.13M → 20.13M | 1.00× | ~1× | 0 |
| ResNet-32 CIFAR-10 86% | 86.6% | 5.1% | 1.86M → 1.86M | 1.00× | ~1× | 0 |
| ResNet-32 CIFAR-10 95% | 87.9% | 0.9% | 1.86M → 1.86M | 1.00× | ~1× | 0 |
| ResNet-32 CIFAR-100 85% | 85.2% | 0.2% | 1.87M → 1.87M | 1.00× | ~1× | 0 |
| ResNet-32 CIFAR-100 86% | 86.3% | 0.4% | 1.87M → 1.87M | 1.00× | ~1× | 0 |
| ResNet-32 TinyImageNet | 89.4% | 0.6% | 1.89M → 1.89M | 1.00× | ~1× | 0 |
| DenseNet-40 | 84.5% | 1.0% | 1.06M → 1.06M | 1.00× | ~1× | 0 |

† LeNets are tiny → GPU time is kernel-launch-bound, so the harvest shows in
params/energy/CPU, not GPU throughput. ResNet/DenseNet: Torch-Pruning found **0**
losslessly-removable channels (residual/concat coupling), confirming the survey.

### Findings from Table 1
1. **Structured harvest from unstructured pruning is real and lossless** — up to
   **12× smaller and 4× GPU throughput with zero accuracy change** (float64-identical).
2. **Sparsity threshold (~95–97%):** below it, unstructured pruning does *not*
   emergently kill whole channels (VGG @ 90% harvests nothing); above it, 60–80% of
   channels die and the model collapses 10–12×.
3. **Architecture matters:** feedforward nets (VGG, LeNet) harvest hugely; **residual/
   dense nets (ResNet, DenseNet) resist entirely** (<6% dead even at ~89% sparsity) —
   they need 2:4 structured sparsity or a sparse accelerator (#3).

---

## Table 2 — Sparsity-aware accelerator (#3, Sparseloop)

Each conv/linear layer modeled as a GEMM on the 256-PE GF4-Engine (45 nm, Accelergy),
with the residual unstructured sparsity exploited via `skipping` of zero-weight ops.
Two objectives: **energy-first** (energy axis) and **delay-first** (throughput axis).
VGG/LeNet use the harvested compact model's residual sparsity; ResNet/DenseNet (which
cannot be harvested) use the original sparse model.

| Model | Energy: dense → sparse | Energy ratio | Throughput: dense → sparse (cycles) | Speedup |
|---|---|---:|---|---:|
| **VGG-19 CIFAR-100 98%** | 677.3 → 536.8 µJ | **0.79× (−21%)** | 5,320,792 → 1,447,057 | **3.68×** |
| VGG-19 CIFAR-10 99% | — | — | — | *(overnight sweep)* |
| VGG-19 TinyImageNet | — | — | — | *(overnight sweep)* |
| LeNet-5 / LeNet-300 | — | — | — | *(overnight sweep)* |
| ResNet-32 (×5) | — | — | — | *(overnight sweep)* |
| DenseNet-40 | — | — | — | *(overnight sweep)* |

*(Remaining rows are produced by `timeloop_gf4/overnight_sweep.sh`; fill from
`sparse_sweep_results.txt`.)*

### VGG-19 CIFAR-100 98% — per-layer Pareto (worked example)

| Layer | dense energy (µJ) | dense cycles | sparse energy (µJ) | sparse cycles | throughput |
|---|---:|---:|---:|---:|---:|
| conv0 | 4.96 | 59,904 | 3.92 | 6,656 | 9.00× |
| conv3 | 84.3 | 679,680 | 60.2 | 77,010 | 8.83× |
| conv4 | 65.7 | 116,112 | 52.8 | 14,144 | 8.21× |
| conv7 | 97.4 | 171,200 | 78.7 | 22,633 | 7.56× |
| conv8 | 38.4 | 729,312 | 38.4 | 99,190 | 7.35× |
| conv15 | 13.9 | 79,016 | 11.8 | 6,543 | 12.08× |
| conv2 / conv11–13 | — | — | ≈ dense | ≈ dense | 1.00× |
| **whole VGG** | **677.3** | **5,320,792** | **536.8** | **1,447,057** | **3.68×** |

### Findings from Table 2
1. Exploiting the residual unstructured sparsity buys **~3.7× throughput and ~21%
   energy** on the GF4-Engine — gains a GPU cannot capture.
2. **`skipping` (not `gating`)** yields the throughput win; it must be paired with a
   **delay-first** mapper objective, else the energy-optimal mapping ignores sparsity.
3. **Weight compression only works at the per-PE register** — placing it on DRAM/global
   buffer (above the 16×16 spatial fanout) makes *every* mapping invalid, because
   compressed weights cannot scatter across the 256 PEs. A real dataflow constraint.
4. Large per-layer variance (1× to 12×): the sparsity payoff is mapping-dependent.

---

## Overall conclusion

**Structured sparsity (harvest) pays off broadly and portably** — compute, memory,
energy, on any hardware, losslessly. **Fine-grained unstructured sparsity adds a further
throughput + modest energy gain, but only on a zero-skipping accelerator and only at the
right dataflow level.** Residual/dense architectures resist structured harvest entirely
and depend on the accelerator path for any sparsity benefit.

## Reproduction
- `harvest_all.py` → `harvest_all_results.csv` (Table 1; RTX 4080).
- `timeloop_gf4/export_all_hwsim.py` → `models/*_hwsim.json` (per-layer GEMM + density).
- `timeloop_gf4/overnight_sweep.sh` (in the Timeloop docker) → `sparse_sweep_results.txt`
  (Table 2). Uses `gen_problems.py`, `run_pareto_sweep.sh`, `sparse_opt/skip_zeros.yaml`.
