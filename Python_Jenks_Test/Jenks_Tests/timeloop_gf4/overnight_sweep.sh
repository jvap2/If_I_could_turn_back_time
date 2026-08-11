#!/usr/bin/env bash
# Overnight Sparseloop Pareto sweep across EVERY harvested/sparse model.
# Runs run_pareto_sweep.sh (4 configs/layer) per model; saves per-model logs +
# a combined summary. Launch on ricky-bobby INSIDE the timeloop docker, detached:
#     nohup bash overnight_sweep.sh > overnight.out 2>&1 &
# Runtime is long (~1200+ mapper runs) — it's an overnight job. Most-informative
# models are first, so partial results are still useful if it doesn't fully finish.
set -u
cd "$(dirname "$0")"
python3 gen_problems.py

MODELS="
vgg19_cifar100_98_sparsity
vgg19_cifar10_99_sparsity
vgg19_tinyimagenet
lenet5
lenet300
resnet32_cifar10_95_sparsity
resnet32_cifar100_85_sparsity
resnet32_cifar10_86_sparsity
resnet32_cifar100_86_sparsity
resnet32_tinyimagenet
densenet40
vgg19_cifar10_90_sparsity
vgg19_cifar100_90_sparsity
vgg19_test
"
RES=sparse_sweep_results.txt; : > "$RES"
echo "sweep started $(date)" | tee -a "$RES"
for m in $MODELS; do
  set -- prob/${m}__*.yaml            # expand glob into $@ (robust for many files)
  [ -e "$1" ] || { echo "skip $m (no problems)"; continue; }
  echo "==================================================================" | tee -a "$RES"
  echo "### $m  $(date +%F_%H:%M:%S)" | tee -a "$RES"
  bash run_pareto_sweep.sh "$m" 2>&1 | tee "log_${m}.txt" | grep -E "^L[0-9]|^conv|ENERGY|THROUGHPUT" | tee -a "$RES"
done
echo "==================================================================" | tee -a "$RES"
echo "sweep DONE $(date)" | tee -a "$RES"
echo
echo "Combined summary is in $RES ; per-model detail in log_<model>.txt"
