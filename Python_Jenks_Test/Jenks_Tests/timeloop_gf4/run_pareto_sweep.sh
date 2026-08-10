#!/usr/bin/env bash
# Sparse-harvest Sparseloop Pareto sweep for a harvested model's conv problems.
# 4 configs per layer: {dense,sparse} x {energy-first,delay-first} -> report the
# energy axis (energy-first dense vs sparse) AND the throughput axis (delay-first
# dense vs sparse). No `bc` (awk does the math).
#   bash run_pareto_sweep.sh [MODEL_KEY]     (default vgg19_cifar100_98_sparsity)
set -u
MODEL=${1:-vgg19_cifar100_98_sparsity}
A="arch/gf4_engine.yaml components/fp4_mac.yaml components/lut_fp4_mac.yaml components/lut_regfile.yaml components/lut_sram.yaml"
SP="sparse_opt/skip_zeros.yaml"

sed 's/optimization-metrics:.*/optimization-metrics: [ "energy", "delay" ]/' mapper/mapper.yaml > map_e.yaml
sed 's/optimization-metrics:.*/optimization-metrics: [ "delay", "energy" ]/' mapper/mapper.yaml > map_d.yaml

cyc(){ grep -oP 'Cycles:\s*\K[0-9]+'      "$1/timeloop-mapper.stats.txt" 2>/dev/null | tail -1; }
enj(){ grep -oP 'Energy:\s*\K[0-9.eE+-]+' "$1/timeloop-mapper.stats.txt" 2>/dev/null | tail -1; }
run(){ mkdir -p "$2"; timeloop-mapper $A "$3" "$1" ${4:+$4} -o "$2" >/dev/null 2>&1; }

printf "%-8s %12s %12s %12s %12s %8s\n" layer dEner_uJ dDel_cyc sEner_uJ sDel_cyc thru
TdE=0; TsE=0; TdC=0; TsC=0
for p in prob/${MODEL}__*.yaml; do
  [ -e "$p" ] || { echo "no problems for $MODEL"; exit 1; }
  b=$(basename "$p" .yaml | sed 's/.*__//')
  run map_e.yaml out/${MODEL}/de/$b "$p"
  run map_d.yaml out/${MODEL}/dd/$b "$p"
  run map_e.yaml out/${MODEL}/se/$b "$p" "$SP"
  run map_d.yaml out/${MODEL}/sd/$b "$p" "$SP"
  dE=$(enj out/${MODEL}/de/$b); dC=$(cyc out/${MODEL}/dd/$b)
  sE=$(enj out/${MODEL}/se/$b); sC=$(cyc out/${MODEL}/sd/$b)
  thr=$(awk "BEGIN{if(${sC:-0}>0)printf \"%.2fx\",${dC:-0}/${sC:-1};else print \"-\"}")
  printf "%-8s %12s %12s %12s %12s %8s\n" "$b" "${dE:--}" "${dC:--}" "${sE:--}" "${sC:--}" "$thr"
  TdE=$(awk "BEGIN{print $TdE+${dE:-0}}"); TsE=$(awk "BEGIN{print $TsE+${sE:-0}}")
  TdC=$((TdC+${dC:-0})); TsC=$((TsC+${sC:-0}))
done
echo "------------------------------------------------------------------------------"
awk "BEGIN{printf \"ENERGY (energy-first)   dense=%.3f uJ  sparse=%.3f uJ  -> sparse/dense = %.2fx\n\",$TdE,$TsE,($TdE>0?$TsE/$TdE:0)}"
awk "BEGIN{printf \"THROUGHPUT (delay-first) dense=%d cyc  sparse=%d cyc  -> speedup = %.2fx\n\",$TdC,$TsC,($TsC>0?$TdC/$TsC:0)}"
