#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# Phase 3 (interface) synthesis: the valid/ready streaming wrapper around
# the full phase1+phase2 pipeline -- this is the final, top-level
# synthesizable unit for the whole RTL prototype: real backpressure on
# both the input (FP16 activation + weight code stream) and output
# (accumulated result) sides, with nothing about the underlying arithmetic
# changed from what's already been verified in phases 0-2.
#
# One synthesis target: gf4_block_pipeline_stream, which pulls in every
# other module from every phase. One testbench gate: tb_gf4_block_pipeline_
# stream.v, which (unlike an earlier mislabeled file of the same name) now
# genuinely matches this module's in_valid/in_ready/out_valid/out_ready
# port list, and exercises real backpressure across three scenarios
# (baseline, producer stall, consumer stall) on top of the two already-
# verified end-to-end arithmetic results (81 and 0).
#
# Usage:
#   source ~/eda/oss-cad-suite/environment
#   bash 02d_synth_phase3_interface.sh /path/to/phase1_lutxlut \
#        /path/to/phase2_encoder /path/to/phase3_interface
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

PHASE1_DIR="${1:?Usage: $0 /path/to/phase1_lutxlut /path/to/phase2_encoder /path/to/phase3_interface}"
PHASE2_DIR="${2:?Usage: $0 /path/to/phase1_lutxlut /path/to/phase2_encoder /path/to/phase3_interface}"
PHASE3_DIR="${3:?Usage: $0 /path/to/phase1_lutxlut /path/to/phase2_encoder /path/to/phase3_interface}"
EDA_DIR="${EDA_DIR:-$HOME/eda}"
NANGATE_LIB="$EDA_DIR/nangate45/lib/NangateOpenCellLibrary_typical.lib"
OUT_DIR="synth_out/phase3"
mkdir -p "$OUT_DIR"

TABLE="$PHASE1_DIR/gf4_joint_product_table.v"
BSMAC="$PHASE1_DIR/gf4_block_scaled_mac.v"
FP16="$PHASE2_DIR/gf4_fp16_to_fixed.v"
ISQRT="$PHASE2_DIR/gf4_isqrt.v"
SUMSQ="$PHASE2_DIR/gf4_sumsq.v"
ENCODER="$PHASE2_DIR/gf4_activation_encoder.v"
PIPELINE="$PHASE2_DIR/gf4_block_pipeline.v"
STREAM="$PHASE3_DIR/gf4_block_pipeline_stream.v"
TB_STREAM="$PHASE3_DIR/tb_gf4_block_pipeline_stream.v"

ALL_RTL=("$TABLE" "$BSMAC" "$FP16" "$SUMSQ" "$ISQRT" "$ENCODER" "$PIPELINE" "$STREAM")

for f in "${ALL_RTL[@]}" "$TB_STREAM"; do
  [ -f "$f" ] || { echo "Missing expected file: $f"; exit 1; }
done
command -v yosys >/dev/null 2>&1 || { echo "yosys not on PATH — source ~/eda/oss-cad-suite/environment first"; exit 1; }
[ -f "$NANGATE_LIB" ] || { echo "Nangate45 liberty not found at $NANGATE_LIB"; exit 1; }

echo "=== Functional sanity check: tb_gf4_block_pipeline_stream (3 backpressure scenarios) ==="
iverilog -g2012 -o "$OUT_DIR/sim_stream.vvp" "${ALL_RTL[@]}" "$TB_STREAM"
vvp "$OUT_DIR/sim_stream.vvp" | tee "$OUT_DIR/sim_stream.log"
if ! grep -q "ALL CHECKS PASSED" "$OUT_DIR/sim_stream.log"; then
  echo ""
  echo "tb_gf4_block_pipeline_stream did not report ALL CHECKS PASSED — stopping before synthesis."
  exit 1
fi
echo ""

YS="$OUT_DIR/synth_stream.ys"
{
  for f in "${ALL_RTL[@]}"; do echo "read_verilog -sv \"$f\""; done
  echo "hierarchy -top gf4_block_pipeline_stream"
  echo "proc; opt; fsm; opt; memory; opt"
  echo "techmap; opt"
  echo "dfflibmap -liberty \"$NANGATE_LIB\""
  echo "abc -liberty \"$NANGATE_LIB\""
  echo "clean"
  echo "tee -o $OUT_DIR/area_stream.txt stat -liberty \"$NANGATE_LIB\""
} > "$YS"

echo "--- Synthesizing: gf4_block_pipeline_stream (full project, top-level) ---"
yosys -s "$YS" > "$OUT_DIR/synth_stream.log" 2>&1 || { echo "FAILED — see $OUT_DIR/synth_stream.log"; exit 1; }

A_STREAM=$(grep -oP "Chip area for.*:\s*\K[0-9.]+" "$OUT_DIR/area_stream.txt" | tail -1)

echo ""
echo "==================================================================="
echo "Phase 3 / full-project area result (Nangate45, um^2)"
echo "-------------------------------------------------------------------"
echo "gf4_block_pipeline_stream (entire project, top-level): $A_STREAM"
echo ""
echo "For reference, phase2's gf4_block_pipeline (no handshake) measured"
echo "33391.246 um^2. The delta here is purely Phase 3's control-plane"
echo "plumbing -- the latch registers, output holding register, and"
echo "in_ready/out_valid logic -- with zero change to the underlying"
echo "arithmetic (already proven identical: this testbench's block A/all-"
echo "zero results match gf4_block_pipeline.v's own testbench exactly)."
echo "==================================================================="