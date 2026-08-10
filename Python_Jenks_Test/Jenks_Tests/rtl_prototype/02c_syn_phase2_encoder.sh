#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# Phase 2 (encoder) synthesis: FP16->fixed converter, integer sqrt, sum-of-
# squares, the full activation encoder, and the first real cross-phase
# integration (gf4_block_pipeline, which pulls in phase1's block-scaled MAC).
#
# Five synthesis targets:
#   (A) gf4_fp16_to_fixed  -- combinational FP16 bit-pattern -> Q8.8 converter
#   (B) gf4_isqrt          -- iterative restoring integer sqrt (18-cycle latency)
#   (C) gf4_sumsq          -- sum-of-squares accumulator over one 16-elem block
#   (D) gf4_activation_encoder -- (C)+(B) plus threshold-ladder GF4 encode
#   (E) gf4_block_pipeline -- (A)+(D)+phase1's gf4_block_scaled_mac, full
#       FP16-in to accumulated-dot-product-out path
#
# All five targets now have a real testbench run as a gate below, including
# (E) gf4_block_pipeline via tb_gf4_block_pipeline.v (written after an
# earlier mix-up where a file claiming to be this testbench turned out to
# actually be tb_gf4_block_pipeline_stream.v, driving a different module
# with a different, valid/ready port list). tb_gf4_block_pipeline_stream.v
# itself still can't run yet -- it depends on gf4_block_pipeline_stream.v,
# which lives in phase3_interface and hasn't been synthesized/tested here.
#
# Usage:
#   source ~/eda/oss-cad-suite/environment
#   bash 02c_synth_phase2_encoder.sh /path/to/phase1_lutxlut /path/to/phase2_encoder
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

PHASE1_DIR="${1:?Usage: $0 /path/to/phase1_lutxlut /path/to/phase2_encoder}"
PHASE2_DIR="${2:?Usage: $0 /path/to/phase1_lutxlut /path/to/phase2_encoder}"
EDA_DIR="${EDA_DIR:-$HOME/eda}"
NANGATE_LIB="$EDA_DIR/nangate45/lib/NangateOpenCellLibrary_typical.lib"
OUT_DIR="synth_out/phase2"
mkdir -p "$OUT_DIR"

TABLE="$PHASE1_DIR/gf4_joint_product_table.v"
BSMAC="$PHASE1_DIR/gf4_block_scaled_mac.v"

FP16="$PHASE2_DIR/gf4_fp16_to_fixed.v"
ISQRT="$PHASE2_DIR/gf4_isqrt.v"
SUMSQ="$PHASE2_DIR/gf4_sumsq.v"
ENCODER="$PHASE2_DIR/gf4_activation_encoder.v"
PIPELINE="$PHASE2_DIR/gf4_block_pipeline.v"

TB_FP16="$PHASE2_DIR/tb_gf4_fp16_to_fixed.v"
TB_ISQRT="$PHASE2_DIR/tb_gf4_isqrt.v"
TB_SUMSQ="$PHASE2_DIR/tb_gf4_sumsq.v"
TB_ENCODER="$PHASE2_DIR/tb_gf4_activation_encoder.v"
TB_PIPELINE="$PHASE2_DIR/tb_gf4_block_pipeline.v"

for f in "$TABLE" "$BSMAC" "$FP16" "$ISQRT" "$SUMSQ" "$ENCODER" "$PIPELINE" \
         "$TB_FP16" "$TB_ISQRT" "$TB_SUMSQ" "$TB_ENCODER" "$TB_PIPELINE"; do
  [ -f "$f" ] || { echo "Missing expected file: $f"; exit 1; }
done
command -v yosys >/dev/null 2>&1 || { echo "yosys not on PATH — source ~/eda/oss-cad-suite/environment first"; exit 1; }
[ -f "$NANGATE_LIB" ] || { echo "Nangate45 liberty not found at $NANGATE_LIB"; exit 1; }

run_tb () {
  local label="$1"; shift
  echo "=== Functional sanity check: $label ==="
  iverilog -g2012 -o "$OUT_DIR/${label}.vvp" "$@"
  vvp "$OUT_DIR/${label}.vvp" | tee "$OUT_DIR/${label}.log"
  if ! grep -q "ALL CHECKS PASSED" "$OUT_DIR/${label}.log"; then
    echo ""
    echo "$label did not report ALL CHECKS PASSED — stopping before synthesis."
    exit 1
  fi
  echo ""
}

run_tb "sim_fp16_to_fixed" "$FP16" "$TB_FP16"
run_tb "sim_isqrt"         "$ISQRT" "$TB_ISQRT"
run_tb "sim_sumsq"         "$SUMSQ" "$ISQRT" "$TB_SUMSQ"
run_tb "sim_encoder"       "$SUMSQ" "$ISQRT" "$ENCODER" "$TB_ENCODER"
run_tb "sim_pipeline"      "$TABLE" "$BSMAC" "$FP16" "$SUMSQ" "$ISQRT" "$ENCODER" "$PIPELINE" "$TB_PIPELINE"

synth_one () {
  local label="$1" top="$2" out_txt="$3"; shift 3
  local ys="$OUT_DIR/${label}.ys"
  {
    for f in "$@"; do echo "read_verilog -sv \"$f\""; done
    echo "hierarchy -top $top"
    echo "proc; opt; fsm; opt; memory; opt"
    echo "techmap; opt"
    echo "dfflibmap -liberty \"$NANGATE_LIB\""
    echo "abc -liberty \"$NANGATE_LIB\""
    echo "clean"
    echo "tee -o $OUT_DIR/$out_txt stat -liberty \"$NANGATE_LIB\""
  } > "$ys"
  echo "--- Synthesizing: $label (top=$top) ---"
  yosys -s "$ys" > "$OUT_DIR/${label}.log" 2>&1 || { echo "FAILED — see $OUT_DIR/${label}.log"; exit 1; }
}

synth_one "fp16_to_fixed" "gf4_fp16_to_fixed"     "area_fp16_to_fixed.txt" "$FP16"
synth_one "isqrt"         "gf4_isqrt"             "area_isqrt.txt"        "$ISQRT"
synth_one "sumsq"         "gf4_sumsq"             "area_sumsq.txt"        "$SUMSQ"
synth_one "encoder"       "gf4_activation_encoder" "area_encoder.txt"     "$SUMSQ" "$ISQRT" "$ENCODER"
synth_one "block_pipeline" "gf4_block_pipeline"   "area_block_pipeline.txt" \
  "$TABLE" "$BSMAC" "$FP16" "$SUMSQ" "$ISQRT" "$ENCODER" "$PIPELINE"

get_area () { grep -oP "Chip area for.*:\s*\K[0-9.]+" "$OUT_DIR/$1" | tail -1; }

A_FP16=$(get_area area_fp16_to_fixed.txt)
A_ISQRT=$(get_area area_isqrt.txt)
A_SUMSQ=$(get_area area_sumsq.txt)
A_ENCODER=$(get_area area_encoder.txt)
A_PIPELINE=$(get_area area_block_pipeline.txt)

echo ""
echo "==================================================================="
echo "Phase 2 area results (Nangate45, um^2)"
echo "-------------------------------------------------------------------"
printf "%-42s %12s\n" "gf4_fp16_to_fixed (combinational):" "$A_FP16"
printf "%-42s %12s\n" "gf4_isqrt (18-cycle iterative):" "$A_ISQRT"
printf "%-42s %12s\n" "gf4_sumsq (standalone):" "$A_SUMSQ"
printf "%-42s %12s\n" "gf4_activation_encoder (sumsq+isqrt+ladder):" "$A_ENCODER"
printf "%-42s %12s\n" "gf4_block_pipeline (full FP16->acc path):" "$A_PIPELINE"
echo "==================================================================="