#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# Phase 1 (LUTxLUT) synthesis: the joint weight-code x activation-code
# product table, the per-element multiplier-free PE built on it, and the
# block-scaled MAC that factors the per-block scale out to once-per-16
# rather than once-per-element.
#
# Three synthesis targets:
#   (A) gf4_joint_product_table alone      -- the 64-entry LUTxLUT core
#   (B) gf4_pe_lutxlut (table + accumulate) -- per-element, no multiplier
#   (C) gf4_block_scaled_mac (table + block-scale multiply once per 16)
#
# NOTE: gf4_pe_lutxlut has no dedicated testbench (same gap as phase0's
# gf4_pe) -- its synthesis result below has no independent functional check
# behind it. (A) and (C) do, via tb_gf4_joint_product.v and
# tb_gf4_block_scaled_mac.v respectively, both run as gates below.
#
# NOTE: (B) is NOT yet a fair diff against phase0's gf4_pe -- gf4_pe only
# decodes one operand and multiplies against a raw operand_b, it doesn't
# decode two coded operands. A true "decode-then-multiply, both sides
# coded" baseline doesn't exist yet in either phase folder. This script
# reports (B)'s absolute area; treat any comparison against phase0 numbers
# as provisional until that baseline exists.
#
# Usage:
#   source ~/eda/oss-cad-suite/environment
#   bash 02b_synth_phase1_lutxlut.sh /path/to/phase1_lutxlut
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

PHASE1_DIR="${1:?Usage: $0 /path/to/phase1_lutxlut}"
EDA_DIR="${EDA_DIR:-$HOME/eda}"
NANGATE_LIB="$EDA_DIR/nangate45/lib/NangateOpenCellLibrary_typical.lib"
OUT_DIR="synth_out/phase1"
mkdir -p "$OUT_DIR"

TABLE="$PHASE1_DIR/gf4_joint_product_table.v"
BSMAC="$PHASE1_DIR/gf4_block_scaled_mac.v"
PELUT="$PHASE1_DIR/gf4_pe_lutxlut.v"
TB_TABLE="$PHASE1_DIR/tb_gf4_joint_product.v"
TB_BSMAC="$PHASE1_DIR/tb_gf4_block_scaled_mac.v"

for f in "$TABLE" "$BSMAC" "$PELUT" "$TB_TABLE" "$TB_BSMAC"; do
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

run_tb "sim_joint_product" "$TABLE" "$TB_TABLE"
run_tb "sim_block_scaled_mac" "$TABLE" "$BSMAC" "$TB_BSMAC"

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

synth_one "joint_table"      "gf4_joint_product_table" "area_joint_table.txt"      "$TABLE"
synth_one "pe_lutxlut"       "gf4_pe_lutxlut"          "area_pe_lutxlut.txt"       "$TABLE" "$PELUT"
synth_one "block_scaled_mac" "gf4_block_scaled_mac"    "area_block_scaled_mac.txt" "$TABLE" "$BSMAC"

get_area () { grep -oP "Chip area for.*:\s*\K[0-9.]+" "$OUT_DIR/$1" | tail -1; }

A_TABLE=$(get_area area_joint_table.txt)
A_PE_LUTXLUT=$(get_area area_pe_lutxlut.txt)
A_BSMAC=$(get_area area_block_scaled_mac.txt)

echo ""
echo "==================================================================="
echo "Phase 1 area results (Nangate45, um^2)"
echo "-------------------------------------------------------------------"
printf "%-38s %12s\n" "Joint product table alone (64-entry):" "$A_TABLE"
printf "%-38s %12s\n" "PE, LUTxLUT (table + accumulate):    " "$A_PE_LUTXLUT"
printf "%-38s %12s\n" "Block-scaled MAC (table + 1 mult/16): " "$A_BSMAC"
echo ""
echo "For reference, phase0's gf4_pe (PROGRAMMABLE=1, decode+multiply,"
echo "single coded operand) measured 1620.738 um^2 in the last phase0 run."
DELTA_PROVISIONAL=$(awk -v a=1620.738 -v b="$A_PE_LUTXLUT" 'BEGIN{printf "%+.2f", ((b-a)/a)*100}')
echo "PE, LUTxLUT vs phase0's gf4_pe: ${DELTA_PROVISIONAL}% -- PROVISIONAL,"
echo "see the note at the top of this script: not yet a fair like-for-like"
echo "diff, since gf4_pe only decodes one of its two operands."
echo "==================================================================="