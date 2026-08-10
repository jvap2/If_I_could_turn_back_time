#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# Phase 0 (decode) synthesis: reproduces Table 3's fixed-vs-programmable
# decode comparison at the RTL/synthesis level, at two granularities:
#
#   (A) Decode table alone:  gf4_decode_fixed  vs  gf4_decode_programmable
#   (B) Full PE (decode+MAC): gf4_pe with PROGRAMMABLE=0 vs PROGRAMMABLE=1
#
# (B) is what the paper's "per-MAC" column in Table 3 describes (decode
# placed inline with every multiply) — NOT the headline "decode-at-ingress"
# ≤1.1%/≤0.01% number, which assumes decode happens once per weight-stationary
# PE load, not every cycle. gf4_pe as written decodes every cycle, so expect
# this delta to land closer to the paper's own +14.6%/+22.7%/+21.2%
# "per-MAC" figures, not the ≤1.1% figure.
#
# NOTE: tb_gf4_decode.v only exercises gf4_decode_fixed/gf4_decode_programmable
# directly — there is no existing testbench for gf4_pe itself. The sanity
# check below therefore only validates the decode tables; gf4_pe's synthesis
# result has no independent functional check behind it yet.
#
# Usage:
#   source ~/eda/oss-cad-suite/environment
#   bash 02a_synth_phase0_decode.sh /path/to/phase0_decode
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

PHASE0_DIR="${1:?Usage: $0 /path/to/phase0_decode}"
EDA_DIR="${EDA_DIR:-$HOME/eda}"
NANGATE_LIB="$EDA_DIR/nangate45/lib/NangateOpenCellLibrary_typical.lib"
OUT_DIR="synth_out/phase0"
mkdir -p "$OUT_DIR"

FIXED="$PHASE0_DIR/gf4_decode_fixed.v"
PROG="$PHASE0_DIR/gf4_decode_programmable.v"
PE="$PHASE0_DIR/gf4_pe.v"
TB="$PHASE0_DIR/tb_gf4_decode.v"

for f in "$FIXED" "$PROG" "$PE" "$TB"; do
  [ -f "$f" ] || { echo "Missing expected file: $f"; exit 1; }
done
command -v yosys >/dev/null 2>&1 || { echo "yosys not on PATH — source ~/eda/oss-cad-suite/environment first"; exit 1; }
[ -f "$NANGATE_LIB" ] || { echo "Nangate45 liberty not found at $NANGATE_LIB"; exit 1; }

# ---- 0. Functional sanity check (decode tables only — see note above) ----
echo "=== Functional sanity check: tb_gf4_decode.v (Icarus Verilog) ==="
iverilog -g2012 -o "$OUT_DIR/sim.vvp" "$FIXED" "$PROG" "$TB"
vvp "$OUT_DIR/sim.vvp" | tee "$OUT_DIR/sim.log"
if ! grep -q "ALL CHECKS PASSED" "$OUT_DIR/sim.log"; then
  echo ""
  echo "Testbench did not report ALL CHECKS PASSED — stopping before synthesis."
  echo "Synthesizing a decode implementation that fails its own known-good"
  echo "checks would produce a meaningless area number."
  exit 1
fi
echo ""

# ---- helper: run one Yosys synth target, extract chip area ----
synth_one () {
  local label="$1" top="$2" out_txt="$3"; shift 3
  local ys="$OUT_DIR/${label}.ys"
  local extra_chparam="${EXTRA_CHPARAM:-}"
  {
    for f in "$@"; do echo "read_verilog -sv \"$f\""; done
    if [ -n "$extra_chparam" ]; then echo "$extra_chparam"; fi
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

# ---- (A) Decode table alone ----
synth_one "decode_fixed"       "gf4_decode_fixed"       "area_decode_fixed.txt"       "$FIXED"
synth_one "decode_programmable" "gf4_decode_programmable" "area_decode_programmable.txt" "$PROG"

# ---- (B) Full PE: decode + MAC, PROGRAMMABLE flipped ----
EXTRA_CHPARAM='chparam -set PROGRAMMABLE 0 -set MAG_WIDTH 8 gf4_pe' \
  synth_one "pe_fixed" "gf4_pe" "area_pe_fixed.txt" "$FIXED" "$PE"

EXTRA_CHPARAM='chparam -set PROGRAMMABLE 1 -set MAG_WIDTH 9 gf4_pe' \
  synth_one "pe_programmable" "gf4_pe" "area_pe_programmable.txt" "$PROG" "$PE"

# ---- extract "Chip area for module" lines and compute deltas ----
get_area () { grep -oP "Chip area for.*:\s*\K[0-9.]+" "$OUT_DIR/$1" | tail -1; }

A_DECODE_FIXED=$(get_area area_decode_fixed.txt)
A_DECODE_PROG=$(get_area area_decode_programmable.txt)
A_PE_FIXED=$(get_area area_pe_fixed.txt)
A_PE_PROG=$(get_area area_pe_programmable.txt)

echo ""
echo "==================================================================="
echo "Phase 0 area results (Nangate45, um^2)"
echo "-------------------------------------------------------------------"
printf "%-32s %12s\n" "Decode table alone:"
printf "%-32s %12s\n" "  fixed (E2M1 ROM)       " "$A_DECODE_FIXED"
printf "%-32s %12s\n" "  programmable (GF4)     " "$A_DECODE_PROG"
DELTA_DECODE=$(awk -v a="$A_DECODE_FIXED" -v b="$A_DECODE_PROG" 'BEGIN{printf "%+.2f", ((b-a)/a)*100}')
echo "  delta:                    ${DELTA_DECODE}%"
echo ""
printf "%-32s %12s\n" "Full PE (decode + MAC), decode-per-MAC placement:"
printf "%-32s %12s\n" "  PROGRAMMABLE=0 (baseline)" "$A_PE_FIXED"
printf "%-32s %12s\n" "  PROGRAMMABLE=1 (GF4)     " "$A_PE_PROG"
DELTA_PE=$(awk -v a="$A_PE_FIXED" -v b="$A_PE_PROG" 'BEGIN{printf "%+.2f", ((b-a)/a)*100}')
echo "  delta:                    ${DELTA_PE}%"
echo "  (compare to the paper's per-MAC column, +14.6/+22.7/+21.2 percent --"
echo "   not the +/-1.1 percent headline, which assumes decode-at-ingress"
echo "   placement, a different architecture than this PE synthesizes)"
echo "==================================================================="
echo ""
echo "Raw reports in $OUT_DIR/area_*.txt if you want full cell breakdowns."