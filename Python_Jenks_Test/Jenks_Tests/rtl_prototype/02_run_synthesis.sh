#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# Run the actual GF4 RTL synthesis: functional sanity check, then Yosys
# synthesis to Nangate45, then an area report you can set against Table 3's
# Timeloop/Accelergy ≤0.01% area claim.
#
# BEFORE RUNNING:
#   source ~/eda/oss-cad-suite/environment
#
# Usage:
#   ./02_run_synthesis.sh <top_module> <file1.v> [file2.v ...]
#
# Example (fill in your real filenames/module name):
#   ./02_run_synthesis.sh gf4_decode_top \
#       rtl/lutxlut_mac.v rtl/block_scaled_mac.v rtl/gf4_activation_encoder.v \
#       rtl/fp16_ingest.v rtl/gf4_decode_top.v
#
# If you also have the Icarus/Verilator testbench used for the Section 2.4
# correctness claims, pass it separately as TESTBENCH=path/to/tb.v below —
# that gets run first as a sanity check so you know synthesis is operating
# on the same design that passed your existing verification.
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <top_module> <rtl_file1.v> [rtl_file2.v ...]"
  exit 1
fi

TOP="$1"; shift
RTL_FILES=("$@")
TESTBENCH="${TESTBENCH:-}"

EDA_DIR="${EDA_DIR:-$HOME/eda}"
NANGATE_LIB="$EDA_DIR/nangate45/lib/NangateOpenCellLibrary_typical.lib"
OUT_DIR="synth_out"
mkdir -p "$OUT_DIR"

command -v yosys >/dev/null 2>&1 || { echo "yosys not on PATH — did you 'source ~/eda/oss-cad-suite/environment'?"; exit 1; }
[ -f "$NANGATE_LIB" ] || { echo "Nangate45 liberty not found at $NANGATE_LIB — rerun 01_setup_toolchain.sh"; exit 1; }

# ---- 0. Optional: functional sanity check against your existing testbench ----
if [ -n "$TESTBENCH" ]; then
  echo "=== Functional sanity check (Icarus Verilog) ==="
  iverilog -g2012 -o "$OUT_DIR/sim.vvp" "$TESTBENCH" "${RTL_FILES[@]}"
  vvp "$OUT_DIR/sim.vvp" | tee "$OUT_DIR/sim.log"
  echo "If this doesn't match the pass/fail behavior you already verified,"
  echo "stop here — synthesis on a broken checkout tells you nothing."
  echo ""
fi

# ---- 1. Yosys synthesis script ----
YS_SCRIPT="$OUT_DIR/synth.ys"
{
  for f in "${RTL_FILES[@]}"; do
    echo "read_verilog -sv \"$f\""
  done
  echo "hierarchy -top $TOP"
  echo "proc; opt; fsm; opt; memory; opt"
  echo "techmap; opt"
  echo "dfflibmap -liberty \"$NANGATE_LIB\""
  echo "abc -liberty \"$NANGATE_LIB\""
  echo "clean"
  echo "write_verilog $OUT_DIR/${TOP}_synth.v"
  echo "write_json $OUT_DIR/${TOP}_synth.json"
  echo "tee -o $OUT_DIR/area_report.txt stat -liberty \"$NANGATE_LIB\""
} > "$YS_SCRIPT"

echo "=== Running Yosys synthesis (top: $TOP) ==="
yosys -s "$YS_SCRIPT"

echo ""
echo "==================================================================="
echo "Area report:  $OUT_DIR/area_report.txt"
echo "Synth netlist: $OUT_DIR/${TOP}_synth.v"
echo "==================================================================="
cat "$OUT_DIR/area_report.txt"

echo ""
echo "--- Next: power estimation (needs a switching-activity trace) ---"
echo "Yosys only gives area for free. For a power number to compare against"
echo "the <=1.1% energy claim, you need real toggle activity, not just area:"
echo ""
echo "1. Dump a VCD from your existing testbench sim (add \$dumpfile/\$dumpvars"
echo "   to $TESTBENCH, or add them to a copy), e.g.:"
echo "     iverilog -g2012 -o $OUT_DIR/sim_vcd.vvp $TESTBENCH ${RTL_FILES[*]}"
echo "     vvp $OUT_DIR/sim_vcd.vvp    # produces dump.vcd if TB has \$dumpvars"
echo ""
echo "2. Feed the synthesized netlist + VCD into OpenSTA (bundled in OSS CAD"
echo "   Suite as 'sta') for a liberty-based power report:"
echo "     sta -exit 03_power_report.tcl"
echo "   (see 03_power_report.tcl in this folder — fill in the VCD path and"
echo "   your clock period before running)"
echo ""
echo "This gives a real per-cell dynamic power number from actual switching,"
echo "not an architectural estimate — the direct counterpart to Accelergy's"
echo "modeled ~1 pJ/access figure in Table 3."