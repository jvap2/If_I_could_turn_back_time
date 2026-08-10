#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# synth_smoke_test.sh
#
# Open-source (Yosys-only) sanity check for the two GF4/E2M1 decode variants
# and the gf4_pe wrapper. This is NOT a PDK synthesis run -- it uses Yosys's
# generic cell library (no `abc` gate mapping; abc hangs in this sandbox, see
# NOTES_rtl_synthesis.md), so the cell counts below are a rough structural
# proxy (mux/DFF/gate counts), not calibrated area/power/timing numbers.
# Use this to catch syntax/elaboration regressions quickly; use a real PDK
# flow (Design Compiler / Fusion Compiler + a real standard-cell library) for
# any number that goes in the paper.
#
# Usage:
#   chmod +x synth_smoke_test.sh
#   ./synth_smoke_test.sh          (do NOT run as `sh synth_smoke_test.sh` --
#                                    that forces dash, which doesn't support
#                                    `set -o pipefail` below; ./ or `bash
#                                    synth_smoke_test.sh` respects the bash
#                                    shebang on line 1)
# -----------------------------------------------------------------------------
set -eu
if [ -n "${BASH_VERSION:-}" ]; then
  set -o pipefail   # bash-only option; only touched when actually running under bash
fi

# Prefer a native `yosys` install (e.g. `sudo apt install yosys`) if present --
# on a real machine (not this sandbox's WASM/wasmtime build) it likely has a
# working `abc` gate-mapping step, which could get you further than the
# generic-cell-only numbers in NOTES_rtl_synthesis.md. Falls back to the
# pip-installable WASM-packaged `yowasp-yosys` if that's what you have instead.
if command -v yosys >/dev/null 2>&1; then
  YOSYS=yosys
elif command -v yowasp-yosys >/dev/null 2>&1; then
  YOSYS=yowasp-yosys
else
  echo "No Yosys found. Install one of:"
  echo "  sudo apt install yosys        (native, recommended if available)"
  echo "  pip install --user yowasp-yosys   (no root needed, WASM-packaged)"
  exit 1
fi
echo "Using: $YOSYS ($(command -v "$YOSYS"))"

echo "=== [1/4] Elaborate + parse-check testbench ==="
$YOSYS -p "read_verilog -sv gf4_decode_fixed.v gf4_decode_programmable.v tb_gf4_decode.v; hierarchy -check -top tb_gf4_decode; proc; opt" \
  2>&1 | grep -E "ERROR|Successfully|\\\$finish" || true

echo
echo "=== [2/4] gf4_decode_fixed (E2M1 ROM) generic-cell stat ==="
$YOSYS -p "read_verilog gf4_decode_fixed.v; hierarchy -check -top gf4_decode_fixed; proc; opt; memory; opt; techmap; opt; stat" \
  2>&1 | awk '/Printing statistics/,0'

echo
echo "=== [3/4] gf4_decode_programmable (GF4 table) generic-cell stat ==="
$YOSYS -p "read_verilog gf4_decode_programmable.v; hierarchy -check -top gf4_decode_programmable; proc; opt; memory; opt; techmap; opt; stat" \
  2>&1 | awk '/Printing statistics/,0'

echo
echo "=== [4/4] gf4_pe -- both PROGRAMMABLE variants ==="
echo "--- PROGRAMMABLE=1 (Q0.8, MAG_WIDTH=9) ---"
$YOSYS -p "read_verilog gf4_decode_fixed.v gf4_decode_programmable.v gf4_pe.v; hierarchy -check -top gf4_pe -chparam PROGRAMMABLE 1 -chparam MAG_WIDTH 9; proc; opt; memory; opt; techmap; opt; stat" \
  2>&1 | awk '/Printing statistics/,0'

echo "--- PROGRAMMABLE=0 (Q4.4, MAG_WIDTH=8) ---"
$YOSYS -p "read_verilog gf4_decode_fixed.v gf4_decode_programmable.v gf4_pe.v; hierarchy -check -top gf4_pe -chparam PROGRAMMABLE 0 -chparam MAG_WIDTH 8; proc; opt; memory; opt; techmap; opt; stat" \
  2>&1 | awk '/Printing statistics/,0'

echo
echo "=== [5/6] Phase 1: elaborate + parse-check joint product table testbench ==="
$YOSYS -p "read_verilog -sv gf4_joint_product_table.v tb_gf4_joint_product.v; hierarchy -check -top tb_gf4_joint_product; proc; opt" \
  2>&1 | grep -E "ERROR|Successfully|\\\$finish" || true

echo
echo "=== [6/6] Phase 1: gf4_joint_product_table (standalone) and gf4_pe_lutxlut generic-cell stat ==="
echo "--- gf4_joint_product_table standalone ---"
$YOSYS -p "read_verilog gf4_joint_product_table.v; hierarchy -check -top gf4_joint_product_table; proc; opt; memory; opt; techmap; opt; stat" \
  2>&1 | awk '/Printing statistics/,0'

echo "--- gf4_pe_lutxlut (joint table + accumulate, no multiplier) ---"
$YOSYS -p "read_verilog gf4_joint_product_table.v gf4_pe_lutxlut.v; hierarchy -check -top gf4_pe_lutxlut; proc; opt; memory; opt; techmap; opt; stat" \
  2>&1 | awk '/Printing statistics/,0'

echo
echo "=== [7/8] W4A16: gf4_pe with OPERAND_WIDTH=16 (decoded weight code x wide/FP16-stand-in operand) ==="
echo "--- PROGRAMMABLE=1, OPERAND_WIDTH=16 ---"
$YOSYS -p "read_verilog gf4_decode_fixed.v gf4_decode_programmable.v gf4_pe.v; hierarchy -check -top gf4_pe -chparam PROGRAMMABLE 1 -chparam MAG_WIDTH 9 -chparam OPERAND_WIDTH 16; proc; opt; memory; opt; techmap; opt; stat" \
  2>&1 | awk '/Printing statistics/,0'

echo "--- PROGRAMMABLE=0, OPERAND_WIDTH=16 ---"
$YOSYS -p "read_verilog gf4_decode_fixed.v gf4_decode_programmable.v gf4_pe.v; hierarchy -check -top gf4_pe -chparam PROGRAMMABLE 0 -chparam MAG_WIDTH 8 -chparam OPERAND_WIDTH 16; proc; opt; memory; opt; techmap; opt; stat" \
  2>&1 | awk '/Printing statistics/,0'

echo
echo "=== [8/8] Block-scaled MAC: elaborate testbench + gf4_block_scaled_mac generic-cell stat ==="
echo "--- elaborate + parse-check tb_gf4_block_scaled_mac ---"
$YOSYS -p "read_verilog -sv gf4_joint_product_table.v gf4_block_scaled_mac.v tb_gf4_block_scaled_mac.v; hierarchy -check -top tb_gf4_block_scaled_mac; proc; opt" \
  2>&1 | grep -E "ERROR|Successfully|\\\$finish" || true
echo "--- gf4_block_scaled_mac generic-cell stat ---"
$YOSYS -p "read_verilog gf4_joint_product_table.v gf4_block_scaled_mac.v; hierarchy -check -top gf4_block_scaled_mac; proc; opt; memory; opt; techmap; opt; stat" \
  2>&1 | awk '/Printing statistics/,0'

echo
echo "=== [9/9] Phase 2: elaborate + parse-check gf4_isqrt testbench, and its generic-cell stat ==="
echo "--- elaborate + parse-check tb_gf4_isqrt ---"
$YOSYS -p "read_verilog -sv gf4_isqrt.v tb_gf4_isqrt.v; hierarchy -check -top tb_gf4_isqrt; proc; opt" \
  2>&1 | grep -E "ERROR|Successfully|\\\$finish" || true
echo "--- gf4_isqrt generic-cell stat ---"
$YOSYS -p "read_verilog gf4_isqrt.v; hierarchy -check -top gf4_isqrt; proc; opt; memory; opt; techmap; opt; stat" \
  2>&1 | awk '/Printing statistics/,0'

echo
echo "=== [10/10] Phase 2: elaborate + parse-check gf4_sumsq (chained w/ gf4_isqrt) testbench, and its generic-cell stat ==="
echo "--- elaborate + parse-check tb_gf4_sumsq ---"
$YOSYS -p "read_verilog -sv gf4_sumsq.v gf4_isqrt.v tb_gf4_sumsq.v; hierarchy -check -top tb_gf4_sumsq; proc; opt" \
  2>&1 | grep -E "ERROR|Successfully|\\\$finish" || true
echo "--- gf4_sumsq generic-cell stat (standalone) ---"
$YOSYS -p "read_verilog gf4_sumsq.v; hierarchy -check -top gf4_sumsq; proc; opt; memory; opt; techmap; opt; stat" \
  2>&1 | awk '/Printing statistics/,0'

echo
echo "=== [11/11] Phase 2: elaborate + parse-check gf4_activation_encoder (final Phase 2 piece) testbench, and its generic-cell stat ==="
echo "--- elaborate + parse-check tb_gf4_activation_encoder ---"
$YOSYS -p "read_verilog -sv gf4_sumsq.v gf4_isqrt.v gf4_activation_encoder.v tb_gf4_activation_encoder.v; hierarchy -check -top tb_gf4_activation_encoder; proc; opt" \
  2>&1 | grep -E "ERROR|Successfully|\\\$finish" || true
echo "--- gf4_activation_encoder generic-cell stat (includes gf4_sumsq + gf4_isqrt submodules) ---"
$YOSYS -p "read_verilog gf4_sumsq.v gf4_isqrt.v gf4_activation_encoder.v; hierarchy -check -top gf4_activation_encoder; proc; opt; memory; opt; techmap; opt; stat" \
  2>&1 | awk '/Printing statistics/,0'

echo
echo "=== [12/12] Phase 2: elaborate + parse-check gf4_fp16_to_fixed (real FP16 ingest) testbench, and its generic-cell stat ==="
echo "--- elaborate + parse-check tb_gf4_fp16_to_fixed ---"
$YOSYS -p "read_verilog -sv gf4_fp16_to_fixed.v tb_gf4_fp16_to_fixed.v; hierarchy -check -top tb_gf4_fp16_to_fixed; proc; opt" \
  2>&1 | grep -E "ERROR|Successfully|\\\$finish" || true
echo "--- gf4_fp16_to_fixed generic-cell stat (standalone) ---"
$YOSYS -p "read_verilog gf4_fp16_to_fixed.v; hierarchy -check -top gf4_fp16_to_fixed; proc; opt; memory; opt; techmap; opt; stat" \
  2>&1 | awk '/Printing statistics/,0'

echo
echo "Done. Cross-check these generic-cell counts against NOTES_rtl_synthesis.md before citing any number."
echo "Notes:"
echo "  - gf4_pe_lutxlut trades the multiplier in gf4_pe.v for a much larger table"
echo "    (64 x 17-bit vs. 8 x 9-bit) -- compare cell counts to see the real tradeoff,"
echo "    not just 'no multiplier'."
echo "  - gf4_block_scaled_mac's cell count is dominated by its two multipliers"
echo "    (Yosys's generic techmap maps multiply to a full gate array, not a real"
echo "    multiplier/DSP cell) -- do not read this as a realistic area number."
echo "  - gf4_isqrt is sequential (18-cycle latency per sqrt), unlike everything"
echo "    else in this project so far -- its cell count is not directly comparable"
echo "    to the combinational/single-cycle modules above."
echo "  - gf4_fp16_to_fixed is purely combinational (a barrel shift + clamp, not"
echo "    floating-point arithmetic) -- notice it has NO multiplier in its cell"
echo "    breakdown (all \$_MUX_/\$_AND_/\$_OR_/\$_XOR_/\$_NOT_), unlike every other"
echo "    module above that touches a scale factor."