# ─────────────────────────────────────────────────────────────────────────
# OpenSTA power report — run with: sta -exit 03_power_report.tcl
# (the 'sta' binary ships in OSS CAD Suite; source its environment first)
#
# Fill in the four paths/values below, then run. This reads the synthesized
# gate-level netlist from 02_run_synthesis.sh, applies real switching
# activity from a VCD you captured during your existing Icarus/Verilator
# testbench run, and reports per-cell dynamic + leakage power from the
# Nangate45 liberty tables — a measured counterpart to Table 3's modeled
# ~1 pJ/access Accelergy figure.
# ─────────────────────────────────────────────────────────────────────────

set NANGATE_LIB   "$env(HOME)/eda/nangate45/lib/NangateOpenCellLibrary_typical.lib"
set NETLIST       "synth_out/CHANGE_ME_synth.v"   ;# from 02_run_synthesis.sh output
set TOP_MODULE    "CHANGE_ME_top"                  ;# same name you passed to 02_run_synthesis.sh
set VCD_FILE      "synth_out/dump.vcd"             ;# from your testbench $dumpvars run
set CLOCK_PORT    "clk"                            ;# your top-level clock signal name
set CLOCK_PERIOD  2.0                              ;# ns — match whatever your testbench actually drives

read_liberty $NANGATE_LIB
read_verilog $NETLIST
link_design  $TOP_MODULE

create_clock -name core_clock -period $CLOCK_PERIOD [get_ports $CLOCK_PORT]

# Anneal real switching activity from simulation onto the netlist instead of
# using OpenSTA's default activity guesses — this is the step that makes the
# result a measurement rather than a second-order estimate.
read_vcd -scope $TOP_MODULE $VCD_FILE

report_checks -path_delay max
report_power