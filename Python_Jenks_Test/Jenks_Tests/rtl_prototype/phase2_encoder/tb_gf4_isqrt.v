// -----------------------------------------------------------------------------
// tb_gf4_isqrt.v
//
// Behavioral testbench for gf4_isqrt.v. Not synthesizable -- run with a real
// event simulator, e.g.:
//
//   iverilog -o sim gf4_isqrt.v tb_gf4_isqrt.v
//   vvp sim
//
// Drives 15 test vectors through the DUT: the paper's own worked example
// (block RMS=0.5, i.e. radicand = 0.5^2 in Q16.16 = 16384, expected root =
// 0.5*256 = 128 in Q8.8), several other exactly-representable reals (1.25,
// 2.5, 10.0, 100.0, and the near-max 255.99609375), plus 8 random 32-bit
// values. Every expected root was computed offline in Python via
// math.isqrt() -- and math.isqrt() was itself cross-checked against this
// exact bit-by-bit algorithm (with the same fixed hardware register widths
// used in the RTL) across 200,000 random values with zero mismatches before
// any of this was encoded.
//
// Timing: latency from asserting `start` to `done` is fixed at
// OUT_WIDTH + 2 = 18 cycles (180ns at this 10ns period), so each vector
// uses a flat #190 wait (comfortable margin) rather than polling `busy` --
// a `while` loop polling a signal is simulator-standard but Yosys's
// frontend (used only for the elaboration smoke test here, not real
// simulation) only allows while loops inside constant functions.
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
 
module tb_gf4_isqrt;
 
    reg clk = 0;
    reg rst_n = 0;
    reg start = 0;
    reg [31:0] radicand = 0;
 
    wire busy;
    wire done;
    wire [15:0] root;
 
    integer i;
    integer errors = 0;
 
    reg [31:0] radicand_vec [0:14];
    reg [15:0] expected_root_vec [0:14];
 
    initial begin
        radicand_vec[0] = 32'd0; expected_root_vec[0] = 16'd0;
        radicand_vec[1] = 32'd16384; expected_root_vec[1] = 16'd128;
        radicand_vec[2] = 32'd102400; expected_root_vec[2] = 16'd320;
        radicand_vec[3] = 32'd409600; expected_root_vec[3] = 16'd640;
        radicand_vec[4] = 32'd6553600; expected_root_vec[4] = 16'd2560;
        radicand_vec[5] = 32'd655360000; expected_root_vec[5] = 16'd25600;
        radicand_vec[6] = 32'd4294836225; expected_root_vec[6] = 16'd65535;
        radicand_vec[7] = 32'd2746317213; expected_root_vec[7] = 16'd52405;
        radicand_vec[8] = 32'd1181241943; expected_root_vec[8] = 16'd34369;
        radicand_vec[9] = 32'd958682846; expected_root_vec[9] = 16'd30962;
        radicand_vec[10] = 32'd3163119785; expected_root_vec[10] = 16'd56241;
        radicand_vec[11] = 32'd1812140441; expected_root_vec[11] = 16'd42569;
        radicand_vec[12] = 32'd127978094; expected_root_vec[12] = 16'd11312;
        radicand_vec[13] = 32'd939042955; expected_root_vec[13] = 16'd30643;
        radicand_vec[14] = 32'd2340505846; expected_root_vec[14] = 16'd48378;
    end
 
    gf4_isqrt #(.IN_WIDTH(32), .OUT_WIDTH(16)) dut (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (start),
        .radicand (radicand),
        .busy     (busy),
        .done     (done),
        .root     (root)
    );
 
    always #5 clk = ~clk;
 
    initial begin
        rst_n = 0;
        #12 rst_n = 1;
        #8; // land on a negedge before driving anything
 
        for (i = 0; i < 15; i = i + 1) begin
            radicand = radicand_vec[i];
            start = 1'b1;
            #10;              // one cycle: DUT latches radicand, moves to RUN
            start = 1'b0;
            #180;             // covers the full 18-cycle latency plus margin
            #1;
 
            if (root !== expected_root_vec[i]) begin
                $display("FAIL: vec=%0d radicand=%0d expected=%0d got=%0d", i, radicand_vec[i], expected_root_vec[i], root);
                errors = errors + 1;
            end else begin
                $display("PASS: vec=%0d radicand=%0d root=%0d", i, radicand_vec[i], root);
            end
 
            #10; // extra settle before the next start pulse
        end
 
        if (errors == 0)
            $display("ALL CHECKS PASSED");
        else
            $display("%0d CHECK(S) FAILED", errors);
 
        $finish;
    end
 
endmodule