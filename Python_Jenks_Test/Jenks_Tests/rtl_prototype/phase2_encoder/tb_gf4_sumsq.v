// -----------------------------------------------------------------------------
// tb_gf4_sumsq.v
//
// Testbench for gf4_sumsq.v, chained directly into gf4_isqrt.v to demonstrate
// the full block-scale pipeline: 16 streamed Q8.8 activations -> Q16.16
// mean-of-squares -> Q8.8 RMS(x_b) (Eq. 8's sqrt((1/b) Sum(x_i^2))).
//
// 11 blocks are driven through: a constant-magnitude block reproducing the
// paper's own worked example (RMS(x_b)=0.5, see tb_gf4_decode.v), an
// alternating-sign block with the same RMS (confirms squaring erases sign),
// a non-trivial 8-nonzero/8-zero construction also targeting RMS=0.5, and
// eight random 16-element blocks (activations in +/-4.0) cross-checked against
// a bit-exact Python model of this module's accumulate-then-shift arithmetic
// and math.isqrt() for the chained gf4_isqrt result.
//
// Timing note: inputs are changed on delay-based waits (#10, aligned to
// negedges), not `@(negedge clk)` event control, for the same reason as
// tb_gf4_block_scaled_mac.v -- this keeps the file parseable by Yosys's
// frontend for the elaboration smoke test. gf4_sumsq's mean_sq is a single
// registered stage (valid the cycle after block_last, same as
// gf4_block_scaled_mac.v's block_sum_reg), so it settles by the end of the
// last element's #10 window with no extra wait needed; the chained
// gf4_isqrt result needs one more cycle (for isqrt to register-capture
// mean_sq/mean_sq_valid across the module boundary) plus its own fixed
// 18-cycle latency, so a #200 margin (20 cycles) is used before checking
// rms_root, generously covering both.
//
// Deliberately avoids SystemVerilog array-literal syntax ('{...}) for the
// per-block activation data below -- every element is assigned individually,
// matching this project's existing testbenches, so this file runs unmodified
// under plain Icarus Verilog (no -sv needed) and Verilator.
//
// Run with:
//   iverilog -o sim gf4_sumsq.v gf4_isqrt.v tb_gf4_sumsq.v && vvp sim
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
 
module tb_gf4_sumsq;
 
    reg clk = 0;
    reg rst_n = 0;
    always #5 clk = ~clk;
 
    reg signed [15:0] x_in = 0;
    reg elem_valid = 0;
    reg block_last = 0;
 
    wire mean_sq_valid;
    wire [31:0] mean_sq;
 
    wire isqrt_busy, isqrt_done;
    wire [15:0] rms_root;
 
    integer errors = 0;
    integer i, v;
 
    gf4_sumsq dut_sumsq (
        .clk           (clk),
        .rst_n         (rst_n),
        .x_in          (x_in),
        .elem_valid    (elem_valid),
        .block_last    (block_last),
        .mean_sq_valid (mean_sq_valid),
        .mean_sq       (mean_sq)
    );
 
    // Directly chained: gf4_sumsq's mean_sq/mean_sq_valid feed gf4_isqrt's
    // radicand/start, exactly as intended in the real block-scale pipeline.
    gf4_isqrt #(.IN_WIDTH(32), .OUT_WIDTH(16)) dut_isqrt (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (mean_sq_valid),
        .radicand (mean_sq),
        .busy     (isqrt_busy),
        .done     (isqrt_done),
        .root     (rms_root)
    );
 
    reg [31:0] expected_mean_sq [0:10];
    reg [15:0] expected_root    [0:10];
    reg signed [15:0] block_x   [0:10][0:15];
 
    initial begin
        // --- block 0: constant_0p5_matches_paper_RMS_worked_example (acc=262144) ---
        block_x[0][0] = 16'sh0080;
        block_x[0][1] = 16'sh0080;
        block_x[0][2] = 16'sh0080;
        block_x[0][3] = 16'sh0080;
        block_x[0][4] = 16'sh0080;
        block_x[0][5] = 16'sh0080;
        block_x[0][6] = 16'sh0080;
        block_x[0][7] = 16'sh0080;
        block_x[0][8] = 16'sh0080;
        block_x[0][9] = 16'sh0080;
        block_x[0][10] = 16'sh0080;
        block_x[0][11] = 16'sh0080;
        block_x[0][12] = 16'sh0080;
        block_x[0][13] = 16'sh0080;
        block_x[0][14] = 16'sh0080;
        block_x[0][15] = 16'sh0080;
        expected_mean_sq[0] = 32'd16384;
        expected_root[0]    = 16'd128;
 
        // --- block 1: alternating_pm0p5_same_RMS_as_above (acc=262144) ---
        block_x[1][0] = 16'sh0080;
        block_x[1][1] = 16'shff80;
        block_x[1][2] = 16'sh0080;
        block_x[1][3] = 16'shff80;
        block_x[1][4] = 16'sh0080;
        block_x[1][5] = 16'shff80;
        block_x[1][6] = 16'sh0080;
        block_x[1][7] = 16'shff80;
        block_x[1][8] = 16'sh0080;
        block_x[1][9] = 16'shff80;
        block_x[1][10] = 16'sh0080;
        block_x[1][11] = 16'shff80;
        block_x[1][12] = 16'sh0080;
        block_x[1][13] = 16'shff80;
        block_x[1][14] = 16'sh0080;
        block_x[1][15] = 16'shff80;
        expected_mean_sq[1] = 32'd16384;
        expected_root[1]    = 16'd128;
 
        // --- block 2: 8_nonzero_8_zero_also_targets_RMS_0p5 (acc=262088) ---
        block_x[2][0] = 16'sh00b5;
        block_x[2][1] = 16'sh00b5;
        block_x[2][2] = 16'sh00b5;
        block_x[2][3] = 16'sh00b5;
        block_x[2][4] = 16'sh00b5;
        block_x[2][5] = 16'sh00b5;
        block_x[2][6] = 16'sh00b5;
        block_x[2][7] = 16'sh00b5;
        block_x[2][8] = 16'sh0000;
        block_x[2][9] = 16'sh0000;
        block_x[2][10] = 16'sh0000;
        block_x[2][11] = 16'sh0000;
        block_x[2][12] = 16'sh0000;
        block_x[2][13] = 16'sh0000;
        block_x[2][14] = 16'sh0000;
        block_x[2][15] = 16'sh0000;
        expected_mean_sq[2] = 32'd16380;
        expected_root[2]    = 16'd127;
 
        // --- block 3: random_block_0 (acc=5997528) ---
        block_x[3][0] = 16'sh011e;
        block_x[3][1] = 16'shfc33;
        block_x[3][2] = 16'shfe33;
        block_x[3][3] = 16'shfdc9;
        block_x[3][4] = 16'sh01e4;
        block_x[3][5] = 16'sh016a;
        block_x[3][6] = 16'sh0323;
        block_x[3][7] = 16'shfcb2;
        block_x[3][8] = 16'shff60;
        block_x[3][9] = 16'shfc3d;
        block_x[3][10] = 16'shfdc0;
        block_x[3][11] = 16'sh000b;
        block_x[3][12] = 16'shfc36;
        block_x[3][13] = 16'shfd97;
        block_x[3][14] = 16'sh0133;
        block_x[3][15] = 16'sh005c;
        expected_mean_sq[3] = 32'd374845;
        expected_root[3]    = 16'd612;
 
        // --- block 4: random_block_1 (acc=6482554) ---
        block_x[4][0] = 16'shfdc3;
        block_x[4][1] = 16'sh00b7;
        block_x[4][2] = 16'sh027a;
        block_x[4][3] = 16'shfc0d;
        block_x[4][4] = 16'sh0272;
        block_x[4][5] = 16'sh0196;
        block_x[4][6] = 16'shfeb9;
        block_x[4][7] = 16'shfd3e;
        block_x[4][8] = 16'sh03a8;
        block_x[4][9] = 16'shfeb1;
        block_x[4][10] = 16'shfcbe;
        block_x[4][11] = 16'shfcc6;
        block_x[4][12] = 16'sh02c8;
        block_x[4][13] = 16'sh00d4;
        block_x[4][14] = 16'sh0275;
        block_x[4][15] = 16'sh01d6;
        expected_mean_sq[4] = 32'd405159;
        expected_root[4]    = 16'd636;
 
        // --- block 5: random_block_2 (acc=5558121) ---
        block_x[5][0] = 16'sh004a;
        block_x[5][1] = 16'sh03c9;
        block_x[5][2] = 16'shff07;
        block_x[5][3] = 16'sh006b;
        block_x[5][4] = 16'sh02a3;
        block_x[5][5] = 16'sh00f3;
        block_x[5][6] = 16'sh02e5;
        block_x[5][7] = 16'sh009e;
        block_x[5][8] = 16'sh01a3;
        block_x[5][9] = 16'shfc5e;
        block_x[5][10] = 16'shfdd3;
        block_x[5][11] = 16'shfe51;
        block_x[5][12] = 16'shfca3;
        block_x[5][13] = 16'shfddd;
        block_x[5][14] = 16'shfccf;
        block_x[5][15] = 16'shfe39;
        expected_mean_sq[5] = 32'd347382;
        expected_root[5]    = 16'd589;
 
        // --- block 6: random_block_3 (acc=4202462) ---
        block_x[6][0] = 16'sh0116;
        block_x[6][1] = 16'shfeeb;
        block_x[6][2] = 16'shfef6;
        block_x[6][3] = 16'shfdad;
        block_x[6][4] = 16'shfe23;
        block_x[6][5] = 16'sh037e;
        block_x[6][6] = 16'sh012f;
        block_x[6][7] = 16'sh00e0;
        block_x[6][8] = 16'shfd5e;
        block_x[6][9] = 16'sh01d5;
        block_x[6][10] = 16'shfd4f;
        block_x[6][11] = 16'shff09;
        block_x[6][12] = 16'sh03eb;
        block_x[6][13] = 16'sh011f;
        block_x[6][14] = 16'sh0075;
        block_x[6][15] = 16'sh017a;
        expected_mean_sq[6] = 32'd262653;
        expected_root[6]    = 16'd512;
 
        // --- block 7: random_block_4 (acc=5695694) ---
        block_x[7][0] = 16'sh02be;
        block_x[7][1] = 16'sh0235;
        block_x[7][2] = 16'shfdd5;
        block_x[7][3] = 16'shfc42;
        block_x[7][4] = 16'shfe86;
        block_x[7][5] = 16'shfe24;
        block_x[7][6] = 16'shfdb0;
        block_x[7][7] = 16'sh038b;
        block_x[7][8] = 16'sh0303;
        block_x[7][9] = 16'shfe84;
        block_x[7][10] = 16'sh013e;
        block_x[7][11] = 16'shff2a;
        block_x[7][12] = 16'sh0351;
        block_x[7][13] = 16'shffac;
        block_x[7][14] = 16'shfe1e;
        block_x[7][15] = 16'shfdf9;
        expected_mean_sq[7] = 32'd355980;
        expected_root[7]    = 16'd596;
 
        // --- block 8: random_block_5 (acc=5868683) ---
        block_x[8][0] = 16'sh007e;
        block_x[8][1] = 16'shfe1a;
        block_x[8][2] = 16'sh00ad;
        block_x[8][3] = 16'sh032f;
        block_x[8][4] = 16'shff32;
        block_x[8][5] = 16'shfdc1;
        block_x[8][6] = 16'sh03fb;
        block_x[8][7] = 16'sh0014;
        block_x[8][8] = 16'shfcba;
        block_x[8][9] = 16'shfc60;
        block_x[8][10] = 16'shfce1;
        block_x[8][11] = 16'sh0105;
        block_x[8][12] = 16'sh0256;
        block_x[8][13] = 16'shff61;
        block_x[8][14] = 16'shfc82;
        block_x[8][15] = 16'shff0e;
        expected_mean_sq[8] = 32'd366792;
        expected_root[8]    = 16'd605;
 
        // --- block 9: random_block_6 (acc=6525434) ---
        block_x[9][0] = 16'sh03f8;
        block_x[9][1] = 16'sh003c;
        block_x[9][2] = 16'sh03c5;
        block_x[9][3] = 16'sh02e3;
        block_x[9][4] = 16'shfc18;
        block_x[9][5] = 16'sh01c4;
        block_x[9][6] = 16'sh0174;
        block_x[9][7] = 16'sh004c;
        block_x[9][8] = 16'shfe22;
        block_x[9][9] = 16'sh0121;
        block_x[9][10] = 16'shfce4;
        block_x[9][11] = 16'shff7a;
        block_x[9][12] = 16'shffa1;
        block_x[9][13] = 16'sh03a1;
        block_x[9][14] = 16'sh0302;
        block_x[9][15] = 16'shfe1b;
        expected_mean_sq[9] = 32'd407839;
        expected_root[9]    = 16'd638;
 
        // --- block 10: random_block_7 (acc=6073031) ---
        block_x[10][0] = 16'sh0001;
        block_x[10][1] = 16'shfd6e;
        block_x[10][2] = 16'sh034d;
        block_x[10][3] = 16'sh02f7;
        block_x[10][4] = 16'shfe63;
        block_x[10][5] = 16'sh011d;
        block_x[10][6] = 16'sh00df;
        block_x[10][7] = 16'shfd39;
        block_x[10][8] = 16'sh021a;
        block_x[10][9] = 16'sh0051;
        block_x[10][10] = 16'sh023b;
        block_x[10][11] = 16'sh003e;
        block_x[10][12] = 16'shfc01;
        block_x[10][13] = 16'shfe98;
        block_x[10][14] = 16'shfc28;
        block_x[10][15] = 16'sh036f;
        expected_mean_sq[10] = 32'd379564;
        expected_root[10]    = 16'd616;
 
    end
 
    initial begin
        rst_n = 0;
        elem_valid = 0;
        block_last = 0;
        #12 rst_n = 1;
        #8;                    // t=20, a negedge -- matches tb_gf4_block_scaled_mac.v's convention
 
        for (v = 0; v < 11; v = v + 1) begin
            for (i = 0; i < 16; i = i + 1) begin
                x_in       = block_x[v][i];
                elem_valid = 1'b1;
                block_last = (i == 15);
                #10;                // hold stable for one full period, set at a negedge
            end
            elem_valid = 1'b0;
            block_last = 1'b0;
 
            // mean_sq is a single registered stage (valid the cycle after
            // block_last, same timing as gf4_block_scaled_mac.v's block_sum_reg)
            // so it has already settled by this point in the loop.
            if (mean_sq !== expected_mean_sq[v]) begin
                $display("FAIL: block %0d mean_sq expected=%0d got=%0d", v, expected_mean_sq[v], mean_sq);
                errors = errors + 1;
            end else begin
                $display("PASS: block %0d mean_sq=%0d", v, mean_sq);
            end
 
            // Margin for the chained gf4_isqrt: one cycle for it to register-
            // capture mean_sq/mean_sq_valid across the module boundary, plus
            // its own fixed 18-cycle (180ns) latency -- #200 (20 cycles) is a
            // generous cushion over the ~190ns minimum.
            #200;
            if (rms_root !== expected_root[v]) begin
                $display("FAIL: block %0d chained rms_root expected=%0d got=%0d", v, expected_root[v], rms_root);
                errors = errors + 1;
            end else begin
                $display("PASS: block %0d chained rms_root=%0d", v, rms_root);
            end
        end
 
        if (errors == 0)
            $display("ALL CHECKS PASSED");
        else
            $display("%0d CHECK(S) FAILED", errors);
 
        $finish;
    end
 
endmodule