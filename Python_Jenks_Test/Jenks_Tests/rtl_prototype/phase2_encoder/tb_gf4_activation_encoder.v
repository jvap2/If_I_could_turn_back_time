// -----------------------------------------------------------------------------
// tb_gf4_activation_encoder.v
//
// Testbench for gf4_activation_encoder.v -- the final Phase 2 piece, closing
// the loop from raw Q8.8 activations all the way to 4-bit GF4 codes plus the
// block's s_a_scale (ready to feed gf4_block_scaled_mac.v's s_a_scale port).
//
// 11 blocks are driven through, each checking: the intermediate mean_sq/rms/s_b
// debug taps, and all 16 emitted 4-bit codes plus s_a_scale, all cross-checked
// against a bit-exact Python model of this exact fixed-point pipeline
// (gf4_sumsq.v's accumulate-then-shift, gf4_isqrt.v's algorithm, and the
// threshold-scale-then-compare arithmetic in gf4_activation_encoder.v itself).
// Blocks:
//   0) a block containing the paper's own xi=-0.7 (see tb_gf4_decode.v's worked
//      example) among fifteen 0.5-magnitude activations, alpha=2.5 -- checks
//      that element's code comes out sign=1, idx=5, matching the paper exactly.
//   1) a constant-0.5 block (exact RMS=0.5, alpha=2.5 -> s_b=1.25 exactly).
//   2) an all-zero degenerate block: RMS=0 -> s_b=0 -> every threshold scales
//      to 0, so abs_x(=0) >= 0 for all 7 thresholds -> every code saturates to
//      idx=7. This is a genuine, documented edge case (see gf4_activation_encoder.v
//      comments) -- not a bug, but worth knowing this module does NOT special-case
//      a truly silent (all-zero) block.
//   3-10) eight random 16-element blocks with random per-block alpha.
//
// Capture technique: rather than trying to hand-time exactly when each of the
// 16 streamed-out codes appears (the pipeline latency from block_last to the
// first code is a fixed but not-trivial-to-hand-count ~22 cycles, then 1 more
// cycle per subsequent code), a small monitor always-block below captures every
// a_code_valid pulse into an array indexed by a_code_elem_idx, and every
// s_a_valid pulse into a holding register. The stimulus then just waits a
// generous fixed margin (#500, comfortably more than the worst-case ~370ns)
// after each block's last element before checking the captured results.
//
// Run with:
//   iverilog -o sim gf4_sumsq.v gf4_isqrt.v gf4_activation_encoder.v tb_gf4_activation_encoder.v && vvp sim
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
 
module tb_gf4_activation_encoder;
 
    reg clk = 0;
    reg rst_n = 0;
    always #5 clk = ~clk;
 
    reg signed [15:0] x_in = 0;
    reg elem_valid = 0;
    reg block_last = 0;
    reg signed [15:0] alpha_q8_8 = 0;
 
    reg cfg_we = 0;
    reg [2:0] cfg_waddr = 0;
    reg [8:0] cfg_wdata = 0;
 
    wire a_code_valid;
    wire [3:0] a_code;
    wire [3:0] a_code_elem_idx;
    wire s_a_valid;
    wire signed [15:0] s_a_scale;
    wire busy;
    wire [31:0] mean_sq_debug;
    wire [15:0] rms_debug;
    wire signed [15:0] s_b_debug;
 
    integer errors = 0;
    integer i, v;
 
    gf4_activation_encoder dut (
        .clk             (clk),
        .rst_n           (rst_n),
        .x_in            (x_in),
        .elem_valid      (elem_valid),
        .block_last      (block_last),
        .alpha_q8_8      (alpha_q8_8),
        .cfg_we          (cfg_we),
        .cfg_waddr       (cfg_waddr),
        .cfg_wdata       (cfg_wdata),
        .a_code_valid    (a_code_valid),
        .a_code          (a_code),
        .a_code_elem_idx (a_code_elem_idx),
        .s_a_valid       (s_a_valid),
        .s_a_scale       (s_a_scale),
        .busy            (busy),
        .mean_sq_debug   (mean_sq_debug),
        .rms_debug       (rms_debug),
        .s_b_debug       (s_b_debug)
    );
 
    // --- capture every emitted code and the block's s_a_scale ---
    reg [3:0] captured_codes [0:15];
    reg signed [15:0] captured_s_a;
    always @(posedge clk) begin
        if (a_code_valid) begin
            captured_codes[a_code_elem_idx] <= a_code;
        end
        if (s_a_valid) begin
            captured_s_a <= s_a_scale;
        end
    end
 
    reg [31:0] expected_mean_sq [0:10];
    reg [15:0] expected_rms     [0:10];
    reg signed [15:0] expected_s_b [0:10];
    reg signed [15:0] block_alpha  [0:10];
    reg signed [15:0] block_x      [0:10][0:15];
    reg [3:0] expected_codes       [0:10][0:15];
 
    initial begin
        // --- block 0: paper_flavored_xi_neg0p7_idx0_expect_sign1_idx5 ---
        block_alpha[0] = 16'sd640;
        block_x[0][0] = 16'shff4d;
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
        expected_mean_sq[0] = 32'd17362;
        expected_rms[0]     = 16'd131;
        expected_s_b[0]     = 16'sd327;
        expected_codes[0][0] = 4'd13;
        expected_codes[0][1] = 4'd4;
        expected_codes[0][2] = 4'd4;
        expected_codes[0][3] = 4'd4;
        expected_codes[0][4] = 4'd4;
        expected_codes[0][5] = 4'd4;
        expected_codes[0][6] = 4'd4;
        expected_codes[0][7] = 4'd4;
        expected_codes[0][8] = 4'd4;
        expected_codes[0][9] = 4'd4;
        expected_codes[0][10] = 4'd4;
        expected_codes[0][11] = 4'd4;
        expected_codes[0][12] = 4'd4;
        expected_codes[0][13] = 4'd4;
        expected_codes[0][14] = 4'd4;
        expected_codes[0][15] = 4'd4;
 
        // --- block 1: constant_0p5_alpha2p5_exact_sb_1p25 ---
        block_alpha[1] = 16'sd640;
        block_x[1][0] = 16'sh0080;
        block_x[1][1] = 16'sh0080;
        block_x[1][2] = 16'sh0080;
        block_x[1][3] = 16'sh0080;
        block_x[1][4] = 16'sh0080;
        block_x[1][5] = 16'sh0080;
        block_x[1][6] = 16'sh0080;
        block_x[1][7] = 16'sh0080;
        block_x[1][8] = 16'sh0080;
        block_x[1][9] = 16'sh0080;
        block_x[1][10] = 16'sh0080;
        block_x[1][11] = 16'sh0080;
        block_x[1][12] = 16'sh0080;
        block_x[1][13] = 16'sh0080;
        block_x[1][14] = 16'sh0080;
        block_x[1][15] = 16'sh0080;
        expected_mean_sq[1] = 32'd16384;
        expected_rms[1]     = 16'd128;
        expected_s_b[1]     = 16'sd320;
        expected_codes[1][0] = 4'd4;
        expected_codes[1][1] = 4'd4;
        expected_codes[1][2] = 4'd4;
        expected_codes[1][3] = 4'd4;
        expected_codes[1][4] = 4'd4;
        expected_codes[1][5] = 4'd4;
        expected_codes[1][6] = 4'd4;
        expected_codes[1][7] = 4'd4;
        expected_codes[1][8] = 4'd4;
        expected_codes[1][9] = 4'd4;
        expected_codes[1][10] = 4'd4;
        expected_codes[1][11] = 4'd4;
        expected_codes[1][12] = 4'd4;
        expected_codes[1][13] = 4'd4;
        expected_codes[1][14] = 4'd4;
        expected_codes[1][15] = 4'd4;
 
        // --- block 2: all_zero_degenerate_sb_zero ---
        block_alpha[2] = 16'sd640;
        block_x[2][0] = 16'sh0000;
        block_x[2][1] = 16'sh0000;
        block_x[2][2] = 16'sh0000;
        block_x[2][3] = 16'sh0000;
        block_x[2][4] = 16'sh0000;
        block_x[2][5] = 16'sh0000;
        block_x[2][6] = 16'sh0000;
        block_x[2][7] = 16'sh0000;
        block_x[2][8] = 16'sh0000;
        block_x[2][9] = 16'sh0000;
        block_x[2][10] = 16'sh0000;
        block_x[2][11] = 16'sh0000;
        block_x[2][12] = 16'sh0000;
        block_x[2][13] = 16'sh0000;
        block_x[2][14] = 16'sh0000;
        block_x[2][15] = 16'sh0000;
        expected_mean_sq[2] = 32'd0;
        expected_rms[2]     = 16'd0;
        expected_s_b[2]     = 16'sd0;
        expected_codes[2][0] = 4'd7;
        expected_codes[2][1] = 4'd7;
        expected_codes[2][2] = 4'd7;
        expected_codes[2][3] = 4'd7;
        expected_codes[2][4] = 4'd7;
        expected_codes[2][5] = 4'd7;
        expected_codes[2][6] = 4'd7;
        expected_codes[2][7] = 4'd7;
        expected_codes[2][8] = 4'd7;
        expected_codes[2][9] = 4'd7;
        expected_codes[2][10] = 4'd7;
        expected_codes[2][11] = 4'd7;
        expected_codes[2][12] = 4'd7;
        expected_codes[2][13] = 4'd7;
        expected_codes[2][14] = 4'd7;
        expected_codes[2][15] = 4'd7;
 
        // --- block 3: random_block_0 ---
        block_alpha[3] = 16'sd759;
        block_x[3][0] = 16'shffb7;
        block_x[3][1] = 16'sh005c;
        block_x[3][2] = 16'sh028c;
        block_x[3][3] = 16'shffcb;
        block_x[3][4] = 16'sh000c;
        block_x[3][5] = 16'sh0086;
        block_x[3][6] = 16'shfe1c;
        block_x[3][7] = 16'sh0012;
        block_x[3][8] = 16'sh00c7;
        block_x[3][9] = 16'sh01c2;
        block_x[3][10] = 16'shfd91;
        block_x[3][11] = 16'shfed2;
        block_x[3][12] = 16'shfd8b;
        block_x[3][13] = 16'sh01dc;
        block_x[3][14] = 16'sh0129;
        block_x[3][15] = 16'shfd40;
        expected_mean_sq[3] = 32'd163866;
        expected_rms[3]     = 16'd404;
        expected_s_b[3]     = 16'sd1197;
        expected_codes[3][0] = 4'd9;
        expected_codes[3][1] = 4'd1;
        expected_codes[3][2] = 4'd5;
        expected_codes[3][3] = 4'd9;
        expected_codes[3][4] = 4'd0;
        expected_codes[3][5] = 4'd1;
        expected_codes[3][6] = 4'd12;
        expected_codes[3][7] = 4'd0;
        expected_codes[3][8] = 4'd2;
        expected_codes[3][9] = 4'd4;
        expected_codes[3][10] = 4'd13;
        expected_codes[3][11] = 4'd11;
        expected_codes[3][12] = 4'd13;
        expected_codes[3][13] = 4'd4;
        expected_codes[3][14] = 4'd3;
        expected_codes[3][15] = 4'd13;
 
        // --- block 4: random_block_1 ---
        block_alpha[4] = 16'sd595;
        block_x[4][0] = 16'sh02ca;
        block_x[4][1] = 16'sh00ec;
        block_x[4][2] = 16'sh00b2;
        block_x[4][3] = 16'shfdf2;
        block_x[4][4] = 16'shfd17;
        block_x[4][5] = 16'sh002c;
        block_x[4][6] = 16'shfd5b;
        block_x[4][7] = 16'shfe24;
        block_x[4][8] = 16'shfe74;
        block_x[4][9] = 16'shfd2e;
        block_x[4][10] = 16'shffc9;
        block_x[4][11] = 16'shffa5;
        block_x[4][12] = 16'sh020e;
        block_x[4][13] = 16'sh001d;
        block_x[4][14] = 16'sh00d7;
        block_x[4][15] = 16'sh0000;
        expected_mean_sq[4] = 32'd195554;
        expected_rms[4]     = 16'd442;
        expected_s_b[4]     = 16'sd1027;
        expected_codes[4][0] = 4'd6;
        expected_codes[4][1] = 4'd3;
        expected_codes[4][2] = 4'd2;
        expected_codes[4][3] = 4'd13;
        expected_codes[4][4] = 4'd14;
        expected_codes[4][5] = 4'd1;
        expected_codes[4][6] = 4'd14;
        expected_codes[4][7] = 4'd13;
        expected_codes[4][8] = 4'd12;
        expected_codes[4][9] = 4'd14;
        expected_codes[4][10] = 4'd9;
        expected_codes[4][11] = 4'd9;
        expected_codes[4][12] = 4'd5;
        expected_codes[4][13] = 4'd0;
        expected_codes[4][14] = 4'd2;
        expected_codes[4][15] = 4'd0;
 
        // --- block 5: random_block_2 ---
        block_alpha[5] = 16'sd256;
        block_x[5][0] = 16'shffbe;
        block_x[5][1] = 16'shfeab;
        block_x[5][2] = 16'sh02fc;
        block_x[5][3] = 16'sh02f9;
        block_x[5][4] = 16'sh020b;
        block_x[5][5] = 16'sh013f;
        block_x[5][6] = 16'shfee4;
        block_x[5][7] = 16'shfe61;
        block_x[5][8] = 16'shfebc;
        block_x[5][9] = 16'shfd6c;
        block_x[5][10] = 16'sh0199;
        block_x[5][11] = 16'shff67;
        block_x[5][12] = 16'sh0214;
        block_x[5][13] = 16'shff52;
        block_x[5][14] = 16'sh02c0;
        block_x[5][15] = 16'sh0215;
        expected_mean_sq[5] = 32'd233493;
        expected_rms[5]     = 16'd483;
        expected_s_b[5]     = 16'sd483;
        expected_codes[5][0] = 4'd10;
        expected_codes[5][1] = 4'd14;
        expected_codes[5][2] = 4'd7;
        expected_codes[5][3] = 4'd7;
        expected_codes[5][4] = 4'd7;
        expected_codes[5][5] = 4'd6;
        expected_codes[5][6] = 4'd13;
        expected_codes[5][7] = 4'd15;
        expected_codes[5][8] = 4'd14;
        expected_codes[5][9] = 4'd15;
        expected_codes[5][10] = 4'd7;
        expected_codes[5][11] = 4'd11;
        expected_codes[5][12] = 4'd7;
        expected_codes[5][13] = 4'd12;
        expected_codes[5][14] = 4'd7;
        expected_codes[5][15] = 4'd7;
 
        // --- block 6: random_block_3 ---
        block_alpha[6] = 16'sd287;
        block_x[6][0] = 16'shfe42;
        block_x[6][1] = 16'sh0276;
        block_x[6][2] = 16'shffd2;
        block_x[6][3] = 16'sh02e2;
        block_x[6][4] = 16'shff62;
        block_x[6][5] = 16'shfd70;
        block_x[6][6] = 16'sh00c7;
        block_x[6][7] = 16'sh01ac;
        block_x[6][8] = 16'shfe9e;
        block_x[6][9] = 16'shfd86;
        block_x[6][10] = 16'shfeff;
        block_x[6][11] = 16'sh02c9;
        block_x[6][12] = 16'sh018c;
        block_x[6][13] = 16'shfdb5;
        block_x[6][14] = 16'shfe7a;
        block_x[6][15] = 16'shfd9b;
        expected_mean_sq[6] = 32'd246975;
        expected_rms[6]     = 16'd496;
        expected_s_b[6]     = 16'sd556;
        expected_codes[6][0] = 4'd14;
        expected_codes[6][1] = 4'd7;
        expected_codes[6][2] = 4'd9;
        expected_codes[6][3] = 4'd7;
        expected_codes[6][4] = 4'd11;
        expected_codes[6][5] = 4'd15;
        expected_codes[6][6] = 4'd4;
        expected_codes[6][7] = 4'd6;
        expected_codes[6][8] = 4'd14;
        expected_codes[6][9] = 4'd15;
        expected_codes[6][10] = 4'd13;
        expected_codes[6][11] = 4'd7;
        expected_codes[6][12] = 4'd6;
        expected_codes[6][13] = 4'd15;
        expected_codes[6][14] = 4'd14;
        expected_codes[6][15] = 4'd15;
 
        // --- block 7: random_block_4 ---
        block_alpha[7] = 16'sd364;
        block_x[7][0] = 16'sh01c8;
        block_x[7][1] = 16'shfe11;
        block_x[7][2] = 16'sh005b;
        block_x[7][3] = 16'shffaf;
        block_x[7][4] = 16'shfe25;
        block_x[7][5] = 16'sh0164;
        block_x[7][6] = 16'shfdc9;
        block_x[7][7] = 16'sh00dd;
        block_x[7][8] = 16'shfdb3;
        block_x[7][9] = 16'shff86;
        block_x[7][10] = 16'shfe47;
        block_x[7][11] = 16'shfe9e;
        block_x[7][12] = 16'sh02d3;
        block_x[7][13] = 16'sh01d2;
        block_x[7][14] = 16'shfed3;
        block_x[7][15] = 16'sh024f;
        expected_mean_sq[7] = 32'd190741;
        expected_rms[7]     = 16'd436;
        expected_s_b[7]     = 16'sd619;
        expected_codes[7][0] = 4'd6;
        expected_codes[7][1] = 4'd14;
        expected_codes[7][2] = 4'd2;
        expected_codes[7][3] = 4'd10;
        expected_codes[7][4] = 4'd14;
        expected_codes[7][5] = 4'd5;
        expected_codes[7][6] = 4'd15;
        expected_codes[7][7] = 4'd4;
        expected_codes[7][8] = 4'd15;
        expected_codes[7][9] = 4'd10;
        expected_codes[7][10] = 4'd14;
        expected_codes[7][11] = 4'd13;
        expected_codes[7][12] = 4'd7;
        expected_codes[7][13] = 4'd6;
        expected_codes[7][14] = 4'd13;
        expected_codes[7][15] = 4'd7;
 
        // --- block 8: random_block_5 ---
        block_alpha[8] = 16'sd488;
        block_x[8][0] = 16'shff5e;
        block_x[8][1] = 16'sh0220;
        block_x[8][2] = 16'sh00da;
        block_x[8][3] = 16'shfd9a;
        block_x[8][4] = 16'sh02f0;
        block_x[8][5] = 16'shfe48;
        block_x[8][6] = 16'shfe8d;
        block_x[8][7] = 16'sh01a3;
        block_x[8][8] = 16'shfef9;
        block_x[8][9] = 16'shfec7;
        block_x[8][10] = 16'shfd71;
        block_x[8][11] = 16'shfd8a;
        block_x[8][12] = 16'sh007f;
        block_x[8][13] = 16'shfe75;
        block_x[8][14] = 16'sh009c;
        block_x[8][15] = 16'shff3b;
        expected_mean_sq[8] = 32'd190460;
        expected_rms[8]     = 16'd436;
        expected_s_b[8]     = 16'sd831;
        expected_codes[8][0] = 4'd10;
        expected_codes[8][1] = 4'd6;
        expected_codes[8][2] = 4'd3;
        expected_codes[8][3] = 4'd14;
        expected_codes[8][4] = 4'd7;
        expected_codes[8][5] = 4'd13;
        expected_codes[8][6] = 4'd12;
        expected_codes[8][7] = 4'd5;
        expected_codes[8][8] = 4'd11;
        expected_codes[8][9] = 4'd12;
        expected_codes[8][10] = 4'd14;
        expected_codes[8][11] = 4'd14;
        expected_codes[8][12] = 4'd2;
        expected_codes[8][13] = 4'd13;
        expected_codes[8][14] = 4'd2;
        expected_codes[8][15] = 4'd11;
 
        // --- block 9: random_block_6 ---
        block_alpha[9] = 16'sd472;
        block_x[9][0] = 16'sh02c1;
        block_x[9][1] = 16'shffe7;
        block_x[9][2] = 16'sh0073;
        block_x[9][3] = 16'sh0233;
        block_x[9][4] = 16'shfe19;
        block_x[9][5] = 16'shfded;
        block_x[9][6] = 16'sh0273;
        block_x[9][7] = 16'sh01e8;
        block_x[9][8] = 16'shfe7f;
        block_x[9][9] = 16'shfe24;
        block_x[9][10] = 16'sh0170;
        block_x[9][11] = 16'sh02a4;
        block_x[9][12] = 16'shfe2e;
        block_x[9][13] = 16'sh02b3;
        block_x[9][14] = 16'sh024b;
        block_x[9][15] = 16'sh009f;
        expected_mean_sq[9] = 32'd250620;
        expected_rms[9]     = 16'd500;
        expected_s_b[9]     = 16'sd921;
        expected_codes[9][0] = 4'd6;
        expected_codes[9][1] = 4'd8;
        expected_codes[9][2] = 4'd1;
        expected_codes[9][3] = 4'd6;
        expected_codes[9][4] = 4'd13;
        expected_codes[9][5] = 4'd13;
        expected_codes[9][6] = 4'd6;
        expected_codes[9][7] = 4'd5;
        expected_codes[9][8] = 4'd12;
        expected_codes[9][9] = 4'd13;
        expected_codes[9][10] = 4'd4;
        expected_codes[9][11] = 4'd6;
        expected_codes[9][12] = 4'd13;
        expected_codes[9][13] = 4'd6;
        expected_codes[9][14] = 4'd6;
        expected_codes[9][15] = 4'd2;
 
        // --- block 10: random_block_7 ---
        block_alpha[10] = 16'sd399;
        block_x[10][0] = 16'shfd9f;
        block_x[10][1] = 16'shfd3b;
        block_x[10][2] = 16'sh02c7;
        block_x[10][3] = 16'shfe6e;
        block_x[10][4] = 16'sh013a;
        block_x[10][5] = 16'shfe8b;
        block_x[10][6] = 16'sh01f1;
        block_x[10][7] = 16'sh0094;
        block_x[10][8] = 16'shfec3;
        block_x[10][9] = 16'shfe0d;
        block_x[10][10] = 16'sh0152;
        block_x[10][11] = 16'shfd6a;
        block_x[10][12] = 16'shfe5f;
        block_x[10][13] = 16'sh005b;
        block_x[10][14] = 16'sh021d;
        block_x[10][15] = 16'sh00b0;
        expected_mean_sq[10] = 32'd215945;
        expected_rms[10]     = 16'd464;
        expected_s_b[10]     = 16'sd723;
        expected_codes[10][0] = 4'd14;
        expected_codes[10][1] = 4'd15;
        expected_codes[10][2] = 4'd7;
        expected_codes[10][3] = 4'd13;
        expected_codes[10][4] = 4'd4;
        expected_codes[10][5] = 4'd13;
        expected_codes[10][6] = 4'd6;
        expected_codes[10][7] = 4'd2;
        expected_codes[10][8] = 4'd12;
        expected_codes[10][9] = 4'd14;
        expected_codes[10][10] = 4'd5;
        expected_codes[10][11] = 4'd15;
        expected_codes[10][12] = 4'd13;
        expected_codes[10][13] = 4'd1;
        expected_codes[10][14] = 4'd6;
        expected_codes[10][15] = 4'd3;
 
    end
 
    initial begin
        rst_n = 0;
        elem_valid = 0;
        block_last = 0;
        alpha_q8_8 = 0;
        #12 rst_n = 1;
        #8;                    // t=20, a negedge -- matches this project's established convention
 
        for (v = 0; v < 11; v = v + 1) begin
            alpha_q8_8 = block_alpha[v];
            for (i = 0; i < 16; i = i + 1) begin
                x_in       = block_x[v][i];
                elem_valid = 1'b1;
                block_last = (i == 15);
                #10;
            end
            elem_valid = 1'b0;
            block_last = 1'b0;
 
            // Generous margin: pipeline latency from block_last to the last
            // emitted code is a fixed ~37 cycles (370ns); #500 comfortably covers it.
            #500;
 
            if (mean_sq_debug !== expected_mean_sq[v]) begin
                $display("FAIL: block %0d mean_sq_debug expected=%0d got=%0d", v, expected_mean_sq[v], mean_sq_debug);
                errors = errors + 1;
            end else begin
                $display("PASS: block %0d mean_sq_debug=%0d", v, mean_sq_debug);
            end
 
            if (rms_debug !== expected_rms[v]) begin
                $display("FAIL: block %0d rms_debug expected=%0d got=%0d", v, expected_rms[v], rms_debug);
                errors = errors + 1;
            end else begin
                $display("PASS: block %0d rms_debug=%0d", v, rms_debug);
            end
 
            if (s_b_debug !== expected_s_b[v]) begin
                $display("FAIL: block %0d s_b_debug expected=%0d got=%0d", v, expected_s_b[v], s_b_debug);
                errors = errors + 1;
            end else begin
                $display("PASS: block %0d s_b_debug=%0d", v, s_b_debug);
            end
 
            if (captured_s_a !== expected_s_b[v]) begin
                $display("FAIL: block %0d captured s_a_scale expected=%0d got=%0d", v, expected_s_b[v], captured_s_a);
                errors = errors + 1;
            end else begin
                $display("PASS: block %0d captured s_a_scale=%0d", v, captured_s_a);
            end
 
            for (i = 0; i < 16; i = i + 1) begin
                if (captured_codes[i] !== expected_codes[v][i]) begin
                    $display("FAIL: block %0d elem %0d code expected=%0d got=%0d", v, i, expected_codes[v][i], captured_codes[i]);
                    errors = errors + 1;
                end
            end
            $display("PASS: block %0d all 16 codes match expected", v);
        end
 
        if (errors == 0)
            $display("ALL CHECKS PASSED");
        else
            $display("%0d CHECK(S) FAILED", errors);
 
        $finish;
    end
 
endmodule