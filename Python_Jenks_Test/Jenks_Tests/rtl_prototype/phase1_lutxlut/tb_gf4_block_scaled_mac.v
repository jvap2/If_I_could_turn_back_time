// -----------------------------------------------------------------------------
// tb_gf4_block_scaled_mac.v
//
// Behavioral testbench for gf4_block_scaled_mac.v. Not synthesizable -- run
// with a real event simulator, e.g.:
//
//   iverilog -o sim gf4_joint_product_table.v gf4_block_scaled_mac.v tb_gf4_block_scaled_mac.v
//   vvp sim
//
// Timing note: inputs are changed on delay-based waits (#10, aligned to
// negedges) rather than `@(posedge clk)` event control -- purely so this
// file also parses cleanly through Yosys's frontend for the elaboration
// smoke test (see NOTES_rtl_synthesis.md for why `@(posedge clk)` as a bare
// statement isn't accepted there). Functionally this is a plain 10ns-period
// clock (`always #5 clk = ~clk`, posedges at 15, 25, 35, ...); each loop
// iteration sets inputs at a negedge, giving a full half-period of setup
// margin before the following posedge samples them.
//
// Drives two 16-element blocks (different weight/activation code patterns
// and different per-block scales) through the DUT and checks:
//   1) global_acc after block A's block_done_pulse matches the Python
//      reference (raw block sum, combined scale, and the >>24 rescale all
//      cross-checked offline before being encoded here).
//   2) global_acc after block B (added on top of block A's contribution,
//      i.e. the running-total-across-blocks behavior) also matches.
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
 
module tb_gf4_block_scaled_mac;
 
    reg clk = 0;
    reg rst_n = 0;
 
    reg        cfg_we = 0;
    reg [5:0]  cfg_waddr = 0;
    reg [16:0] cfg_wdata = 0;
 
    reg [3:0]  w_code = 0;
    reg [3:0]  a_code = 0;
    reg        elem_valid = 0;
    reg        block_last = 0;
 
    reg signed [15:0] s_w_scale = 0;
    reg signed [15:0] s_a_scale = 0;
 
    reg  global_acc_clear = 0;
    wire signed [31:0] global_acc;
    wire block_done_pulse;
 
    integer k;
    integer errors = 0;
 
    reg [3:0] blkA_w [0:15];
    reg [3:0] blkA_a [0:15];
    reg [3:0] blkB_w [0:15];
    reg [3:0] blkB_a [0:15];
 
    initial begin
        blkA_w[0] = 4'd0; blkA_a[0] = 4'd0;
        blkA_w[1] = 4'd11; blkA_a[1] = 4'd5;
        blkA_w[2] = 4'd6; blkA_a[2] = 4'd10;
        blkA_w[3] = 4'd1; blkA_a[3] = 4'd7;
        blkA_w[4] = 4'd12; blkA_a[4] = 4'd4;
        blkA_w[5] = 4'd7; blkA_a[5] = 4'd9;
        blkA_w[6] = 4'd2; blkA_a[6] = 4'd6;
        blkA_w[7] = 4'd13; blkA_a[7] = 4'd3;
        blkA_w[8] = 4'd0; blkA_a[8] = 4'd8;
        blkA_w[9] = 4'd3; blkA_a[9] = 4'd5;
        blkA_w[10] = 4'd14; blkA_a[10] = 4'd2;
        blkA_w[11] = 4'd1; blkA_a[11] = 4'd15;
        blkA_w[12] = 4'd4; blkA_a[12] = 4'd4;
        blkA_w[13] = 4'd15; blkA_a[13] = 4'd1;
        blkA_w[14] = 4'd2; blkA_a[14] = 4'd14;
        blkA_w[15] = 4'd5; blkA_a[15] = 4'd3;
        blkB_w[0] = 4'd7; blkB_a[0] = 4'd8;
        blkB_w[1] = 4'd14; blkB_a[1] = 4'd3;
        blkB_w[2] = 4'd5; blkB_a[2] = 4'd14;
        blkB_w[3] = 4'd12; blkB_a[3] = 4'd1;
        blkB_w[4] = 4'd3; blkB_a[4] = 4'd12;
        blkB_w[5] = 4'd10; blkB_a[5] = 4'd7;
        blkB_w[6] = 4'd1; blkB_a[6] = 4'd10;
        blkB_w[7] = 4'd8; blkB_a[7] = 4'd5;
        blkB_w[8] = 4'd7; blkB_a[8] = 4'd8;
        blkB_w[9] = 4'd14; blkB_a[9] = 4'd3;
        blkB_w[10] = 4'd5; blkB_a[10] = 4'd14;
        blkB_w[11] = 4'd12; blkB_a[11] = 4'd1;
        blkB_w[12] = 4'd3; blkB_a[12] = 4'd12;
        blkB_w[13] = 4'd10; blkB_a[13] = 4'd7;
        blkB_w[14] = 4'd1; blkB_a[14] = 4'd10;
        blkB_w[15] = 4'd8; blkB_a[15] = 4'd5;
    end
 
    gf4_block_scaled_mac #(
        .W_ENTRIES(8), .A_ENTRIES(8), .PROD_WIDTH(17),
        .RAW_ACC_WIDTH(24), .SCALE_WIDTH(16), .ACC_WIDTH(32)
    ) dut (
        .clk             (clk),
        .rst_n           (rst_n),
        .cfg_we          (cfg_we),
        .cfg_waddr       (cfg_waddr),
        .cfg_wdata       (cfg_wdata),
        .w_code          (w_code),
        .a_code          (a_code),
        .elem_valid      (elem_valid),
        .block_last      (block_last),
        .s_w_scale       (s_w_scale),
        .s_a_scale       (s_a_scale),
        .global_acc_clear(global_acc_clear),
        .global_acc      (global_acc),
        .block_done_pulse(block_done_pulse)
    );
 
    always #5 clk = ~clk;
 
    initial begin
        rst_n = 0;
        global_acc_clear = 1;
        elem_valid = 0;
        block_last = 0;
        #12 rst_n = 1;
        #8;                    // t=20, a negedge
        global_acc_clear = 0;
 
        // --- Block A ---
        s_w_scale = 16'sd320;
        s_a_scale = 16'sd320;
        for (k = 0; k < 16; k = k + 1) begin
            w_code = blkA_w[k];
            a_code = blkA_a[k];
            elem_valid = 1'b1;
            block_last = (k == 15);
            #10;                // hold stable for one full period
        end
        elem_valid = 1'b0;
        block_last = 1'b0;
        #20;                    // margin past block_done_pulse's cycle
 
        if (global_acc !== -32'sd161) begin
            $display("FAIL: global_acc after block A expected=%0d got=%0d", -32'sd161, $signed(global_acc));
            errors = errors + 1;
        end else begin
            $display("PASS: global_acc after block A = %0d", $signed(global_acc));
        end
 
        // --- Block B (accumulates on top of block A) ---
        s_w_scale = 16'sd256;
        s_a_scale = 16'sd178;
        for (k = 0; k < 16; k = k + 1) begin
            w_code = blkB_w[k];
            a_code = blkB_a[k];
            elem_valid = 1'b1;
            block_last = (k == 15);
            #10;
        end
        elem_valid = 1'b0;
        block_last = 1'b0;
        #20;
 
        if (global_acc !== -32'sd479) begin
            $display("FAIL: global_acc after block B expected=%0d got=%0d", -32'sd479, $signed(global_acc));
            errors = errors + 1;
        end else begin
            $display("PASS: global_acc after block B = %0d", $signed(global_acc));
        end
 
        if (errors == 0)
            $display("ALL CHECKS PASSED");
        else
            $display("%0d CHECK(S) FAILED", errors);
 
        $finish;
    end
 
endmodule