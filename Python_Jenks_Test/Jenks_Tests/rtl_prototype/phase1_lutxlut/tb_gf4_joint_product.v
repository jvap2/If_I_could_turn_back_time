// -----------------------------------------------------------------------------
// tb_gf4_joint_product.v
//
// Behavioral testbench for gf4_joint_product_table.v. Not synthesizable --
// run with a real event simulator, e.g.:
//
//   iverilog -o sim gf4_joint_product_table.v tb_gf4_joint_product.v
//   vvp sim
//
// Checks:
//   1) Exhaustive readback of all 64 table entries against the outer
//      product of the GF4 levels L_int = {0,20,45,72,101,134,178,256}
//      (Q0.8 each factor, Q0.16 product), computed offline in Python and
//      cross-checked byte-for-byte against gf4_joint_product_table.v's
//      default table contents before being encoded here as this testbench's
//      independent reference (see NOTES_rtl_synthesis.md).
//   2) Sign combination: prod_sign must equal w_sign XOR a_sign, checked
//      across all four sign combinations for one representative entry.
//   3) Config write port overwrites the default table.
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
 
module tb_gf4_joint_product;
 
    reg        clk = 0;
    reg        rst_n = 0;
    reg        cfg_we = 0;
    reg [5:0]  cfg_waddr = 0;
    reg [16:0] cfg_wdata = 0;
 
    reg [2:0]  w_idx = 0;
    reg        w_sign = 0;
    reg [2:0]  a_idx = 0;
    reg        a_sign = 0;
 
    wire        prod_sign;
    wire [16:0] prod_mag;
 
    integer i, w, a;
    integer errors = 0;
 
    reg [16:0] expected_prod [0:63];
 
    initial begin
        expected_prod[6'd0] = 17'd0;
        expected_prod[6'd1] = 17'd0;
        expected_prod[6'd2] = 17'd0;
        expected_prod[6'd3] = 17'd0;
        expected_prod[6'd4] = 17'd0;
        expected_prod[6'd5] = 17'd0;
        expected_prod[6'd6] = 17'd0;
        expected_prod[6'd7] = 17'd0;
        expected_prod[6'd8] = 17'd0;
        expected_prod[6'd9] = 17'd400;
        expected_prod[6'd10] = 17'd900;
        expected_prod[6'd11] = 17'd1440;
        expected_prod[6'd12] = 17'd2020;
        expected_prod[6'd13] = 17'd2680;
        expected_prod[6'd14] = 17'd3560;
        expected_prod[6'd15] = 17'd5120;
        expected_prod[6'd16] = 17'd0;
        expected_prod[6'd17] = 17'd900;
        expected_prod[6'd18] = 17'd2025;
        expected_prod[6'd19] = 17'd3240;
        expected_prod[6'd20] = 17'd4545;
        expected_prod[6'd21] = 17'd6030;
        expected_prod[6'd22] = 17'd8010;
        expected_prod[6'd23] = 17'd11520;
        expected_prod[6'd24] = 17'd0;
        expected_prod[6'd25] = 17'd1440;
        expected_prod[6'd26] = 17'd3240;
        expected_prod[6'd27] = 17'd5184;
        expected_prod[6'd28] = 17'd7272;
        expected_prod[6'd29] = 17'd9648;
        expected_prod[6'd30] = 17'd12816;
        expected_prod[6'd31] = 17'd18432;
        expected_prod[6'd32] = 17'd0;
        expected_prod[6'd33] = 17'd2020;
        expected_prod[6'd34] = 17'd4545;
        expected_prod[6'd35] = 17'd7272;
        expected_prod[6'd36] = 17'd10201;
        expected_prod[6'd37] = 17'd13534;
        expected_prod[6'd38] = 17'd17978;
        expected_prod[6'd39] = 17'd25856;
        expected_prod[6'd40] = 17'd0;
        expected_prod[6'd41] = 17'd2680;
        expected_prod[6'd42] = 17'd6030;
        expected_prod[6'd43] = 17'd9648;
        expected_prod[6'd44] = 17'd13534;
        expected_prod[6'd45] = 17'd17956;
        expected_prod[6'd46] = 17'd23852;
        expected_prod[6'd47] = 17'd34304;
        expected_prod[6'd48] = 17'd0;
        expected_prod[6'd49] = 17'd3560;
        expected_prod[6'd50] = 17'd8010;
        expected_prod[6'd51] = 17'd12816;
        expected_prod[6'd52] = 17'd17978;
        expected_prod[6'd53] = 17'd23852;
        expected_prod[6'd54] = 17'd31684;
        expected_prod[6'd55] = 17'd45568;
        expected_prod[6'd56] = 17'd0;
        expected_prod[6'd57] = 17'd5120;
        expected_prod[6'd58] = 17'd11520;
        expected_prod[6'd59] = 17'd18432;
        expected_prod[6'd60] = 17'd25856;
        expected_prod[6'd61] = 17'd34304;
        expected_prod[6'd62] = 17'd45568;
        expected_prod[6'd63] = 17'd65536;
    end
 
    gf4_joint_product_table #(
        .W_ENTRIES(8),
        .A_ENTRIES(8),
        .OUT_WIDTH(17)
    ) dut (
        .clk      (clk),
        .rst_n    (rst_n),
        .cfg_we   (cfg_we),
        .cfg_waddr(cfg_waddr),
        .cfg_wdata(cfg_wdata),
        .w_idx    (w_idx),
        .w_sign   (w_sign),
        .a_idx    (a_idx),
        .a_sign   (a_sign),
        .prod_sign(prod_sign),
        .prod_mag (prod_mag)
    );
 
    always #5 clk = ~clk;
 
    initial begin
        rst_n = 0;
        #12 rst_n = 1;
 
        // --- Check 1: exhaustive readback of all 64 entries ---
        for (w = 0; w < 8; w = w + 1) begin
            for (a = 0; a < 8; a = a + 1) begin
                w_idx = w[2:0];
                a_idx = a[2:0];
                w_sign = 1'b0;
                a_sign = 1'b0;
                #1;
                i = w*8 + a;
                if (prod_mag !== expected_prod[i]) begin
                    $display("FAIL: w_idx=%0d a_idx=%0d expected=%0d got=%0d", w, a, expected_prod[i], prod_mag);
                    errors = errors + 1;
                end else begin
                    $display("PASS: w_idx=%0d a_idx=%0d prod_mag=%0d", w, a, prod_mag);
                end
            end
        end
 
        // --- Check 2: sign combination (representative entry w_idx=5, a_idx=5) ---
        w_idx = 3'd5;
        a_idx = 3'd5;
        w_sign = 1'b0; a_sign = 1'b0; #1;
        if (prod_sign !== 1'b0) begin $display("FAIL: sign(0,0) expected=0 got=%0d", prod_sign); errors = errors + 1; end
        w_sign = 1'b0; a_sign = 1'b1; #1;
        if (prod_sign !== 1'b1) begin $display("FAIL: sign(0,1) expected=1 got=%0d", prod_sign); errors = errors + 1; end
        w_sign = 1'b1; a_sign = 1'b0; #1;
        if (prod_sign !== 1'b1) begin $display("FAIL: sign(1,0) expected=1 got=%0d", prod_sign); errors = errors + 1; end
        w_sign = 1'b1; a_sign = 1'b1; #1;
        if (prod_sign !== 1'b0) begin $display("FAIL: sign(1,1) expected=0 got=%0d", prod_sign); errors = errors + 1; end
        if (errors == 0)
            $display("PASS: sign XOR combination correct across all 4 cases");
 
        // --- Check 3: config write port overwrites the default table ---
        cfg_we    = 1'b1;
        cfg_waddr = 6'd0;   // (w_idx=0, a_idx=0), normally 0
        cfg_wdata = 17'd99999 % 131072;  // arbitrary nonzero test value within 17-bit range
        #10;
        cfg_we    = 1'b0;
        w_idx = 3'd0; a_idx = 3'd0;
        #1;
        if (prod_mag !== (17'd99999 % 131072)) begin
            $display("FAIL: cfg write to addr=0 expected=%0d got=%0d", (17'd99999 % 131072), prod_mag);
            errors = errors + 1;
        end else begin
            $display("PASS: cfg write readback addr=0 -> %0d", prod_mag);
        end
 
        if (errors == 0)
            $display("ALL CHECKS PASSED");
        else
            $display("%0d CHECK(S) FAILED", errors);
 
        $finish;
    end
 
endmodule