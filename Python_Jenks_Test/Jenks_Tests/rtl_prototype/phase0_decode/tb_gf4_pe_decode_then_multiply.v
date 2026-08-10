// -----------------------------------------------------------------------------
// tb_gf4_pe_decode_then_multiply.v
//
// Testbench for gf4_pe_decode_then_multiply.v. Deliberately reuses
// tb_gf4_joint_product.v's own already-verified 64-entry expected-product
// table (the outer product of the GF4 levels L_int =
// {0,20,45,72,101,134,178,256} with themselves, Q0.8 x Q0.8 -> Q0.16) as
// this file's reference, rather than deriving a second, independent set of
// expected values -- gf4_pe_decode_then_multiply computes the exact same
// mathematical quantity as gf4_joint_product_table.v (decode two codes,
// multiply the magnitudes, XOR the signs), just via decode-then-multiply
// instead of a single table lookup, so the two implementations should be
// bit-identical for every one of the 64 (w_idx, a_idx) combinations.
//
// Checks:
//   1) All 64 (w_idx, a_idx) combinations, sign=0 on both operands: clear
//      the accumulator, apply one element, check mac_acc equals
//      expected_prod[w_idx*8 + a_idx] exactly.
//   2) Sign combination on one representative entry (w_idx=5, a_idx=5,
//      expected magnitude 17956, matching tb_gf4_joint_product.v's own
//      check on the same entry): all four sign combinations, confirming
//      the accumulated result's sign follows w_sign XOR a_sign.
//
// Run with:
//   iverilog -o sim gf4_decode_fixed.v gf4_decode_programmable.v \
//     gf4_pe_decode_then_multiply.v tb_gf4_pe_decode_then_multiply.v
//   vvp sim
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module tb_gf4_pe_decode_then_multiply;

    reg clk = 0;
    reg rst_n = 0;
    always #5 clk = ~clk;

    reg        cfg_we = 0;
    reg [2:0]  cfg_waddr = 0;
    reg [8:0]  cfg_wdata = 0;

    reg [3:0]  w_code = 0;
    reg [3:0]  a_code = 0;
    reg        valid = 0;
    reg        acc_clear = 0;

    wire signed [31:0] mac_acc;

    integer errors = 0;
    integer w, a, i;

    // --- same 64-entry reference table as tb_gf4_joint_product.v ---
    reg [16:0] expected_prod [0:63];

    initial begin
        expected_prod[6'd0]  = 17'd0;    expected_prod[6'd1]  = 17'd0;
        expected_prod[6'd2]  = 17'd0;    expected_prod[6'd3]  = 17'd0;
        expected_prod[6'd4]  = 17'd0;    expected_prod[6'd5]  = 17'd0;
        expected_prod[6'd6]  = 17'd0;    expected_prod[6'd7]  = 17'd0;
        expected_prod[6'd8]  = 17'd0;    expected_prod[6'd9]  = 17'd400;
        expected_prod[6'd10] = 17'd900;  expected_prod[6'd11] = 17'd1440;
        expected_prod[6'd12] = 17'd2020; expected_prod[6'd13] = 17'd2680;
        expected_prod[6'd14] = 17'd3560; expected_prod[6'd15] = 17'd5120;
        expected_prod[6'd16] = 17'd0;    expected_prod[6'd17] = 17'd900;
        expected_prod[6'd18] = 17'd2025; expected_prod[6'd19] = 17'd3240;
        expected_prod[6'd20] = 17'd4545; expected_prod[6'd21] = 17'd6030;
        expected_prod[6'd22] = 17'd8010; expected_prod[6'd23] = 17'd11520;
        expected_prod[6'd24] = 17'd0;    expected_prod[6'd25] = 17'd1440;
        expected_prod[6'd26] = 17'd3240; expected_prod[6'd27] = 17'd5184;
        expected_prod[6'd28] = 17'd7272; expected_prod[6'd29] = 17'd9648;
        expected_prod[6'd30] = 17'd12816; expected_prod[6'd31] = 17'd18432;
        expected_prod[6'd32] = 17'd0;    expected_prod[6'd33] = 17'd2020;
        expected_prod[6'd34] = 17'd4545; expected_prod[6'd35] = 17'd7272;
        expected_prod[6'd36] = 17'd10201; expected_prod[6'd37] = 17'd13534;
        expected_prod[6'd38] = 17'd17978; expected_prod[6'd39] = 17'd25856;
        expected_prod[6'd40] = 17'd0;    expected_prod[6'd41] = 17'd2680;
        expected_prod[6'd42] = 17'd6030; expected_prod[6'd43] = 17'd9648;
        expected_prod[6'd44] = 17'd13534; expected_prod[6'd45] = 17'd17956;
        expected_prod[6'd46] = 17'd23852; expected_prod[6'd47] = 17'd34304;
        expected_prod[6'd48] = 17'd0;    expected_prod[6'd49] = 17'd3560;
        expected_prod[6'd50] = 17'd8010; expected_prod[6'd51] = 17'd12816;
        expected_prod[6'd52] = 17'd17978; expected_prod[6'd53] = 17'd23852;
        expected_prod[6'd54] = 17'd31684; expected_prod[6'd55] = 17'd45568;
        expected_prod[6'd56] = 17'd0;    expected_prod[6'd57] = 17'd5120;
        expected_prod[6'd58] = 17'd11520; expected_prod[6'd59] = 17'd18432;
        expected_prod[6'd60] = 17'd25856; expected_prod[6'd61] = 17'd34304;
        expected_prod[6'd62] = 17'd45568; expected_prod[6'd63] = 17'd65536;
    end

    gf4_pe_decode_then_multiply #(
        .PROGRAMMABLE(1), .MAG_WIDTH(9), .ACC_WIDTH(32)
    ) dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .cfg_we    (cfg_we),
        .cfg_waddr (cfg_waddr),
        .cfg_wdata (cfg_wdata),
        .w_code    (w_code),
        .a_code    (a_code),
        .valid     (valid),
        .acc_clear (acc_clear),
        .mac_acc   (mac_acc)
    );

    initial begin
        rst_n = 0;
        valid = 0;
        acc_clear = 0;
        #12 rst_n = 1;
        #8; // land on a negedge, matching this project's convention

        // --- Check 1: exhaustive 64-combination readback, sign=0 both sides ---
        for (w = 0; w < 8; w = w + 1) begin
            for (a = 0; a < 8; a = a + 1) begin
                acc_clear = 1'b1;
                w_code = {1'b0, w[2:0]};
                a_code = {1'b0, a[2:0]};
                #10;
                acc_clear = 1'b0;
                valid = 1'b1;
                #10;
                valid = 1'b0;
                #1;
                i = w*8 + a;
                if (mac_acc !== $signed({1'b0, expected_prod[i]})) begin
                    $display("FAIL: w_idx=%0d a_idx=%0d expected=%0d got=%0d", w, a, expected_prod[i], $signed(mac_acc));
                    errors = errors + 1;
                end else begin
                    $display("PASS: w_idx=%0d a_idx=%0d mac_acc=%0d", w, a, $signed(mac_acc));
                end
            end
        end

        // --- Check 2: sign combination on w_idx=5, a_idx=5 (expected magnitude 17956) ---
        acc_clear = 1'b1;
        w_code = {1'b0, 3'd5}; a_code = {1'b0, 3'd5};
        #10; acc_clear = 1'b0; valid = 1'b1; #10; valid = 1'b0; #1;
        if (mac_acc !== 32'sd17956) begin
            $display("FAIL: sign(0,0) expected=17956 got=%0d", $signed(mac_acc));
            errors = errors + 1;
        end else begin
            $display("PASS: sign(0,0) mac_acc=17956");
        end

        acc_clear = 1'b1;
        w_code = {1'b0, 3'd5}; a_code = {1'b1, 3'd5};
        #10; acc_clear = 1'b0; valid = 1'b1; #10; valid = 1'b0; #1;
        if (mac_acc !== -32'sd17956) begin
            $display("FAIL: sign(0,1) expected=-17956 got=%0d", $signed(mac_acc));
            errors = errors + 1;
        end else begin
            $display("PASS: sign(0,1) mac_acc=-17956");
        end

        acc_clear = 1'b1;
        w_code = {1'b1, 3'd5}; a_code = {1'b0, 3'd5};
        #10; acc_clear = 1'b0; valid = 1'b1; #10; valid = 1'b0; #1;
        if (mac_acc !== -32'sd17956) begin
            $display("FAIL: sign(1,0) expected=-17956 got=%0d", $signed(mac_acc));
            errors = errors + 1;
        end else begin
            $display("PASS: sign(1,0) mac_acc=-17956");
        end

        acc_clear = 1'b1;
        w_code = {1'b1, 3'd5}; a_code = {1'b1, 3'd5};
        #10; acc_clear = 1'b0; valid = 1'b1; #10; valid = 1'b0; #1;
        if (mac_acc !== 32'sd17956) begin
            $display("FAIL: sign(1,1) expected=17956 got=%0d", $signed(mac_acc));
            errors = errors + 1;
        end else begin
            $display("PASS: sign(1,1) mac_acc=17956");
        end

        if (errors == 0)
            $display("ALL CHECKS PASSED");
        else
            $display("%0d CHECK(S) FAILED", errors);

        $finish;
    end

endmodule