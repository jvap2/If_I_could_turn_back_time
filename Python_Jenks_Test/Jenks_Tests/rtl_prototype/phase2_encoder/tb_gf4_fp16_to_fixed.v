// -----------------------------------------------------------------------------
// tb_gf4_fp16_to_fixed.v
//
// Testbench for gf4_fp16_to_fixed.v. Purely combinational DUT, so this is a
// simple drive-and-check loop -- no clock needed on the DUT itself, though a
// clock is still used here to pace the stimulus loop (consistent with every
// other testbench in this project).
//
// 30 vectors: worked examples (+/-0, +/-1.0, 0.5, 100.0, a representative
// activation value, the largest normal FP16 (65504), min/max subnormal, min
// normal, +/-Inf, a NaN pattern, values that overflow the Q8.8 range in both
// directions), plus 12 random 16-bit patterns for broad coverage. Every
// expected value came from an independent from-scratch Python FP16 decoder
// cross-checked against this exact shift-and-clamp hardware model across 2,384
// bit patterns (see gf4_fp16_to_fixed.v's header) before any of this was
// encoded here.
//
// Run with:
//   iverilog -o sim gf4_fp16_to_fixed.v tb_gf4_fp16_to_fixed.v && vvp sim
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
 
module tb_gf4_fp16_to_fixed;
 
    reg clk = 0;
    always #5 clk = ~clk;
 
    reg [15:0] fp16_in = 0;
    wire signed [15:0] fixed_out;
    wire is_zero;
    wire is_special;
 
    integer errors = 0;
    integer v;
 
    gf4_fp16_to_fixed dut (
        .fp16_in    (fp16_in),
        .fixed_out  (fixed_out),
        .is_zero    (is_zero),
        .is_special (is_special)
    );
 
    reg [15:0] test_bits        [0:29];
    reg signed [15:0] expected_raw [0:29];
    reg expected_zero            [0:29];
    reg expected_special         [0:29];
 
    initial begin
        test_bits[0]      = 16'h0000; // plus_zero
        expected_raw[0]   = 16'sd0;
        expected_zero[0]  = 1'b1;
        expected_special[0] = 1'b0;
        test_bits[1]      = 16'h8000; // minus_zero
        expected_raw[1]   = 16'sd0;
        expected_zero[1]  = 1'b1;
        expected_special[1] = 1'b0;
        test_bits[2]      = 16'h3c00; // plus_one
        expected_raw[2]   = 16'sd256;
        expected_zero[2]  = 1'b0;
        expected_special[2] = 1'b0;
        test_bits[3]      = 16'hbc00; // minus_one
        expected_raw[3]   = -16'sd256;
        expected_zero[3]  = 1'b0;
        expected_special[3] = 1'b0;
        test_bits[4]      = 16'h3800; // plus_half
        expected_raw[4]   = 16'sd128;
        expected_zero[4]  = 1'b0;
        expected_special[4] = 1'b0;
        test_bits[5]      = 16'h5640; // plus_100
        expected_raw[5]   = 16'sd25600;
        expected_zero[5]  = 1'b0;
        expected_special[5] = 1'b0;
        test_bits[6]      = 16'hca66; // minus_12p8ish
        expected_raw[6]   = -16'sd3276;
        expected_zero[6]  = 1'b0;
        expected_special[6] = 1'b0;
        test_bits[7]      = 16'h7bff; // max_normal_65504
        expected_raw[7]   = 16'sd32767;
        expected_zero[7]  = 1'b0;
        expected_special[7] = 1'b0;
        test_bits[8]      = 16'h0001; // min_subnormal
        expected_raw[8]   = 16'sd0;
        expected_zero[8]  = 1'b1;
        expected_special[8] = 1'b0;
        test_bits[9]      = 16'h03ff; // max_subnormal
        expected_raw[9]   = 16'sd0;
        expected_zero[9]  = 1'b1;
        expected_special[9] = 1'b0;
        test_bits[10]      = 16'h0400; // min_normal
        expected_raw[10]   = 16'sd0;
        expected_zero[10]  = 1'b0;
        expected_special[10] = 1'b0;
        test_bits[11]      = 16'h7c00; // plus_inf
        expected_raw[11]   = 16'sd32767;
        expected_zero[11]  = 1'b0;
        expected_special[11] = 1'b1;
        test_bits[12]      = 16'hfc00; // minus_inf
        expected_raw[12]   = -16'sd32767;
        expected_zero[12]  = 1'b0;
        expected_special[12] = 1'b1;
        test_bits[13]      = 16'h7e00; // a_nan_pattern
        expected_raw[13]   = 16'sd32767;
        expected_zero[13]  = 1'b0;
        expected_special[13] = 1'b1;
        test_bits[14]      = 16'h40c0; // plausible_activation_2p375
        expected_raw[14]   = 16'sd608;
        expected_zero[14]  = 1'b0;
        expected_special[14] = 1'b0;
        test_bits[15]      = 16'hb900; // plausible_activation_neg0p625
        expected_raw[15]   = -16'sd160;
        expected_zero[15]  = 1'b0;
        expected_special[15] = 1'b0;
        test_bits[16]      = 16'h5a40; // value_that_overflows_q8_8_range
        expected_raw[16]   = 16'sd32767;
        expected_zero[16]  = 1'b0;
        expected_special[16] = 1'b0;
        test_bits[17]      = 16'hdfd0; // value_that_overflows_negative
        expected_raw[17]   = -16'sd32767;
        expected_zero[17]  = 1'b0;
        expected_special[17] = 1'b0;
        test_bits[18]      = 16'h946f; // random_bits_0
        expected_raw[18]   = 16'sd0;
        expected_zero[18]  = 1'b0;
        expected_special[18] = 1'b0;
        test_bits[19]      = 16'h2ac7; // random_bits_1
        expected_raw[19]   = 16'sd13;
        expected_zero[19]  = 1'b0;
        expected_special[19] = 1'b0;
        test_bits[20]      = 16'h08be; // random_bits_2
        expected_raw[20]   = 16'sd0;
        expected_zero[20]  = 1'b0;
        expected_special[20] = 1'b0;
        test_bits[21]      = 16'h9d0d; // random_bits_3
        expected_raw[21]   = -16'sd1;
        expected_zero[21]  = 1'b0;
        expected_special[21] = 1'b0;
        test_bits[22]      = 16'hd8f5; // random_bits_4
        expected_raw[22]   = -16'sd32767;
        expected_zero[22]  = 1'b0;
        expected_special[22] = 1'b0;
        test_bits[23]      = 16'hc224; // random_bits_5
        expected_raw[23]   = -16'sd786;
        expected_zero[23]  = 1'b0;
        expected_special[23] = 1'b0;
        test_bits[24]      = 16'hb756; // random_bits_6
        expected_raw[24]   = -16'sd117;
        expected_zero[24]  = 1'b0;
        expected_special[24] = 1'b0;
        test_bits[25]      = 16'h42b7; // random_bits_7
        expected_raw[25]   = 16'sd859;
        expected_zero[25]  = 1'b0;
        expected_special[25] = 1'b0;
        test_bits[26]      = 16'h624d; // random_bits_8
        expected_raw[26]   = 16'sd32767;
        expected_zero[26]  = 1'b0;
        expected_special[26] = 1'b0;
        test_bits[27]      = 16'h88e6; // random_bits_9
        expected_raw[27]   = 16'sd0;
        expected_zero[27]  = 1'b0;
        expected_special[27] = 1'b0;
        test_bits[28]      = 16'he39f; // random_bits_10
        expected_raw[28]   = -16'sd32767;
        expected_zero[28]  = 1'b0;
        expected_special[28] = 1'b0;
        test_bits[29]      = 16'h0734; // random_bits_11
        expected_raw[29]   = 16'sd0;
        expected_zero[29]  = 1'b0;
        expected_special[29] = 1'b0;
    end
 
    initial begin
        #12;
        for (v = 0; v < 30; v = v + 1) begin
            fp16_in = test_bits[v];
            #10; // purely combinational DUT -- one full clock period is generous settle time
 
            if (fixed_out !== expected_raw[v]) begin
                $display("FAIL: vec=%0d bits=%04h expected_raw=%0d got=%0d", v, test_bits[v], expected_raw[v], fixed_out);
                errors = errors + 1;
            end else if (is_zero !== expected_zero[v]) begin
                $display("FAIL: vec=%0d bits=%04h is_zero expected=%0d got=%0d", v, test_bits[v], expected_zero[v], is_zero);
                errors = errors + 1;
            end else if (is_special !== expected_special[v]) begin
                $display("FAIL: vec=%0d bits=%04h is_special expected=%0d got=%0d", v, test_bits[v], expected_special[v], is_special);
                errors = errors + 1;
            end else begin
                $display("PASS: vec=%0d bits=%04h -> raw=%0d", v, test_bits[v], fixed_out);
            end
        end
 
        if (errors == 0)
            $display("ALL CHECKS PASSED");
        else
            $display("%0d CHECK(S) FAILED", errors);
 
        $finish;
    end
 
endmodule