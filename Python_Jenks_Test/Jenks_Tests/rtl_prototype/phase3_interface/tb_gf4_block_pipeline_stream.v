// -----------------------------------------------------------------------------
// tb_gf4_block_pipeline_stream.v
//
// Testbench for gf4_block_pipeline_stream.v -- the Phase 3 valid/ready wrapper.
// The underlying arithmetic (mean_sq/RMS/s_b, joint-table lookup, block-scaled
// accumulation) was already exhaustively verified through gf4_block_pipeline.v;
// this testbench's job is specifically to prove the NEW control-plane behavior
// -- the valid/ready handshake and its backpressure -- actually works, using the
// same already-Python-and-Icarus-verified block data (block A: the paper's own
// xi=-0.7 case, expected global_acc/contribution=81; block 2: the all-zero
// degenerate case, expected 0) so any test failure here points at the NEW
// handshake logic, not at arithmetic that's already been proven correct.
//
// Three scenarios:
//   1) Baseline: block A driven with in_valid/out_ready held high the whole
//      time (no stalls) -- confirms the wrapper is functionally equivalent to
//      gf4_block_pipeline.v when nothing stalls. Since out_ready is already
//      high, out_valid pulses for exactly one cycle and is immediately
//      consumed (correct handshake behavior, not a bug) -- so this scenario
//      checks a small monitor's CAPTURED transfer, not the live out_valid
//      signal at some later fixed time (which would legitimately have
//      already dropped back to 0 by then).
//   2) Producer stall: block A again, but in_valid is deliberately deasserted
//      for 4 cycles between elements 7 and 8 of the fill phase -- confirms the
//      module correctly waits (no transfer happens while in_valid is low,
//      regardless of in_ready) and the final result is unchanged (still 81).
//      Also checked via the captured-transfer monitor, same reasoning as (1).
//   3) Consumer stall: block 2, with out_ready deliberately held low for a
//      long fixed window after the block completes -- confirms out_valid is
//      HELD (not just pulsed) throughout that window, and that in_ready
//      correctly stays low the whole time too (a new block cannot start until
//      the pending result is consumed) -- genuine end-to-end backpressure, not
//      just a passthrough wire.
//
// Timing note: all stall durations below are fixed, predetermined delays (not
// reactive polling on in_ready/out_valid), so this file stays parseable by
// Yosys's frontend for the elaboration smoke test, consistent with every other
// testbench in this project -- the exact latencies were established when
// gf4_block_pipeline.v (and its own testbench) were first verified.
//
// Run with:
//   iverilog -o sim gf4_fp16_to_fixed.v gf4_sumsq.v gf4_isqrt.v gf4_activation_encoder.v gf4_joint_product_table.v gf4_block_scaled_mac.v gf4_block_pipeline.v gf4_block_pipeline_stream.v tb_gf4_block_pipeline_stream.v && vvp sim
// -----------------------------------------------------------------------------
`timescale 1ns/1ps

module tb_gf4_block_pipeline_stream;

    reg clk = 0;
    reg rst_n = 0;
    always #5 clk = ~clk;

    reg in_valid = 0;
    wire in_ready;
    reg [15:0] fp16_x_in = 0;
    reg [3:0]  w_code_in = 0;
    reg in_last = 0;
    reg signed [15:0] alpha_q8_8 = 0;
    reg signed [15:0] s_w_scale = 0;
    reg global_acc_clear = 0;

    wire out_valid;
    reg  out_ready = 0;
    wire signed [31:0] out_contribution;
    wire signed [31:0] out_global_acc;
    wire [31:0] mean_sq_debug;
    wire [15:0] rms_debug;
    wire signed [15:0] s_b_debug;

    integer errors = 0;
    integer i;

    // Capture the transfer as it actually happens (out_valid && out_ready
    // together), rather than snapshotting out_valid at some later fixed
    // time -- a fast (always-ready) consumer accepts the moment out_valid
    // pulses and out_valid correctly drops again the very next cycle, so
    // checking the LIVE out_valid signal 700ns later would see it already
    // deasserted (and rightly so). This monitor is the same technique used
    // in tb_gf4_sumsq.v/tb_gf4_activation_encoder.v for pulsed valid
    // signals; scenario 3 below (a genuinely slow, momentarily-not-ready
    // consumer) is the one case where checking the live signal at a fixed
    // time is actually meaningful, since out_valid is deliberately HELD
    // there.
    reg transfer_seen;
    reg signed [31:0] captured_contribution;
    reg signed [31:0] captured_global_acc;
    always @(posedge clk) begin
        if (out_valid && out_ready) begin
            transfer_seen         <= 1'b1;
            captured_contribution <= out_contribution;
            captured_global_acc   <= out_global_acc;
        end
    end

    gf4_block_pipeline_stream dut (
        .clk              (clk),
        .rst_n            (rst_n),
        .in_valid         (in_valid),
        .in_ready         (in_ready),
        .fp16_x_in        (fp16_x_in),
        .w_code_in        (w_code_in),
        .in_last          (in_last),
        .alpha_q8_8       (alpha_q8_8),
        .s_w_scale        (s_w_scale),
        .global_acc_clear (global_acc_clear),
        .out_valid        (out_valid),
        .out_ready        (out_ready),
        .out_contribution (out_contribution),
        .out_global_acc   (out_global_acc),
        .mean_sq_debug    (mean_sq_debug),
        .rms_debug        (rms_debug),
        .s_b_debug        (s_b_debug)
    );

    reg [15:0] blkA_fp16 [0:15];
    reg [3:0]  blkA_w    [0:15];
    reg [15:0] blk2_fp16 [0:15];
    reg [3:0]  blk2_w    [0:15];

    initial begin
        blkA_fp16[0] = 16'hb99a; blkA_w[0] = 4'd0;
        blkA_fp16[1] = 16'h3800; blkA_w[1] = 4'd11;
        blkA_fp16[2] = 16'h3800; blkA_w[2] = 4'd6;
        blkA_fp16[3] = 16'h3800; blkA_w[3] = 4'd1;
        blkA_fp16[4] = 16'h3800; blkA_w[4] = 4'd12;
        blkA_fp16[5] = 16'h3800; blkA_w[5] = 4'd7;
        blkA_fp16[6] = 16'h3800; blkA_w[6] = 4'd2;
        blkA_fp16[7] = 16'h3800; blkA_w[7] = 4'd13;
        blkA_fp16[8] = 16'h3800; blkA_w[8] = 4'd0;
        blkA_fp16[9] = 16'h3800; blkA_w[9] = 4'd3;
        blkA_fp16[10] = 16'h3800; blkA_w[10] = 4'd14;
        blkA_fp16[11] = 16'h3800; blkA_w[11] = 4'd1;
        blkA_fp16[12] = 16'h3800; blkA_w[12] = 4'd4;
        blkA_fp16[13] = 16'h3800; blkA_w[13] = 4'd15;
        blkA_fp16[14] = 16'h3800; blkA_w[14] = 4'd2;
        blkA_fp16[15] = 16'h3800; blkA_w[15] = 4'd5;
        blk2_fp16[0] = 16'h0000; blk2_w[0] = 4'd3;
        blk2_fp16[1] = 16'h0000; blk2_w[1] = 4'd3;
        blk2_fp16[2] = 16'h0000; blk2_w[2] = 4'd3;
        blk2_fp16[3] = 16'h0000; blk2_w[3] = 4'd3;
        blk2_fp16[4] = 16'h0000; blk2_w[4] = 4'd3;
        blk2_fp16[5] = 16'h0000; blk2_w[5] = 4'd3;
        blk2_fp16[6] = 16'h0000; blk2_w[6] = 4'd3;
        blk2_fp16[7] = 16'h0000; blk2_w[7] = 4'd3;
        blk2_fp16[8] = 16'h0000; blk2_w[8] = 4'd3;
        blk2_fp16[9] = 16'h0000; blk2_w[9] = 4'd3;
        blk2_fp16[10] = 16'h0000; blk2_w[10] = 4'd3;
        blk2_fp16[11] = 16'h0000; blk2_w[11] = 4'd3;
        blk2_fp16[12] = 16'h0000; blk2_w[12] = 4'd3;
        blk2_fp16[13] = 16'h0000; blk2_w[13] = 4'd3;
        blk2_fp16[14] = 16'h0000; blk2_w[14] = 4'd3;
        blk2_fp16[15] = 16'h0000; blk2_w[15] = 4'd3;
    end

    initial begin
        rst_n = 0;
        in_valid = 0;
        in_last = 0;
        out_ready = 1'b1; // always-ready consumer for scenarios 1 and 2
        global_acc_clear = 0;
        transfer_seen = 1'b0;
        #12 rst_n = 1;
        #8; // t=20, a negedge -- matches this project's established convention

        // ================= Scenario 1: baseline, no stalls =================
        transfer_seen = 1'b0;
        global_acc_clear = 1'b1; #10; global_acc_clear = 1'b0; #10;
        alpha_q8_8 = 16'sd640;
        s_w_scale  = 16'sd320;
        for (i = 0; i < 16; i = i + 1) begin
            fp16_x_in = blkA_fp16[i];
            w_code_in = blkA_w[i];
            in_last   = (i == 15);
            in_valid  = 1'b1;
            #10; // in_ready is expected high throughout an idle fill phase
        end
        in_valid = 1'b0;
        in_last  = 1'b0;
        #700; // generous margin, matches tb_gf4_block_pipeline.v's own timing budget

        if (!transfer_seen || captured_contribution !== 32'sd81 || captured_global_acc !== 32'sd81) begin
            $display("FAIL: scenario 1 expected a transfer with contribution=81 global_acc=81, got transfer_seen=%0d contribution=%0d global_acc=%0d", transfer_seen, $signed(captured_contribution), $signed(captured_global_acc));
            errors = errors + 1;
        end else begin
            $display("PASS: scenario 1 (baseline) contribution=%0d global_acc=%0d", $signed(captured_contribution), $signed(captured_global_acc));
        end
        #20; // let the always-high out_ready consume the pending result before the next scenario

        // ================= Scenario 2: producer stall mid-fill =================
        transfer_seen = 1'b0;
        global_acc_clear = 1'b1; #10; global_acc_clear = 1'b0; #10;
        alpha_q8_8 = 16'sd640;
        s_w_scale  = 16'sd320;
        for (i = 0; i < 16; i = i + 1) begin
            fp16_x_in = blkA_fp16[i];
            w_code_in = blkA_w[i];
            in_last   = (i == 15);
            in_valid  = 1'b1;
            #10;
            if (i == 7) begin
                // producer goes idle for 4 cycles right after element 7 --
                // a fixed, predetermined stall (not reactive polling), so the
                // DUT must simply wait; no transfer happens while in_valid=0
                // regardless of in_ready.
                in_valid = 1'b0;
                #40;
            end
        end
        in_valid = 1'b0;
        in_last  = 1'b0;
        #700;

        if (!transfer_seen || captured_contribution !== 32'sd81 || captured_global_acc !== 32'sd81) begin
            $display("FAIL: scenario 2 (producer stall) expected a transfer with contribution=81 global_acc=81, got transfer_seen=%0d contribution=%0d global_acc=%0d", transfer_seen, $signed(captured_contribution), $signed(captured_global_acc));
            errors = errors + 1;
        end else begin
            $display("PASS: scenario 2 (producer stall, unaffected result) contribution=%0d global_acc=%0d", $signed(captured_contribution), $signed(captured_global_acc));
        end
        #20;

        // ================= Scenario 3: consumer stall after block completes =================
        out_ready = 1'b0; // consumer goes idle for this whole scenario until told otherwise
        global_acc_clear = 1'b1; #10; global_acc_clear = 1'b0; #10;
        alpha_q8_8 = 16'sd640;
        s_w_scale  = 16'sd256;
        for (i = 0; i < 16; i = i + 1) begin
            fp16_x_in = blk2_fp16[i];
            w_code_in = blk2_w[i];
            in_last   = (i == 15);
            in_valid  = 1'b1;
            #10;
        end
        in_valid = 1'b0;
        in_last  = 1'b0;

        // out_ready stays low for a long, fixed window -- long enough to be well
        // past the ~380-400ns worst-case completion latency established when
        // gf4_block_pipeline.v was first verified.
        #500;
        if (!out_valid) begin
            $display("FAIL: scenario 3 out_valid should be held high (block done, awaiting consumption) at t+500 -- got out_valid=%0d", out_valid);
            errors = errors + 1;
        end else if (out_contribution !== 32'sd0 || out_global_acc !== 32'sd0) begin
            $display("FAIL: scenario 3 held data corrupted -- expected contribution=0 global_acc=0, got contribution=%0d global_acc=%0d", $signed(out_contribution), $signed(out_global_acc));
            errors = errors + 1;
        end else begin
            $display("PASS: scenario 3 out_valid correctly held high mid-stall, data intact (contribution=%0d, global_acc=%0d)", $signed(out_contribution), $signed(out_global_acc));
        end
        if (in_ready !== 1'b0) begin
            $display("FAIL: scenario 3 in_ready should be low while a result is pending consumption -- got in_ready=%0d", in_ready);
            errors = errors + 1;
        end else begin
            $display("PASS: scenario 3 in_ready correctly held low mid-stall -- genuine backpressure confirmed");
        end

        #300; // continue holding out_ready low a while longer
        if (!out_valid) begin
            $display("FAIL: scenario 3 out_valid dropped before being consumed (checkpoint 2) -- got out_valid=%0d", out_valid);
            errors = errors + 1;
        end else begin
            $display("PASS: scenario 3 out_valid still held high at checkpoint 2, unconsumed result was never lost");
        end

        out_ready = 1'b1; // consumer finally ready
        #20;
        if (out_valid !== 1'b0) begin
            $display("FAIL: scenario 3 out_valid should drop once consumed -- got out_valid=%0d", out_valid);
            errors = errors + 1;
        end else if (in_ready !== 1'b1) begin
            $display("FAIL: scenario 3 in_ready should return high once the result is consumed and the pipeline is idle -- got in_ready=%0d", in_ready);
            errors = errors + 1;
        end else begin
            $display("PASS: scenario 3 out_valid dropped and in_ready returned high after out_ready arrived -- backpressure released correctly");
        end

        if (errors == 0)
            $display("ALL CHECKS PASSED");
        else
            $display("%0d CHECK(S) FAILED", errors);

        $finish;
    end

endmodule
