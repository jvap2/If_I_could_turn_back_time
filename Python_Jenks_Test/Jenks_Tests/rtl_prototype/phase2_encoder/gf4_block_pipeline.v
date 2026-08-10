// -----------------------------------------------------------------------------
// gf4_block_pipeline.v
//
// Closes the full Phase 1 + 1.5 + 2 loop: wires gf4_fp16_to_fixed.v ->
// gf4_activation_encoder.v -> gf4_block_scaled_mac.v together into one
// synthesizable block. Until now, every one of those modules had only ever
// been simulated standalone -- this is the first module in the whole
// project where raw FP16 activation bits go in one end and a scaled,
// accumulated dot-product result (`global_acc`) comes out the other,
// with weight codes (assumed already produced offline by the
// Hessian-guided weight quantizer -- not computed in this RTL) folded in
// along the way.
//
// The wiring problem this module actually solves: gf4_activation_encoder.v
// cannot emit a block's activation codes until it knows that block's s_b,
// which means there's a real, non-trivial latency (~37 cycles) between
// when a block's raw activations finish streaming in and when its 16
// encoded codes stream back out (one per cycle, in original element
// order, via a_code/a_code_valid/a_code_elem_idx). But gf4_block_scaled_mac.v
// needs BOTH a weight code and an activation code for the same element at
// the same time. Weight codes are available immediately (they come from
// an offline quantization step, not a live computation), so this module
// buffers the block's 16 weight codes in lockstep with the activation
// fill phase, then replays them out of that buffer -- indexed by
// `a_code_elem_idx` -- in sync with the encoder's delayed code emission.
// This buffer-and-replay pattern is genuine hardware a real
// implementation would need here, not a testbench convenience.
//
// Timing contract this module places on its caller (beyond what
// gf4_activation_encoder.v already requires): `s_w_scale` must stay valid
// for the ENTIRE non-pipelined duration of a block -- roughly 16 (fill) +
// ~21 (RMS/threshold compute) + 16 (emit) = ~53 cycles -- not just during
// the 16-cycle fill phase, since it's only actually consumed by
// gf4_block_scaled_mac.v once each element's code is replayed during
// emit. The simplest way to satisfy this: hold `s_w_scale` constant for
// the whole time `busy` is high.
//
// Everything below this module's boundary (gf4_joint_product_table.v via
// gf4_block_scaled_mac.v, the codebook levels, the Q-format conventions)
// is unchanged from its already-verified standalone behavior -- this
// module only adds plumbing (the FP16 converter and the weight-code
// buffer), no new arithmetic.
// -----------------------------------------------------------------------------
module gf4_block_pipeline #(
    parameter X_WIDTH     = 16,  // Q8.8 signed activation width
    parameter SCALE_WIDTH = 16,  // Q8.8 signed (alpha, s_w_scale, s_a_scale)
    parameter PROD_WIDTH  = 17,  // gf4_joint_product_table's OUT_WIDTH
    parameter ACC_WIDTH   = 32,  // global running accumulator width
    parameter BLOCK_SIZE  = 16
) (
    input  wire                          clk,
    input  wire                          rst_n,
 
    // raw activation stream in, as real FP16 bits (not pre-converted fixed point)
    input  wire [15:0]                   fp16_x_in,
    // this element's weight code (assumed already quantized offline)
    input  wire [3:0]                    w_code_in,
    input  wire                          elem_valid,
    input  wire                          block_last,
 
    // static per-layer alpha (activation RMS scale multiplier)
    input  wire signed [SCALE_WIDTH-1:0] alpha_q8_8,
    // this block's weight scale (assumed already computed offline, held
    // stable for the whole ~53-cycle non-pipelined block duration -- see
    // header comment)
    input  wire signed [SCALE_WIDTH-1:0] s_w_scale,
 
    input  wire                          global_acc_clear,
 
    output wire signed [ACC_WIDTH-1:0]   global_acc,
    output wire                          block_done_pulse,
    // Phase 3 addition: pass-through of gf4_block_scaled_mac.v's per-block
    // scaled contribution (see that module's header) -- purely additive,
    // existing callers unaffected. Valid alongside block_done_pulse.
    output wire signed [ACC_WIDTH-1:0]   scaled_contribution,
    output wire                          busy,
 
    // debug taps, passed through from the internal activation encoder
    output wire [31:0]                   mean_sq_debug,
    output wire [15:0]                   rms_debug,
    output wire signed [SCALE_WIDTH-1:0] s_b_debug
);
 
    // --- stage 1: real FP16 bits -> Q8.8 fixed point (combinational, no latency) ---
    wire signed [X_WIDTH-1:0] x_fixed;
    wire fp16_is_zero, fp16_is_special; // unused here, available for a caller that wants them
 
    gf4_fp16_to_fixed #(.OUT_WIDTH(X_WIDTH), .OUT_FRAC(8)) u_fp16_conv (
        .fp16_in    (fp16_x_in),
        .fixed_out  (x_fixed),
        .is_zero    (fp16_is_zero),
        .is_special (fp16_is_special)
    );
 
    // --- stage 2: activation encoder (buffers the block, computes s_b, emits codes) ---
    wire                        a_code_valid;
    wire [3:0]                  a_code;
    wire [3:0]                  a_code_elem_idx;
    wire                        s_a_valid;
    wire signed [SCALE_WIDTH-1:0] s_a_scale;
 
    gf4_activation_encoder #(
        .X_WIDTH(X_WIDTH), .SCALE_WIDTH(SCALE_WIDTH), .BLOCK_SIZE(BLOCK_SIZE)
    ) u_encoder (
        .clk             (clk),
        .rst_n           (rst_n),
        .x_in            (x_fixed),
        .elem_valid      (elem_valid),
        .block_last      (block_last),
        .alpha_q8_8      (alpha_q8_8),
        .cfg_we          (1'b0),
        .cfg_waddr       (3'd0),
        .cfg_wdata       (9'd0),
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
 
    // --- weight-code buffer: fills in lockstep with the activation fill phase,
    //     replays (indexed by a_code_elem_idx) in lockstep with the encoder's
    //     delayed emit phase ---
    reg [3:0] w_buf [0:BLOCK_SIZE-1];
    reg [3:0] w_fill_idx;
 
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w_fill_idx <= 4'd0;
        end else if (elem_valid) begin
            w_buf[w_fill_idx] <= w_code_in;
            if (block_last) begin
                w_fill_idx <= 4'd0;
            end else begin
                w_fill_idx <= w_fill_idx + 1'b1;
            end
        end
    end
 
    wire [3:0] w_code_replay = w_buf[a_code_elem_idx];
 
    // --- stage 3: block-scaled MAC (joint-table lookup, per-block scale, running accumulator) ---
    gf4_block_scaled_mac #(
        .PROD_WIDTH(PROD_WIDTH), .SCALE_WIDTH(SCALE_WIDTH), .ACC_WIDTH(ACC_WIDTH)
    ) u_mac (
        .clk             (clk),
        .rst_n           (rst_n),
        .cfg_we          (1'b0),
        .cfg_waddr       (6'd0),
        .cfg_wdata       ({PROD_WIDTH{1'b0}}),
        .w_code          (w_code_replay),
        .a_code          (a_code),
        .elem_valid      (a_code_valid),
        .block_last      (a_code_valid && (a_code_elem_idx == BLOCK_SIZE-1)),
        .s_w_scale       (s_w_scale),
        .s_a_scale       (s_a_scale),
        .global_acc_clear(global_acc_clear),
        .global_acc      (global_acc),
        .block_done_pulse(block_done_pulse),
        .scaled_contribution_out(scaled_contribution)
    );
 
endmodule