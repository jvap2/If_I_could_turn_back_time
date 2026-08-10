// -----------------------------------------------------------------------------
// gf4_block_pipeline_stream.v
//
// Phase 3: wraps gf4_block_pipeline.v (which itself already wraps
// gf4_fp16_to_fixed.v -> gf4_activation_encoder.v -> gf4_block_scaled_mac.v)
// with a real valid/ready streaming interface on both sides, replacing
// this project's previous "caller must count cycles and respect an
// undocumented latency" contract with a standard backpressure-capable
// handshake -- the actual mechanism behind the original "portable across
// accelerators" goal from way back at the start of this RTL work. Nothing
// about the underlying arithmetic changes here; this module is pure
// control-plane plumbing on top of already-verified datapath logic.
//
// Input side (producer -> this module): a transfer happens on any cycle
// where `in_valid && in_ready` are both high. `in_ready` is this module's
// own signal, computed below -- the caller never needs to count cycles or
// consult a latency table, only watch `in_ready`. `in_last` marks the
// 16th (final) element of a block, exactly like `block_last` everywhere
// else in this project.
//
// Standard valid/ready convention (same as AXI-Stream): the producer must
// hold fp16_x_in/w_code_in/in_last/in_valid stable until it observes
// in_ready high on some cycle -- that's the cycle the transfer actually
// happens, and only then may the producer move on to the next element.
// in_ready itself only ever depends on this module's OWN registered state
// (never on in_valid), so there's no combinational dependency loop.
//
// A meaningful simplification over gf4_block_pipeline.v's own contract:
// `alpha_q8_8`/`s_w_scale` are LATCHED internally on the first transfer of
// each block (in_valid && in_ready, seen while not already mid-block), so
// the caller only needs to present them correctly on that one cycle --
// not hold them stable for the whole ~53-cycle non-pipelined block
// duration underneath. That burden is now this module's problem, not the
// caller's, which is the point of a real "pluggable IP" interface.
//
// Output side (this module -> consumer): once a block's result is ready
// (gf4_block_pipeline.v's block_done_pulse), it's latched into a holding
// register and `out_valid` is asserted and HELD (not just pulsed) until
// the consumer asserts `out_ready` on some later cycle -- proper
// "valid-holds-until-accepted" semantics, unlike a one-cycle pulse a slow
// consumer could simply miss. Two results are exposed: `out_contribution`
// (this block's own scaled contribution, useful if the surrounding system
// -- e.g. a GEMM engine's own K-dimension accumulator -- wants to combine
// results itself) and `out_global_acc` (the running total this module
// maintains internally regardless, for backward-compatible use exactly
// like gf4_block_pipeline.v's own output).
//
// Backpressure is genuinely end-to-end, by design, at the cost of a
// documented simplification: this module holds only ONE pending result at
// a time (no internal FIFO). `in_ready` is gated on the output holding
// register being empty (`!out_valid_reg`) as well as the inner pipeline
// being idle (`~busy`) -- so a new block cannot start filling until the
// PREVIOUS block's result has actually been consumed via `out_ready`. A
// production version wanting to overlap "filling block N+1" with "N's
// result still awaiting pickup" would need a small output FIFO (depth 2+)
// instead of this single holding register; left out here as an explicit,
// documented scope decision, consistent with gf4_activation_encoder.v's
// own "no fill/emit overlap" simplification underneath.
// -----------------------------------------------------------------------------
module gf4_block_pipeline_stream #(
    parameter X_WIDTH     = 16,
    parameter SCALE_WIDTH = 16,
    parameter PROD_WIDTH  = 17,
    parameter ACC_WIDTH   = 32,
    parameter BLOCK_SIZE  = 16
) (
    input  wire                          clk,
    input  wire                          rst_n,
 
    // --- input stream: valid/ready handshake, one element per accepted transfer ---
    input  wire                          in_valid,
    output wire                          in_ready,
    input  wire [15:0]                   fp16_x_in,
    input  wire [3:0]                    w_code_in,
    input  wire                          in_last,          // marks the 16th element of a block
 
    // only need to be correct on the cycle of the FIRST accepted transfer of
    // each block -- latched internally, may change/don't-care afterward
    input  wire signed [SCALE_WIDTH-1:0] alpha_q8_8,
    input  wire signed [SCALE_WIDTH-1:0] s_w_scale,
 
    input  wire                          global_acc_clear,
 
    // --- output stream: valid/ready handshake, one transfer per completed block ---
    output wire                          out_valid,
    input  wire                          out_ready,
    output wire signed [ACC_WIDTH-1:0]   out_contribution, // this block's own scaled contribution
    output wire signed [ACC_WIDTH-1:0]   out_global_acc,   // running total (same convention as gf4_block_pipeline.v)
 
    // debug taps, passed straight through
    output wire [31:0]                   mean_sq_debug,
    output wire [15:0]                   rms_debug,
    output wire signed [SCALE_WIDTH-1:0] s_b_debug
);
 
    // --- latch alpha/s_w_scale on the first accepted transfer of each block ---
    reg mid_block;
    reg signed [SCALE_WIDTH-1:0] alpha_reg;
    reg signed [SCALE_WIDTH-1:0] s_w_scale_reg;
 
    wire inner_busy;
    wire accept_xfer = in_valid && in_ready; // a genuine, accepted transfer this cycle
 
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mid_block     <= 1'b0;
            alpha_reg     <= {SCALE_WIDTH{1'b0}};
            s_w_scale_reg <= {SCALE_WIDTH{1'b0}};
        end else begin
            if (accept_xfer) begin
                if (!mid_block) begin
                    // first element of a new block: latch the scales now,
                    // the caller need not hold them beyond this cycle
                    alpha_reg     <= alpha_q8_8;
                    s_w_scale_reg <= s_w_scale;
                end
                mid_block <= in_last ? 1'b0 : 1'b1;
            end
        end
    end
 
    // --- output holding register: latches on block_done_pulse, holds until out_ready ---
    //
    // IMPORTANT ASYMMETRY, found by running this on a real simulator (Icarus
    // caught a genuine bug here -- worth leaving this note rather than
    // silently fixing it): `scaled_contribution` inside gf4_block_scaled_mac.v
    // is a purely COMBINATIONAL wire, already correct the instant
    // block_done_pulse first goes high. `global_acc` is a REGISTERED
    // accumulator that only picks up this block's contribution ONE CYCLE
    // AFTER block_done_pulse first goes high (its own internal stage-2
    // update happens at that same edge, using block_done_pulse's
    // just-before-this-edge value). Latching BOTH into holding registers at
    // the same edge (keyed off inner_block_done) grabbed the fresh
    // contribution but a stale, pre-update global_acc.
    //
    // Fix: don't latch global_acc into a separate register at all. It's
    // already a stable, persistent register inside the wrapped pipeline
    // that only changes once per completed block and holds steady
    // otherwise -- by the time ANY reader looks at it (out_valid_reg itself
    // only becomes visible starting the same edge global_acc's update
    // becomes visible, since both are non-blocking-assigned at that same
    // edge), it's already correct. So this is a live combinational
    // pass-through, not something that needs holding.
    reg                          out_valid_reg;
    reg signed [ACC_WIDTH-1:0]   out_contribution_reg;
 
    wire inner_block_done;
    wire signed [ACC_WIDTH-1:0] inner_contribution;
    wire signed [ACC_WIDTH-1:0] inner_global_acc;
 
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid_reg         <= 1'b0;
            out_contribution_reg  <= {ACC_WIDTH{1'b0}};
        end else begin
            if (inner_block_done) begin
                // by construction (in_ready gates on !out_valid_reg below),
                // a new block_done_pulse cannot arrive while a previous
                // result is still pending -- so an unconditional set here
                // never clobbers an unconsumed result.
                out_contribution_reg <= inner_contribution;
                out_valid_reg        <= 1'b1;
            end else if (out_valid_reg && out_ready) begin
                out_valid_reg <= 1'b0; // consumed
            end
        end
    end
 
    assign out_valid        = out_valid_reg;
    assign out_contribution = out_contribution_reg;
    assign out_global_acc   = inner_global_acc; // live pass-through, see note above
 
    // --- real, end-to-end backpressure: only accept new block data once the
    //     inner pipeline is idle AND the previous result has been consumed ---
    assign in_ready = (~inner_busy) && (!out_valid_reg);
 
    // --- the wrapped pipeline itself, unmodified ---
    gf4_block_pipeline #(
        .X_WIDTH(X_WIDTH), .SCALE_WIDTH(SCALE_WIDTH),
        .PROD_WIDTH(PROD_WIDTH), .ACC_WIDTH(ACC_WIDTH), .BLOCK_SIZE(BLOCK_SIZE)
    ) u_pipeline (
        .clk              (clk),
        .rst_n            (rst_n),
        .fp16_x_in        (fp16_x_in),
        .w_code_in        (w_code_in),
        .elem_valid       (accept_xfer),
        .block_last       (accept_xfer && in_last),
        .alpha_q8_8       (alpha_reg),
        .s_w_scale        (s_w_scale_reg),
        .global_acc_clear (global_acc_clear),
        .global_acc       (inner_global_acc),
        .block_done_pulse (inner_block_done),
        .scaled_contribution (inner_contribution),
        .busy             (inner_busy),
        .mean_sq_debug    (mean_sq_debug),
        .rms_debug        (rms_debug),
        .s_b_debug        (s_b_debug)
    );
 
endmodule