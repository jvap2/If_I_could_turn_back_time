// -----------------------------------------------------------------------------
// gf4_activation_encoder.v
//
// Phase 2, piece 3 of 3 (final piece): closes the activation-encode pipeline.
// Buffers one 16-element block of raw Q8.8 activations, computes its block
// scale s_b = alpha * RMS(x_b) using the already-verified gf4_sumsq.v and
// gf4_isqrt.v, threshold-scales the 7 fixed GF4 codebook midpoints by s_b
// (once per block, NOT once per element -- no division anywhere), then
// streams out one 4-bit GF4 code (sign + 3-bit index) per buffered element
// via a comparator ladder against those 7 scaled thresholds. `s_a_scale`
// (=s_b) is exactly what gf4_block_scaled_mac.v's `s_a_scale` port has been
// waiting for since Phase 1.5 -- wiring this module's outputs into that
// port's inputs (a_code -> a_code, s_a_scale -> s_a_scale, timed off
// s_a_valid/a_code_valid) closes the full Phase 1 + 1.5 + 2 loop.
//
// Why a comparator ladder instead of division: computing |x_i|/s_b per
// element and comparing to fixed codebook thresholds would need a divider
// per element (expensive). Instead, the 7 fixed threshold midpoints
// (offline-computed once, between each pair of adjacent GF4 codebook
// levels) are multiplied by s_b ONCE PER BLOCK, and every element's raw
// |x_i| is compared directly against those pre-scaled thresholds -- 7
// multiplies per block (16 elements) instead of up to 16 divisions.
//
// Midpoints: derived from the exact Q0.8 integer levels already used and
// verified in gf4_decode_programmable.v / tb_gf4_decode.v
// (L = 0,20,45,72,101,134,178,256), as round((L_i + L_i+1)/2) for each
// adjacent pair -- deliberately reusing that same source of truth rather
// than re-deriving from the real-valued levels, to avoid a second
// independently-rounded copy of the same numbers:
//   {10, 33, 59, 87, 118, 156, 217}   (default cfg_wdata table contents)
// A comparator ladder that counts how many of these 7 (monotonically
// increasing) thresholds |x_i| meets or exceeds gives exactly the nearest-
// level index for a nearest-neighbor quantizer against a monotonic
// codebook -- no separate distance computation needed.
//
// Fixed-point bookkeeping (same conventions as gf4_isqrt.v / gf4_sumsq.v /
// gf4_block_scaled_mac.v):
//   - alpha_q8_8 / s_a_scale / s_b are Q8.8 signed (SCALE_WIDTH=16), matching
//     s_w_scale/s_a_scale elsewhere.
//   - RMS(x_b) (gf4_isqrt's `root`) is unsigned Q8.8 (always >= 0).
//   - s_b = alpha * RMS(x_b): Q8.8 * Q8.8 = Q16.16, rescaled back down to
//     Q8.8 with an explicit `>>> 8` (arithmetic right shift) then truncated
//     -- unlike gf4_block_scaled_mac.v's top-bit-slice trick, this uses an
//     explicit shift so the required shift amount doesn't have to be solved
//     for via the operand widths; functionally equivalent for a
//     non-negative two's-complement value.
//   - Threshold scaling: each fixed Q0.8-ish midpoint (THRESH_WIDTH=9,
//     matching gf4_decode_programmable.v's WIDTH=9, needed there because
//     level 256=1.000 needs a 9th bit) times s_b (Q8.8) gives Q8.16,
//     rescaled the same way back down to Q8.8 for direct comparison
//     against |x_i| (also Q8.8).
//
// Verified in Python before being encoded here:
//   - The paper's own worked example, reproduced through the quantization
//     stage specifically (the mean_sq->RMS->s_b chain was already verified
//     end to end in tb_gf4_sumsq.v): with s_b=1.25 exactly (Q8.8=320, the
//     paper's own alpha=2.5 x RMS=0.5), xi=-0.7 threshold-scales and
//     compares to give sign=1, idx=5 exactly -- matching tb_gf4_decode.v's
//     own worked-example check (nearest level 0.525, idx 5).
//   - 200 random 16-element blocks (3200 total per-element codes) against a
//     real-valued (floating point) nearest-neighbor reference: 49/3200
//     (~1.5%) codes differ, and every difference is off by exactly one
//     codebook index (never more) -- the expected, bounded rounding error
//     from finite Q8.8 precision in RMS/threshold computation, the same
//     category of "small expected fixed-point rounding gap" already
//     documented in tb_gf4_decode.v's own worked example, not a bug.
//
// Timing / non-pipelined limitation (documented, not fixed): this module
// does NOT overlap filling the next block with emitting the current one --
// a full block must finish emitting before the next block's elements are
// accepted (`busy` is high the whole time). A production version would
// double-buffer to overlap fill(N+1) with emit(N); left as a known
// simplification for this proof-of-concept, exactly like
// gf4_block_scaled_mac.v's lack of saturation handling.
// -----------------------------------------------------------------------------
module gf4_activation_encoder #(
    parameter X_WIDTH      = 16,  // Q8.8 signed activation width
    parameter SCALE_WIDTH  = 16,  // Q8.8 signed (alpha, s_b/s_a_scale)
    parameter THRESH_WIDTH = 9,   // unsigned codebook-midpoint width (matches gf4_decode_programmable.v)
    parameter BLOCK_SIZE   = 16
) (
    input  wire                        clk,
    input  wire                        rst_n,
 
    // raw activation stream in (one block at a time)
    input  wire signed [X_WIDTH-1:0]   x_in,
    input  wire                        elem_valid,
    input  wire                        block_last,
 
    // static per-layer alpha (held stable; not part of the per-block stream)
    input  wire signed [SCALE_WIDTH-1:0] alpha_q8_8,
 
    // config port for the 7 codebook-midpoint thresholds (addr 0..6)
    input  wire                        cfg_we,
    input  wire [2:0]                  cfg_waddr,
    input  wire [THRESH_WIDTH-1:0]     cfg_wdata,
 
    // encoded activation-code stream out (one per buffered element, in order)
    output reg                         a_code_valid,
    output reg  [3:0]                  a_code,          // {sign, 3-bit index}, same convention as w_code/a_code elsewhere
    output reg  [3:0]                  a_code_elem_idx, // which of the 16 buffered elements this code is for
 
    // the block's scale, valid once per block (holds stable through the whole emit phase)
    output reg                         s_a_valid,
    output reg  signed [SCALE_WIDTH-1:0] s_a_scale,
 
    output wire                        busy,
 
    // debug/cross-check outputs (not needed by a real consumer, only for verification)
    output wire [31:0]                 mean_sq_debug,
    output wire [15:0]                 rms_debug,
    output wire signed [SCALE_WIDTH-1:0] s_b_debug
);
 
    localparam [2:0] S_FILL        = 3'd0;
    localparam [2:0] S_WAIT_RMS    = 3'd1;
    localparam [2:0] S_COMPUTE_TH  = 3'd2;
    localparam [2:0] S_EMIT        = 3'd3;
 
    reg [2:0] state;
    assign busy = (state != S_FILL);
 
    // --- 16-element activation buffer (must hold the whole block until s_b is known) ---
    reg signed [X_WIDTH-1:0] buf_mem [0:BLOCK_SIZE-1];
    reg [3:0] fill_idx;
    reg [3:0] emit_idx;
 
    // --- 7-entry threshold table, default-initialized to the codebook midpoints ---
    reg [THRESH_WIDTH-1:0] thresh_table [0:6];
    initial begin
        thresh_table[0] = 9'd10;
        thresh_table[1] = 9'd33;
        thresh_table[2] = 9'd59;
        thresh_table[3] = 9'd87;
        thresh_table[4] = 9'd118;
        thresh_table[5] = 9'd156;
        thresh_table[6] = 9'd217;
    end
 
    // --- internal sum-of-squares + sqrt chain (already-verified modules) ---
    wire mean_sq_valid;
    wire [31:0] mean_sq;
 
    gf4_sumsq #(.X_WIDTH(X_WIDTH)) u_sumsq (
        .clk           (clk),
        .rst_n         (rst_n),
        .x_in          (x_in),
        .elem_valid    (elem_valid),
        .block_last    (block_last),
        .mean_sq_valid (mean_sq_valid),
        .mean_sq       (mean_sq)
    );
 
    wire isqrt_busy, isqrt_done;
    wire [15:0] rms_root;
 
    gf4_isqrt #(.IN_WIDTH(32), .OUT_WIDTH(16)) u_isqrt (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (mean_sq_valid),
        .radicand (mean_sq),
        .busy     (isqrt_busy),
        .done     (isqrt_done),
        .root     (rms_root)
    );
 
    assign mean_sq_debug = mean_sq;
    assign rms_debug      = rms_root;
 
    // --- s_b = alpha * RMS(x_b): Q8.8 * Q8.8 = Q16.16, shifted back to Q8.8 ---
    wire signed [16:0]  rms_ext       = {1'b0, rms_root};              // zero-extend unsigned Q8.8 to signed 17-bit
    wire signed [32:0]  sb_full_prod  = alpha_q8_8 * rms_ext;          // Q16.16, 33 bits
    wire signed [32:0]  sb_shifted    = sb_full_prod >>> 8;            // drop 8 of 16 fractional bits
    wire signed [SCALE_WIDTH-1:0] sb_comb = sb_shifted[SCALE_WIDTH-1:0]; // truncate to Q8.8 (no overflow check -- proof-of-concept)
 
    reg signed [SCALE_WIDTH-1:0] s_b_reg;
    assign s_b_debug = s_b_reg;
 
    // --- threshold scaling: each Q0.8-ish midpoint * s_b (Q8.8) -> Q8.16 -> shift back to Q8.8 ---
    wire signed [9:0]  mid0_signed = {1'b0, thresh_table[0]};
    wire signed [9:0]  mid1_signed = {1'b0, thresh_table[1]};
    wire signed [9:0]  mid2_signed = {1'b0, thresh_table[2]};
    wire signed [9:0]  mid3_signed = {1'b0, thresh_table[3]};
    wire signed [9:0]  mid4_signed = {1'b0, thresh_table[4]};
    wire signed [9:0]  mid5_signed = {1'b0, thresh_table[5]};
    wire signed [9:0]  mid6_signed = {1'b0, thresh_table[6]};
 
    wire signed [25:0] th0_full = mid0_signed * s_b_reg;
    wire signed [25:0] th1_full = mid1_signed * s_b_reg;
    wire signed [25:0] th2_full = mid2_signed * s_b_reg;
    wire signed [25:0] th3_full = mid3_signed * s_b_reg;
    wire signed [25:0] th4_full = mid4_signed * s_b_reg;
    wire signed [25:0] th5_full = mid5_signed * s_b_reg;
    wire signed [25:0] th6_full = mid6_signed * s_b_reg;
 
    wire signed [15:0] th0_comb = (th0_full >>> 8);
    wire signed [15:0] th1_comb = (th1_full >>> 8);
    wire signed [15:0] th2_comb = (th2_full >>> 8);
    wire signed [15:0] th3_comb = (th3_full >>> 8);
    wire signed [15:0] th4_comb = (th4_full >>> 8);
    wire signed [15:0] th5_comb = (th5_full >>> 8);
    wire signed [15:0] th6_comb = (th6_full >>> 8);
 
    reg signed [15:0] th0_reg, th1_reg, th2_reg, th3_reg, th4_reg, th5_reg, th6_reg;
 
    // --- comparator ladder for the element currently being emitted ---
    wire signed [X_WIDTH-1:0] cur_x    = buf_mem[emit_idx];
    wire                      cur_sign = cur_x[X_WIDTH-1];
    wire signed [X_WIDTH-1:0] cur_abs  = cur_sign ? -cur_x : cur_x;
 
    wire ge0 = (cur_abs >= th0_reg);
    wire ge1 = (cur_abs >= th1_reg);
    wire ge2 = (cur_abs >= th2_reg);
    wire ge3 = (cur_abs >= th3_reg);
    wire ge4 = (cur_abs >= th4_reg);
    wire ge5 = (cur_abs >= th5_reg);
    wire ge6 = (cur_abs >= th6_reg);
 
    wire [2:0] idx_comb = ge0 + ge1 + ge2 + ge3 + ge4 + ge5 + ge6; // monotonic thresholds -> count = nearest-level index
 
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_FILL;
            fill_idx        <= 4'd0;
            emit_idx        <= 4'd0;
            a_code_valid    <= 1'b0;
            a_code          <= 4'd0;
            a_code_elem_idx <= 4'd0;
            s_a_valid       <= 1'b0;
            s_a_scale       <= {SCALE_WIDTH{1'b0}};
            s_b_reg         <= {SCALE_WIDTH{1'b0}};
            th0_reg <= 16'sd0; th1_reg <= 16'sd0; th2_reg <= 16'sd0; th3_reg <= 16'sd0;
            th4_reg <= 16'sd0; th5_reg <= 16'sd0; th6_reg <= 16'sd0;
        end else begin
            a_code_valid <= 1'b0; // default: clear one-cycle pulses each cycle
            s_a_valid    <= 1'b0;
 
            if (cfg_we) begin
                thresh_table[cfg_waddr] <= cfg_wdata; // config write, not on the encode critical path
            end
 
            case (state)
                S_FILL: begin
                    if (elem_valid) begin
                        buf_mem[fill_idx] <= x_in;
                        if (block_last) begin
                            fill_idx <= 4'd0;
                            state    <= S_WAIT_RMS;
                        end else begin
                            fill_idx <= fill_idx + 1'b1;
                        end
                    end
                end
 
                S_WAIT_RMS: begin
                    if (isqrt_done) begin
                        s_b_reg <= sb_comb; // valid now: rms_root (isqrt's registered output) is stable this same cycle
                        state   <= S_COMPUTE_TH;
                    end
                end
 
                S_COMPUTE_TH: begin
                    // s_b_reg became valid last cycle; the th*_comb wires (driven off
                    // s_b_reg) are now correct, so latch all 7 in this one cycle.
                    th0_reg <= th0_comb; th1_reg <= th1_comb; th2_reg <= th2_comb; th3_reg <= th3_comb;
                    th4_reg <= th4_comb; th5_reg <= th5_comb; th6_reg <= th6_comb;
                    emit_idx <= 4'd0;
                    state    <= S_EMIT;
                end
 
                S_EMIT: begin
                    // th*_reg became valid last cycle, so idx_comb/cur_sign (driven off
                    // buf_mem[emit_idx] and th*_reg) are correct this cycle.
                    a_code_valid    <= 1'b1;
                    a_code          <= {cur_sign, idx_comb};
                    a_code_elem_idx <= emit_idx;
                    if (emit_idx == 4'd0) begin
                        s_a_valid <= 1'b1;
                        s_a_scale <= s_b_reg;
                    end
                    if (emit_idx == BLOCK_SIZE-1) begin
                        state <= S_FILL;
                    end else begin
                        emit_idx <= emit_idx + 1'b1;
                    end
                end
 
                default: state <= S_FILL;
            endcase
        end
    end
 
endmodule