// -----------------------------------------------------------------------------
// gf4_block_scaled_mac.v
//
// Completes the link between Phase 1 (gf4_joint_product_table.v /
// gf4_pe_lutxlut.v) and Phase 2 (the per-block scale, s_b = alpha*RMS(x_b)):
// since both operands' block scales are constant across a 16-element block,
// they factor out of the dot-product sum entirely --
//   Sum_i(x_hat_i * w_hat_i) = s_w * s_a * Sum_i(+-l_w_i * l_a_i)
// -- so this module accumulates the 16 raw, unscaled signed joint-table
// products for one block, and applies a SINGLE s_w*s_a multiply once per
// block, rather than once per element. This is the standard approach used
// by real block-scaled tensor cores (e.g. NVFP4/MXFP4): low-precision
// products accumulate in a narrow local accumulator per block; the block
// scale is applied once; the scaled per-block result adds into a wider
// running total across many blocks (the reduction/K dimension).
//
// Fixed-point bookkeeping (all Q-formats relative to the paper's own GF4
// representation, Eq. 7-8):
//   - gf4_joint_product_table entries are Q0.16 (JOINT_FRAC=16 fractional
//     bits; PROD_WIDTH=17 unsigned bits cover 0..65536 = 2^16 exactly).
//   - s_w_scale / s_a_scale are Q8.8 signed (SCALE_WIDTH=16, SCALE_FRAC=8),
//     the same convention already used for s_b elsewhere in this project.
//     As with "FP16" elsewhere, this is a fixed-point stand-in, not real
//     floating-point hardware.
//   - raw_block_acc sums up to 16 signed Q0.16 terms -- same Q0.16 format,
//     just widened (RAW_ACC_WIDTH=24) for headroom against overflow.
//   - combined_scale = s_w_scale * s_a_scale is Q16.16 (SCALE_WIDTH
//     fractional bits from each factor combine additively).
//   - product2 = raw_block_acc * combined_scale is therefore Q?.32
//     (JOINT_FRAC + SCALE_WIDTH = 16+16 = 32 fractional bits implied).
//   - The per-block scaled contribution is truncated back down to
//     FINAL_FRAC=8 fractional bits (matching the Q8.8 convention) by taking
//     product2's top (RAW_ACC_WIDTH+2*SCALE_WIDTH-SHIFT) bits, where
//     SHIFT = JOINT_FRAC + SCALE_WIDTH - FINAL_FRAC = 16+16-8 = 24. Taking a
//     contiguous top bit-slice of a two's-complement value is exactly an
//     arithmetic right shift + truncate, so no explicit >>> is needed --
//     see the assignment below.
//   - global_acc (ACC_WIDTH=32, Q24.8-ish) accumulates these per-block
//     contributions across many blocks (the full reduction dimension).
//     No saturation/overflow handling is implemented -- acceptable for a
//     proof-of-concept, not for a production accumulator.
//
// Timing: this is a 2-stage pipeline. On the cycle where elem_valid &&
// block_last, the just-completed element is folded into raw_block_acc,
// block_sum_reg captures the FULL 16-element total, and s_w_scale/s_a_scale
// are snapshotted into holding registers (so the caller need not keep them
// stable beyond that one cycle). One cycle later, block_done_pulse is high
// and the scale multiply + shift + add into global_acc happens.
// -----------------------------------------------------------------------------
module gf4_block_scaled_mac #(
    parameter W_ENTRIES    = 8,
    parameter A_ENTRIES    = 8,
    parameter PROD_WIDTH   = 17,   // must match gf4_joint_product_table's OUT_WIDTH
    parameter RAW_ACC_WIDTH = 24,  // local per-block accumulator (Q0.16, widened)
    parameter SCALE_WIDTH  = 16,   // s_w_scale / s_a_scale width (Q8.8 signed)
    parameter ACC_WIDTH    = 32    // global running accumulator (Q24.8-ish)
) (
    input  wire                        clk,
    input  wire                        rst_n,
 
    // configuration port for the joint product table (not on the MAC critical path)
    input  wire                        cfg_we,
    input  wire [5:0]                  cfg_waddr,
    input  wire [PROD_WIDTH-1:0]       cfg_wdata,
 
    // per-cycle operands: one weight/activation code pair per element
    input  wire [3:0]                  w_code,
    input  wire [3:0]                  a_code,
    input  wire                        elem_valid,   // accumulate this element into the current block
    input  wire                        block_last,   // this is the 16th (final) element of the block
 
    // per-block scales -- must be valid on the same cycle block_last fires;
    // snapshotted internally, so they need not be held any longer than that.
    input  wire signed [SCALE_WIDTH-1:0] s_w_scale,
    input  wire signed [SCALE_WIDTH-1:0] s_a_scale,
 
    input  wire                        global_acc_clear,
 
    output wire signed [ACC_WIDTH-1:0] global_acc,
    output wire                        block_done_pulse, // one-cycle pulse: global_acc just updated
 
    // Phase 3 addition: exposes this block's own scaled contribution (already
    // computed internally below as `scaled_contribution`), separate from the
    // ever-growing running total in `global_acc`. Purely additive -- existing
    // callers that don't connect this port are unaffected; nothing about
    // global_acc/block_done_pulse's behavior changes. Useful for a generic
    // "pluggable into someone else's accumulator" story (see
    // gf4_block_pipeline_stream.v), where a surrounding system may want this
    // block's delta rather than a running total this module maintains
    // internally. Valid on the same cycle as block_done_pulse.
    output wire signed [ACC_WIDTH-1:0] scaled_contribution_out
);
 
    localparam integer JOINT_FRAC = PROD_WIDTH - 1;                      // 16
    localparam integer FINAL_FRAC = SCALE_WIDTH / 2;                     // 8
    localparam integer SHIFT      = JOINT_FRAC + SCALE_WIDTH - FINAL_FRAC; // 24
    localparam integer PROD2_WIDTH = RAW_ACC_WIDTH + 2*SCALE_WIDTH;      // 56
 
    // --- joint product lookup for this element ---
    wire        prod_sign;
    wire [PROD_WIDTH-1:0] prod_mag;
 
    gf4_joint_product_table #(
        .W_ENTRIES(W_ENTRIES),
        .A_ENTRIES(A_ENTRIES),
        .OUT_WIDTH(PROD_WIDTH)
    ) u_joint_table (
        .clk      (clk),
        .rst_n    (rst_n),
        .cfg_we   (cfg_we),
        .cfg_waddr(cfg_waddr),
        .cfg_wdata(cfg_wdata),
        .w_idx    (w_code[2:0]),
        .w_sign   (w_code[3]),
        .a_idx    (a_code[2:0]),
        .a_sign   (a_code[3]),
        .prod_sign(prod_sign),
        .prod_mag (prod_mag)
    );
 
    wire signed [PROD_WIDTH:0] prod_signed = prod_sign
        ? -$signed({1'b0, prod_mag})
        :  $signed({1'b0, prod_mag});
 
    // --- stage 1: within-block raw accumulation (Q0.16, widened) ---
    reg signed [RAW_ACC_WIDTH-1:0] raw_block_acc;
    reg signed [RAW_ACC_WIDTH-1:0] block_sum_reg;
    reg signed [SCALE_WIDTH-1:0]   s_w_scale_reg;
    reg signed [SCALE_WIDTH-1:0]   s_a_scale_reg;
    reg                            block_done_reg;
 
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            raw_block_acc   <= {RAW_ACC_WIDTH{1'b0}};
            block_sum_reg   <= {RAW_ACC_WIDTH{1'b0}};
            s_w_scale_reg   <= {SCALE_WIDTH{1'b0}};
            s_a_scale_reg   <= {SCALE_WIDTH{1'b0}};
            block_done_reg  <= 1'b0;
        end else begin
            block_done_reg <= 1'b0;
            if (elem_valid) begin
                if (block_last) begin
                    // Fold in the final element, capture the completed
                    // 16-element total, snapshot the scales, and reset the
                    // running accumulator for the next block.
                    block_sum_reg  <= raw_block_acc + prod_signed;
                    s_w_scale_reg  <= s_w_scale;
                    s_a_scale_reg  <= s_a_scale;
                    block_done_reg <= 1'b1;
                    raw_block_acc  <= {RAW_ACC_WIDTH{1'b0}};
                end else begin
                    raw_block_acc <= raw_block_acc + prod_signed;
                end
            end
        end
    end
 
    // --- stage 2: apply the single per-block scale, one cycle after block_done_reg ---
    wire signed [2*SCALE_WIDTH-1:0] combined_scale = s_w_scale_reg * s_a_scale_reg;      // Q16.16
    wire signed [PROD2_WIDTH-1:0]   product2       = block_sum_reg * combined_scale;      // Q?.32
 
    // Truncating to the top (PROD2_WIDTH-SHIFT) bits is exactly an
    // arithmetic right shift by SHIFT for a two's-complement value -- this
    // rescales from 32 implied fractional bits down to FINAL_FRAC (8),
    // matching the Q8.8-ish convention used for global_acc.
    wire signed [ACC_WIDTH-1:0] scaled_contribution = product2[PROD2_WIDTH-1 -: ACC_WIDTH];
 
    reg signed [ACC_WIDTH-1:0] global_acc_reg;
 
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            global_acc_reg <= {ACC_WIDTH{1'b0}};
        else if (global_acc_clear)
            global_acc_reg <= {ACC_WIDTH{1'b0}};
        else if (block_done_reg)
            global_acc_reg <= global_acc_reg + scaled_contribution;
    end
 
    assign global_acc      = global_acc_reg;
    assign block_done_pulse = block_done_reg;
    assign scaled_contribution_out = scaled_contribution;
 
endmodule