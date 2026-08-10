// -----------------------------------------------------------------------------
// gf4_sumsq.v
//
// Phase 2, piece 2 of 3: sum-of-squares accumulator over one 16-element
// activation block, producing the Q16.16 mean-of-squares that feeds directly
// into gf4_isqrt.v's `radicand` input (RMS(x_b) = sqrt(mean(x_i^2)), Eq. 8).
//
// Convention: activations x_i arrive as signed Q8.8 fixed point (X_WIDTH=16),
// the same "FP16-class stand-in" width used for s_w_scale/s_a_scale in
// gf4_block_scaled_mac.v. Squaring a signed Q8.8 value gives an *unsigned*
// Q16.16 value (Q8.8 * Q8.8 = Q16.16, and a square is never negative), so no
// sign handling is needed past this point -- this mirrors gf4_isqrt.v's own
// "raw bits in, raw bits out" fixed-point trick.
//
// Streaming interface, matching gf4_block_scaled_mac.v's style: drive
// `elem_valid` for one cycle per activation (16 total per block), pulsing
// `block_last` together with `elem_valid` on the 16th. One cycle after
// `block_last`, `mean_sq_valid` pulses with the Q16.16 mean-of-squares for
// the block on `mean_sq`; feed that directly into gf4_isqrt's `radicand`
// (with `start`) to get RMS(x_b) in Q8.8.
//
// Dividing the 16-term sum by BLOCK_SIZE=16 is done with a plain arithmetic
// right shift (SHIFT_BITS=4, since 16 is a power of two) -- no divider
// anywhere in this module, consistent with the rest of this project's
// "shift instead of divide wherever the divisor is a fixed power of two"
// convention (see gf4_block_scaled_mac.v's rescaling shifts).
//
// Numerically verified in Python before being encoded here: a constant
// block of sixteen 0.5-magnitude activations (matching the paper's own
// RMS(x_b)=0.5 worked example, see tb_gf4_decode.v) produces
// mean_sq=16384 (=0.25 * 65536 exactly), which gf4_isqrt.v's algorithm
// reduces to root=128 (=0.5 in Q8.8) exactly -- i.e. this module's output
// chains into gf4_isqrt.v to reproduce the paper's own example end to end.
// Eight additional random 16-element blocks (activations up to +/-4.0) were
// also cross-checked against a bit-exact Python model of this same
// accumulate-then-shift arithmetic, with the fixed hardware register widths
// (ACC_WIDTH=36) checked for headroom (max possible sum is
// 16 * 32768^2 = 2^34, comfortably inside 36 bits).
// -----------------------------------------------------------------------------
module gf4_sumsq #(
    parameter X_WIDTH    = 16,              // Q8.8 signed activation width
    parameter SQ_WIDTH   = 2 * X_WIDTH,     // 32 -- one squared term, Q16.16 (always >= 0)
    parameter SHIFT_BITS = 4,               // log2(BLOCK_SIZE); BLOCK_SIZE must be a power of 2
    parameter ACC_WIDTH  = SQ_WIDTH + SHIFT_BITS  // 36 -- headroom for summing BLOCK_SIZE terms
) (
    input  wire                       clk,
    input  wire                       rst_n,
 
    input  wire signed [X_WIDTH-1:0]  x_in,
    input  wire                       elem_valid,  // pulse for each of the 16 activations
    input  wire                       block_last,  // pulse with elem_valid on the 16th element
 
    output reg                        mean_sq_valid, // one-cycle pulse, one cycle after block_last
    output reg  [SQ_WIDTH-1:0]        mean_sq        // Q16.16 mean-of-squares for the block
);
 
    // x_in * x_in as a signed product is always non-negative (it's a square),
    // so its raw bit pattern is safely reinterpreted as the unsigned Q16.16
    // magnitude -- no separate abs()/sign logic needed.
    wire signed [SQ_WIDTH-1:0] x_sq_signed = x_in * x_in;
    wire        [SQ_WIDTH-1:0] x_sq        = x_sq_signed[SQ_WIDTH-1:0];
    wire        [ACC_WIDTH-1:0] x_sq_ext   = {{(ACC_WIDTH-SQ_WIDTH){1'b0}}, x_sq};
 
    reg [ACC_WIDTH-1:0] acc;
    wire [ACC_WIDTH-1:0] acc_next = acc + x_sq_ext;
 
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc           <= {ACC_WIDTH{1'b0}};
            mean_sq_valid <= 1'b0;
            mean_sq       <= {SQ_WIDTH{1'b0}};
        end else begin
            mean_sq_valid <= 1'b0; // default: clear the one-cycle pulse
 
            if (elem_valid) begin
                if (block_last) begin
                    // Final element: fold it in, divide-by-shift, publish, and
                    // reset the accumulator for the next block.
                    mean_sq       <= acc_next[ACC_WIDTH-1:SHIFT_BITS]; // implicit truncation to SQ_WIDTH bits
                    mean_sq_valid <= 1'b1;
                    acc           <= {ACC_WIDTH{1'b0}};
                end else begin
                    acc <= acc_next;
                end
            end
        end
    end
 
endmodule