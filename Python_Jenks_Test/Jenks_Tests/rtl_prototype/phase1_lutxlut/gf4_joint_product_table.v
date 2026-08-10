// -----------------------------------------------------------------------------
// gf4_joint_product_table.v
//
// Phase 1 of the "LUTxLUT" W4A4 datapath: a joint outer-product table that
// replaces decode-then-multiply with a single lookup. Instead of decoding a
// weight code and an activation code separately and feeding both into a
// multiplier (as gf4_pe.v does), this module directly stores the precomputed
// product of every possible (weight-level, activation-level) pair, so the
// runtime datapath is just a table read plus a sign XOR -- no multiplier.
//
// Addressing: the weight codebook and activation codebook are each 8 entries
// (matching the GF4 magnitude table L, Eq. 7 of the paper), so the joint
// table has W_ENTRIES * A_ENTRIES = 64 entries, addressed by
// {w_idx[2:0], a_idx[2:0]} (6 bits).
//
// Value representation: each factor is Q0.8 (magnitude * 256, range 0..256,
// per gf4_decode_programmable.v). The product of two Q0.8 values is Q0.16
// (range 0..65536), which needs 17 bits unsigned -- confirmed by computing
// the full 8x8 outer product offline (max value 256*256 = 65536 = 2^16,
// bit_length() = 17). OUT_WIDTH defaults to 17 accordingly.
//
// Sign is NOT stored in the table -- it's cheap to compute separately as
// prod_sign = w_sign ^ a_sign, so the table only needs to hold 64 unsigned
// magnitudes rather than 64 signed values, keeping it the same size as if
// sign were ignored entirely.
//
// Like gf4_decode_programmable.v, the table defaults to precomputed content
// (here: the outer product of the GF4 levels L with themselves, i.e. this
// instance assumes weights and activations share the same GF4 codebook) but
// is writable via a config port that is not on the per-MAC critical path --
// a real deployment would reload it once per model/layer if a different
// activation codebook were used.
// -----------------------------------------------------------------------------
module gf4_joint_product_table #(
    parameter W_ENTRIES = 8,
    parameter A_ENTRIES = 8,
    parameter OUT_WIDTH = 17
) (
    input  wire                  clk,
    input  wire                  rst_n,
 
    // configuration (write) port -- loaded once, not on the MAC critical path
    input  wire                  cfg_we,
    input  wire [5:0]            cfg_waddr,   // {w_idx[2:0], a_idx[2:0]}, 64 entries
    input  wire [OUT_WIDTH-1:0]  cfg_wdata,
 
    // lookup (read) port
    input  wire [2:0]            w_idx,
    input  wire                  w_sign,
    input  wire [2:0]            a_idx,
    input  wire                  a_sign,
 
    output wire                  prod_sign,
    output wire [OUT_WIDTH-1:0]  prod_mag
);
 
    reg [OUT_WIDTH-1:0] table_mem [0:(W_ENTRIES*A_ENTRIES)-1];
 
    // Default-initialize to the outer product of the GF4 levels (Eq. 7) with
    // themselves, in Q0.16 fixed point (L_int[w] * L_int[a], where L_int =
    // {0, 20, 45, 72, 101, 134, 178, 256} is the same Q0.8 table used by
    // gf4_decode_programmable.v). Computed offline and cross-checked in
    // Python before being encoded here (see NOTES_rtl_synthesis.md).
    // Address = w_idx*8 + a_idx, i.e. {w_idx, a_idx} concatenated.
    initial begin
        // w_idx = 0 (L=0)
        table_mem[6'd0]  = 17'd0;     table_mem[6'd1]  = 17'd0;     table_mem[6'd2]  = 17'd0;
        table_mem[6'd3]  = 17'd0;     table_mem[6'd4]  = 17'd0;     table_mem[6'd5]  = 17'd0;
        table_mem[6'd6]  = 17'd0;     table_mem[6'd7]  = 17'd0;
        // w_idx = 1 (L=20)
        table_mem[6'd8]  = 17'd0;     table_mem[6'd9]  = 17'd400;   table_mem[6'd10] = 17'd900;
        table_mem[6'd11] = 17'd1440;  table_mem[6'd12] = 17'd2020;  table_mem[6'd13] = 17'd2680;
        table_mem[6'd14] = 17'd3560;  table_mem[6'd15] = 17'd5120;
        // w_idx = 2 (L=45)
        table_mem[6'd16] = 17'd0;     table_mem[6'd17] = 17'd900;   table_mem[6'd18] = 17'd2025;
        table_mem[6'd19] = 17'd3240;  table_mem[6'd20] = 17'd4545;  table_mem[6'd21] = 17'd6030;
        table_mem[6'd22] = 17'd8010;  table_mem[6'd23] = 17'd11520;
        // w_idx = 3 (L=72)
        table_mem[6'd24] = 17'd0;     table_mem[6'd25] = 17'd1440;  table_mem[6'd26] = 17'd3240;
        table_mem[6'd27] = 17'd5184;  table_mem[6'd28] = 17'd7272;  table_mem[6'd29] = 17'd9648;
        table_mem[6'd30] = 17'd12816; table_mem[6'd31] = 17'd18432;
        // w_idx = 4 (L=101)
        table_mem[6'd32] = 17'd0;     table_mem[6'd33] = 17'd2020;  table_mem[6'd34] = 17'd4545;
        table_mem[6'd35] = 17'd7272;  table_mem[6'd36] = 17'd10201; table_mem[6'd37] = 17'd13534;
        table_mem[6'd38] = 17'd17978; table_mem[6'd39] = 17'd25856;
        // w_idx = 5 (L=134)
        table_mem[6'd40] = 17'd0;     table_mem[6'd41] = 17'd2680;  table_mem[6'd42] = 17'd6030;
        table_mem[6'd43] = 17'd9648;  table_mem[6'd44] = 17'd13534; table_mem[6'd45] = 17'd17956;
        table_mem[6'd46] = 17'd23852; table_mem[6'd47] = 17'd34304;
        // w_idx = 6 (L=178)
        table_mem[6'd48] = 17'd0;     table_mem[6'd49] = 17'd3560;  table_mem[6'd50] = 17'd8010;
        table_mem[6'd51] = 17'd12816; table_mem[6'd52] = 17'd17978; table_mem[6'd53] = 17'd23852;
        table_mem[6'd54] = 17'd31684; table_mem[6'd55] = 17'd45568;
        // w_idx = 7 (L=256)
        table_mem[6'd56] = 17'd0;     table_mem[6'd57] = 17'd5120;  table_mem[6'd58] = 17'd11520;
        table_mem[6'd59] = 17'd18432; table_mem[6'd60] = 17'd25856; table_mem[6'd61] = 17'd34304;
        table_mem[6'd62] = 17'd45568; table_mem[6'd63] = 17'd65536;
    end
 
    always @(posedge clk) begin
        if (cfg_we)
            table_mem[cfg_waddr] <= cfg_wdata;
    end
 
    // Combinational read + cheap sign combine -- this is the entire runtime
    // datapath: one table read, one XOR. No multiplier.
    assign prod_mag  = table_mem[{w_idx, a_idx}];
    assign prod_sign = w_sign ^ a_sign;
 
endmodule