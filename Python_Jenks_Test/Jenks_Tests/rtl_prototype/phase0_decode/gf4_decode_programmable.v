// -----------------------------------------------------------------------------
// gf4_decode_programmable.v
//
// Programmable GF4 codebook decode: an 8-entry writable register file holding
// the Gaussian-optimal quantile levels L = {0.000, 0.080, 0.174, 0.283, 0.395,
// 0.525, 0.696, 1.000} (paper Eq. 7), represented in Q0.8 unsigned fixed
// point (mag_out = round(value * 256), range 0..256, hence a 9-bit field).
//
// This is the "programmable table" half of the GF4-vs-NVFP4 cost comparison
// (paper Table II): identical read path to gf4_decode_fixed.v, but the table
// contents are loaded once via cfg_we/cfg_waddr/cfg_wdata rather than
// hard-wired, so the marginal synthesis cost of *this* module over
// gf4_decode_fixed.v is the actual "decode-at-ingress" table cost being
// priced.
//
// Reset/config values default to the GF4 levels below so the table is
// functionally correct even before any cfg_we write, which keeps the
// testbench simple; a real deployment would load the table once per model
// (or per layer, if per-layer codebooks are used) at configuration time --
// i.e. this write port is NOT on the per-MAC critical path, matching the
// paper's "decode-at-ingress" placement (Table II) rather than
// "decode-per-MAC".
// -----------------------------------------------------------------------------
module gf4_decode_programmable #(
    parameter ENTRIES = 8,
    parameter WIDTH   = 9
) (
    input  wire             clk,
    input  wire             rst_n,

    // configuration (write) port -- loaded once, not on the MAC critical path
    input  wire             cfg_we,
    input  wire [2:0]       cfg_waddr,
    input  wire [WIDTH-1:0] cfg_wdata,

    // decode (read) port
    input  wire [2:0]       idx,
    input  wire             sign_in,
    output wire             sign_out,
    output wire [WIDTH-1:0] mag_out
);

    reg [WIDTH-1:0] table_mem [0:ENTRIES-1];

    // Default-initialize to the paper's GF4 levels (Eq. 7), Q0.8 fixed point:
    // L = {0.000, 0.080, 0.174, 0.283, 0.395, 0.525, 0.696, 1.000}
    initial begin
        table_mem[0] = 9'd0;    // 0.000
        table_mem[1] = 9'd20;   // 0.080
        table_mem[2] = 9'd45;   // 0.174
        table_mem[3] = 9'd72;   // 0.283
        table_mem[4] = 9'd101;  // 0.395
        table_mem[5] = 9'd134;  // 0.525
        table_mem[6] = 9'd178;  // 0.696
        table_mem[7] = 9'd256;  // 1.000
    end

    always @(posedge clk) begin
        if (cfg_we)
            table_mem[cfg_waddr] <= cfg_wdata;
    end

    // Combinational read: this is the block whose area/energy the paper's
    // "decode-at-ingress vs. decode-per-MAC" comparison (Table II) is about --
    // synthesize this module and gf4_decode_fixed.v separately and diff area
    // to reproduce that comparison at the RTL level.
    assign mag_out  = table_mem[idx];
    assign sign_out = sign_in;

endmodule
