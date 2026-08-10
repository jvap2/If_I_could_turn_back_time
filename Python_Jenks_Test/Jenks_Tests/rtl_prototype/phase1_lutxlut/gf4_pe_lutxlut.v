// -----------------------------------------------------------------------------
// gf4_pe_lutxlut.v
//
// Phase 1 "LUTxLUT" processing element: consumes a weight code and an
// activation code directly, looks up their precomputed product in
// gf4_joint_product_table.v, and accumulates -- no multiplier anywhere in
// this datapath. Contrast with gf4_pe.v, which decodes one operand via a
// table and then multiplies against a conventional operand; this module
// removes that multiplier entirely for the case where BOTH operands are
// LUT-coded (the W4A4 setting), replacing it with one table read + one XOR.
//
// This is the RTL-level demonstration of the "precompute the possibilities
// of multiplying the two lookup tables into an output table" mechanism
// discussed for the paper's W4A4 datapath, as distinct from (and cheaper
// than) a full chunked LUT-GEMM engine (e.g. LUT Tensor Core/T-MAN/MxGLUT),
// which precomputes partial dot products over multi-element chunks rather
// than a single scalar pair.
// -----------------------------------------------------------------------------
module gf4_pe_lutxlut #(
    parameter W_ENTRIES = 8,
    parameter A_ENTRIES = 8,
    parameter PROD_WIDTH = 17,   // must match gf4_joint_product_table's OUT_WIDTH
    parameter ACC_WIDTH  = 32
) (
    input  wire                        clk,
    input  wire                        rst_n,
 
    // configuration port for the joint product table (not on the MAC critical path)
    input  wire                        cfg_we,
    input  wire [5:0]                  cfg_waddr,
    input  wire [PROD_WIDTH-1:0]       cfg_wdata,
 
    // per-cycle operands: both weight and activation arrive as 4-bit codes
    // ({sign, 3-bit index}), matching the W4A4 setting -- neither side is
    // ever reconstructed to a full-precision value in this datapath.
    input  wire [3:0]                  w_code,
    input  wire [3:0]                  a_code,
    input  wire                        valid,
    input  wire                        acc_clear,
 
    output wire signed [ACC_WIDTH-1:0] mac_acc
);
 
    wire        w_sign = w_code[3];
    wire [2:0]  w_idx  = w_code[2:0];
    wire        a_sign = a_code[3];
    wire [2:0]  a_idx  = a_code[2:0];
 
    wire                   prod_sign;
    wire [PROD_WIDTH-1:0]  prod_mag;
 
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
        .w_idx    (w_idx),
        .w_sign   (w_sign),
        .a_idx    (a_idx),
        .a_sign   (a_sign),
        .prod_sign(prod_sign),
        .prod_mag (prod_mag)
    );
 
    // Apply sign to the looked-up magnitude -- the only arithmetic in this
    // datapath besides the accumulate itself.
    wire signed [PROD_WIDTH:0] prod_signed = prod_sign
        ? -$signed({1'b0, prod_mag})
        :  $signed({1'b0, prod_mag});
 
    reg signed [ACC_WIDTH-1:0] acc_reg;
 
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            acc_reg <= {ACC_WIDTH{1'b0}};
        else if (acc_clear)
            acc_reg <= {ACC_WIDTH{1'b0}};
        else if (valid)
            acc_reg <= acc_reg + prod_signed;
    end
 
    assign mac_acc = acc_reg;
 
endmodule