// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vtb_gf4_block_scaled_mac.h for the primary calling header

#ifndef VERILATED_VTB_GF4_BLOCK_SCALED_MAC___024ROOT_H_
#define VERILATED_VTB_GF4_BLOCK_SCALED_MAC___024ROOT_H_  // guard

#include "verilated.h"
#include "verilated_timing.h"


class Vtb_gf4_block_scaled_mac__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vtb_gf4_block_scaled_mac___024root final {
  public:

    // DESIGN SPECIFIC STATE
    CData/*0:0*/ tb_gf4_block_scaled_mac__DOT__clk;
    CData/*0:0*/ tb_gf4_block_scaled_mac__DOT__rst_n;
    CData/*3:0*/ tb_gf4_block_scaled_mac__DOT__w_code;
    CData/*3:0*/ tb_gf4_block_scaled_mac__DOT__a_code;
    CData/*0:0*/ tb_gf4_block_scaled_mac__DOT__elem_valid;
    CData/*0:0*/ tb_gf4_block_scaled_mac__DOT__block_last;
    CData/*0:0*/ tb_gf4_block_scaled_mac__DOT__global_acc_clear;
    CData/*0:0*/ tb_gf4_block_scaled_mac__DOT__dut__DOT__block_done_reg;
    CData/*0:0*/ __VstlFirstIteration;
    CData/*0:0*/ __VstlPhaseResult;
    CData/*0:0*/ __Vtrigprevexpr___TOP__tb_gf4_block_scaled_mac__DOT__clk__0;
    CData/*0:0*/ __Vtrigprevexpr___TOP__tb_gf4_block_scaled_mac__DOT__rst_n__0;
    CData/*0:0*/ __VactPhaseResult;
    CData/*0:0*/ __VinactPhaseResult;
    CData/*0:0*/ __VnbaPhaseResult;
    SData/*15:0*/ tb_gf4_block_scaled_mac__DOT__s_w_scale;
    SData/*15:0*/ tb_gf4_block_scaled_mac__DOT__s_a_scale;
    SData/*15:0*/ tb_gf4_block_scaled_mac__DOT__dut__DOT__s_w_scale_reg;
    SData/*15:0*/ tb_gf4_block_scaled_mac__DOT__dut__DOT__s_a_scale_reg;
    IData/*31:0*/ tb_gf4_block_scaled_mac__DOT__errors;
    IData/*17:0*/ tb_gf4_block_scaled_mac__DOT__dut__DOT__prod_signed;
    IData/*23:0*/ tb_gf4_block_scaled_mac__DOT__dut__DOT__raw_block_acc;
    IData/*23:0*/ tb_gf4_block_scaled_mac__DOT__dut__DOT__block_sum_reg;
    IData/*31:0*/ tb_gf4_block_scaled_mac__DOT__dut__DOT__global_acc_reg;
    IData/*31:0*/ __VactIterCount;
    IData/*31:0*/ __VinactIterCount;
    IData/*31:0*/ __Vi;
    VlUnpacked<CData/*3:0*/, 16> tb_gf4_block_scaled_mac__DOT__blkA_w;
    VlUnpacked<CData/*3:0*/, 16> tb_gf4_block_scaled_mac__DOT__blkA_a;
    VlUnpacked<CData/*3:0*/, 16> tb_gf4_block_scaled_mac__DOT__blkB_w;
    VlUnpacked<CData/*3:0*/, 16> tb_gf4_block_scaled_mac__DOT__blkB_a;
    VlUnpacked<IData/*16:0*/, 64> tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem;
    VlUnpacked<QData/*63:0*/, 1> __VstlTriggered;
    VlUnpacked<QData/*63:0*/, 1> __VactTriggered;
    VlUnpacked<QData/*63:0*/, 1> __VactTriggeredAcc;
    VlUnpacked<QData/*63:0*/, 1> __VnbaTriggered;
    VlDelayScheduler __VdlySched;

    // INTERNAL VARIABLES
    Vtb_gf4_block_scaled_mac__Syms* vlSymsp;
    const char* vlNamep;

    // CONSTRUCTORS
    Vtb_gf4_block_scaled_mac___024root(Vtb_gf4_block_scaled_mac__Syms* symsp, const char* namep);
    ~Vtb_gf4_block_scaled_mac___024root();
    VL_UNCOPYABLE(Vtb_gf4_block_scaled_mac___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
