// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vtb_gf4_joint_product.h for the primary calling header

#ifndef VERILATED_VTB_GF4_JOINT_PRODUCT___024ROOT_H_
#define VERILATED_VTB_GF4_JOINT_PRODUCT___024ROOT_H_  // guard

#include "verilated.h"
#include "verilated_timing.h"


class Vtb_gf4_joint_product__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vtb_gf4_joint_product___024root final {
  public:

    // DESIGN SPECIFIC STATE
    CData/*0:0*/ tb_gf4_joint_product__DOT__clk;
    CData/*0:0*/ tb_gf4_joint_product__DOT__cfg_we;
    CData/*5:0*/ tb_gf4_joint_product__DOT__cfg_waddr;
    CData/*2:0*/ tb_gf4_joint_product__DOT__w_idx;
    CData/*0:0*/ tb_gf4_joint_product__DOT__w_sign;
    CData/*2:0*/ tb_gf4_joint_product__DOT__a_idx;
    CData/*0:0*/ tb_gf4_joint_product__DOT__a_sign;
    CData/*0:0*/ tb_gf4_joint_product__DOT__prod_sign;
    CData/*0:0*/ __VstlFirstIteration;
    CData/*0:0*/ __VstlPhaseResult;
    CData/*0:0*/ __Vtrigprevexpr___TOP__tb_gf4_joint_product__DOT__clk__0;
    CData/*0:0*/ __VactPhaseResult;
    CData/*0:0*/ __VinactPhaseResult;
    CData/*0:0*/ __VnbaPhaseResult;
    IData/*16:0*/ tb_gf4_joint_product__DOT__cfg_wdata;
    IData/*31:0*/ tb_gf4_joint_product__DOT__i;
    IData/*31:0*/ tb_gf4_joint_product__DOT__a;
    IData/*31:0*/ tb_gf4_joint_product__DOT__errors;
    IData/*31:0*/ __VactIterCount;
    IData/*31:0*/ __VinactIterCount;
    IData/*31:0*/ __Vi;
    VlUnpacked<IData/*16:0*/, 64> tb_gf4_joint_product__DOT__expected_prod;
    VlUnpacked<IData/*16:0*/, 64> tb_gf4_joint_product__DOT__dut__DOT__table_mem;
    VlUnpacked<QData/*63:0*/, 1> __VstlTriggered;
    VlUnpacked<QData/*63:0*/, 1> __VactTriggered;
    VlUnpacked<QData/*63:0*/, 1> __VactTriggeredAcc;
    VlUnpacked<QData/*63:0*/, 1> __VnbaTriggered;
    VlDelayScheduler __VdlySched;

    // INTERNAL VARIABLES
    Vtb_gf4_joint_product__Syms* vlSymsp;
    const char* vlNamep;

    // CONSTRUCTORS
    Vtb_gf4_joint_product___024root(Vtb_gf4_joint_product__Syms* symsp, const char* namep);
    ~Vtb_gf4_joint_product___024root();
    VL_UNCOPYABLE(Vtb_gf4_joint_product___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
