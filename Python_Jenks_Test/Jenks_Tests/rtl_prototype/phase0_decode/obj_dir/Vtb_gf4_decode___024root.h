// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vtb_gf4_decode.h for the primary calling header

#ifndef VERILATED_VTB_GF4_DECODE___024ROOT_H_
#define VERILATED_VTB_GF4_DECODE___024ROOT_H_  // guard

#include "verilated.h"
#include "verilated_timing.h"


class Vtb_gf4_decode__Syms;

class alignas(VL_CACHE_LINE_BYTES) Vtb_gf4_decode___024root final {
  public:

    // DESIGN SPECIFIC STATE
    CData/*0:0*/ tb_gf4_decode__DOT__clk;
    CData/*0:0*/ tb_gf4_decode__DOT__cfg_we;
    CData/*2:0*/ tb_gf4_decode__DOT__cfg_waddr;
    CData/*2:0*/ tb_gf4_decode__DOT__idx;
    CData/*7:0*/ tb_gf4_decode__DOT__mag_out_fixed;
    CData/*0:0*/ __VstlFirstIteration;
    CData/*0:0*/ __VstlPhaseResult;
    CData/*0:0*/ __Vtrigprevexpr___TOP__tb_gf4_decode__DOT__clk__0;
    CData/*0:0*/ __VactPhaseResult;
    CData/*0:0*/ __VinactPhaseResult;
    CData/*0:0*/ __VnbaPhaseResult;
    SData/*8:0*/ tb_gf4_decode__DOT__cfg_wdata;
    IData/*31:0*/ tb_gf4_decode__DOT__errors;
    IData/*31:0*/ __VactIterCount;
    IData/*31:0*/ __VinactIterCount;
    IData/*31:0*/ __Vi;
    VlUnpacked<SData/*8:0*/, 8> tb_gf4_decode__DOT__expected_gf4;
    VlUnpacked<CData/*7:0*/, 8> tb_gf4_decode__DOT__expected_e2m1;
    VlUnpacked<SData/*8:0*/, 8> tb_gf4_decode__DOT__dut_prog__DOT__table_mem;
    VlUnpacked<QData/*63:0*/, 1> __VstlTriggered;
    VlUnpacked<QData/*63:0*/, 1> __VactTriggered;
    VlUnpacked<QData/*63:0*/, 1> __VactTriggeredAcc;
    VlUnpacked<QData/*63:0*/, 1> __VnbaTriggered;
    VlDelayScheduler __VdlySched;

    // INTERNAL VARIABLES
    Vtb_gf4_decode__Syms* vlSymsp;
    const char* vlNamep;

    // CONSTRUCTORS
    Vtb_gf4_decode___024root(Vtb_gf4_decode__Syms* symsp, const char* namep);
    ~Vtb_gf4_decode___024root();
    VL_UNCOPYABLE(Vtb_gf4_decode___024root);

    // INTERNAL METHODS
    void __Vconfigure(bool first);
};


#endif  // guard
