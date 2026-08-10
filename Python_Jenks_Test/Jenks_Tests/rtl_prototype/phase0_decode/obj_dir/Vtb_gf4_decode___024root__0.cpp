// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtb_gf4_decode.h for the primary calling header

#include "Vtb_gf4_decode__pch.h"

VlCoroutine Vtb_gf4_decode___024root___eval_initial__TOP__Vtiming__0(Vtb_gf4_decode___024root* vlSelf);
VlCoroutine Vtb_gf4_decode___024root___eval_initial__TOP__Vtiming__1(Vtb_gf4_decode___024root* vlSelf);

void Vtb_gf4_decode___024root___eval_initial(Vtb_gf4_decode___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_decode___024root___eval_initial\n"); );
    Vtb_gf4_decode__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    {
        // Inlined CFunc: _eval_initial__TOP
        vlSelfRef.tb_gf4_decode__DOT__expected_gf4[0U] = 0U;
        vlSelfRef.tb_gf4_decode__DOT__expected_gf4[1U] = 0x0014U;
        vlSelfRef.tb_gf4_decode__DOT__expected_gf4[2U] = 0x002dU;
        vlSelfRef.tb_gf4_decode__DOT__expected_gf4[3U] = 0x0048U;
        vlSelfRef.tb_gf4_decode__DOT__expected_gf4[4U] = 0x0065U;
        vlSelfRef.tb_gf4_decode__DOT__expected_gf4[5U] = 0x0086U;
        vlSelfRef.tb_gf4_decode__DOT__expected_gf4[6U] = 0x00b2U;
        vlSelfRef.tb_gf4_decode__DOT__expected_gf4[7U] = 0x0100U;
        vlSelfRef.tb_gf4_decode__DOT__expected_e2m1[0U] = 0U;
        vlSelfRef.tb_gf4_decode__DOT__expected_e2m1[1U] = 8U;
        vlSelfRef.tb_gf4_decode__DOT__expected_e2m1[2U] = 0x10U;
        vlSelfRef.tb_gf4_decode__DOT__expected_e2m1[3U] = 0x18U;
        vlSelfRef.tb_gf4_decode__DOT__expected_e2m1[4U] = 0x20U;
        vlSelfRef.tb_gf4_decode__DOT__expected_e2m1[5U] = 0x30U;
        vlSelfRef.tb_gf4_decode__DOT__expected_e2m1[6U] = 0x40U;
        vlSelfRef.tb_gf4_decode__DOT__expected_e2m1[7U] = 0x60U;
        vlSelfRef.tb_gf4_decode__DOT__dut_prog__DOT__table_mem[0U] = 0U;
        vlSelfRef.tb_gf4_decode__DOT__dut_prog__DOT__table_mem[1U] = 0x0014U;
        vlSelfRef.tb_gf4_decode__DOT__dut_prog__DOT__table_mem[2U] = 0x002dU;
        vlSelfRef.tb_gf4_decode__DOT__dut_prog__DOT__table_mem[3U] = 0x0048U;
        vlSelfRef.tb_gf4_decode__DOT__dut_prog__DOT__table_mem[4U] = 0x0065U;
        vlSelfRef.tb_gf4_decode__DOT__dut_prog__DOT__table_mem[5U] = 0x0086U;
        vlSelfRef.tb_gf4_decode__DOT__dut_prog__DOT__table_mem[6U] = 0x00b2U;
        vlSelfRef.tb_gf4_decode__DOT__dut_prog__DOT__table_mem[7U] = 0x0100U;
    }
    Vtb_gf4_decode___024root___eval_initial__TOP__Vtiming__0(vlSelf);
    Vtb_gf4_decode___024root___eval_initial__TOP__Vtiming__1(vlSelf);
}

VlCoroutine Vtb_gf4_decode___024root___eval_initial__TOP__Vtiming__0(Vtb_gf4_decode___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_decode___024root___eval_initial__TOP__Vtiming__0\n"); );
    Vtb_gf4_decode__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    IData/*31:0*/ tb_gf4_decode__DOT__i;
    tb_gf4_decode__DOT__i = 0;
    SData/*15:0*/ tb_gf4_decode__DOT__sb_q8_8;
    tb_gf4_decode__DOT__sb_q8_8 = 0;
    IData/*31:0*/ tb_gf4_decode__DOT__recon_q8_16;
    tb_gf4_decode__DOT__recon_q8_16 = 0;
    // Body
    co_await vlSelfRef.__VdlySched.delay(0x0000000000002ee0ULL, 
                                         nullptr, "tb_gf4_decode.v", 
                                         99);
    tb_gf4_decode__DOT__i = 0U;
    while (VL_GTS_III(32, 8U, tb_gf4_decode__DOT__i)) {
        vlSelfRef.tb_gf4_decode__DOT__idx = (7U & tb_gf4_decode__DOT__i);
        co_await vlSelfRef.__VdlySched.delay(0x00000000000003e8ULL, 
                                             nullptr, 
                                             "tb_gf4_decode.v", 
                                             104);
        if ((vlSelfRef.tb_gf4_decode__DOT__dut_prog__DOT__table_mem
             [vlSelfRef.tb_gf4_decode__DOT__idx] != vlSelfRef.tb_gf4_decode__DOT__expected_gf4
             [(7U & tb_gf4_decode__DOT__i)])) {
            VL_WRITEF_NX("FAIL: GF4 idx=%0d expected=%0d got=%0d\n",3
                         , '~',32,tb_gf4_decode__DOT__i
                         , '#',9,vlSelfRef.tb_gf4_decode__DOT__expected_gf4
                         [(7U & tb_gf4_decode__DOT__i)]
                         , '#',9,vlSelfRef.tb_gf4_decode__DOT__dut_prog__DOT__table_mem
                         [vlSelfRef.tb_gf4_decode__DOT__idx]);
            vlSelfRef.tb_gf4_decode__DOT__errors = 
                ((IData)(1U) + vlSelfRef.tb_gf4_decode__DOT__errors);
        } else {
            VL_WRITEF_NX("PASS: GF4 idx=%0d mag_q0_8=%0d\n",2
                         , '~',32,tb_gf4_decode__DOT__i
                         , '#',9,vlSelfRef.tb_gf4_decode__DOT__dut_prog__DOT__table_mem
                         [vlSelfRef.tb_gf4_decode__DOT__idx]);
        }
        tb_gf4_decode__DOT__i = ((IData)(1U) + tb_gf4_decode__DOT__i);
    }
    tb_gf4_decode__DOT__i = 0U;
    while (VL_GTS_III(32, 8U, tb_gf4_decode__DOT__i)) {
        vlSelfRef.tb_gf4_decode__DOT__idx = (7U & tb_gf4_decode__DOT__i);
        co_await vlSelfRef.__VdlySched.delay(0x00000000000003e8ULL, 
                                             nullptr, 
                                             "tb_gf4_decode.v", 
                                             116);
        if (((IData)(vlSelfRef.tb_gf4_decode__DOT__mag_out_fixed) 
             != vlSelfRef.tb_gf4_decode__DOT__expected_e2m1
             [(7U & tb_gf4_decode__DOT__i)])) {
            VL_WRITEF_NX("FAIL: E2M1 idx=%0d expected=%0d got=%0d\n",3
                         , '~',32,tb_gf4_decode__DOT__i
                         , '#',8,vlSelfRef.tb_gf4_decode__DOT__expected_e2m1
                         [(7U & tb_gf4_decode__DOT__i)]
                         , '#',8,vlSelfRef.tb_gf4_decode__DOT__mag_out_fixed);
            vlSelfRef.tb_gf4_decode__DOT__errors = 
                ((IData)(1U) + vlSelfRef.tb_gf4_decode__DOT__errors);
        } else {
            VL_WRITEF_NX("PASS: E2M1 idx=%0d mag_q4_4=%0d\n",2
                         , '~',32,tb_gf4_decode__DOT__i
                         , '#',8,(IData)(vlSelfRef.tb_gf4_decode__DOT__mag_out_fixed));
        }
        tb_gf4_decode__DOT__i = ((IData)(1U) + tb_gf4_decode__DOT__i);
    }
    vlSelfRef.tb_gf4_decode__DOT__idx = 5U;
    co_await vlSelfRef.__VdlySched.delay(0x00000000000003e8ULL, 
                                         nullptr, "tb_gf4_decode.v", 
                                         128);
    if (VL_UNLIKELY(((vlSelfRef.tb_gf4_decode__DOT__dut_prog__DOT__table_mem
                      [vlSelfRef.tb_gf4_decode__DOT__idx] 
                      != vlSelfRef.tb_gf4_decode__DOT__expected_gf4[5U])))) {
        VL_WRITEF_NX("FAIL: worked-example idx=5 expected=%0d got=%0d\n",2
                     , '#',9,vlSelfRef.tb_gf4_decode__DOT__expected_gf4[5U]
                     , '#',9,vlSelfRef.tb_gf4_decode__DOT__dut_prog__DOT__table_mem
                     [vlSelfRef.tb_gf4_decode__DOT__idx]);
        vlSelfRef.tb_gf4_decode__DOT__errors = ((IData)(1U) 
                                                + vlSelfRef.tb_gf4_decode__DOT__errors);
    }
    tb_gf4_decode__DOT__sb_q8_8 = 0x0140U;
    tb_gf4_decode__DOT__recon_q8_16 = ((IData)(tb_gf4_decode__DOT__sb_q8_8) 
                                       * vlSelfRef.tb_gf4_decode__DOT__dut_prog__DOT__table_mem
                                       [vlSelfRef.tb_gf4_decode__DOT__idx]);
    VL_WRITEF_NX("Worked example: mag_q0_8=%0d, sb_q8_8=%0d, recon_q8_16=%0d\n",3
                 , '#',9,vlSelfRef.tb_gf4_decode__DOT__dut_prog__DOT__table_mem
                 [vlSelfRef.tb_gf4_decode__DOT__idx]
                 , '#',16,tb_gf4_decode__DOT__sb_q8_8
                 , '#',32,tb_gf4_decode__DOT__recon_q8_16);
    if (VL_UNLIKELY((((0x0000a85cU < tb_gf4_decode__DOT__recon_q8_16) 
                      | (0x0000a6ccU > tb_gf4_decode__DOT__recon_q8_16))))) {
        VL_WRITEF_NX("FAIL: worked-example reconstruction out of expected fixed-point range\n",0);
        vlSelfRef.tb_gf4_decode__DOT__errors = ((IData)(1U) 
                                                + vlSelfRef.tb_gf4_decode__DOT__errors);
    }
    vlSelfRef.tb_gf4_decode__DOT__cfg_we = 1U;
    vlSelfRef.tb_gf4_decode__DOT__cfg_waddr = 3U;
    vlSelfRef.tb_gf4_decode__DOT__cfg_wdata = 0x00c8U;
    co_await vlSelfRef.__VdlySched.delay(0x0000000000002710ULL, 
                                         nullptr, "tb_gf4_decode.v", 
                                         155);
    vlSelfRef.tb_gf4_decode__DOT__cfg_we = 0U;
    vlSelfRef.tb_gf4_decode__DOT__idx = 3U;
    co_await vlSelfRef.__VdlySched.delay(0x00000000000003e8ULL, 
                                         nullptr, "tb_gf4_decode.v", 
                                         163);
    if ((0x00c8U != vlSelfRef.tb_gf4_decode__DOT__dut_prog__DOT__table_mem
         [vlSelfRef.tb_gf4_decode__DOT__idx])) {
        VL_WRITEF_NX("FAIL: cfg write to idx=3 expected=200 got=%0d\n",1
                     , '#',9,vlSelfRef.tb_gf4_decode__DOT__dut_prog__DOT__table_mem
                     [vlSelfRef.tb_gf4_decode__DOT__idx]);
        vlSelfRef.tb_gf4_decode__DOT__errors = ((IData)(1U) 
                                                + vlSelfRef.tb_gf4_decode__DOT__errors);
    } else {
        VL_WRITEF_NX("PASS: cfg write readback idx=3 -> %0d\n",1
                     , '#',9,vlSelfRef.tb_gf4_decode__DOT__dut_prog__DOT__table_mem
                     [vlSelfRef.tb_gf4_decode__DOT__idx]);
    }
    if ((0U == vlSelfRef.tb_gf4_decode__DOT__errors)) {
        VL_WRITEF_NX("ALL CHECKS PASSED\n",0);
    } else {
        VL_WRITEF_NX("%0d CHECK(S) FAILED\n",1, '~',32,vlSelfRef.tb_gf4_decode__DOT__errors);
    }
    VL_FINISH_MT("tb_gf4_decode.v", 176, "");
    co_return;
}

VlCoroutine Vtb_gf4_decode___024root___eval_initial__TOP__Vtiming__1(Vtb_gf4_decode___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_decode___024root___eval_initial__TOP__Vtiming__1\n"); );
    Vtb_gf4_decode__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    while (VL_LIKELY(!vlSymsp->_vm_contextp__->gotFinish())) {
        co_await vlSelfRef.__VdlySched.delay(0x0000000000001388ULL, 
                                             nullptr, 
                                             "tb_gf4_decode.v", 
                                             95);
        vlSelfRef.tb_gf4_decode__DOT__clk = (1U & (~ (IData)(vlSelfRef.tb_gf4_decode__DOT__clk)));
    }
    co_return;
}

bool Vtb_gf4_decode___024root___trigger_anySet__act(const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_decode___024root___trigger_anySet__act\n"); );
    // Locals
    IData/*31:0*/ n;
    // Body
    n = 0U;
    do {
        if (in[n]) {
            return (1U);
        }
        n = ((IData)(1U) + n);
    } while ((1U > n));
    return (0U);
}

void Vtb_gf4_decode___024root___trigger_orInto__act_vec_vec(VlUnpacked<QData/*63:0*/, 1> &out, const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_decode___024root___trigger_orInto__act_vec_vec\n"); );
    // Locals
    IData/*31:0*/ n;
    // Body
    n = 0U;
    do {
        out[n] = (out[n] | in[n]);
        n = ((IData)(1U) + n);
    } while ((0U >= n));
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_gf4_decode___024root___dump_triggers__act(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag);
#endif  // VL_DEBUG
extern const VlWide<8>/*255:0*/ Vtb_gf4_decode__ConstPool__CONST_h55009368_0;

bool Vtb_gf4_decode___024root___eval_phase__act(Vtb_gf4_decode___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_decode___024root___eval_phase__act\n"); );
    Vtb_gf4_decode__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VactExecute;
    // Body
    {
        // Inlined CFunc: _eval_triggers_vec__act
        vlSelfRef.__VactTriggered[0U] = (QData)((IData)(
                                                        ((vlSelfRef.__VdlySched.awaitingCurrentTime() 
                                                          << 1U) 
                                                         | ((IData)(vlSelfRef.tb_gf4_decode__DOT__clk) 
                                                            & (~ (IData)(vlSelfRef.__Vtrigprevexpr___TOP__tb_gf4_decode__DOT__clk__0))))));
        vlSelfRef.__Vtrigprevexpr___TOP__tb_gf4_decode__DOT__clk__0 
            = vlSelfRef.tb_gf4_decode__DOT__clk;
    }
    Vtb_gf4_decode___024root___trigger_orInto__act_vec_vec(vlSelfRef.__VactTriggered, vlSelfRef.__VactTriggeredAcc);
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vtb_gf4_decode___024root___dump_triggers__act(vlSelfRef.__VactTriggered, "act"s);
    }
#endif
    Vtb_gf4_decode___024root___trigger_orInto__act_vec_vec(vlSelfRef.__VnbaTriggered, vlSelfRef.__VactTriggered);
    __VactExecute = Vtb_gf4_decode___024root___trigger_anySet__act(vlSelfRef.__VactTriggered);
    if (__VactExecute) {
        vlSelfRef.__VactTriggeredAcc.fill(0ULL);
        {
            // Inlined CFunc: _timing_resume
            if ((2ULL & vlSelfRef.__VactTriggered[0U])) {
                vlSelfRef.__VdlySched.resume();
            }
        }
        {
            // Inlined CFunc: _eval_act
            if ((2ULL & vlSelfRef.__VactTriggered[0U])) {
                {
                    // Inlined CFunc: _act_sequent__TOP__0
                    vlSelfRef.tb_gf4_decode__DOT__mag_out_fixed 
                        = (0x000000ffU & Vtb_gf4_decode__ConstPool__CONST_h55009368_0
                           [(0x07ffffffU & (IData)(vlSelfRef.tb_gf4_decode__DOT__idx))]);
                }
            }
        }
    }
    return (__VactExecute);
}

bool Vtb_gf4_decode___024root___eval_phase__inact(Vtb_gf4_decode___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_decode___024root___eval_phase__inact\n"); );
    Vtb_gf4_decode__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VinactExecute;
    // Body
    __VinactExecute = vlSelfRef.__VdlySched.awaitingZeroDelay();
    if (__VinactExecute) {
        VL_FATAL_MT("tb_gf4_decode.v", 26, "", "ZERODLY: Design Verilated with '--no-sched-zero-delay', but #0 delay executed at runtime");
    }
    return (__VinactExecute);
}

void Vtb_gf4_decode___024root___trigger_clear__act(VlUnpacked<QData/*63:0*/, 1> &out) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_decode___024root___trigger_clear__act\n"); );
    // Locals
    IData/*31:0*/ n;
    // Body
    n = 0U;
    do {
        out[n] = 0ULL;
        n = ((IData)(1U) + n);
    } while ((1U > n));
}

bool Vtb_gf4_decode___024root___eval_phase__nba(Vtb_gf4_decode___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_decode___024root___eval_phase__nba\n"); );
    Vtb_gf4_decode__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = Vtb_gf4_decode___024root___trigger_anySet__act(vlSelfRef.__VnbaTriggered);
    if (__VnbaExecute) {
        {
            // Inlined CFunc: _eval_nba
            if ((1ULL & vlSelfRef.__VnbaTriggered[0U])) {
                {
                    // Inlined CFunc: _nba_sequent__TOP__0
                    SData/*8:0*/ __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___VdlyVal__tb_gf4_decode__DOT__dut_prog__DOT__table_mem__v0;
                    __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___VdlyVal__tb_gf4_decode__DOT__dut_prog__DOT__table_mem__v0 = 0;
                    CData/*2:0*/ __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___VdlyDim0__tb_gf4_decode__DOT__dut_prog__DOT__table_mem__v0;
                    __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___VdlyDim0__tb_gf4_decode__DOT__dut_prog__DOT__table_mem__v0 = 0;
                    CData/*0:0*/ __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___VdlySet__tb_gf4_decode__DOT__dut_prog__DOT__table_mem__v0;
                    __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___VdlySet__tb_gf4_decode__DOT__dut_prog__DOT__table_mem__v0 = 0;
                    __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___VdlySet__tb_gf4_decode__DOT__dut_prog__DOT__table_mem__v0 = 0U;
                    if (vlSelfRef.tb_gf4_decode__DOT__cfg_we) {
                        __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___VdlyVal__tb_gf4_decode__DOT__dut_prog__DOT__table_mem__v0 
                            = vlSelfRef.tb_gf4_decode__DOT__cfg_wdata;
                        __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___VdlyDim0__tb_gf4_decode__DOT__dut_prog__DOT__table_mem__v0 
                            = vlSelfRef.tb_gf4_decode__DOT__cfg_waddr;
                        __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___VdlySet__tb_gf4_decode__DOT__dut_prog__DOT__table_mem__v0 = 1U;
                    }
                    if (__Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___VdlySet__tb_gf4_decode__DOT__dut_prog__DOT__table_mem__v0) {
                        vlSelfRef.tb_gf4_decode__DOT__dut_prog__DOT__table_mem[__Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___VdlyDim0__tb_gf4_decode__DOT__dut_prog__DOT__table_mem__v0] 
                            = __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___VdlyVal__tb_gf4_decode__DOT__dut_prog__DOT__table_mem__v0;
                    }
                }
            }
            if ((2ULL & vlSelfRef.__VnbaTriggered[0U])) {
                {
                    // Inlined CFunc: _act_sequent__TOP__0
                    vlSelfRef.tb_gf4_decode__DOT__mag_out_fixed 
                        = (0x000000ffU & Vtb_gf4_decode__ConstPool__CONST_h55009368_0
                           [(0x07ffffffU & (IData)(vlSelfRef.tb_gf4_decode__DOT__idx))]);
                }
            }
        }
        Vtb_gf4_decode___024root___trigger_clear__act(vlSelfRef.__VnbaTriggered);
    }
    return (__VnbaExecute);
}

void Vtb_gf4_decode___024root___eval(Vtb_gf4_decode___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_decode___024root___eval\n"); );
    Vtb_gf4_decode__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    IData/*31:0*/ __VnbaIterCount;
    // Body
    __VnbaIterCount = 0U;
    do {
        if (VL_UNLIKELY(((0x00002710U < __VnbaIterCount)))) {
#ifdef VL_DEBUG
            Vtb_gf4_decode___024root___dump_triggers__act(vlSelfRef.__VnbaTriggered, "nba"s);
#endif
            VL_FATAL_MT("tb_gf4_decode.v", 26, "", "DIDNOTCONVERGE: NBA region did not converge after '--converge-limit' of 10000 tries");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        vlSelfRef.__VinactIterCount = 0U;
        do {
            if (VL_UNLIKELY(((0x00002710U < vlSelfRef.__VinactIterCount)))) {
                VL_FATAL_MT("tb_gf4_decode.v", 26, "", "DIDNOTCONVERGE: Inactive region did not converge after '--converge-limit' of 10000 tries");
            }
            vlSelfRef.__VinactIterCount = ((IData)(1U) 
                                           + vlSelfRef.__VinactIterCount);
            vlSelfRef.__VactIterCount = 0U;
            do {
                if (VL_UNLIKELY(((0x00002710U < vlSelfRef.__VactIterCount)))) {
#ifdef VL_DEBUG
                    Vtb_gf4_decode___024root___dump_triggers__act(vlSelfRef.__VactTriggered, "act"s);
#endif
                    VL_FATAL_MT("tb_gf4_decode.v", 26, "", "DIDNOTCONVERGE: Active region did not converge after '--converge-limit' of 10000 tries");
                }
                vlSelfRef.__VactIterCount = ((IData)(1U) 
                                             + vlSelfRef.__VactIterCount);
                vlSelfRef.__VactPhaseResult = Vtb_gf4_decode___024root___eval_phase__act(vlSelf);
            } while (vlSelfRef.__VactPhaseResult);
            vlSelfRef.__VinactPhaseResult = Vtb_gf4_decode___024root___eval_phase__inact(vlSelf);
        } while (vlSelfRef.__VinactPhaseResult);
        vlSelfRef.__VnbaPhaseResult = Vtb_gf4_decode___024root___eval_phase__nba(vlSelf);
    } while (vlSelfRef.__VnbaPhaseResult);
}

#ifdef VL_DEBUG
void Vtb_gf4_decode___024root___eval_debug_assertions(Vtb_gf4_decode___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_decode___024root___eval_debug_assertions\n"); );
    Vtb_gf4_decode__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}
#endif  // VL_DEBUG
