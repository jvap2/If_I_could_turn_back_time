// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtb_gf4_joint_product.h for the primary calling header

#include "Vtb_gf4_joint_product__pch.h"

VL_ATTR_COLD void Vtb_gf4_joint_product___024root___eval_initial__TOP(Vtb_gf4_joint_product___024root* vlSelf);
VlCoroutine Vtb_gf4_joint_product___024root___eval_initial__TOP__Vtiming__0(Vtb_gf4_joint_product___024root* vlSelf);
VlCoroutine Vtb_gf4_joint_product___024root___eval_initial__TOP__Vtiming__1(Vtb_gf4_joint_product___024root* vlSelf);

void Vtb_gf4_joint_product___024root___eval_initial(Vtb_gf4_joint_product___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_joint_product___024root___eval_initial\n"); );
    Vtb_gf4_joint_product__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    Vtb_gf4_joint_product___024root___eval_initial__TOP(vlSelf);
    Vtb_gf4_joint_product___024root___eval_initial__TOP__Vtiming__0(vlSelf);
    Vtb_gf4_joint_product___024root___eval_initial__TOP__Vtiming__1(vlSelf);
}

VlCoroutine Vtb_gf4_joint_product___024root___eval_initial__TOP__Vtiming__0(Vtb_gf4_joint_product___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_joint_product___024root___eval_initial__TOP__Vtiming__0\n"); );
    Vtb_gf4_joint_product__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    IData/*31:0*/ tb_gf4_joint_product__DOT__w;
    tb_gf4_joint_product__DOT__w = 0;
    // Body
    co_await vlSelfRef.__VdlySched.delay(0x0000000000002ee0ULL, 
                                         nullptr, "tb_gf4_joint_product.v", 
                                         133);
    tb_gf4_joint_product__DOT__w = 0U;
    while (VL_GTS_III(32, 8U, tb_gf4_joint_product__DOT__w)) {
        vlSelfRef.tb_gf4_joint_product__DOT__a = 0U;
        while (VL_GTS_III(32, 8U, vlSelfRef.tb_gf4_joint_product__DOT__a)) {
            vlSelfRef.tb_gf4_joint_product__DOT__w_idx 
                = (7U & tb_gf4_joint_product__DOT__w);
            vlSelfRef.tb_gf4_joint_product__DOT__a_idx 
                = (7U & vlSelfRef.tb_gf4_joint_product__DOT__a);
            vlSelfRef.tb_gf4_joint_product__DOT__w_sign = 0U;
            vlSelfRef.tb_gf4_joint_product__DOT__a_sign = 0U;
            co_await vlSelfRef.__VdlySched.delay(0x00000000000003e8ULL, 
                                                 nullptr, 
                                                 "tb_gf4_joint_product.v", 
                                                 142);
            vlSelfRef.tb_gf4_joint_product__DOT__i 
                = (VL_MULS_III(32, (IData)(8U), tb_gf4_joint_product__DOT__w) 
                   + vlSelfRef.tb_gf4_joint_product__DOT__a);
            if ((vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem
                 [(((IData)(vlSelfRef.tb_gf4_joint_product__DOT__w_idx) 
                    << 3U) | (IData)(vlSelfRef.tb_gf4_joint_product__DOT__a_idx))] 
                 != vlSelfRef.tb_gf4_joint_product__DOT__expected_prod
                 [(0x0000003fU & vlSelfRef.tb_gf4_joint_product__DOT__i)])) {
                VL_WRITEF_NX("FAIL: w_idx=%0d a_idx=%0d expected=%0d got=%0d\n",4
                             , '~',32,tb_gf4_joint_product__DOT__w
                             , '~',32,vlSelfRef.tb_gf4_joint_product__DOT__a
                             , '#',17,vlSelfRef.tb_gf4_joint_product__DOT__expected_prod
                             [(0x0000003fU & vlSelfRef.tb_gf4_joint_product__DOT__i)]
                             , '#',17,vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem
                             [(((IData)(vlSelfRef.tb_gf4_joint_product__DOT__w_idx) 
                                << 3U) | (IData)(vlSelfRef.tb_gf4_joint_product__DOT__a_idx))]);
                vlSelfRef.tb_gf4_joint_product__DOT__errors 
                    = ((IData)(1U) + vlSelfRef.tb_gf4_joint_product__DOT__errors);
            } else {
                VL_WRITEF_NX("PASS: w_idx=%0d a_idx=%0d prod_mag=%0d\n",3
                             , '~',32,tb_gf4_joint_product__DOT__w
                             , '~',32,vlSelfRef.tb_gf4_joint_product__DOT__a
                             , '#',17,vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem
                             [(((IData)(vlSelfRef.tb_gf4_joint_product__DOT__w_idx) 
                                << 3U) | (IData)(vlSelfRef.tb_gf4_joint_product__DOT__a_idx))]);
            }
            vlSelfRef.tb_gf4_joint_product__DOT__a 
                = ((IData)(1U) + vlSelfRef.tb_gf4_joint_product__DOT__a);
        }
        tb_gf4_joint_product__DOT__w = ((IData)(1U) 
                                        + tb_gf4_joint_product__DOT__w);
    }
    vlSelfRef.tb_gf4_joint_product__DOT__w_idx = 5U;
    vlSelfRef.tb_gf4_joint_product__DOT__a_idx = 5U;
    vlSelfRef.tb_gf4_joint_product__DOT__w_sign = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__a_sign = 0U;
    co_await vlSelfRef.__VdlySched.delay(0x00000000000003e8ULL, 
                                         nullptr, "tb_gf4_joint_product.v", 
                                         156);
    vlSelfRef.tb_gf4_joint_product__DOT__w_sign = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__a_sign = 1U;
    if (VL_UNLIKELY((vlSelfRef.tb_gf4_joint_product__DOT__prod_sign))) {
        VL_WRITEF_NX("FAIL: sign(0,0) expected=0 got=%0d\n",1
                     , '#',1,vlSelfRef.tb_gf4_joint_product__DOT__prod_sign);
        vlSelfRef.tb_gf4_joint_product__DOT__errors 
            = ((IData)(1U) + vlSelfRef.tb_gf4_joint_product__DOT__errors);
    }
    co_await vlSelfRef.__VdlySched.delay(0x00000000000003e8ULL, 
                                         nullptr, "tb_gf4_joint_product.v", 
                                         158);
    vlSelfRef.tb_gf4_joint_product__DOT__w_sign = 1U;
    vlSelfRef.tb_gf4_joint_product__DOT__a_sign = 0U;
    if (VL_UNLIKELY(((1U & (~ (IData)(vlSelfRef.tb_gf4_joint_product__DOT__prod_sign)))))) {
        VL_WRITEF_NX("FAIL: sign(0,1) expected=1 got=%0d\n",1
                     , '#',1,vlSelfRef.tb_gf4_joint_product__DOT__prod_sign);
        vlSelfRef.tb_gf4_joint_product__DOT__errors 
            = ((IData)(1U) + vlSelfRef.tb_gf4_joint_product__DOT__errors);
    }
    co_await vlSelfRef.__VdlySched.delay(0x00000000000003e8ULL, 
                                         nullptr, "tb_gf4_joint_product.v", 
                                         160);
    if (VL_UNLIKELY(((1U & (~ (IData)(vlSelfRef.tb_gf4_joint_product__DOT__prod_sign)))))) {
        VL_WRITEF_NX("FAIL: sign(1,0) expected=1 got=%0d\n",1
                     , '#',1,vlSelfRef.tb_gf4_joint_product__DOT__prod_sign);
        vlSelfRef.tb_gf4_joint_product__DOT__errors 
            = ((IData)(1U) + vlSelfRef.tb_gf4_joint_product__DOT__errors);
    }
    vlSelfRef.tb_gf4_joint_product__DOT__w_sign = 1U;
    vlSelfRef.tb_gf4_joint_product__DOT__a_sign = 1U;
    co_await vlSelfRef.__VdlySched.delay(0x00000000000003e8ULL, 
                                         nullptr, "tb_gf4_joint_product.v", 
                                         162);
    if (VL_UNLIKELY((vlSelfRef.tb_gf4_joint_product__DOT__prod_sign))) {
        VL_WRITEF_NX("FAIL: sign(1,1) expected=0 got=%0d\n",1
                     , '#',1,vlSelfRef.tb_gf4_joint_product__DOT__prod_sign);
        vlSelfRef.tb_gf4_joint_product__DOT__errors 
            = ((IData)(1U) + vlSelfRef.tb_gf4_joint_product__DOT__errors);
    }
    vlSelfRef.tb_gf4_joint_product__DOT__cfg_we = 1U;
    vlSelfRef.tb_gf4_joint_product__DOT__cfg_waddr = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__cfg_wdata = 0x0001869fU;
    if (VL_UNLIKELY(((0U == vlSelfRef.tb_gf4_joint_product__DOT__errors)))) {
        VL_WRITEF_NX("PASS: sign XOR combination correct across all 4 cases\n",0);
    }
    co_await vlSelfRef.__VdlySched.delay(0x0000000000002710ULL, 
                                         nullptr, "tb_gf4_joint_product.v", 
                                         171);
    vlSelfRef.tb_gf4_joint_product__DOT__cfg_we = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__w_idx = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__a_idx = 0U;
    co_await vlSelfRef.__VdlySched.delay(0x00000000000003e8ULL, 
                                         nullptr, "tb_gf4_joint_product.v", 
                                         174);
    if ((0x0001869fU != vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem
         [(((IData)(vlSelfRef.tb_gf4_joint_product__DOT__w_idx) 
            << 3U) | (IData)(vlSelfRef.tb_gf4_joint_product__DOT__a_idx))])) {
        VL_WRITEF_NX("FAIL: cfg write to addr=0 expected=99999 got=%0d\n",1
                     , '#',17,vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem
                     [(((IData)(vlSelfRef.tb_gf4_joint_product__DOT__w_idx) 
                        << 3U) | (IData)(vlSelfRef.tb_gf4_joint_product__DOT__a_idx))]);
        vlSelfRef.tb_gf4_joint_product__DOT__errors 
            = ((IData)(1U) + vlSelfRef.tb_gf4_joint_product__DOT__errors);
    } else {
        VL_WRITEF_NX("PASS: cfg write readback addr=0 -> %0d\n",1
                     , '#',17,vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem
                     [(((IData)(vlSelfRef.tb_gf4_joint_product__DOT__w_idx) 
                        << 3U) | (IData)(vlSelfRef.tb_gf4_joint_product__DOT__a_idx))]);
    }
    if ((0U == vlSelfRef.tb_gf4_joint_product__DOT__errors)) {
        VL_WRITEF_NX("ALL CHECKS PASSED\n",0);
    } else {
        VL_WRITEF_NX("%0d CHECK(S) FAILED\n",1, '~',32,vlSelfRef.tb_gf4_joint_product__DOT__errors);
    }
    VL_FINISH_MT("tb_gf4_joint_product.v", 187, "");
    co_return;
}

VlCoroutine Vtb_gf4_joint_product___024root___eval_initial__TOP__Vtiming__1(Vtb_gf4_joint_product___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_joint_product___024root___eval_initial__TOP__Vtiming__1\n"); );
    Vtb_gf4_joint_product__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    while (VL_LIKELY(!vlSymsp->_vm_contextp__->gotFinish())) {
        co_await vlSelfRef.__VdlySched.delay(0x0000000000001388ULL, 
                                             nullptr, 
                                             "tb_gf4_joint_product.v", 
                                             129);
        vlSelfRef.tb_gf4_joint_product__DOT__clk = 
            (1U & (~ (IData)(vlSelfRef.tb_gf4_joint_product__DOT__clk)));
    }
    co_return;
}

bool Vtb_gf4_joint_product___024root___trigger_anySet__act(const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_joint_product___024root___trigger_anySet__act\n"); );
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

void Vtb_gf4_joint_product___024root___trigger_orInto__act_vec_vec(VlUnpacked<QData/*63:0*/, 1> &out, const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_joint_product___024root___trigger_orInto__act_vec_vec\n"); );
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
VL_ATTR_COLD void Vtb_gf4_joint_product___024root___dump_triggers__act(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag);
#endif  // VL_DEBUG

bool Vtb_gf4_joint_product___024root___eval_phase__act(Vtb_gf4_joint_product___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_joint_product___024root___eval_phase__act\n"); );
    Vtb_gf4_joint_product__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VactExecute;
    // Body
    {
        // Inlined CFunc: _eval_triggers_vec__act
        vlSelfRef.__VactTriggered[0U] = (QData)((IData)(
                                                        ((vlSelfRef.__VdlySched.awaitingCurrentTime() 
                                                          << 1U) 
                                                         | ((IData)(vlSelfRef.tb_gf4_joint_product__DOT__clk) 
                                                            & (~ (IData)(vlSelfRef.__Vtrigprevexpr___TOP__tb_gf4_joint_product__DOT__clk__0))))));
        vlSelfRef.__Vtrigprevexpr___TOP__tb_gf4_joint_product__DOT__clk__0 
            = vlSelfRef.tb_gf4_joint_product__DOT__clk;
    }
    Vtb_gf4_joint_product___024root___trigger_orInto__act_vec_vec(vlSelfRef.__VactTriggered, vlSelfRef.__VactTriggeredAcc);
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vtb_gf4_joint_product___024root___dump_triggers__act(vlSelfRef.__VactTriggered, "act"s);
    }
#endif
    Vtb_gf4_joint_product___024root___trigger_orInto__act_vec_vec(vlSelfRef.__VnbaTriggered, vlSelfRef.__VactTriggered);
    __VactExecute = Vtb_gf4_joint_product___024root___trigger_anySet__act(vlSelfRef.__VactTriggered);
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
                    vlSelfRef.tb_gf4_joint_product__DOT__prod_sign 
                        = ((IData)(vlSelfRef.tb_gf4_joint_product__DOT__w_sign) 
                           ^ (IData)(vlSelfRef.tb_gf4_joint_product__DOT__a_sign));
                }
            }
        }
    }
    return (__VactExecute);
}

bool Vtb_gf4_joint_product___024root___eval_phase__inact(Vtb_gf4_joint_product___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_joint_product___024root___eval_phase__inact\n"); );
    Vtb_gf4_joint_product__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VinactExecute;
    // Body
    __VinactExecute = vlSelfRef.__VdlySched.awaitingZeroDelay();
    if (__VinactExecute) {
        VL_FATAL_MT("tb_gf4_joint_product.v", 23, "", "ZERODLY: Design Verilated with '--no-sched-zero-delay', but #0 delay executed at runtime");
    }
    return (__VinactExecute);
}

void Vtb_gf4_joint_product___024root___trigger_clear__act(VlUnpacked<QData/*63:0*/, 1> &out) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_joint_product___024root___trigger_clear__act\n"); );
    // Locals
    IData/*31:0*/ n;
    // Body
    n = 0U;
    do {
        out[n] = 0ULL;
        n = ((IData)(1U) + n);
    } while ((1U > n));
}

bool Vtb_gf4_joint_product___024root___eval_phase__nba(Vtb_gf4_joint_product___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_joint_product___024root___eval_phase__nba\n"); );
    Vtb_gf4_joint_product__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = Vtb_gf4_joint_product___024root___trigger_anySet__act(vlSelfRef.__VnbaTriggered);
    if (__VnbaExecute) {
        {
            // Inlined CFunc: _eval_nba
            if ((1ULL & vlSelfRef.__VnbaTriggered[0U])) {
                {
                    // Inlined CFunc: _nba_sequent__TOP__0
                    IData/*16:0*/ __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___VdlyVal__tb_gf4_joint_product__DOT__dut__DOT__table_mem__v0;
                    __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___VdlyVal__tb_gf4_joint_product__DOT__dut__DOT__table_mem__v0 = 0;
                    CData/*5:0*/ __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___VdlyDim0__tb_gf4_joint_product__DOT__dut__DOT__table_mem__v0;
                    __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___VdlyDim0__tb_gf4_joint_product__DOT__dut__DOT__table_mem__v0 = 0;
                    CData/*0:0*/ __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___VdlySet__tb_gf4_joint_product__DOT__dut__DOT__table_mem__v0;
                    __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___VdlySet__tb_gf4_joint_product__DOT__dut__DOT__table_mem__v0 = 0;
                    __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___VdlySet__tb_gf4_joint_product__DOT__dut__DOT__table_mem__v0 = 0U;
                    if (vlSelfRef.tb_gf4_joint_product__DOT__cfg_we) {
                        __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___VdlyVal__tb_gf4_joint_product__DOT__dut__DOT__table_mem__v0 
                            = vlSelfRef.tb_gf4_joint_product__DOT__cfg_wdata;
                        __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___VdlyDim0__tb_gf4_joint_product__DOT__dut__DOT__table_mem__v0 
                            = vlSelfRef.tb_gf4_joint_product__DOT__cfg_waddr;
                        __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___VdlySet__tb_gf4_joint_product__DOT__dut__DOT__table_mem__v0 = 1U;
                    }
                    if (__Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___VdlySet__tb_gf4_joint_product__DOT__dut__DOT__table_mem__v0) {
                        vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[__Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___VdlyDim0__tb_gf4_joint_product__DOT__dut__DOT__table_mem__v0] 
                            = __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___VdlyVal__tb_gf4_joint_product__DOT__dut__DOT__table_mem__v0;
                    }
                }
            }
            if ((2ULL & vlSelfRef.__VnbaTriggered[0U])) {
                {
                    // Inlined CFunc: _act_sequent__TOP__0
                    vlSelfRef.tb_gf4_joint_product__DOT__prod_sign 
                        = ((IData)(vlSelfRef.tb_gf4_joint_product__DOT__w_sign) 
                           ^ (IData)(vlSelfRef.tb_gf4_joint_product__DOT__a_sign));
                }
            }
        }
        Vtb_gf4_joint_product___024root___trigger_clear__act(vlSelfRef.__VnbaTriggered);
    }
    return (__VnbaExecute);
}

void Vtb_gf4_joint_product___024root___eval(Vtb_gf4_joint_product___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_joint_product___024root___eval\n"); );
    Vtb_gf4_joint_product__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    IData/*31:0*/ __VnbaIterCount;
    // Body
    __VnbaIterCount = 0U;
    do {
        if (VL_UNLIKELY(((0x00002710U < __VnbaIterCount)))) {
#ifdef VL_DEBUG
            Vtb_gf4_joint_product___024root___dump_triggers__act(vlSelfRef.__VnbaTriggered, "nba"s);
#endif
            VL_FATAL_MT("tb_gf4_joint_product.v", 23, "", "DIDNOTCONVERGE: NBA region did not converge after '--converge-limit' of 10000 tries");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        vlSelfRef.__VinactIterCount = 0U;
        do {
            if (VL_UNLIKELY(((0x00002710U < vlSelfRef.__VinactIterCount)))) {
                VL_FATAL_MT("tb_gf4_joint_product.v", 23, "", "DIDNOTCONVERGE: Inactive region did not converge after '--converge-limit' of 10000 tries");
            }
            vlSelfRef.__VinactIterCount = ((IData)(1U) 
                                           + vlSelfRef.__VinactIterCount);
            vlSelfRef.__VactIterCount = 0U;
            do {
                if (VL_UNLIKELY(((0x00002710U < vlSelfRef.__VactIterCount)))) {
#ifdef VL_DEBUG
                    Vtb_gf4_joint_product___024root___dump_triggers__act(vlSelfRef.__VactTriggered, "act"s);
#endif
                    VL_FATAL_MT("tb_gf4_joint_product.v", 23, "", "DIDNOTCONVERGE: Active region did not converge after '--converge-limit' of 10000 tries");
                }
                vlSelfRef.__VactIterCount = ((IData)(1U) 
                                             + vlSelfRef.__VactIterCount);
                vlSelfRef.__VactPhaseResult = Vtb_gf4_joint_product___024root___eval_phase__act(vlSelf);
            } while (vlSelfRef.__VactPhaseResult);
            vlSelfRef.__VinactPhaseResult = Vtb_gf4_joint_product___024root___eval_phase__inact(vlSelf);
        } while (vlSelfRef.__VinactPhaseResult);
        vlSelfRef.__VnbaPhaseResult = Vtb_gf4_joint_product___024root___eval_phase__nba(vlSelf);
    } while (vlSelfRef.__VnbaPhaseResult);
}

#ifdef VL_DEBUG
void Vtb_gf4_joint_product___024root___eval_debug_assertions(Vtb_gf4_joint_product___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_joint_product___024root___eval_debug_assertions\n"); );
    Vtb_gf4_joint_product__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}
#endif  // VL_DEBUG
