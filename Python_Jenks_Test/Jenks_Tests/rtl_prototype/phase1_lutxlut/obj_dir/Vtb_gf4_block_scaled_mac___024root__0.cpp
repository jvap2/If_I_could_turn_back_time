// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtb_gf4_block_scaled_mac.h for the primary calling header

#include "Vtb_gf4_block_scaled_mac__pch.h"

VL_ATTR_COLD void Vtb_gf4_block_scaled_mac___024root___eval_initial__TOP(Vtb_gf4_block_scaled_mac___024root* vlSelf);
VlCoroutine Vtb_gf4_block_scaled_mac___024root___eval_initial__TOP__Vtiming__0(Vtb_gf4_block_scaled_mac___024root* vlSelf);
VlCoroutine Vtb_gf4_block_scaled_mac___024root___eval_initial__TOP__Vtiming__1(Vtb_gf4_block_scaled_mac___024root* vlSelf);

void Vtb_gf4_block_scaled_mac___024root___eval_initial(Vtb_gf4_block_scaled_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_block_scaled_mac___024root___eval_initial\n"); );
    Vtb_gf4_block_scaled_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    Vtb_gf4_block_scaled_mac___024root___eval_initial__TOP(vlSelf);
    Vtb_gf4_block_scaled_mac___024root___eval_initial__TOP__Vtiming__0(vlSelf);
    Vtb_gf4_block_scaled_mac___024root___eval_initial__TOP__Vtiming__1(vlSelf);
}

VlCoroutine Vtb_gf4_block_scaled_mac___024root___eval_initial__TOP__Vtiming__0(Vtb_gf4_block_scaled_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_block_scaled_mac___024root___eval_initial__TOP__Vtiming__0\n"); );
    Vtb_gf4_block_scaled_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    IData/*31:0*/ tb_gf4_block_scaled_mac__DOT__k;
    tb_gf4_block_scaled_mac__DOT__k = 0;
    // Body
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__rst_n = 0U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__global_acc_clear = 1U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__elem_valid = 0U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__block_last = 0U;
    co_await vlSelfRef.__VdlySched.delay(0x0000000000002ee0ULL, 
                                         nullptr, "tb_gf4_block_scaled_mac.v", 
                                         120);
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__rst_n = 1U;
    co_await vlSelfRef.__VdlySched.delay(0x0000000000001f40ULL, 
                                         nullptr, "tb_gf4_block_scaled_mac.v", 
                                         121);
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__global_acc_clear = 0U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__s_w_scale = 0x0140U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__s_a_scale = 0x0140U;
    tb_gf4_block_scaled_mac__DOT__k = 0U;
    while (VL_GTS_III(32, 0x00000010U, tb_gf4_block_scaled_mac__DOT__k)) {
        vlSelfRef.tb_gf4_block_scaled_mac__DOT__w_code 
            = vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_w
            [(0x0000000fU & tb_gf4_block_scaled_mac__DOT__k)];
        vlSelfRef.tb_gf4_block_scaled_mac__DOT__a_code 
            = vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_a
            [(0x0000000fU & tb_gf4_block_scaled_mac__DOT__k)];
        vlSelfRef.tb_gf4_block_scaled_mac__DOT__elem_valid = 1U;
        vlSelfRef.tb_gf4_block_scaled_mac__DOT__block_last 
            = (0x0000000fU == tb_gf4_block_scaled_mac__DOT__k);
        co_await vlSelfRef.__VdlySched.delay(0x0000000000002710ULL, 
                                             nullptr, 
                                             "tb_gf4_block_scaled_mac.v", 
                                             132);
        tb_gf4_block_scaled_mac__DOT__k = ((IData)(1U) 
                                           + tb_gf4_block_scaled_mac__DOT__k);
    }
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__elem_valid = 0U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__block_last = 0U;
    co_await vlSelfRef.__VdlySched.delay(0x0000000000004e20ULL, 
                                         nullptr, "tb_gf4_block_scaled_mac.v", 
                                         136);
    if ((0xffffff5fU != vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__global_acc_reg)) {
        VL_WRITEF_NX("FAIL: global_acc after block A expected=-161 got=%0d\n",1
                     , '~',32,vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__global_acc_reg);
        vlSelfRef.tb_gf4_block_scaled_mac__DOT__errors 
            = ((IData)(1U) + vlSelfRef.tb_gf4_block_scaled_mac__DOT__errors);
    } else {
        VL_WRITEF_NX("PASS: global_acc after block A = %0d\n",1
                     , '~',32,vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__global_acc_reg);
    }
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__s_w_scale = 0x0100U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__s_a_scale = 0x00b2U;
    tb_gf4_block_scaled_mac__DOT__k = 0U;
    while (VL_GTS_III(32, 0x00000010U, tb_gf4_block_scaled_mac__DOT__k)) {
        vlSelfRef.tb_gf4_block_scaled_mac__DOT__w_code 
            = vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_w
            [(0x0000000fU & tb_gf4_block_scaled_mac__DOT__k)];
        vlSelfRef.tb_gf4_block_scaled_mac__DOT__a_code 
            = vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_a
            [(0x0000000fU & tb_gf4_block_scaled_mac__DOT__k)];
        vlSelfRef.tb_gf4_block_scaled_mac__DOT__elem_valid = 1U;
        vlSelfRef.tb_gf4_block_scaled_mac__DOT__block_last 
            = (0x0000000fU == tb_gf4_block_scaled_mac__DOT__k);
        co_await vlSelfRef.__VdlySched.delay(0x0000000000002710ULL, 
                                             nullptr, 
                                             "tb_gf4_block_scaled_mac.v", 
                                             153);
        tb_gf4_block_scaled_mac__DOT__k = ((IData)(1U) 
                                           + tb_gf4_block_scaled_mac__DOT__k);
    }
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__elem_valid = 0U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__block_last = 0U;
    co_await vlSelfRef.__VdlySched.delay(0x0000000000004e20ULL, 
                                         nullptr, "tb_gf4_block_scaled_mac.v", 
                                         157);
    if ((0xfffffe21U != vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__global_acc_reg)) {
        VL_WRITEF_NX("FAIL: global_acc after block B expected=-479 got=%0d\n",1
                     , '~',32,vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__global_acc_reg);
        vlSelfRef.tb_gf4_block_scaled_mac__DOT__errors 
            = ((IData)(1U) + vlSelfRef.tb_gf4_block_scaled_mac__DOT__errors);
    } else {
        VL_WRITEF_NX("PASS: global_acc after block B = %0d\n",1
                     , '~',32,vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__global_acc_reg);
    }
    if ((0U == vlSelfRef.tb_gf4_block_scaled_mac__DOT__errors)) {
        VL_WRITEF_NX("ALL CHECKS PASSED\n",0);
    } else {
        VL_WRITEF_NX("%0d CHECK(S) FAILED\n",1, '~',32,vlSelfRef.tb_gf4_block_scaled_mac__DOT__errors);
    }
    VL_FINISH_MT("tb_gf4_block_scaled_mac.v", 171, "");
    co_return;
}

VlCoroutine Vtb_gf4_block_scaled_mac___024root___eval_initial__TOP__Vtiming__1(Vtb_gf4_block_scaled_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_block_scaled_mac___024root___eval_initial__TOP__Vtiming__1\n"); );
    Vtb_gf4_block_scaled_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    while (VL_LIKELY(!vlSymsp->_vm_contextp__->gotFinish())) {
        co_await vlSelfRef.__VdlySched.delay(0x0000000000001388ULL, 
                                             nullptr, 
                                             "tb_gf4_block_scaled_mac.v", 
                                             113);
        vlSelfRef.tb_gf4_block_scaled_mac__DOT__clk 
            = (1U & (~ (IData)(vlSelfRef.tb_gf4_block_scaled_mac__DOT__clk)));
    }
    co_return;
}

bool Vtb_gf4_block_scaled_mac___024root___trigger_anySet__act(const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_block_scaled_mac___024root___trigger_anySet__act\n"); );
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

void Vtb_gf4_block_scaled_mac___024root___trigger_orInto__act_vec_vec(VlUnpacked<QData/*63:0*/, 1> &out, const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_block_scaled_mac___024root___trigger_orInto__act_vec_vec\n"); );
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
VL_ATTR_COLD void Vtb_gf4_block_scaled_mac___024root___dump_triggers__act(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag);
#endif  // VL_DEBUG

bool Vtb_gf4_block_scaled_mac___024root___eval_phase__act(Vtb_gf4_block_scaled_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_block_scaled_mac___024root___eval_phase__act\n"); );
    Vtb_gf4_block_scaled_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VactExecute;
    // Body
    {
        // Inlined CFunc: _eval_triggers_vec__act
        vlSelfRef.__VactTriggered[0U] = (QData)((IData)(
                                                        ((vlSelfRef.__VdlySched.awaitingCurrentTime() 
                                                          << 2U) 
                                                         | ((((~ (IData)(vlSelfRef.tb_gf4_block_scaled_mac__DOT__rst_n)) 
                                                              & (IData)(vlSelfRef.__Vtrigprevexpr___TOP__tb_gf4_block_scaled_mac__DOT__rst_n__0)) 
                                                             << 1U) 
                                                            | ((IData)(vlSelfRef.tb_gf4_block_scaled_mac__DOT__clk) 
                                                               & (~ (IData)(vlSelfRef.__Vtrigprevexpr___TOP__tb_gf4_block_scaled_mac__DOT__clk__0)))))));
        vlSelfRef.__Vtrigprevexpr___TOP__tb_gf4_block_scaled_mac__DOT__clk__0 
            = vlSelfRef.tb_gf4_block_scaled_mac__DOT__clk;
        vlSelfRef.__Vtrigprevexpr___TOP__tb_gf4_block_scaled_mac__DOT__rst_n__0 
            = vlSelfRef.tb_gf4_block_scaled_mac__DOT__rst_n;
    }
    Vtb_gf4_block_scaled_mac___024root___trigger_orInto__act_vec_vec(vlSelfRef.__VactTriggered, vlSelfRef.__VactTriggeredAcc);
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vtb_gf4_block_scaled_mac___024root___dump_triggers__act(vlSelfRef.__VactTriggered, "act"s);
    }
#endif
    Vtb_gf4_block_scaled_mac___024root___trigger_orInto__act_vec_vec(vlSelfRef.__VnbaTriggered, vlSelfRef.__VactTriggered);
    __VactExecute = Vtb_gf4_block_scaled_mac___024root___trigger_anySet__act(vlSelfRef.__VactTriggered);
    if (__VactExecute) {
        vlSelfRef.__VactTriggeredAcc.fill(0ULL);
        {
            // Inlined CFunc: _timing_resume
            if ((4ULL & vlSelfRef.__VactTriggered[0U])) {
                vlSelfRef.__VdlySched.resume();
            }
        }
        {
            // Inlined CFunc: _eval_act
            if ((4ULL & vlSelfRef.__VactTriggered[0U])) {
                {
                    // Inlined CFunc: _act_sequent__TOP__0
                    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__prod_signed 
                        = (0x0003ffffU & ((8U & ((IData)(vlSelfRef.tb_gf4_block_scaled_mac__DOT__w_code) 
                                                 ^ (IData)(vlSelfRef.tb_gf4_block_scaled_mac__DOT__a_code)))
                                           ? (- vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem
                                              [((0x00000038U 
                                                 & ((IData)(vlSelfRef.tb_gf4_block_scaled_mac__DOT__w_code) 
                                                    << 3U)) 
                                                | (7U 
                                                   & (IData)(vlSelfRef.tb_gf4_block_scaled_mac__DOT__a_code)))])
                                           : vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem
                                          [((0x00000038U 
                                             & ((IData)(vlSelfRef.tb_gf4_block_scaled_mac__DOT__w_code) 
                                                << 3U)) 
                                            | (7U & (IData)(vlSelfRef.tb_gf4_block_scaled_mac__DOT__a_code)))]));
                }
            }
        }
    }
    return (__VactExecute);
}

bool Vtb_gf4_block_scaled_mac___024root___eval_phase__inact(Vtb_gf4_block_scaled_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_block_scaled_mac___024root___eval_phase__inact\n"); );
    Vtb_gf4_block_scaled_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VinactExecute;
    // Body
    __VinactExecute = vlSelfRef.__VdlySched.awaitingZeroDelay();
    if (__VinactExecute) {
        VL_FATAL_MT("tb_gf4_block_scaled_mac.v", 29, "", "ZERODLY: Design Verilated with '--no-sched-zero-delay', but #0 delay executed at runtime");
    }
    return (__VinactExecute);
}

void Vtb_gf4_block_scaled_mac___024root___trigger_clear__act(VlUnpacked<QData/*63:0*/, 1> &out) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_block_scaled_mac___024root___trigger_clear__act\n"); );
    // Locals
    IData/*31:0*/ n;
    // Body
    n = 0U;
    do {
        out[n] = 0ULL;
        n = ((IData)(1U) + n);
    } while ((1U > n));
}

bool Vtb_gf4_block_scaled_mac___024root___eval_phase__nba(Vtb_gf4_block_scaled_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_block_scaled_mac___024root___eval_phase__nba\n"); );
    Vtb_gf4_block_scaled_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VnbaExecute;
    // Body
    __VnbaExecute = Vtb_gf4_block_scaled_mac___024root___trigger_anySet__act(vlSelfRef.__VnbaTriggered);
    if (__VnbaExecute) {
        {
            // Inlined CFunc: _eval_nba
            if ((3ULL & vlSelfRef.__VnbaTriggered[0U])) {
                {
                    // Inlined CFunc: _nba_sequent__TOP__0
                    IData/*23:0*/ __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___Vdly__tb_gf4_block_scaled_mac__DOT__dut__DOT__raw_block_acc;
                    __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___Vdly__tb_gf4_block_scaled_mac__DOT__dut__DOT__raw_block_acc = 0;
                    IData/*31:0*/ __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___Vdly__tb_gf4_block_scaled_mac__DOT__dut__DOT__global_acc_reg;
                    __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___Vdly__tb_gf4_block_scaled_mac__DOT__dut__DOT__global_acc_reg = 0;
                    __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___Vdly__tb_gf4_block_scaled_mac__DOT__dut__DOT__raw_block_acc 
                        = vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__raw_block_acc;
                    __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___Vdly__tb_gf4_block_scaled_mac__DOT__dut__DOT__global_acc_reg 
                        = vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__global_acc_reg;
                    if (vlSelfRef.tb_gf4_block_scaled_mac__DOT__rst_n) {
                        if (vlSelfRef.tb_gf4_block_scaled_mac__DOT__global_acc_clear) {
                            __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___Vdly__tb_gf4_block_scaled_mac__DOT__dut__DOT__global_acc_reg = 0U;
                        } else if (vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__block_done_reg) {
                            __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___Vdly__tb_gf4_block_scaled_mac__DOT__dut__DOT__global_acc_reg 
                                = (vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__global_acc_reg 
                                   + (IData)((0x00000000ffffffffULL 
                                              & (VL_MULS_QQQ(56, 
                                                             (0x00ffffffffffffffULL 
                                                              & VL_EXTENDS_QI(56,24, vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__block_sum_reg)), 
                                                             (0x00ffffffffffffffULL 
                                                              & VL_EXTENDS_QI(56,32, 
                                                                              VL_MULS_III(32, 
                                                                                VL_EXTENDS_II(32,16, (IData)(vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__s_w_scale_reg)), 
                                                                                VL_EXTENDS_II(32,16, (IData)(vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__s_a_scale_reg)))))) 
                                                 >> 0x00000018U))));
                        }
                        vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__block_done_reg = 0U;
                        if (vlSelfRef.tb_gf4_block_scaled_mac__DOT__elem_valid) {
                            if (vlSelfRef.tb_gf4_block_scaled_mac__DOT__block_last) {
                                vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__block_done_reg = 1U;
                                vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__s_w_scale_reg 
                                    = vlSelfRef.tb_gf4_block_scaled_mac__DOT__s_w_scale;
                                vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__s_a_scale_reg 
                                    = vlSelfRef.tb_gf4_block_scaled_mac__DOT__s_a_scale;
                                vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__block_sum_reg 
                                    = (0x00ffffffU 
                                       & (vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__raw_block_acc 
                                          + VL_EXTENDS_II(24,18, vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__prod_signed)));
                                __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___Vdly__tb_gf4_block_scaled_mac__DOT__dut__DOT__raw_block_acc = 0U;
                            } else {
                                __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___Vdly__tb_gf4_block_scaled_mac__DOT__dut__DOT__raw_block_acc 
                                    = (0x00ffffffU 
                                       & (vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__raw_block_acc 
                                          + VL_EXTENDS_II(24,18, vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__prod_signed)));
                            }
                        }
                    } else {
                        __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___Vdly__tb_gf4_block_scaled_mac__DOT__dut__DOT__global_acc_reg = 0U;
                        vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__block_done_reg = 0U;
                        vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__s_w_scale_reg = 0U;
                        vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__s_a_scale_reg = 0U;
                        __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___Vdly__tb_gf4_block_scaled_mac__DOT__dut__DOT__raw_block_acc = 0U;
                        vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__block_sum_reg = 0U;
                    }
                    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__global_acc_reg 
                        = __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___Vdly__tb_gf4_block_scaled_mac__DOT__dut__DOT__global_acc_reg;
                    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__raw_block_acc 
                        = __Vinline_0__eval_nba___Vinline_0__nba_sequent__TOP__0___Vdly__tb_gf4_block_scaled_mac__DOT__dut__DOT__raw_block_acc;
                }
            }
            if ((4ULL & vlSelfRef.__VnbaTriggered[0U])) {
                {
                    // Inlined CFunc: _act_sequent__TOP__0
                    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__prod_signed 
                        = (0x0003ffffU & ((8U & ((IData)(vlSelfRef.tb_gf4_block_scaled_mac__DOT__w_code) 
                                                 ^ (IData)(vlSelfRef.tb_gf4_block_scaled_mac__DOT__a_code)))
                                           ? (- vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem
                                              [((0x00000038U 
                                                 & ((IData)(vlSelfRef.tb_gf4_block_scaled_mac__DOT__w_code) 
                                                    << 3U)) 
                                                | (7U 
                                                   & (IData)(vlSelfRef.tb_gf4_block_scaled_mac__DOT__a_code)))])
                                           : vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem
                                          [((0x00000038U 
                                             & ((IData)(vlSelfRef.tb_gf4_block_scaled_mac__DOT__w_code) 
                                                << 3U)) 
                                            | (7U & (IData)(vlSelfRef.tb_gf4_block_scaled_mac__DOT__a_code)))]));
                }
            }
        }
        Vtb_gf4_block_scaled_mac___024root___trigger_clear__act(vlSelfRef.__VnbaTriggered);
    }
    return (__VnbaExecute);
}

void Vtb_gf4_block_scaled_mac___024root___eval(Vtb_gf4_block_scaled_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_block_scaled_mac___024root___eval\n"); );
    Vtb_gf4_block_scaled_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    IData/*31:0*/ __VnbaIterCount;
    // Body
    __VnbaIterCount = 0U;
    do {
        if (VL_UNLIKELY(((0x00002710U < __VnbaIterCount)))) {
#ifdef VL_DEBUG
            Vtb_gf4_block_scaled_mac___024root___dump_triggers__act(vlSelfRef.__VnbaTriggered, "nba"s);
#endif
            VL_FATAL_MT("tb_gf4_block_scaled_mac.v", 29, "", "DIDNOTCONVERGE: NBA region did not converge after '--converge-limit' of 10000 tries");
        }
        __VnbaIterCount = ((IData)(1U) + __VnbaIterCount);
        vlSelfRef.__VinactIterCount = 0U;
        do {
            if (VL_UNLIKELY(((0x00002710U < vlSelfRef.__VinactIterCount)))) {
                VL_FATAL_MT("tb_gf4_block_scaled_mac.v", 29, "", "DIDNOTCONVERGE: Inactive region did not converge after '--converge-limit' of 10000 tries");
            }
            vlSelfRef.__VinactIterCount = ((IData)(1U) 
                                           + vlSelfRef.__VinactIterCount);
            vlSelfRef.__VactIterCount = 0U;
            do {
                if (VL_UNLIKELY(((0x00002710U < vlSelfRef.__VactIterCount)))) {
#ifdef VL_DEBUG
                    Vtb_gf4_block_scaled_mac___024root___dump_triggers__act(vlSelfRef.__VactTriggered, "act"s);
#endif
                    VL_FATAL_MT("tb_gf4_block_scaled_mac.v", 29, "", "DIDNOTCONVERGE: Active region did not converge after '--converge-limit' of 10000 tries");
                }
                vlSelfRef.__VactIterCount = ((IData)(1U) 
                                             + vlSelfRef.__VactIterCount);
                vlSelfRef.__VactPhaseResult = Vtb_gf4_block_scaled_mac___024root___eval_phase__act(vlSelf);
            } while (vlSelfRef.__VactPhaseResult);
            vlSelfRef.__VinactPhaseResult = Vtb_gf4_block_scaled_mac___024root___eval_phase__inact(vlSelf);
        } while (vlSelfRef.__VinactPhaseResult);
        vlSelfRef.__VnbaPhaseResult = Vtb_gf4_block_scaled_mac___024root___eval_phase__nba(vlSelf);
    } while (vlSelfRef.__VnbaPhaseResult);
}

#ifdef VL_DEBUG
void Vtb_gf4_block_scaled_mac___024root___eval_debug_assertions(Vtb_gf4_block_scaled_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_block_scaled_mac___024root___eval_debug_assertions\n"); );
    Vtb_gf4_block_scaled_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}
#endif  // VL_DEBUG
