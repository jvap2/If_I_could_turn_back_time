// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtb_gf4_joint_product.h for the primary calling header

#include "Vtb_gf4_joint_product__pch.h"

VL_ATTR_COLD void Vtb_gf4_joint_product___024root___eval_static(Vtb_gf4_joint_product___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_joint_product___024root___eval_static\n"); );
    Vtb_gf4_joint_product__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    {
        // Inlined CFunc: _eval_static__TOP
        vlSelfRef.tb_gf4_joint_product__DOT__clk = 0U;
        vlSelfRef.tb_gf4_joint_product__DOT__cfg_we = 0U;
        vlSelfRef.tb_gf4_joint_product__DOT__cfg_waddr = 0U;
        vlSelfRef.tb_gf4_joint_product__DOT__cfg_wdata = 0U;
        vlSelfRef.tb_gf4_joint_product__DOT__w_idx = 0U;
        vlSelfRef.tb_gf4_joint_product__DOT__w_sign = 0U;
        vlSelfRef.tb_gf4_joint_product__DOT__a_idx = 0U;
        vlSelfRef.tb_gf4_joint_product__DOT__a_sign = 0U;
        vlSelfRef.tb_gf4_joint_product__DOT__errors = 0U;
    }
    vlSelfRef.__Vtrigprevexpr___TOP__tb_gf4_joint_product__DOT__clk__0 = 0U;
    do {
        vlSelfRef.__VactTriggeredAcc[vlSelfRef.__Vi] 
            = vlSelfRef.__VactTriggered[vlSelfRef.__Vi];
        vlSelfRef.__Vi = ((IData)(1U) + vlSelfRef.__Vi);
    } while ((0U >= vlSelfRef.__Vi));
}

VL_ATTR_COLD void Vtb_gf4_joint_product___024root___eval_initial__TOP(Vtb_gf4_joint_product___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_joint_product___024root___eval_initial__TOP\n"); );
    Vtb_gf4_joint_product__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[0U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[1U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[2U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[3U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[4U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[5U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[6U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[7U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[8U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[9U] = 0x00000190U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[10U] = 0x00000384U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[11U] = 0x000005a0U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[12U] = 0x000007e4U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[13U] = 0x00000a78U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[14U] = 0x00000de8U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[15U] = 0x00001400U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[16U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[17U] = 0x00000384U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[18U] = 0x000007e9U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[19U] = 0x00000ca8U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[20U] = 0x000011c1U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[21U] = 0x0000178eU;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[22U] = 0x00001f4aU;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[23U] = 0x00002d00U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[24U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[25U] = 0x000005a0U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[26U] = 0x00000ca8U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[27U] = 0x00001440U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[28U] = 0x00001c68U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[29U] = 0x000025b0U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[30U] = 0x00003210U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[31U] = 0x00004800U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[32U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[33U] = 0x000007e4U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[34U] = 0x000011c1U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[35U] = 0x00001c68U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[36U] = 0x000027d9U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[37U] = 0x000034deU;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[38U] = 0x0000463aU;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[39U] = 0x00006500U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[40U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[41U] = 0x00000a78U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[42U] = 0x0000178eU;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[43U] = 0x000025b0U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[44U] = 0x000034deU;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[45U] = 0x00004624U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[46U] = 0x00005d2cU;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[47U] = 0x00008600U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[48U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[49U] = 0x00000de8U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[50U] = 0x00001f4aU;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[51U] = 0x00003210U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[52U] = 0x0000463aU;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[53U] = 0x00005d2cU;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[54U] = 0x00007bc4U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[55U] = 0x0000b200U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[56U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[57U] = 0x00001400U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[58U] = 0x00002d00U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[59U] = 0x00004800U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[60U] = 0x00006500U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[61U] = 0x00008600U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[62U] = 0x0000b200U;
    vlSelfRef.tb_gf4_joint_product__DOT__expected_prod[63U] = 0x00010000U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[0U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[1U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[2U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[3U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[4U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[5U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[6U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[7U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[8U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[9U] = 0x00000190U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[10U] = 0x00000384U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[11U] = 0x000005a0U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[12U] = 0x000007e4U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[13U] = 0x00000a78U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[14U] = 0x00000de8U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[15U] = 0x00001400U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[16U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[17U] = 0x00000384U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[18U] = 0x000007e9U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[19U] = 0x00000ca8U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[20U] = 0x000011c1U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[21U] = 0x0000178eU;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[22U] = 0x00001f4aU;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[23U] = 0x00002d00U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[24U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[25U] = 0x000005a0U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[26U] = 0x00000ca8U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[27U] = 0x00001440U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[28U] = 0x00001c68U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[29U] = 0x000025b0U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[30U] = 0x00003210U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[31U] = 0x00004800U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[32U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[33U] = 0x000007e4U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[34U] = 0x000011c1U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[35U] = 0x00001c68U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[36U] = 0x000027d9U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[37U] = 0x000034deU;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[38U] = 0x0000463aU;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[39U] = 0x00006500U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[40U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[41U] = 0x00000a78U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[42U] = 0x0000178eU;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[43U] = 0x000025b0U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[44U] = 0x000034deU;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[45U] = 0x00004624U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[46U] = 0x00005d2cU;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[47U] = 0x00008600U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[48U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[49U] = 0x00000de8U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[50U] = 0x00001f4aU;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[51U] = 0x00003210U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[52U] = 0x0000463aU;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[53U] = 0x00005d2cU;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[54U] = 0x00007bc4U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[55U] = 0x0000b200U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[56U] = 0U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[57U] = 0x00001400U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[58U] = 0x00002d00U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[59U] = 0x00004800U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[60U] = 0x00006500U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[61U] = 0x00008600U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[62U] = 0x0000b200U;
    vlSelfRef.tb_gf4_joint_product__DOT__dut__DOT__table_mem[63U] = 0x00010000U;
}

VL_ATTR_COLD void Vtb_gf4_joint_product___024root___eval_final(Vtb_gf4_joint_product___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_joint_product___024root___eval_final\n"); );
    Vtb_gf4_joint_product__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_gf4_joint_product___024root___dump_triggers__stl(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag);
#endif  // VL_DEBUG
VL_ATTR_COLD bool Vtb_gf4_joint_product___024root___eval_phase__stl(Vtb_gf4_joint_product___024root* vlSelf);

VL_ATTR_COLD void Vtb_gf4_joint_product___024root___eval_settle(Vtb_gf4_joint_product___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_joint_product___024root___eval_settle\n"); );
    Vtb_gf4_joint_product__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    IData/*31:0*/ __VstlIterCount;
    // Body
    __VstlIterCount = 0U;
    vlSelfRef.__VstlFirstIteration = 1U;
    do {
        if (VL_UNLIKELY(((0x00002710U < __VstlIterCount)))) {
#ifdef VL_DEBUG
            Vtb_gf4_joint_product___024root___dump_triggers__stl(vlSelfRef.__VstlTriggered, "stl"s);
#endif
            VL_FATAL_MT("tb_gf4_joint_product.v", 23, "", "DIDNOTCONVERGE: Settle region did not converge after '--converge-limit' of 10000 tries");
        }
        __VstlIterCount = ((IData)(1U) + __VstlIterCount);
        vlSelfRef.__VstlPhaseResult = Vtb_gf4_joint_product___024root___eval_phase__stl(vlSelf);
        vlSelfRef.__VstlFirstIteration = 0U;
    } while (vlSelfRef.__VstlPhaseResult);
}

VL_ATTR_COLD bool Vtb_gf4_joint_product___024root___trigger_anySet__stl(const VlUnpacked<QData/*63:0*/, 1> &in);

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_gf4_joint_product___024root___dump_triggers__stl(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_joint_product___024root___dump_triggers__stl\n"); );
    // Body
    if ((1U & (~ (IData)(Vtb_gf4_joint_product___024root___trigger_anySet__stl(triggers))))) {
        VL_DBG_MSGS("         No '" + tag + "' region triggers active\n");
    }
    if ((1U & (IData)(triggers[0U]))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 0 is active: Internal 'stl' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD bool Vtb_gf4_joint_product___024root___trigger_anySet__stl(const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_joint_product___024root___trigger_anySet__stl\n"); );
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

VL_ATTR_COLD bool Vtb_gf4_joint_product___024root___eval_phase__stl(Vtb_gf4_joint_product___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_joint_product___024root___eval_phase__stl\n"); );
    Vtb_gf4_joint_product__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    CData/*0:0*/ __VstlExecute;
    // Body
    {
        // Inlined CFunc: _eval_triggers_vec__stl
        vlSelfRef.__VstlTriggered[0U] = ((0xfffffffffffffffeULL 
                                          & vlSelfRef.__VstlTriggered[0U]) 
                                         | (IData)((IData)(vlSelfRef.__VstlFirstIteration)));
    }
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vtb_gf4_joint_product___024root___dump_triggers__stl(vlSelfRef.__VstlTriggered, "stl"s);
    }
#endif
    __VstlExecute = Vtb_gf4_joint_product___024root___trigger_anySet__stl(vlSelfRef.__VstlTriggered);
    if (__VstlExecute) {
        {
            // Inlined CFunc: _eval_stl
            if ((1ULL & vlSelfRef.__VstlTriggered[0U])) {
                {
                    // Inlined CFunc: _act_sequent__TOP__0
                    vlSelfRef.tb_gf4_joint_product__DOT__prod_sign 
                        = ((IData)(vlSelfRef.tb_gf4_joint_product__DOT__w_sign) 
                           ^ (IData)(vlSelfRef.tb_gf4_joint_product__DOT__a_sign));
                }
            }
        }
    }
    return (__VstlExecute);
}

bool Vtb_gf4_joint_product___024root___trigger_anySet__act(const VlUnpacked<QData/*63:0*/, 1> &in);

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_gf4_joint_product___024root___dump_triggers__act(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_joint_product___024root___dump_triggers__act\n"); );
    // Body
    if ((1U & (~ (IData)(Vtb_gf4_joint_product___024root___trigger_anySet__act(triggers))))) {
        VL_DBG_MSGS("         No '" + tag + "' region triggers active\n");
    }
    if ((1U & (IData)(triggers[0U]))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 0 is active: @(posedge tb_gf4_joint_product.clk)\n");
    }
    if ((1U & (IData)((triggers[0U] >> 1U)))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 1 is active: @([true] __VdlySched.awaitingCurrentTime())\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vtb_gf4_joint_product___024root___ctor_var_reset(Vtb_gf4_joint_product___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_joint_product___024root___ctor_var_reset\n"); );
    Vtb_gf4_joint_product__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    const uint64_t __VscopeHash = VL_MURMUR64_HASH(vlSelf->vlNamep);
    vlSelf->tb_gf4_joint_product__DOT__prod_sign = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 12090880342584767230ull);
    vlSelf->tb_gf4_joint_product__DOT__i = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 12948000324720138206ull);
    vlSelf->tb_gf4_joint_product__DOT__a = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 17250172397881800616ull);
    for (int __Vi0 = 0; __Vi0 < 64; ++__Vi0) {
        vlSelf->tb_gf4_joint_product__DOT__expected_prod[__Vi0] = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 718835301210890768ull);
    }
    for (int __Vi0 = 0; __Vi0 < 64; ++__Vi0) {
        vlSelf->tb_gf4_joint_product__DOT__dut__DOT__table_mem[__Vi0] = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 748266730666128211ull);
    }
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->__VstlTriggered[__Vi0] = 0;
    }
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->__VactTriggered[__Vi0] = 0;
    }
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->__VactTriggeredAcc[__Vi0] = 0;
    }
    vlSelf->__Vtrigprevexpr___TOP__tb_gf4_joint_product__DOT__clk__0 = 0;
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->__VnbaTriggered[__Vi0] = 0;
    }
    vlSelf->__Vi = 0;
}
