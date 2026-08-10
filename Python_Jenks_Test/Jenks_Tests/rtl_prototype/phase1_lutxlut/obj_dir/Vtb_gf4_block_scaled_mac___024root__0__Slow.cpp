// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtb_gf4_block_scaled_mac.h for the primary calling header

#include "Vtb_gf4_block_scaled_mac__pch.h"

VL_ATTR_COLD void Vtb_gf4_block_scaled_mac___024root___eval_static(Vtb_gf4_block_scaled_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_block_scaled_mac___024root___eval_static\n"); );
    Vtb_gf4_block_scaled_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    {
        // Inlined CFunc: _eval_static__TOP
        vlSelfRef.tb_gf4_block_scaled_mac__DOT__clk = 0U;
        vlSelfRef.tb_gf4_block_scaled_mac__DOT__rst_n = 0U;
        vlSelfRef.tb_gf4_block_scaled_mac__DOT__w_code = 0U;
        vlSelfRef.tb_gf4_block_scaled_mac__DOT__a_code = 0U;
        vlSelfRef.tb_gf4_block_scaled_mac__DOT__elem_valid = 0U;
        vlSelfRef.tb_gf4_block_scaled_mac__DOT__block_last = 0U;
        vlSelfRef.tb_gf4_block_scaled_mac__DOT__s_w_scale = 0U;
        vlSelfRef.tb_gf4_block_scaled_mac__DOT__s_a_scale = 0U;
        vlSelfRef.tb_gf4_block_scaled_mac__DOT__global_acc_clear = 0U;
        vlSelfRef.tb_gf4_block_scaled_mac__DOT__errors = 0U;
    }
    vlSelfRef.__Vtrigprevexpr___TOP__tb_gf4_block_scaled_mac__DOT__clk__0 = 0U;
    vlSelfRef.__Vtrigprevexpr___TOP__tb_gf4_block_scaled_mac__DOT__rst_n__0 = 0U;
    do {
        vlSelfRef.__VactTriggeredAcc[vlSelfRef.__Vi] 
            = vlSelfRef.__VactTriggered[vlSelfRef.__Vi];
        vlSelfRef.__Vi = ((IData)(1U) + vlSelfRef.__Vi);
    } while ((0U >= vlSelfRef.__Vi));
}

VL_ATTR_COLD void Vtb_gf4_block_scaled_mac___024root___eval_initial__TOP(Vtb_gf4_block_scaled_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_block_scaled_mac___024root___eval_initial__TOP\n"); );
    Vtb_gf4_block_scaled_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_w[0U] = 0U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_a[0U] = 0U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_w[1U] = 0x0bU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_a[1U] = 5U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_w[2U] = 6U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_a[2U] = 0x0aU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_w[3U] = 1U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_a[3U] = 7U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_w[4U] = 0x0cU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_a[4U] = 4U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_w[5U] = 7U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_a[5U] = 9U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_w[6U] = 2U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_a[6U] = 6U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_w[7U] = 0x0dU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_a[7U] = 3U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_w[8U] = 0U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_a[8U] = 8U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_w[9U] = 3U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_a[9U] = 5U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_w[10U] = 0x0eU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_a[10U] = 2U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_w[11U] = 1U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_a[11U] = 0x0fU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_w[12U] = 4U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_a[12U] = 4U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_w[13U] = 0x0fU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_a[13U] = 1U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_w[14U] = 2U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_a[14U] = 0x0eU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_w[15U] = 5U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkA_a[15U] = 3U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_w[0U] = 7U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_a[0U] = 8U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_w[1U] = 0x0eU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_a[1U] = 3U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_w[2U] = 5U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_a[2U] = 0x0eU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_w[3U] = 0x0cU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_a[3U] = 1U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_w[4U] = 3U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_a[4U] = 0x0cU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_w[5U] = 0x0aU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_a[5U] = 7U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_w[6U] = 1U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_a[6U] = 0x0aU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_w[7U] = 8U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_a[7U] = 5U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_w[8U] = 7U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_a[8U] = 8U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_w[9U] = 0x0eU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_a[9U] = 3U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_w[10U] = 5U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_a[10U] = 0x0eU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_w[11U] = 0x0cU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_a[11U] = 1U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_w[12U] = 3U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_a[12U] = 0x0cU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_w[13U] = 0x0aU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_a[13U] = 7U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_w[14U] = 1U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_a[14U] = 0x0aU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_w[15U] = 8U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__blkB_a[15U] = 5U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[0U] = 0U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[1U] = 0U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[2U] = 0U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[3U] = 0U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[4U] = 0U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[5U] = 0U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[6U] = 0U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[7U] = 0U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[8U] = 0U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[9U] = 0x00000190U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[10U] = 0x00000384U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[11U] = 0x000005a0U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[12U] = 0x000007e4U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[13U] = 0x00000a78U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[14U] = 0x00000de8U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[15U] = 0x00001400U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[16U] = 0U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[17U] = 0x00000384U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[18U] = 0x000007e9U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[19U] = 0x00000ca8U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[20U] = 0x000011c1U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[21U] = 0x0000178eU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[22U] = 0x00001f4aU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[23U] = 0x00002d00U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[24U] = 0U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[25U] = 0x000005a0U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[26U] = 0x00000ca8U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[27U] = 0x00001440U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[28U] = 0x00001c68U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[29U] = 0x000025b0U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[30U] = 0x00003210U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[31U] = 0x00004800U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[32U] = 0U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[33U] = 0x000007e4U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[34U] = 0x000011c1U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[35U] = 0x00001c68U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[36U] = 0x000027d9U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[37U] = 0x000034deU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[38U] = 0x0000463aU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[39U] = 0x00006500U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[40U] = 0U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[41U] = 0x00000a78U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[42U] = 0x0000178eU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[43U] = 0x000025b0U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[44U] = 0x000034deU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[45U] = 0x00004624U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[46U] = 0x00005d2cU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[47U] = 0x00008600U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[48U] = 0U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[49U] = 0x00000de8U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[50U] = 0x00001f4aU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[51U] = 0x00003210U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[52U] = 0x0000463aU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[53U] = 0x00005d2cU;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[54U] = 0x00007bc4U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[55U] = 0x0000b200U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[56U] = 0U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[57U] = 0x00001400U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[58U] = 0x00002d00U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[59U] = 0x00004800U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[60U] = 0x00006500U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[61U] = 0x00008600U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[62U] = 0x0000b200U;
    vlSelfRef.tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[63U] = 0x00010000U;
}

VL_ATTR_COLD void Vtb_gf4_block_scaled_mac___024root___eval_final(Vtb_gf4_block_scaled_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_block_scaled_mac___024root___eval_final\n"); );
    Vtb_gf4_block_scaled_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_gf4_block_scaled_mac___024root___dump_triggers__stl(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag);
#endif  // VL_DEBUG
VL_ATTR_COLD bool Vtb_gf4_block_scaled_mac___024root___eval_phase__stl(Vtb_gf4_block_scaled_mac___024root* vlSelf);

VL_ATTR_COLD void Vtb_gf4_block_scaled_mac___024root___eval_settle(Vtb_gf4_block_scaled_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_block_scaled_mac___024root___eval_settle\n"); );
    Vtb_gf4_block_scaled_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Locals
    IData/*31:0*/ __VstlIterCount;
    // Body
    __VstlIterCount = 0U;
    vlSelfRef.__VstlFirstIteration = 1U;
    do {
        if (VL_UNLIKELY(((0x00002710U < __VstlIterCount)))) {
#ifdef VL_DEBUG
            Vtb_gf4_block_scaled_mac___024root___dump_triggers__stl(vlSelfRef.__VstlTriggered, "stl"s);
#endif
            VL_FATAL_MT("tb_gf4_block_scaled_mac.v", 29, "", "DIDNOTCONVERGE: Settle region did not converge after '--converge-limit' of 10000 tries");
        }
        __VstlIterCount = ((IData)(1U) + __VstlIterCount);
        vlSelfRef.__VstlPhaseResult = Vtb_gf4_block_scaled_mac___024root___eval_phase__stl(vlSelf);
        vlSelfRef.__VstlFirstIteration = 0U;
    } while (vlSelfRef.__VstlPhaseResult);
}

VL_ATTR_COLD bool Vtb_gf4_block_scaled_mac___024root___trigger_anySet__stl(const VlUnpacked<QData/*63:0*/, 1> &in);

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_gf4_block_scaled_mac___024root___dump_triggers__stl(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_block_scaled_mac___024root___dump_triggers__stl\n"); );
    // Body
    if ((1U & (~ (IData)(Vtb_gf4_block_scaled_mac___024root___trigger_anySet__stl(triggers))))) {
        VL_DBG_MSGS("         No '" + tag + "' region triggers active\n");
    }
    if ((1U & (IData)(triggers[0U]))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 0 is active: Internal 'stl' trigger - first iteration\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD bool Vtb_gf4_block_scaled_mac___024root___trigger_anySet__stl(const VlUnpacked<QData/*63:0*/, 1> &in) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_block_scaled_mac___024root___trigger_anySet__stl\n"); );
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

VL_ATTR_COLD bool Vtb_gf4_block_scaled_mac___024root___eval_phase__stl(Vtb_gf4_block_scaled_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_block_scaled_mac___024root___eval_phase__stl\n"); );
    Vtb_gf4_block_scaled_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
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
        Vtb_gf4_block_scaled_mac___024root___dump_triggers__stl(vlSelfRef.__VstlTriggered, "stl"s);
    }
#endif
    __VstlExecute = Vtb_gf4_block_scaled_mac___024root___trigger_anySet__stl(vlSelfRef.__VstlTriggered);
    if (__VstlExecute) {
        {
            // Inlined CFunc: _eval_stl
            if ((1ULL & vlSelfRef.__VstlTriggered[0U])) {
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
    return (__VstlExecute);
}

bool Vtb_gf4_block_scaled_mac___024root___trigger_anySet__act(const VlUnpacked<QData/*63:0*/, 1> &in);

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtb_gf4_block_scaled_mac___024root___dump_triggers__act(const VlUnpacked<QData/*63:0*/, 1> &triggers, const std::string &tag) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_block_scaled_mac___024root___dump_triggers__act\n"); );
    // Body
    if ((1U & (~ (IData)(Vtb_gf4_block_scaled_mac___024root___trigger_anySet__act(triggers))))) {
        VL_DBG_MSGS("         No '" + tag + "' region triggers active\n");
    }
    if ((1U & (IData)(triggers[0U]))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 0 is active: @(posedge tb_gf4_block_scaled_mac.clk)\n");
    }
    if ((1U & (IData)((triggers[0U] >> 1U)))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 1 is active: @(negedge tb_gf4_block_scaled_mac.rst_n)\n");
    }
    if ((1U & (IData)((triggers[0U] >> 2U)))) {
        VL_DBG_MSGS("         '" + tag + "' region trigger index 2 is active: @([true] __VdlySched.awaitingCurrentTime())\n");
    }
}
#endif  // VL_DEBUG

VL_ATTR_COLD void Vtb_gf4_block_scaled_mac___024root___ctor_var_reset(Vtb_gf4_block_scaled_mac___024root* vlSelf) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtb_gf4_block_scaled_mac___024root___ctor_var_reset\n"); );
    Vtb_gf4_block_scaled_mac__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    auto& vlSelfRef = std::ref(*vlSelf).get();
    // Body
    const uint64_t __VscopeHash = VL_MURMUR64_HASH(vlSelf->vlNamep);
    for (int __Vi0 = 0; __Vi0 < 16; ++__Vi0) {
        vlSelf->tb_gf4_block_scaled_mac__DOT__blkA_w[__Vi0] = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 1827533667763767638ull);
    }
    for (int __Vi0 = 0; __Vi0 < 16; ++__Vi0) {
        vlSelf->tb_gf4_block_scaled_mac__DOT__blkA_a[__Vi0] = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 14831972393273512015ull);
    }
    for (int __Vi0 = 0; __Vi0 < 16; ++__Vi0) {
        vlSelf->tb_gf4_block_scaled_mac__DOT__blkB_w[__Vi0] = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 14614526540529488885ull);
    }
    for (int __Vi0 = 0; __Vi0 < 16; ++__Vi0) {
        vlSelf->tb_gf4_block_scaled_mac__DOT__blkB_a[__Vi0] = VL_SCOPED_RAND_RESET_I(4, __VscopeHash, 3468773299967948299ull);
    }
    vlSelf->tb_gf4_block_scaled_mac__DOT__dut__DOT__prod_signed = VL_SCOPED_RAND_RESET_I(18, __VscopeHash, 12374160685703278069ull);
    vlSelf->tb_gf4_block_scaled_mac__DOT__dut__DOT__raw_block_acc = VL_SCOPED_RAND_RESET_I(24, __VscopeHash, 17374055633637939559ull);
    vlSelf->tb_gf4_block_scaled_mac__DOT__dut__DOT__block_sum_reg = VL_SCOPED_RAND_RESET_I(24, __VscopeHash, 13421205771932535330ull);
    vlSelf->tb_gf4_block_scaled_mac__DOT__dut__DOT__s_w_scale_reg = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 4594807235462376957ull);
    vlSelf->tb_gf4_block_scaled_mac__DOT__dut__DOT__s_a_scale_reg = VL_SCOPED_RAND_RESET_I(16, __VscopeHash, 17973258228555339213ull);
    vlSelf->tb_gf4_block_scaled_mac__DOT__dut__DOT__block_done_reg = VL_SCOPED_RAND_RESET_I(1, __VscopeHash, 14047214195302730577ull);
    vlSelf->tb_gf4_block_scaled_mac__DOT__dut__DOT__global_acc_reg = VL_SCOPED_RAND_RESET_I(32, __VscopeHash, 70918189029657744ull);
    for (int __Vi0 = 0; __Vi0 < 64; ++__Vi0) {
        vlSelf->tb_gf4_block_scaled_mac__DOT__dut__DOT__u_joint_table__DOT__table_mem[__Vi0] = VL_SCOPED_RAND_RESET_I(17, __VscopeHash, 13401408562599292307ull);
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
    vlSelf->__Vtrigprevexpr___TOP__tb_gf4_block_scaled_mac__DOT__clk__0 = 0;
    vlSelf->__Vtrigprevexpr___TOP__tb_gf4_block_scaled_mac__DOT__rst_n__0 = 0;
    for (int __Vi0 = 0; __Vi0 < 1; ++__Vi0) {
        vlSelf->__VnbaTriggered[__Vi0] = 0;
    }
    vlSelf->__Vi = 0;
}
