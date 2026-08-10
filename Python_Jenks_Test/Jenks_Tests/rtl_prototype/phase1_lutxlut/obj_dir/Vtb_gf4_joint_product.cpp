// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Model implementation (design independent parts)

#include "Vtb_gf4_joint_product__pch.h"

//============================================================
// Constructors

Vtb_gf4_joint_product::Vtb_gf4_joint_product(VerilatedContext* _vcontextp__, const char* _vcname__)
    : VerilatedModel{*_vcontextp__}
    , vlSymsp{new Vtb_gf4_joint_product__Syms(contextp(), _vcname__, this)}
    , rootp{&(vlSymsp->TOP)}
{
    // Register model with the context
    contextp()->addModel(this);
}

Vtb_gf4_joint_product::Vtb_gf4_joint_product(const char* _vcname__)
    : Vtb_gf4_joint_product(Verilated::threadContextp(), _vcname__)
{
}

//============================================================
// Destructor

Vtb_gf4_joint_product::~Vtb_gf4_joint_product() {
    delete vlSymsp;
}

//============================================================
// Evaluation function

#ifdef VL_DEBUG
void Vtb_gf4_joint_product___024root___eval_debug_assertions(Vtb_gf4_joint_product___024root* vlSelf);
#endif  // VL_DEBUG
void Vtb_gf4_joint_product___024root___eval_static(Vtb_gf4_joint_product___024root* vlSelf);
void Vtb_gf4_joint_product___024root___eval_initial(Vtb_gf4_joint_product___024root* vlSelf);
void Vtb_gf4_joint_product___024root___eval_settle(Vtb_gf4_joint_product___024root* vlSelf);
void Vtb_gf4_joint_product___024root___eval(Vtb_gf4_joint_product___024root* vlSelf);

void Vtb_gf4_joint_product::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate Vtb_gf4_joint_product::eval_step\n"); );
#ifdef VL_DEBUG
    // Debug assertions
    Vtb_gf4_joint_product___024root___eval_debug_assertions(&(vlSymsp->TOP));
#endif  // VL_DEBUG
    vlSymsp->__Vm_deleter.deleteAll();
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) {
        VL_DEBUG_IF(VL_DBG_MSGF("+ Initial\n"););
        Vtb_gf4_joint_product___024root___eval_static(&(vlSymsp->TOP));
        Vtb_gf4_joint_product___024root___eval_initial(&(vlSymsp->TOP));
        Vtb_gf4_joint_product___024root___eval_settle(&(vlSymsp->TOP));
        vlSymsp->__Vm_didInit = true;
    }
    VL_DEBUG_IF(VL_DBG_MSGF("+ Eval\n"););
    Vtb_gf4_joint_product___024root___eval(&(vlSymsp->TOP));
    // Evaluate cleanup
    Verilated::endOfEval(vlSymsp->__Vm_evalMsgQp);
}

//============================================================
// Events and timing
bool Vtb_gf4_joint_product::eventsPending() { return !vlSymsp->TOP.__VdlySched.empty() && !contextp()->gotFinish(); }

uint64_t Vtb_gf4_joint_product::nextTimeSlot() { return vlSymsp->TOP.__VdlySched.nextTimeSlot(); }

//============================================================
// Utilities

const char* Vtb_gf4_joint_product::name() const {
    return vlSymsp->name();
}

//============================================================
// Invoke final blocks

void Vtb_gf4_joint_product___024root___eval_final(Vtb_gf4_joint_product___024root* vlSelf);

VL_ATTR_COLD void Vtb_gf4_joint_product::final() {
    contextp()->executingFinal(true);
    Vtb_gf4_joint_product___024root___eval_final(&(vlSymsp->TOP));
    contextp()->executingFinal(false);
}

//============================================================
// Implementations of abstract methods from VerilatedModel

const char* Vtb_gf4_joint_product::hierName() const { return vlSymsp->name(); }
const char* Vtb_gf4_joint_product::modelName() const { return "Vtb_gf4_joint_product"; }
unsigned Vtb_gf4_joint_product::threads() const { return 1; }
void Vtb_gf4_joint_product::prepareClone() const { contextp()->prepareClone(); }
void Vtb_gf4_joint_product::atClone() const {
    contextp()->threadPoolpOnClone();
}
