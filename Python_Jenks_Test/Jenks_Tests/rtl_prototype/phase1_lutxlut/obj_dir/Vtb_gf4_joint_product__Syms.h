// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header,
// unless using verilator public meta comments.

#ifndef VERILATED_VTB_GF4_JOINT_PRODUCT__SYMS_H_
#define VERILATED_VTB_GF4_JOINT_PRODUCT__SYMS_H_  // guard

#include "verilated.h"

// INCLUDE MODEL CLASS

#include "Vtb_gf4_joint_product.h"

// INCLUDE MODULE CLASSES
#include "Vtb_gf4_joint_product___024root.h"

// SYMS CLASS (contains all model state)
class alignas(VL_CACHE_LINE_BYTES) Vtb_gf4_joint_product__Syms final : public VerilatedSyms {
  public:
    // INTERNAL STATE
    Vtb_gf4_joint_product* const __Vm_modelp;
    VlDeleter __Vm_deleter;
    bool __Vm_didInit = false;

    // MODULE INSTANCE STATE
    Vtb_gf4_joint_product___024root TOP;

    // CONSTRUCTORS
    Vtb_gf4_joint_product__Syms(VerilatedContext* contextp, const char* namep, Vtb_gf4_joint_product* modelp);
    ~Vtb_gf4_joint_product__Syms();

    // METHODS
    const char* name() const { return TOP.vlNamep; }
};

#endif  // guard
