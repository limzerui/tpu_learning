# Requirements: Kintex-7 Optimized TPU

## v1 Requirements

### Compute Engine
- [ ] **REQ-COMP-01**: 16x16 INT8 Systolic Array (Weight-Stationary).
- [ ] **REQ-COMP-02**: Pipelined MAC Unit (Synthesizable, targeting high frequency).
- [ ] **REQ-COMP-03**: Support for Weight Loading & Activation Streaming in a 16x16 grid.

### Memory & Control
- [ ] **REQ-MEM-01**: High-Bandwidth Multi-banked Memory Controller (64KB+ BRAM).
- [ ] **REQ-MEM-02**: Weight FIFO with double-buffering logic.
- [ ] **REQ-CTRL-01**: Instruction Decoder (32-bit instructions).
- [ ] **REQ-CTRL-02**: Sequencer FSM for matmul (Load -> Compute -> Store).

### Verification & Deployment
- [ ] **REQ-VERIF-01**: Cycle-accurate cocotb testbench for individual modules (PE, Array, Buffer).
- [ ] **REQ-DEP-01**: Kintex-7 FPGA Implementation (Vivado Synthesis, Placement, Routing).
- [ ] **REQ-DEP-02**: Timing closure at target frequency (e.g., 200MHz-300MHz).

---

## v2 Requirements (Deferred)
- **REQ-COMP-04**: BF16 or INT16 precision support.
- **REQ-CTRL-03**: Native hardware Tiling support for larger-than-array matrices.
- **REQ-DEP-03**: AI Inference Benchmarks (e.g., MNIST MLP execution on FPGA).

---

## Out of Scope
- **Non-FPGA Backend**: ASIC flow (standard cell synthesis).
- **Floating-Point Precision**: Keeping complexity low for initial hardware mapping.

---
*Last updated: 2026-03-05*
