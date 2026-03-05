# TPU Learning & Kintex-7 FPGA Deployment — Project Overview

## Core Value
Learn and implement a high-performance TPU from scratch, bridging the gap between a Python/SystemVerilog reference (`tiny-tpu`) and a synthesis-ready design optimized for Kintex-7 FPGAs.

## Context
- **Learning Phase**: Understand the reference implementation (8x8 INT8 systolic array).
- **Expansion Phase**: Enhance throughput (larger array), timing (pipelining), memory architecture, and precision (BF16/INT16).
- **Deployment Phase**: Synthesize for Kintex-7, run benchmarks, and execute AI inference.

## Requirements

### Validated (from existing codebase)
- ✓ **Reference Implementation Mapping**: `tiny-tpu` reference structure identified.
- ✓ **Initial Logic**: Basic modules (`pe.sv`, `systolic_array.sv`, `unified_buffer.sv`, etc.) exist in `src/`.
- ✓ **Environment**: cocotb, Icarus Verilog, and PyTorch toolchain available for testing.

### Active (Hypotheses)
- [ ] **PE & Array Deep Dive**: Thorough understanding of weight-stationary MAC logic and systolic data flow.
- [ ] **Hardware-Ready RTL**: Refactor `src/` modules to be fully synthesizable for Kintex-7 (no simulation-only constructs).
- [ ] **Performance Upgrades**: Scaling the array size (16x16+) and adding pipelining for frequency targets.
- [ ] **Precision Support**: Implementation of BF16 or INT16 MAC units.
- [ ] **Memory Controller Optimization**: Reducing latency and increasing bandwidth for activations/weights.
- [ ] **AI Inference Benchmarks**: Running real AI models and measuring TOPS/W.

### Out of Scope
- **Non-FPGA Backend**: ASIC flow (standard cell synthesis) is not the primary focus.
- **High-Level Frameworks**: Direct CUDA/ROCm-like integration (keeping it bare-metal/assembly for learning).

## Key Decisions
| Decision | Rationale | Outcome |
|----------|-----------|---------|
| **Reference-First Learning** | Must master the existing 8x8 INT8 architecture before scaling. | — Pending |
| **Kintex-7 Target** | Professional-grade FPGA with ample DSP/BRAM for a large systolic array. | — Pending |
| **All Optimizations** | Aiming for a "better than reference" design (size, pipelining, memory, precision). | — Pending |

---
*Last updated: 2026-03-05 after initialization*
