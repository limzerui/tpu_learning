# Stack Research: Kintex-7 FPGA TPU (2025)

## Recommendation
- **Synthesis/Implementation**: Vivado ML Edition (2024.2+) — Best support for Kintex-7 architecture-specific optimizations (DSP48E1, BRAM).
- **Verification**: cocotb (v1.9+) with Icarus Verilog or Verilator (v5.0+) for cycle-accurate sim; Python-based testbench for Pytorch integration.
- **Languages**: SystemVerilog-2012 (strict synthesizable subset) for RTL; Python 3.11+ for toolchain and golden model.
- **Hardware Abstraction**: Xilinx IP (Memory Interface Generator - MIG) for DDR3 interface if using external memory on Kintex-7 board.

## Rationale
- **Vivado**: Necessary for device-specific mapping (DSPs, BRAMs) on Xilinx 7-series.
- **cocotb**: Allows high-level Python modeling (Pytorch) to verify low-level RTL.

## What NOT to use
- **HLS (High-Level Synthesis)**: We want to learn architecture by writing RTL; HLS abstracts away the cycle-by-cycle control we need.
- **Simulation-only constructs**: `initial` blocks for anything other than memory initialization, `#` delays, or non-synthesizable tasks.
