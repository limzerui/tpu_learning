# Research Summary: Kintex-7 TPU

**Stack**: Vivado 2024.2, SystemVerilog, cocotb, DSP48E1 primitives.
**Table Stakes**: 8x8 INT8 weight-stationary, double-buffered weights, unified scratchpad.
**Watch Out For**: Fanout on control signals, DSP mapping efficiency, and memory bandwidth bottlenecks.

## Core Strategy
Start with a **Pipelined PE** mapped to **DSP48E1**, then build a scalable grid that uses **control signal pipelining** to ensure 300MHz+ timing on Kintex-7.
