# Pitfalls Research: Simulation to Hardware (Kintex-7)

## Pitfalls
- **High Fanout on Control Signals**: A 16x16 array has 256 PEs; broadcasting 'reset' or 'clock_enable' across all will fail timing.
- **Inefficient DSP Utilization**: Not following the DSP48E1 template results in LUT-based MACs, killing performance and area.
- **Memory Bottlenecks**: Loading a 32x32 array requires 32 bytes/cycle; if memory is slow, array sits idle.

## Prevention Strategy
- **Control Signal Pipelining**: Insert registers in the control tree (pipeline 'broadcasts').
- **DSP Templates**: Use Xilinx macros or inferred templates that match DSP48E1 exactly.
- **Interleaved Memory**: Use multiple BRAM banks to provide parallel access.
