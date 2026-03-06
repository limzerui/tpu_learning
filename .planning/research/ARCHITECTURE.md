# Architecture Research: Scaled Systolic Array

## Component Boundaries
- **PE Cluster**: Grouping PEs to share control signals and reduce global fanout.
- **DSP Mapping**: Direct mapping of MAC logic to DSP48E1 (Pre-adder/Multiplier/Accumulator).
- **BRAM Layout**: Activations stored in BRAM rows to match array width.

## Data Flow
- **Diagonal Skewing**: Critical for systolic timing; must be implemented in hardware (FIFOs) or software (memory layout).
- **Pipelined Interconnect**: Registering signals between PEs for timing closure at high speed.

## Build Order
1. **Pipelined PE (Hardware-Ready)**
2. **DSP-Optimized Systolic Grid**
3. **High-Bandwidth Memory Controller**
4. **Instruction Decoder & Sequencer**
