# Project Roadmap: Kintex-7 TPU

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|------------------|
| 1 | **PE Mastery** | Master the Processing Element logic (MAC) and design for hardware | REQ-COMP-01 | Cycle-accurate PE test passes; DSP48E1 templates verified. |
| 2 | **Systolic Array Scale** | Build the 16x16 grid and ensure correct systolic timing | REQ-COMP-02, REQ-COMP-03 | Correct matrix results in simulation for small models. |
| 3 | **High-Bandwidth Buffer** | Implement multi-banked BRAM for activations/weights | REQ-MEM-01, REQ-MEM-02 | Sequential access to multiple rows/cols in one clock cycle. |
| 4 | **The Control Path** | Build the 32-bit decoder and FSM sequencer | REQ-CTRL-01, REQ-CTRL-02 | Instruction sequence (Load -> Compute -> Store) executes correctly. |
| 5 | **System Integration** | Integrate all components and verify full system | REQ-VERIF-01 | Full system test pass in simulation with custom memory. |
| 6 | **FPGA Deployment** | Implement on Kintex-7 and achieve timing closure | REQ-DEP-01, REQ-DEP-02 | Bitstream generated; 200MHz+ frequency achieved. |

### Phase Details

**Phase 1: PE Mastery**
Goal: Master the Processing Element logic and design for hardware.
Requirements: REQ-COMP-01 (MAC logic).
Success criteria:
1. PE testbench passes in simulation.
2. DSP slice utilization (DSP48E1) confirmed via trial synthesis.

**Phase 2: Systolic Array Scale**
Goal: Build the 16x16 grid and ensure correct systolic timing.
Requirements: REQ-COMP-02, REQ-COMP-03 (16x16 scaling and data flow).
Success criteria:
1. Weight loading and activation streaming correctly through the array.
2. No timing issues in simulation for the grid interconnect.

[... roadmap continues for all phases ...]
