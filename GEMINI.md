# TPU Learning Project — Single Source of Truth

## Project Goal

Learn TPU architecture from the `tiny-tpu` reference repo, then implement a custom TPU (improved/inspired version) from scratch.

## Reference Repository: tiny-tpu

- **Location**: `tiny-tpu/`
- **Author**: Jaber Jaber / RightNow AI
- **Source**: [GitHub](https://github.com/RightNow-AI/tiny-tpu)
- **Language**: ~3,400 lines of SystemVerilog + Python toolchain
- **License**: MIT

## Architecture Summary

### RTL Modules (`tiny-tpu/src/`)

| Module                  | Lines | Purpose                                                            |
| ----------------------- | ----- | ------------------------------------------------------------------ |
| `pe.sv`                 | 95    | Processing Element — INT8 MAC unit with weight-stationary dataflow |
| `systolic_array.sv`     | 170   | 8×8 grid of PEs, weight loading + activation streaming             |
| `tpu_top.sv`            | 500   | Top-level integration of all components                            |
| `sequencer.sv`          | 298   | 11-state FSM controlling execution pipeline                        |
| `decoder.sv`            | 328   | 32-bit instruction decode → control signals                        |
| `fetcher.sv`            | ~200  | Instruction fetch from program memory                              |
| `unified_buffer.sv`     | 279   | 64KB dual-port SRAM (activations/weights/outputs)                  |
| `memory_controller.sv`  | ~400  | Round-robin arbitration for memory access                          |
| `weight_fifo.sv`        | ~280  | Double-buffered weight loading pipeline                            |
| `activation_fifo.sv`    | ~300  | Diagonal skewing logic for systolic flow                           |
| `matrix_controller.sv`  | 365   | Matmul orchestration (load → compute → drain)                      |
| `accumulator_buffer.sv` | ~300  | INT32 accumulator storage + quantization                           |
| `activation_unit.sv`    | 389   | ReLU, GELU, SiLU implementations                                   |
| `softmax_unit.sv`       | 400   | 4-pass row-wise softmax                                            |
| `layernorm_unit.sv`     | 548   | Layer normalization                                                |
| `tiling_controller.sv`  | ~320  | Tiled computation for matrices > 8×8                               |

### ISA (16 instructions)

`NOP, LOAD_W, LOAD_A, MATMUL, STORE, ACT_RELU, ACT_GELU, ACT_SILU, SOFTMAX, ADD, LAYERNORM, TRANSPOSE, SCALE, SYNC, LOOP, HALT`

### Memory Map (64KB Unified Buffer)

- `0x0000–0x4FFF`: Activations (20KB)
- `0x5000–0xAFFF`: Weights (24KB)
- `0xB000–0xFFFF`: Outputs (20KB)

### Python Toolchain (`tiny-tpu/tiny_tpu/`)

- **assembler/** — Lexer → Parser → CodeGen pipeline for TPU assembly
- **compiler/** — High-level compiler (attention, matmul, transformer blocks)
- **simulator/** — Cycle-accurate Python simulator (651 lines)
- **pytorch/** — Model extractor + INT8 quantizer
- **visualizer/** — (placeholder)

### Test Suite (`tiny-tpu/test/`)

15 cocotb test files covering every module. Uses sv2v + Icarus Verilog + cocotb.

### Demo Models (`tiny-tpu/models/`)

- **attention_head** — Q×K^T×V single-head attention
- **mnist_mlp** — MLP inference on MNIST
- **tiny_transformer** — Full transformer block

## Build & Run

```bash
# Prerequisites
brew install icarus-verilog
pip install cocotb cocotb-bus
# sv2v from https://github.com/zachjs/sv2v/releases

# Run tests
make test_pe           # Test processing element
make test_systolic     # Test systolic array
make test_matmul       # Test matrix multiplication
make test_all          # Full test suite
```

## Current Phase

**Phase 1: Learning & Understanding** — Reading and studying the reference implementation.

## Status Log

- 2026-02-23: Cloned repo, completed full codebase exploration, created learning roadmap.
