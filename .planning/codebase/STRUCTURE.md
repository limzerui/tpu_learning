# Structure

- `src/`: Custom TPU implementation (memory_controller.sv, pe.sv, systolic_array.sv, unified_buffer.sv, weight_fifo.sv)
- `tiny-tpu/`: Reference repository
  - `src/`: Reference RTL
  - `test/`: cocotb test suite
  - `tiny_tpu/`: Python toolchain (assembler, compiler, simulator, pytorch, visualizer)
  - `models/`: Demo models (attention_head, mnist_mlp, tiny_transformer)
