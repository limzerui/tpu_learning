# Features Research: Optimized TPU

## Table Stakes
- **Weight-Stationary MAC**: 8-bit integer support (INT8) for standard quantized models.
- **Double-Buffered Weights**: Hide weight loading latency behind computation.
- **Unified Buffer**: Dedicated scratchpad for activations/outputs.

## Differentiators (Better than Reference)
- **Scaled Array (16x16 / 32x32)**: Leverages 7-series DSP48E1 slices efficiently.
- **BF16 Support**: Higher precision training-compatible inference.
- **Pipelined Pipeline MAC**: Deep pipelining within the PE to hit higher frequencies (300MHz+ on Kintex-7).
- **Tiling Support**: Native support for matrix sizes larger than the physical array.

## Anti-Features
- **FP64 Support**: Overkill for standard AI inference; wastes DSP resources.
- **Dynamic Scheduling**: Keep it statically scheduled (compiler-led) to save area/timing.
