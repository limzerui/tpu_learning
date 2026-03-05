# Architecture

## High-Level Patterns
- Weight-stationary dataflow (Processing Elements)
- Systolic Array architecture (8x8 grid)
- Unified Buffer for activations/weights/outputs

## Component Boundaries
- **PE**: INT8 MAC unit
- **Systolic Array**: Grid of PEs
- **Sequencer/Decoder**: Control logic
- **Memory**: Unified Buffer + Memory Controller
