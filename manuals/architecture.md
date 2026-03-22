# TinyTPU — System Architecture Manual

> **Purpose:** High-level understanding of the entire project.
> Read this first before any individual module manual.

---

## 1. What the TPU Does

A TPU accelerates one operation: **matrix multiplication**.

```
C = A × B
where:
  A is M × K  (input activations)
  B is K × N  (weights, from training)
  C is M × N  (output activations, passed to the next layer)
```

All values are **INT8** (8-bit integers). The hardware computes in **INT32** internally
to avoid overflow during accumulation, then quantizes the result back to INT8 at the end.

---

## 2. System Component Map

```
┌─────────────────────────────────────────────────────────────────────┐
│                            HOST / CPU                               │
│   Writes matrices A, B into SRAM. Sends `start`. Reads C back.     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ start / done / config
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MATRIX CONTROLLER                              │
│   14-state FSM. The conductor. Drives all other modules.            │
│   Delegates tile iteration arithmetic to the Tiling Controller.     │
└────────┬───────────┬──────────────┬──────────────────┬─────────────┘
         │           │              │                  │
         ▼           ▼              ▼                  ▼
   TILING        WEIGHT FIFO   ACTIVATION FIFO    ACCUMULATOR
   CONTROLLER    Prefetches     Prefetches &        BUFFER
                 B tiles from   skews A tiles       Captures INT32
   Iterates      SRAM.          for diagonal        partial sums.
   m, n, k       Double-        dataflow.           Quantizes to
   tile indices  buffered.      Double-buffered.    INT8.
                     │              │                  │
                     └──────────────┤                  │
                                    ▼                  │
                          ┌──────────────────┐         │
                          │  SYSTOLIC ARRAY  │         │
                          │  8×8 PE Grid     │─────────┘
                          │  MACs every clk  │  Results drain south
                          └──────────────────┘
                                    │ (all modules)
                                    ▼
                          ┌──────────────────┐
                          │  UNIFIED BUFFER  │
                          │  (SRAM)          │
                          │  Holds A, B, C   │
                          └──────────────────┘
                                    ▲
                          ┌──────────────────┐
                          │ MEMORY CONTROLLER│
                          │  Arbiter. Grants │
                          │  one port at a   │
                          │  time to modules │
                          └──────────────────┘
```

---

## 3. The Three Processing Phases

Every tile of C goes through three phases:

```
Phase 1: LOAD          Phase 2: COMPUTE         Phase 3: STORE
──────────────         ─────────────────         ──────────────
SRAM → FIFOs           FIFOs → Array             Buffer → SRAM
                        → Accumulator

weight_fifo prefetches  Weights stationary        Accumulator
B tile.                 in PEs.                   quantizes INT32
                                                  → INT8.
activation_fifo         Activations stream
prefetches & skews      left→right.               Memory controller
A tile.                                           DMAs result tile
                        Partial sums flow         back to SRAM.
                        top→bottom and
                        land in accumulator.
```

---

## 4. Data Types and Widths

| Location        | Type     | Width | Notes                        |
|:----------------|:---------|:------|:-----------------------------|
| SRAM (matrices) | INT8     | 8-bit | 4 elements packed per word   |
| PE inputs       | INT8     | 8-bit | Activations + weights        |
| PE multiplier   | INT16    | 16-bit| Product of two INT8s         |
| PE accumulator  | INT32    | 32-bit| Sum of up to K products      |
| Systolic output | INT32    | 32-bit| Partial sums from south edge |
| Accum buffer    | INT32    | 32-bit| Running totals across K-tiles|
| Quantized output| INT8     | 8-bit | After `(sum * scale) >>> 8`  |

---

## 5. Tiled Matrix Multiplication

The systolic array is `N×N` (8×8). Matrices can be much larger.
The solution: chop the problem into tiles and accumulate.

```
C = A × B

         K=16
A(8×16): ┌──────┬──────┐
         │ A₀₀  │ A₀₁  │  ← tile_m=0
         └──────┴──────┘
              ×
         N=8   N=8
B(16×8): ┌──────┐
         │ B₀₀  │  ← tile_k=0
         ├──────┤
         │ B₁₀  │  ← tile_k=1
         └──────┘
              =
C(8×8):  ┌──────┐
         │ C₀₀  │  = A₀₀×B₀₀ + A₀₁×B₁₀
         └──────┘
```

Three nested loops (outer → inner):
```
for tile_m:                 ← rows of C / rows of A
  for tile_n:               ← cols of C / cols of B
    for tile_k:             ← inner dimension (shared)
      C[m][n] += A[m][k] × B[k][n]
```

`first_k_tile=1` → overwrite the accumulator (fresh start for this C cell).  
`last_k_tile=1`  → quantize and write back (C cell is complete).

---

## 6. The Skewing Problem (Why Activation FIFO is Complex)

In a systolic array, Row 0 of activations enters at cycle 0.
Row 1 must enter at cycle 1 (one cycle later). Row 2 at cycle 2.
This "staircase" timing is called **skewing**.

```
Cycle 0:   Row 0 → PE[0][0]          (rows 1,2,... not yet)
Cycle 1:   Row 0 → PE[0][1]
           Row 1 → PE[1][0]
Cycle 2:   Row 0 → PE[0][2]
           Row 1 → PE[1][1]
           Row 2 → PE[2][0]
```

`activation_fifo` uses shift registers to delay each row by its index,
so the data arrives at the array already in the right wave pattern.

---

## 7. The Quantization Step

After all K-tile passes are done for a given C tile:

```
raw_sum (INT32)
    │
    × quant_scale (Q8.8 fixed-point)         ← configured per layer
    │
    ▼
48-bit product
    │
    >>> 8  (arithmetic right shift, removes the fixed-point denominator)
    │
    ▼
32-bit shifted value
    │
    clamp(-128, 127)
    │
    ▼
INT8 output → written to SRAM
```

`quant_scale` is a Q8.8 number: `1.0 = 256`, `0.5 = 128`.
Chosen during model calibration to map the expected range of `raw_sum`
into [-128, 127] without significant loss.

---

## 8. Module Dependency Map

```
tpu_top.sv
├── matrix_controller.sv      ← FSM conductor
│   ├── tiling_controller.sv  ← Loop counter + address math
│   ├── weight_fifo.sv        ← B tile staging
│   ├── activation_fifo.sv    ← A tile staging + skewing
│   ├── systolic_array.sv     ← Compute grid
│   │   └── pe.sv             ← Processing element (MAC unit)
│   └── accumulator_buffer.sv ← Result capture + quantization
└── memory_controller.sv      ← SRAM arbiter
    └── unified_buffer.sv     ← SRAM (Block RAM)
```

---

## 9. Control Signal Quick Reference

| Signal              | Direction         | Meaning                              |
|:--------------------|:------------------|:-------------------------------------|
| `start`             | Host → MatCtrl    | Begin one matmul                     |
| `done`              | MatCtrl → Host    | One-cycle pulse: matmul complete     |
| `busy`              | MatCtrl → Host    | High while working                   |
| `tile_advance`      | MatCtrl → TileCtrl| Move to next tile                    |
| `tile_done`         | TileCtrl → MatCtrl| All tiles exhausted                  |
| `first_k_tile`      | TileCtrl → MatCtrl| First K pass: use overwrite mode     |
| `last_k_tile`       | TileCtrl → MatCtrl| Last K pass: quantize after compute  |
| `weight_prefetch_*` | MatCtrl → WtFIFO  | Load B tile from SRAM                |
| `activation_load_*` | MatCtrl → ActFIFO | Load A tile from SRAM                |
| `array_enable`      | MatCtrl → SysArr  | Clock-gate the array                 |
| `array_weight_load` | MatCtrl → SysArr  | Array: accept weights from FIFO      |
| `acc_results_enable`| MatCtrl → AccBuf  | Capture south-edge outputs           |
| `acc_accumulate_mode`| MatCtrl → AccBuf | Add (vs overwrite) partial sums      |
| `acc_quant_enable`  | MatCtrl → AccBuf  | Trigger quantization FSM             |
| `acc_quant_done`    | AccBuf → MatCtrl  | INT8 results are ready               |
