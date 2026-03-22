# matrix_controller.sv — Module Manual

> **Role:** The FSM conductor. Orchestrates all other modules to execute
> one complete tiled matrix multiplication (`C = A × B`).

---

## 1. What Problem Does This Solve?

You have five powerful components sitting idle:
- `weight_fifo` — can fetch and stage weight tiles
- `activation_fifo` — can fetch and skew activation tiles
- `systolic_array` — can multiply and accumulate at full speed
- `accumulator_buffer` — can hold and quantize partial sums
- `tiling_controller` — knows which tile to process next

**Without this module, nothing happens.** The matrix controller is the
sequence of `start signals` and `wait conditions` that makes all five
work together in the right order.

---

## 2. The 14-State FSM

### State Transition Diagram

```
         ┌────────────┐
  start  │    IDLE    │◄──────────────────────────────────────────┐
 ───────► └─────┬──────┘                                          │
                │                                                  │
                ▼                                                  │
         ┌──────────────┐                                          │
         │ LOAD_WEIGHTS │  ← pulse weight_prefetch_start           │
         └──────┬───────┘                                          │
                ▼                                                  │
         ┌──────────────┐                                          │ done
         │ WAIT_WEIGHTS │  ← stall until weight_fifo ready         │
         └──────┬───────┘                                          │
                ▼                                                  │
         ┌────────────────┐                                        │
         │LOAD_ACTIVATIONS│  ← pulse activation_load_start         │
         └──────┬─────────┘                                        │
                ▼                                                  │
         ┌─────────────────┐                                       │
         │ WAIT_ACTIVATIONS│  ← stall until act_fifo ready         │
         └──────┬──────────┘                                       │
                ▼                                                  │
         ┌─────────────────┐                                       │
         │BROADCAST_WEIGHTS│  ← load weights into PEs (N cycles)   │
         └──────┬──────────┘                                       │
                ▼                                                  │
         ┌──────────────┐                                          │
         │WAIT_BROADCAST │  ← 1-cycle settle; optionally clear acc │
         └──────┬────────┘                                         │
                ▼                                                  │
         ┌──────────────┐                                          │
         │STREAM_COMPUTE │  ← stream activations (2N-1 cycles)     │
         └──────┬────────┘                                         │
                ▼                                                  │
         ┌──────────────┐                                          │
         │DRAIN_PIPELINE │  ← wait 2N cycles for last sums         │
         └──────┬────────┘                                         │
                ▼                                                  │
         ┌────────────┐                                            │
         │ ACCUMULATE │  ← is this the last K-tile?                │
         └──┬───────┬─┘                                            │
            │       │                                              │
      YES   │       │ NO                                           │
  last_k    │       │                                              │
            │       ▼                                              │
            │ ┌────────────┐                                       │
            │ │ NEXT_K_TILE│  ← advance K, loop back to LOAD_WTS  │
            │ └────────────┘                                       │
            │       │                                              │
            │       └──────────► back to LOAD_WEIGHTS              │
            │                                                      │
            ▼                                                      │
         ┌──────────┐                                              │
         │ WRITEBACK│  ← quantize INT32→INT8, then advance tile    │
         └────┬─────┘                                              │
              │                                                    │
    ┌─────────┴──────────┐                                         │
    │ tile_done?         │                                         │
    │  NO → tile_advance │                                         │
    │       → LOAD_WEIGHTS                                         │
    │  YES → MATMUL_DONE │                                         │
    └──────────┬─────────┘                                         │
               ▼                                                   │
         ┌────────────┐                                            │
         │ MATMUL_DONE│  ← assert done=1 for 1 cycle ─────────────┘
         └────────────┘
```

---

## 3. State-by-State Breakdown

### IDLE
Waits for `start`. When received:
- Registers `first_k_tile` and `last_k_tile` — these come from the
  tiling controller combinationally; registering them makes them stable
  throughout all sub-states that follow.
- Sets `busy = 1`.

### LOAD_WEIGHTS
One-cycle state. Pulses `weight_prefetch_start` with:
- `weight_prefetch_addr` = `tile_addr_b` (tiling controller gives B address)
- `weight_prefetch_rows` = N

The "one-cycle pulse" pattern: because `weight_prefetch_start <= 0` is
at the top of the `else` block (default pulse-clear), it is automatically
de-asserted next cycle without needing logic in every other state.

### WAIT_WEIGHTS
Stalls until `weight_prefetch_done OR weight_buffer_ready`.

`weight_buffer_ready` handles the **double-buffer case**: if weights were
already prefetched into the second buffer, there's no need to wait.

### LOAD_ACTIVATIONS
One-cycle state. Pulses `activation_load_start` with:
- `activation_load_addr` = `tile_addr_a`
- `activation_load_rows` = N
- `activation_load_cols` = N
- `activation_load_stride` = `matrix_k >> 2`

**Why `matrix_k >> 2` for stride?**
Matrix A is stored row-major, one byte per element. Memory words are 32-bit
(4 bytes). To skip from row `i` to row `i+1` in the A tile, the FIFO needs
to advance by `matrix_k / 4` words. The `>> 2` is dividing by 4.

### WAIT_ACTIVATIONS
Same pattern as WAIT_WEIGHTS. Resets `weight_row_counter = 0` so BROADCAST
starts from the first weight row.

### BROADCAST_WEIGHTS
Runs for exactly N cycles.
```
Each cycle:
  assert array_weight_load = 1   ← tell array: latch this weight row
  assert weight_drain_enable = 1 ← tell FIFO: output current row
  pulse weight_drain_row_done    ← tell FIFO: advance to next row
  increment weight_row_counter
  if (weight_row_counter >= N-1) → go to WAIT_BROADCAST
```

Timing diagram for N=4:
```
Cycle:            0     1     2     3
weight_load:      1     1     1     1
drain_row_done:   1     1     1     1  (pulses each cycle)
row_counter:      0     1     2     3
                                   └── transition at end
```

### WAIT_BROADCAST
One-cycle settle. De-asserts weight signals.

**Key decision:** If `first_k_tile_reg AND NOT accumulate_mode`:
- Assert `array_clear_acc = 1` (resets PE internal accumulators to 0)
- This is the first K pass for this C tile → start fresh

If it's a subsequent K pass (`first_k_tile_reg = 0`):
- Don't clear → keep existing partial sums in the PEs

### STREAM_COMPUTE
The main compute phase. Asserts three signals simultaneously:

| Signal                   | Why                                          |
|:-------------------------|:---------------------------------------------|
| `array_enable = 1`       | Clock-gate open: PEs compute every cycle     |
| `activation_stream_enable = 1` | FIFO starts sending rows to the array  |
| `acc_results_enable = 1` | Accumulator captures south-edge outputs      |

Sets `acc_accumulate_mode`:
```sv
acc_accumulate_mode <= !first_k_tile_reg || accumulate_mode;
```
- If first K-tile and no global accumulate: overwrite (= 0)
- If subsequent K-tile: add to existing partial sums (= 1)
- If global accumulate_mode set: always add (= 1)

Waits for `activation_stream_done` (fired by activation_fifo after all
rows are sent), then resets `drain_cycle_counter = 0` and moves to DRAIN.

### DRAIN_PIPELINE
After activations stop, partial sums are still moving through the pipeline.
The deepest path through the array is N PEs → waits **2N cycles** to be safe.

```
array_enable stays = 1     ← pipeline is still flushing
acc_results_enable = 1     ← still capturing results

After 2N cycles:
  array_enable <= 0
  acc_results_enable <= 0
  → ACCUMULATE
```

Why 2N and not N? The pipeline has N stages but result valid signals are
also delayed. 2N provides margin for all edge cases.

### ACCUMULATE
Pure decision state (no outputs change):
```
if (last_k_tile_reg) → WRITEBACK     (this C tile is fully computed)
else                 → NEXT_K_TILE   (more K passes needed)
```

### NEXT_K_TILE
Pulses `tile_advance = 1`. The tiling controller responds next cycle with
updated `tile_addr_a`, `tile_addr_b`, and new `last_k_tile` value.

Sets `first_k_tile_reg = 0` (we're no longer on the first pass).
Loops back to `LOAD_WEIGHTS`.

### WRITEBACK
Two-phase:
1. Assert `acc_quant_enable = 1` (one-cycle pulse). Wait for `acc_quant_done`.
2. When done:
   - If `tile_done`: all tiles exhausted → `MATMUL_DONE`
   - Else: pulse `tile_advance`, re-register `first/last_k_tile`, → `LOAD_WEIGHTS`

### MATMUL_DONE
```sv
busy <= 0;
done <= 1;   // one-cycle pulse
state <= IDLE;
```

---

## 4. Key Design Patterns

### Default Pulse-Clear
Signals that are one-cycle pulses are **zeroed at the top of every clock cycle**
before the case statement runs:

```sv
// These lines run every cycle in the else block:
weight_prefetch_start <= 0;
weight_drain_row_done <= 0;
activation_load_start <= 0;
acc_clear             <= 0;
acc_quant_enable      <= 0;
tile_advance          <= 0;
```

This means: if you want a signal to be a pulse, you just assert it once in
the state that needs it. You don't need to clear it anywhere else.

### Registered K-tile Flags
`first_k_tile` and `last_k_tile` come from the tiling controller combinationally.
If you read them directly, they might change mid-operation. So we register them:

```sv
// In IDLE (on start) and WRITEBACK (on tile_advance):
first_k_tile_reg <= first_k_tile;
last_k_tile_reg  <= last_k_tile;
```

These stable copies are used throughout `WAIT_BROADCAST`, `STREAM_COMPUTE`,
and `ACCUMULATE`.

---

## 5. Signal Timing for One K-Tile Pass

```
Cycle:     0       1      2..N  N+1    N+2..3N   3N+1   3N+2
State:    LOAD_W  WAIT_W  ...   BCAST  STREAM    DRAIN  ACCUM
           │       │            │      │          │
           │       wait for     load   compute    flush
           │       prefetch     N rows activations pipe
           │       done
weight_prefetch_start:
           ▔▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
array_weight_load:
           ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁║▔▔▔▔▔▔▔▔▔║▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
                           BROADCAST
array_enable:
           ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁║▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔║▁▁▁
                                  STREAM + DRAIN
acc_results_enable:
           same as array_enable ─────────────────────
```

---

## 6. Port Quick Reference

| Port                      | Dir | Width | Connected To              |
|:--------------------------|:----|:------|:--------------------------|
| `start`                   | in  | 1     | Host/Sequencer            |
| `busy`, `done`            | out | 1     | Host/Sequencer            |
| `matrix_m/n/k`            | in  | 16    | Host (matmul dimensions)  |
| `addr_a/b/c`              | in  | 14    | Tiling controller (base)  |
| `weight_prefetch_start`   | out | 1     | weight_fifo               |
| `weight_drain_enable`     | out | 1     | weight_fifo               |
| `weight_drain_row_done`   | out | 1     | weight_fifo (pulse)       |
| `weight_prefetch_done`    | in  | 1     | weight_fifo               |
| `activation_load_start`   | out | 1     | activation_fifo (pulse)   |
| `activation_stream_enable`| out | 1     | activation_fifo           |
| `activation_stream_done`  | in  | 1     | activation_fifo           |
| `array_enable`            | out | 1     | systolic_array            |
| `array_weight_load`       | out | 1     | systolic_array            |
| `array_clear_acc`         | out | 1     | systolic_array            |
| `acc_results_enable`      | out | 1     | accumulator_buffer        |
| `acc_accumulate_mode`     | out | 1     | accumulator_buffer        |
| `acc_quant_enable`        | out | 1     | accumulator_buffer        |
| `acc_quant_done`          | in  | 1     | accumulator_buffer        |
| `tile_advance`            | out | 1     | tiling_controller (pulse) |
| `tile_done`               | in  | 1     | tiling_controller         |
| `first_k_tile`            | in  | 1     | tiling_controller         |
| `last_k_tile`             | in  | 1     | tiling_controller         |
| `tile_addr_a/b/c`         | in  | 14    | tiling_controller         |
