`default_nettype none
`timescale 1ns/1ns

// =============================================================================
// MATRIX CONTROLLER 
// =============================================================================
//
// WHAT PROBLEM DOES THIS SOLVE?
//   You now have all the building blocks:
//     - weight_fifo   : loads weights from SRAM, stages them for the array
//     - activation_fifo: loads activations, skews them for the array
//     - systolic_array : does the actual multiply-accumulate
//     - accumulator    : holds partial results between K-tile passes
//   That is this module. It is a 14-state FSM that runs the full matmul loop.
//
// THE BIG PICTURE — Tiled Matrix Multiply
//   To multiply two large matrices A(M×K) × B(K×N) = C(M×N):
//   The result doesn't fit in one NxN systolic array pass.
//   So you tile — you chop A and B into NxN blocks and accumulate.
//
//   The three nested loops, which become the tile iteration order:
//     for each M-tile (rows of A / rows of C):          ← outer loop
//       for each N-tile (cols of B / cols of C):        ← middle loop
//         for each K-tile (inner dimension, shared):    ← inner loop
//           C[m][n] += A[m][k] × B[k][n]
//
//   This controller runs ONE full pass through all K-tiles for a given (m,n).
//   The tiling_controller module handles iterating m and n — it tells us
//   the addresses and sizes for each tile via the tile_* ports.
//
// THE 14 STATES IN ORDER:
//   IDLE              → Wait for start pulse
//   LOAD_WEIGHTS      → Kick off weight_fifo prefetch
//   WAIT_WEIGHTS      → Wait until weight_fifo has the full tile
//   LOAD_ACTIVATIONS  → Kick off activation_fifo load
//   WAIT_ACTIVATIONS  → Wait until activation_fifo has the full tile
//   BROADCAST_WEIGHTS → Feed weights from FIFO into the systolic array PEs
//   WAIT_BROADCAST    → One-cycle settle; optionally clear accumulators
//   STREAM_COMPUTE    → Stream activations, array computes MACs each cycle
//   DRAIN_PIPELINE    → Wait for the last partial sum to exit the array
//   ACCUMULATE        → Decide: more K-tiles? or done with this (m,n)?
//   NEXT_K_TILE       → Advance inner loop counter, loop back to LOAD_WEIGHTS
//   NEXT_N_TILE       → Advance middle loop counter (handled by tiling ctrl)
//   NEXT_M_TILE       → Advance outer loop counter (handled by tiling ctrl)
//   WRITEBACK         → Quantize accumulated result, write C tile to memory
//   DONE              → Assert done, return to IDLE
//
// CODER THOUGHT PROCESS:
//   Step 1: Define the ports — what does this module need to CONTROL?
//           It drives every other module. So its outputs are the control
//           signals of weight_fifo, activation_fifo, systolic_array, accumulator.
//           Its inputs are the "done" and "ready" signals back from each.
//   Step 2: Enumerate the states — draw the FSM on paper first.
//           For each state ask: "what do I assert, what do I wait for?"
//   Step 3: Write the state register + reset block (all outputs zeroed).
//   Step 4: Write the default pulse resets (one-cycle-pulse signals at top
//           of else begin, so you don't have to clear them in every state).
//   Step 5: Fill in each case state one by one, referring to the diagram.

// =============================================================================
// STEP 1 — MODULE HEADER + PORTS
// =============================================================================
// Programmer asks: "who does this module talk to?"
//   - Host/sequencer: gives us start/config, we signal busy/done
//   - weight_fifo:    we issue prefetch_start, drain_enable, drain_row_done
//   - activation_fifo: we issue load_start, stream_enable
//   - systolic_array:  we issue array_enable, array_weight_load, array_clear_acc
//   - accumulator:     we issue results_enable, accumulate_mode, quant_enable
//   - memory controller: we issue write requests for writeback
//   - tiling controller: it tells us per-tile addresses; we signal tile_advance

module matrix_controller #(
    parameter N          = 8,   // Systolic array dimension
    parameter ADDR_WIDTH = 14   // Unified buffer address width
) (
    input wire clk,
    input wire reset,

    // -------------------------------------------------------------------------
    // HOST / SEQUENCER INTERFACE
    // -------------------------------------------------------------------------
    input wire  start,            // Pulse: begin one matmul operation
    input wire  accumulate_mode,  // If 1, add to existing C instead of replacing
    output reg  busy,             // High while we are working
    output reg  done,             // One-cycle pulse when finished

    // Matrix dimensions (M × K) × (K × N) = M × N
    input wire [15:0]           matrix_m,
    input wire [15:0]           matrix_n,
    input wire [15:0]           matrix_k,
    input wire [ADDR_WIDTH-1:0] addr_a,   // Base address of matrix A in SRAM
    input wire [ADDR_WIDTH-1:0] addr_b,   // Base address of matrix B in SRAM
    input wire [ADDR_WIDTH-1:0] addr_c,   // Base address of result C in SRAM

    // -------------------------------------------------------------------------
    // WEIGHT FIFO INTERFACE
    // Programmer: "I need to tell weight_fifo to prefetch, then drain
    //              row by row into the systolic array."
    // -------------------------------------------------------------------------
    output reg                  weight_prefetch_start,   // Pulse to begin load
    output reg [ADDR_WIDTH-1:0] weight_prefetch_addr,    // Which SRAM address to fetch from
    output reg [7:0]            weight_prefetch_rows,    // How many rows to fetch
    input wire                  weight_prefetch_done,    // weight_fifo: "tile loaded"
    input wire                  weight_buffer_ready,     // weight_fifo: "ready to drain"
    output reg                  weight_drain_enable,     // Allow weight_fifo to drain
    output reg                  weight_drain_row_done,   // Pulse: PE row loaded, next row please
    input wire                  weight_buffer_empty,     // weight_fifo: "all rows drained"

    // -------------------------------------------------------------------------
    // ACTIVATION FIFO INTERFACE
    // -------------------------------------------------------------------------
    output reg                  activation_load_start,
    output reg [ADDR_WIDTH-1:0] activation_load_addr,
    output reg [7:0]            activation_load_rows,
    output reg [7:0]            activation_load_cols,
    output reg [ADDR_WIDTH-1:0] activation_load_stride,  // Row stride in SRAM words
    input wire                  activation_load_done,
    input wire                  activation_buffer_ready,
    output reg                  activation_stream_enable, // Start streaming to array
    input wire                  activation_stream_done,   // All activations sent

    // -------------------------------------------------------------------------
    // SYSTOLIC ARRAY INTERFACE
    // -------------------------------------------------------------------------
    output reg          array_enable,       // Clock-gate / enable the array
    output reg          array_weight_load,  // Tell array: we're loading weights now
    output reg          array_clear_acc,    // Reset internal PE accumulators to 0
    input wire [N-1:0]  array_result_valid, // Array: "column N outputs valid"

    // -------------------------------------------------------------------------
    // ACCUMULATOR INTERFACE
    // The accumulator holds partial sums between K-tile passes.
    // On the last K-tile, we quantize (scale INT32→INT8) and write back.
    // -------------------------------------------------------------------------
    output reg        acc_results_enable,   // Capture array outputs this cycle
    output reg        acc_accumulate_mode,  // Add to existing (vs overwrite)
    output reg        acc_clear,            // Reset accumulator to 0
    output reg [7:0]  acc_tile_row,         // Which tile row (for writeback addressing)
    input wire        acc_busy,             // Accumulator is processing
    output reg        acc_quant_enable,     // Start quantization step
    input wire        acc_quant_done,       // Quantization complete, data ready

    // -------------------------------------------------------------------------
    // MEMORY CONTROLLER INTERFACE (writeback only)
    // After quantization, we DMA the result tile back to SRAM.
    // -------------------------------------------------------------------------
    output reg                  mem_write_req,
    output reg [ADDR_WIDTH-1:0] mem_write_addr,
    output reg [31:0]           mem_write_data,
    input wire                  mem_write_ack,

    // -------------------------------------------------------------------------
    // TILING CONTROLLER INTERFACE
    // tiling_controller iterates (m, n, k) tile indices and gives us addresses.
    // We pulse tile_advance when done with one tile to get the next.
    // -------------------------------------------------------------------------
    output reg                  tile_advance,    // Pulse: move to next tile
    input wire                  tile_done,       // No more tiles to process
    input wire                  first_k_tile,    // This is the first K-tile for this (m,n)
    input wire                  last_k_tile,     // This is the last K-tile for this (m,n)
    input wire [15:0]           tile_m,          // Rows in current tile
    input wire [15:0]           tile_n,          // Cols in current tile
    input wire [15:0]           tile_k,          // Depth of current tile
    input wire [ADDR_WIDTH-1:0] tile_addr_a,     // SRAM address for A tile
    input wire [ADDR_WIDTH-1:0] tile_addr_b,     // SRAM address for B tile
    input wire [ADDR_WIDTH-1:0] tile_addr_c,     // SRAM address for C tile (writeback)
    input wire [7:0]            tile_rows,        // Valid rows in this tile (may be < N)
    input wire [7:0]            tile_cols,        // Valid cols in this tile (may be < N)

    // -------------------------------------------------------------------------
    // STATUS / DEBUG
    // -------------------------------------------------------------------------
    output reg [3:0]  matmul_state,
    output wire [31:0] debug_cycle_count,
    output wire [3:0]  debug_state
);

// =============================================================================
// STEP 2 — STATE DEFINITIONS
// =============================================================================
// Programmer asks: "what are the discrete phases of one tile matmul?"
// Draw the sequence diagram:
//   prefetch B → prefetch A → load weights into PEs → compute → drain → accumulate → writeback
// Each phase becomes a state. Add "WAIT_*" states for multi-cycle handoffs.
//
// Use 4-bit encoding (14 states fits in 4 bits).
// Use named constants — never use raw 4'b... in the case statement.

localparam IDLE              = 4'd0;
localparam LOAD_WEIGHTS      = 4'd1;
localparam WAIT_WEIGHTS      = 4'd2;
localparam LOAD_ACTIVATIONS  = 4'd3;
localparam WAIT_ACTIVATIONS  = 4'd4;
localparam BROADCAST_WEIGHTS = 4'd5;  // Feed weight_fifo rows into systolic array PEs
localparam WAIT_BROADCAST    = 4'd6;  // One-cycle settle before compute
localparam STREAM_COMPUTE    = 4'd7;  // Stream activations, array computes
localparam DRAIN_PIPELINE    = 4'd8;  // Wait 2N cycles for last sum to drain
localparam ACCUMULATE        = 4'd9;  // Decide: next K tile or writeback?
localparam NEXT_K_TILE       = 4'd10; // Inner loop advance
localparam NEXT_N_TILE       = 4'd11; // Middle loop advance (via tiling controller)
localparam NEXT_M_TILE       = 4'd12; // Outer loop advance (via tiling controller)
localparam WRITEBACK         = 4'd13; // Quantize + DMA result tile to SRAM
localparam MATMUL_DONE       = 4'd14; // Complete, assert done

// =============================================================================
// STEP 3 — INTERNAL STATE REGISTERS
// =============================================================================
// Programmer asks: "what do I need to track between clock cycles?"
//   - The FSM state itself
//   - A cycle counter (performance measurement)
//   - Weight row counter (to know when all N rows have been broadcast to PEs)
//   - Pipeline drain counter (to wait 2N cycles for results to drain)
//   - Copies of first_k_tile and last_k_tile (registered so they're stable)
//   - Writeback position (row/col into the C tile)

reg [3:0]  state;
reg [31:0] cycle_count;
reg [7:0]  weight_row_counter;   // Counts 0..N-1 during BROADCAST_WEIGHTS
reg [7:0]  drain_cycle_counter;  // Counts 0..2N during DRAIN_PIPELINE
reg        first_k_tile_reg;     // Registered copy — stable throughout state machine
reg        last_k_tile_reg;
reg        writeback_row;        // 1-bit: which row of accumulator we're writing
reg [7:0]  writeback_col;        // Column position during writeback

// =============================================================================
// STEP 4 — DEBUG / STATUS COMBINATIONAL OUTPUTS
// =============================================================================
assign debug_state       = state;
assign debug_cycle_count = cycle_count;
// matmul_state drives external status registers — just mirror the internal state

// =============================================================================
// STEP 5 — THE STATE MACHINE
// =============================================================================
// CRITICAL CODING PATTERN: "default pulse reset at top of else begin"
//
// Some signals are one-cycle pulses (e.g. prefetch_start, tile_advance).
// If you set them in a state, you MUST clear them next cycle.
// Rather than clearing in every other state, declare them 0 at the TOP of
// the else block. This way they're always 0 UNLESS the current state asserts them.
// This is standard RTL style for control pulses.
//
// Signals that should be cleared this way:
//   weight_prefetch_start, weight_drain_row_done
//   activation_load_start
//   acc_clear, acc_quant_enable
//   tile_advance

always @(posedge clk) begin
    if (reset) begin
        // =====================================================================
        // RESET: Every output to a safe idle state.
        // Programmer rule: Every reg driven by this always block must appear here.
        // If you forget one, it will power up with unknown value in simulation.
        // =====================================================================
        state <= IDLE;
        busy  <= 1'b0;
        done  <= 1'b0;
        cycle_count <= 32'd0;

        weight_prefetch_start    <= 1'b0;
        weight_prefetch_addr     <= {ADDR_WIDTH{1'b0}};
        weight_prefetch_rows     <= 8'd0;
        weight_drain_enable      <= 1'b0;
        weight_drain_row_done    <= 1'b0;

        activation_load_start    <= 1'b0;
        activation_load_addr     <= {ADDR_WIDTH{1'b0}};
        activation_load_rows     <= 8'd0;
        activation_load_cols     <= 8'd0;
        activation_load_stride   <= {ADDR_WIDTH{1'b0}};
        activation_stream_enable <= 1'b0;

        array_enable             <= 1'b0;
        array_weight_load        <= 1'b0;
        array_clear_acc          <= 1'b0;

        acc_results_enable       <= 1'b0;
        acc_accumulate_mode      <= 1'b0;
        acc_clear                <= 1'b0;
        acc_tile_row             <= 8'd0;
        acc_quant_enable         <= 1'b0;

        mem_write_req            <= 1'b0;
        mem_write_addr           <= {ADDR_WIDTH{1'b0}};
        mem_write_data           <= 32'd0;

        tile_advance             <= 1'b0;
        weight_row_counter       <= 8'd0;
        drain_cycle_counter      <= 8'd0;
        first_k_tile_reg         <= 1'b0;
        last_k_tile_reg          <= 1'b0;
        writeback_row            <= 1'b0;
        writeback_col            <= 8'd0;

    end else begin
        // =====================================================================
        // DEFAULT PULSE CLEAR — runs every cycle before the case statement.
        // Any signal listed here will be 0 unless the current state overrides it.
        // This is equivalent to "deassert by default, assert only when needed."
        // =====================================================================
        weight_prefetch_start <= 1'b0;  // One-cycle pulse
        weight_drain_row_done <= 1'b0;  // One-cycle pulse
        activation_load_start <= 1'b0;  // One-cycle pulse
        acc_clear             <= 1'b0;  // One-cycle pulse
        acc_quant_enable      <= 1'b0;  // One-cycle pulse
        tile_advance          <= 1'b0;  // One-cycle pulse

        // Count cycles while busy (performance counter)
        if (busy) cycle_count <= cycle_count + 1;

        case (state)
            // -----------------------------------------------------------------
            // IDLE
            // Wait for start. When it arrives, register tile info and go.
            // WHY register first_k_tile? The tiling controller drives this wire
            // combinationally. If we move to LOAD_WEIGHTS the same cycle we
            // register it, it's stable throughout all sub-states that follow.
            // -----------------------------------------------------------------
            IDLE: begin
                done        <= 1'b0;
                cycle_count <= 32'd0;

                if (start) begin
                    busy             <= 1'b1;
                    first_k_tile_reg <= first_k_tile;  // Register before tiling_ctrl changes it
                    last_k_tile_reg  <= last_k_tile;
                    state            <= LOAD_WEIGHTS;
                end
            end

            // -----------------------------------------------------------------
            // LOAD_WEIGHTS
            // Tell weight_fifo to prefetch the B tile from SRAM.
            // weight_prefetch_start is a one-cycle pulse — the default clear
            // above handles de-assertion automatically next cycle.
            // tile_addr_b comes from the tiling controller — it computes the
            // correct SRAM address for B[k_tile][n_tile].
            // -----------------------------------------------------------------
            LOAD_WEIGHTS: begin
                // YOUR CODE HERE:
                // 1. Assert weight_prefetch_start for one cycle (it's a pulse)
                // 2. Set weight_prefetch_addr to the current B tile address
                // 3. Set weight_prefetch_rows to N (full tile)
                // 4. Transition to WAIT_WEIGHTS

                // weight_prefetch_start <= ???
                // weight_prefetch_addr  <= ???
                // weight_prefetch_rows  <= ???
                // state                 <= ???
            end

            // -----------------------------------------------------------------
            // WAIT_WEIGHTS
            // Stall until weight_fifo signals the tile is loaded.
            // Two conditions because weight_fifo may have loaded it already
            // (double-buffering) — weight_buffer_ready is also checked.
            // -----------------------------------------------------------------
            WAIT_WEIGHTS: begin
                // YOUR CODE HERE:
                // Wait until weight_prefetch_done OR weight_buffer_ready
                // Then reset weight_row_counter to 0 (used in BROADCAST)
                // Transition to LOAD_ACTIVATIONS

                // if (???) begin
                //     weight_row_counter <= 8'd0;
                //     state <= ???;
                // end
            end

            // -----------------------------------------------------------------
            // LOAD_ACTIVATIONS
            // Tell activation_fifo to load the A tile from SRAM.
            //
            // KEY DESIGN DECISION — activation_load_stride:
            //   Activations are stored row-major in memory. Matrix A has K columns.
            //   Each word holds 4 INT8s, so stride = K/4 = K >> 2.
            //   This is how activation_fifo knows how many words to skip
            //   to go from row i to row i+1 of this tile in memory.
            // -----------------------------------------------------------------
            LOAD_ACTIVATIONS: begin
                // YOUR CODE HERE:
                // 1. Assert activation_load_start (one-cycle pulse)
                // 2. Set activation_load_addr to current A tile address
                // 3. Set activation_load_rows to N
                // 4. Set activation_load_cols to N
                // 5. Set activation_load_stride = matrix_k >> 2  (K/4 words per row)
                // 6. Transition to WAIT_ACTIVATIONS

                // activation_load_start  <= ???
                // activation_load_addr   <= ???
                // activation_load_rows   <= ???
                // activation_load_cols   <= ???
                // activation_load_stride <= ???
                // state                  <= ???
            end

            // -----------------------------------------------------------------
            // WAIT_ACTIVATIONS
            // Same pattern as WAIT_WEIGHTS. Two conditions to check.
            // -----------------------------------------------------------------
            WAIT_ACTIVATIONS: begin
                // YOUR CODE HERE:
                // if (activation_load_done || activation_buffer_ready)
                //     state <= BROADCAST_WEIGHTS;
            end

            // -----------------------------------------------------------------
            // BROADCAST_WEIGHTS
            // Feed weights from weight_fifo into the systolic array PEs.
            //
            // HOW DOES WEIGHT LOADING WORK?
            //   weight_drain_enable = 1 tells weight_fifo: "start outputting rows"
            //   array_weight_load   = 1 tells systolic array: "load the row you see"
            //   weight_drain_row_done = one-cycle pulse each cycle tells weight_fifo
            //   to advance to the next row.
            //
            //   We do this for N cycles — one cycle per row of the weight tile.
            //   weight_row_counter tracks how many rows we've loaded.
            //   When weight_row_counter reaches N-1, next cycle we go to WAIT_BROADCAST.
            //
            // WHY NOT WAIT FOR weight_buffer_empty?
            //   We know exactly how many rows to load (N), so counting is simpler
            //   and doesn't depend on weight_fifo's internal state.
            // -----------------------------------------------------------------
            BROADCAST_WEIGHTS: begin
                // YOUR CODE HERE:
                // Assert array_weight_load = 1
                // Assert weight_drain_enable = 1
                // Assert weight_drain_row_done = 1 (pulse every cycle to advance row)
                // Increment weight_row_counter
                // When weight_row_counter >= N-1, state <= WAIT_BROADCAST

                // array_weight_load      <= ???
                // weight_drain_enable    <= ???
                // weight_drain_row_done  <= ???  (this advances weight_fifo each cycle)
                // weight_row_counter     <= ???
                // if (???) state <= ???;
            end

            // -----------------------------------------------------------------
            // WAIT_BROADCAST
            // De-assert weight loading signals.
            // If this is the FIRST K-tile AND accumulate_mode is off:
            //   assert array_clear_acc for one cycle (reset PE accumulators to 0)
            // If this is a subsequent K-tile: don't clear (we want to accumulate)
            //
            // WHY CLEAR ONLY ON FIRST K-TILE?
            //   C = A[:,0:k]*B[0:k,:] + A[:,k:2k]*B[k:2k,:] + ...
            //   First pass: overwrite (clear=1). Subsequent passes: add (clear=0).
            // -----------------------------------------------------------------
            WAIT_BROADCAST: begin
                // YOUR CODE HERE:
                array_weight_load  <= 1'b0;
                weight_drain_enable <= 1'b0;

                // Only clear accumulator on first K tile (no carry-over expected)
                if (first_k_tile_reg && !accumulate_mode) begin
                    array_clear_acc <= 1'b1;
                end

                state <= STREAM_COMPUTE;
            end

            // -----------------------------------------------------------------
            // STREAM_COMPUTE
            // The main compute phase. Three things happen simultaneously:
            //   1. array_enable = 1           → systolic array is clocked and computing
            //   2. activation_stream_enable=1  → activation_fifo starts streaming rows
            //   3. acc_results_enable = 1      → accumulator captures output columns
            //
            // acc_accumulate_mode:
            //   HIGH if this is not the first K-tile (we're adding to existing C)
            //   HIGH if accumulate_mode is set globally
            //   LOW  if this is the first K-tile and not in accumulate_mode
            //         (first pass should overwrite, not add to garbage)
            //
            // Streaming runs for 2N-1 cycles autonomously inside activation_fifo.
            // We wait for activation_stream_done, then go to DRAIN_PIPELINE.
            // -----------------------------------------------------------------
            STREAM_COMPUTE: begin
                // YOUR CODE HERE:
                // array_enable             <= 1
                // array_clear_acc          <= 0 (clear is now done in WAIT_BROADCAST)
                // activation_stream_enable <= 1
                // acc_results_enable       <= 1
                // acc_accumulate_mode      <= (!first_k_tile_reg || accumulate_mode)
                //   Explanation: accumulate if this is NOT the first K pass,
                //                OR if global accumulate_mode is set.
                //
                // When activation_stream_done:
                //   de-assert activation_stream_enable
                //   reset drain_cycle_counter
                //   state <= DRAIN_PIPELINE
            end

            // -----------------------------------------------------------------
            // DRAIN_PIPELINE
            // After activations stop streaming, partial sums are still flowing
            // through the systolic array pipeline.
            // The deepest path is N PEs, so we wait 2N cycles to be safe.
            //
            // During this time:
            //   array_enable stays HIGH (pipeline is still flushing)
            //   acc_results_enable stays HIGH (still capturing results)
            //
            // When drain_cycle_counter >= 2*N, everything has settled.
            // -----------------------------------------------------------------
            DRAIN_PIPELINE: begin
                // YOUR CODE HERE:
                // drain_cycle_counter <= drain_cycle_counter + 1;
                // if (drain_cycle_counter >= 2*N):
                //   array_enable       <= 0
                //   acc_results_enable <= 0
                //   state <= ACCUMULATE
            end

            // -----------------------------------------------------------------
            // ACCUMULATE
            // Results are now in the accumulator buffer.
            // Decision point: are there more K-tiles?
            //   YES (not last_k_tile_reg): loop back via NEXT_K_TILE
            //   NO  (last_k_tile_reg):     go write back the result
            // -----------------------------------------------------------------
            ACCUMULATE: begin
                // YOUR CODE HERE:
                // if (last_k_tile_reg)
                //     state <= WRITEBACK;
                // else
                //     state <= NEXT_K_TILE;
            end

            // -----------------------------------------------------------------
            // NEXT_K_TILE
            // Advance the inner K loop.
            // tile_advance is a one-cycle pulse to the tiling controller.
            // After it, tile_addr_a/b update to the next K-tile addresses.
            // first_k_tile_reg becomes 0 (we're no longer on the first pass).
            // last_k_tile_reg will be updated from the tiling controller signals
            // next cycle — but we don't register it here; we let IDLE/LOAD do it.
            // -----------------------------------------------------------------
            NEXT_K_TILE: begin
                // YOUR CODE HERE:
                // tile_advance      <= 1 (pulse)
                // first_k_tile_reg  <= 0 (no longer first K tile)
                // state             <= LOAD_WEIGHTS (loop back to top)
            end

            // -----------------------------------------------------------------
            // NEXT_N_TILE / NEXT_M_TILE
            // These advance the tiling controller to the next (n) or (m) tile.
            // In this reference, both just pulse tile_advance and return to
            // LOAD_WEIGHTS — the tiling_controller handles address computation.
            // first_k_tile_reg is reset to 1 because we're starting a new (m,n).
            // Note: NEXT_N_TILE and NEXT_M_TILE are never explicitly transitioned
            // to in this reference FSM — the WRITEBACK → LOAD_WEIGHTS path uses
            // tile_advance directly. These states exist for completeness.
            // -----------------------------------------------------------------
            NEXT_N_TILE: begin
                tile_advance     <= 1'b1;
                first_k_tile_reg <= 1'b1;
                state            <= LOAD_WEIGHTS;
            end

            NEXT_M_TILE: begin
                tile_advance     <= 1'b1;
                first_k_tile_reg <= 1'b1;
                state            <= LOAD_WEIGHTS;
            end

            // -----------------------------------------------------------------
            // WRITEBACK
            // C tile is in the accumulator. Two steps:
            //   1. Quantize: scale INT32 partial sums → INT8 output.
            //      Assert acc_quant_enable (one-cycle pulse). Wait for acc_quant_done.
            //   2. After quant_done: check if there are more tiles.
            //      tile_done = 1 → all done, go to MATMUL_DONE.
            //      tile_done = 0 → advance to next tile, go back to LOAD_WEIGHTS.
            // -----------------------------------------------------------------
            WRITEBACK: begin
                // YOUR CODE HERE:
                // if (!acc_quant_done):
                //   acc_quant_enable <= 1  (will auto-clear next cycle by default)
                // else:
                //   if (tile_done):
                //       state <= MATMUL_DONE
                //   else:
                //       tile_advance      <= 1
                //       first_k_tile_reg  <= first_k_tile  (re-register for new tile)
                //       last_k_tile_reg   <= last_k_tile
                //       state             <= LOAD_WEIGHTS
            end

            // -----------------------------------------------------------------
            // MATMUL_DONE
            // Assert done for one cycle. De-assert busy. Return to IDLE.
            // done is a one-cycle pulse — caller latches it.
            // -----------------------------------------------------------------
            MATMUL_DONE: begin
                // YOUR CODE HERE:
                // busy  <= 0
                // done  <= 1
                // state <= IDLE
            end

            default: state <= IDLE;
        endcase
    end
end

// Drive matmul_state output
always @(posedge clk) begin
    matmul_state <= state;
end

endmodule
