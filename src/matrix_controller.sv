`default_nettype none
`timescale 1ns/1ns

// =============================================================================
// MATRIX CONTROLLER
// =============================================================================
//
// WHAT PROBLEM DOES THIS SOLVE?
//   You have five powerful components sitting idle:
//     - weight_fifo       : fetches and stages weight tiles from SRAM
//     - activation_fifo   : fetches and skews activation tiles
//     - systolic_array    : does the actual multiply-accumulate
//     - accumulator_buffer: holds partial sums between K-tile passes
//     - tiling_controller : knows which tile to process next
//
//   This module is the conductor. It is a 14-state FSM that sequences all five
//   to execute one complete tiled matrix multiply: C = A × B.
//
// THE BIG PICTURE — Tiled Matrix Multiply
//   To multiply A(M×K) × B(K×N) = C(M×N) on an N×N systolic array:
//   Chop A and B into N×N tiles and accumulate partial results:
//
//     for each M-tile (rows of A / rows of C):       ← outer loop
//       for each N-tile (cols of B / cols of C):     ← middle loop
//         for each K-tile (shared inner dimension):  ← inner loop
//           C[m][n] += A[m][k] × B[k][n]
//
//   tiling_controller iterates (m,n,k) and provides tile addresses.
//   This controller drives the load → broadcast → compute → writeback sequence.
//
// THE 14 STATES IN ORDER:
//   IDLE              → Wait for start pulse
//   LOAD_WEIGHTS      → Kick off weight_fifo prefetch of B tile
//   WAIT_WEIGHTS      → Wait until weight_fifo has the full tile
//   LOAD_ACTIVATIONS  → Kick off activation_fifo load of A tile
//   WAIT_ACTIVATIONS  → Wait until activation_fifo has the full tile
//   BROADCAST_WEIGHTS → Feed weight_fifo rows into systolic array PEs (N cycles)
//   WAIT_BROADCAST    → One-cycle settle; clear accumulators on first K-tile
//   STREAM_COMPUTE    → Stream activations; array computes MACs each cycle
//   DRAIN_PIPELINE    → Wait 2N cycles for last partial sum to exit the array
//   ACCUMULATE        → More K-tiles? loop. Otherwise: writeback.
//   NEXT_K_TILE       → Advance inner loop counter, loop back to LOAD_WEIGHTS
//   NEXT_N_TILE       → Advance middle loop counter (via tiling_controller)
//   NEXT_M_TILE       → Advance outer loop counter (via tiling_controller)
//   WRITEBACK         → Quantize accumulated result; write C tile to memory
//   MATMUL_DONE       → Assert done, return to IDLE
//
// =============================================================================

module matrix_controller #(
    parameter N          = 8,   // Systolic array dimension
    parameter ADDR_WIDTH = 14   // Unified buffer address width
) (
    input wire clk,
    input wire reset,

    // -------------------------------------------------------------------------
    // Host / Sequencer Interface
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
    // Weight FIFO Interface
    // -------------------------------------------------------------------------
    output reg                  weight_prefetch_start,   // Pulse to begin load
    output reg [ADDR_WIDTH-1:0] weight_prefetch_addr,    // SRAM address to fetch from
    output reg [7:0]            weight_prefetch_rows,    // Number of rows to fetch
    input wire                  weight_prefetch_done,    // weight_fifo: tile loaded
    input wire                  weight_buffer_ready,     // weight_fifo: ready to drain
    output reg                  weight_drain_enable,     // Allow weight_fifo to drain
    output reg                  weight_drain_row_done,   // Pulse: PE row loaded, advance
    input wire                  weight_buffer_empty,     // weight_fifo: all rows drained

    // -------------------------------------------------------------------------
    // Activation FIFO Interface
    // -------------------------------------------------------------------------
    output reg                  activation_load_start,
    output reg [ADDR_WIDTH-1:0] activation_load_addr,
    output reg [7:0]            activation_load_rows,
    output reg [7:0]            activation_load_cols,
    output reg [ADDR_WIDTH-1:0] activation_load_stride,  // Row stride in SRAM words (K/4)
    input wire                  activation_load_done,
    input wire                  activation_buffer_ready,
    output reg                  activation_stream_enable, // Start streaming to array
    input wire                  activation_stream_done,   // All activations sent

    // -------------------------------------------------------------------------
    // Systolic Array Interface
    // -------------------------------------------------------------------------
    output reg          array_enable,       // Clock-gate / enable the array
    output reg          array_weight_load,  // High during weight broadcast phase
    output reg          array_clear_acc,    // Reset internal PE accumulators to 0
    input wire [N-1:0]  array_result_valid, // Per-column output valid flags

    // -------------------------------------------------------------------------
    // Accumulator Interface
    // Holds partial sums between K-tile passes.
    // On the last K-tile, quantizes (INT32 → INT8) and prepares for writeback.
    // -------------------------------------------------------------------------
    output reg        acc_results_enable,   // Capture array outputs this cycle
    output reg        acc_accumulate_mode,  // Add to existing (vs overwrite)
    output reg        acc_clear,            // Reset accumulator to 0
    output reg [7:0]  acc_tile_row,         // Which tile row (for writeback addressing)
    input wire        acc_busy,             // Accumulator is processing
    output reg        acc_quant_enable,     // Start quantization step
    input wire        acc_quant_done,       // Quantization complete, data ready

    // -------------------------------------------------------------------------
    // Memory Controller Interface (writeback only)
    // After quantization, DMA the result tile back to SRAM.
    // -------------------------------------------------------------------------
    output reg                  mem_write_req,
    output reg [ADDR_WIDTH-1:0] mem_write_addr,
    output reg [31:0]           mem_write_data,
    input wire                  mem_write_ack,

    // -------------------------------------------------------------------------
    // Tiling Controller Interface
    // Iterates (m, n, k) tile indices and provides tile addresses.
    // Pulse tile_advance when done with one tile to get the next.
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
    // Status / Debug
    // -------------------------------------------------------------------------
    output reg [3:0]   matmul_state,
    output wire [31:0] debug_cycle_count,
    output wire [3:0]  debug_state
);

// =============================================================================
// STATE DEFINITIONS
// =============================================================================

localparam IDLE              = 4'd0;
localparam LOAD_WEIGHTS      = 4'd1;
localparam WAIT_WEIGHTS      = 4'd2;
localparam LOAD_ACTIVATIONS  = 4'd3;
localparam WAIT_ACTIVATIONS  = 4'd4;
localparam BROADCAST_WEIGHTS = 4'd5;
localparam WAIT_BROADCAST    = 4'd6;
localparam STREAM_COMPUTE    = 4'd7;
localparam DRAIN_PIPELINE    = 4'd8;
localparam ACCUMULATE        = 4'd9;
localparam NEXT_K_TILE       = 4'd10;
localparam NEXT_N_TILE       = 4'd11;
localparam NEXT_M_TILE       = 4'd12;
localparam WRITEBACK         = 4'd13;
localparam MATMUL_DONE       = 4'd14;

// =============================================================================
// INTERNAL REGISTERS
// =============================================================================

reg [3:0]  state;
reg [31:0] cycle_count;
reg [7:0]  weight_row_counter;   // Counts 0..N-1 during BROADCAST_WEIGHTS
reg [7:0]  drain_cycle_counter;  // Counts up during DRAIN_PIPELINE
reg        first_k_tile_reg;     // Registered copy — stable throughout a tile pass
reg        last_k_tile_reg;
reg        writeback_row;        // Which row of accumulator we're writing back
reg [7:0]  writeback_col;        // Column position during writeback

// =============================================================================
// COMBINATIONAL OUTPUTS
// =============================================================================

assign debug_state       = state;
assign debug_cycle_count = cycle_count;

// =============================================================================
// FSM
// =============================================================================
//
// One-cycle pulse signals are cleared at the top of the else block every cycle.
// They stay 0 unless the active state explicitly asserts them.
// Pulse signals: weight_prefetch_start, weight_drain_row_done,
//                activation_load_start, acc_clear, acc_quant_enable, tile_advance.

always @(posedge clk) begin
    if (reset) begin
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
        // Default: clear all one-cycle pulses
        weight_prefetch_start <= 1'b0;
        weight_drain_row_done <= 1'b0;
        activation_load_start <= 1'b0;
        acc_clear             <= 1'b0;
        acc_quant_enable      <= 1'b0;
        tile_advance          <= 1'b0;

        if (busy) cycle_count <= cycle_count + 1;

        case (state)

            // -----------------------------------------------------------------
            // IDLE — wait for start; register tile flags before entering FSM.
            // WHY register first/last_k_tile? tiling_controller drives them
            // combinationally; latching on entry keeps them stable mid-pass.
            // -----------------------------------------------------------------
            IDLE: begin
                done        <= 1'b0;
                cycle_count <= 32'd0;

                if (start) begin
                    busy             <= 1'b1;
                    first_k_tile_reg <= first_k_tile;
                    last_k_tile_reg  <= last_k_tile;
                    state            <= LOAD_WEIGHTS;
                end
            end

            // -----------------------------------------------------------------
            // LOAD_WEIGHTS — pulse weight_fifo to prefetch the B tile from SRAM.
            // -----------------------------------------------------------------
            LOAD_WEIGHTS: begin
                weight_prefetch_start <= 1'b1;
                weight_prefetch_addr  <= tile_addr_b;
                weight_prefetch_rows  <= N;
                state                 <= WAIT_WEIGHTS;
            end

            // -----------------------------------------------------------------
            // WAIT_WEIGHTS — stall until weight_fifo confirms tile is loaded.
            // weight_buffer_ready handles the case where double-buffering means
            // the tile was already prefetched.
            // -----------------------------------------------------------------
            WAIT_WEIGHTS: begin
                if (weight_prefetch_done || weight_buffer_ready) begin
                    state <= LOAD_ACTIVATIONS;
                end
            end

            // -----------------------------------------------------------------
            // LOAD_ACTIVATIONS — pulse activation_fifo to load the A tile.
            //
            // activation_load_stride = matrix_k >> 2:
            //   A is stored row-major; each SRAM word holds 4 INT8 values.
            //   Stride = K / 4 words to step from one row to the next in memory.
            // -----------------------------------------------------------------
            LOAD_ACTIVATIONS: begin
                activation_load_start  <= 1'b1;
                activation_load_addr   <= tile_addr_a;
                activation_load_rows   <= N;
                activation_load_cols   <= N;
                activation_load_stride <= matrix_k >> 2;
                state                  <= WAIT_ACTIVATIONS;
            end

            // -----------------------------------------------------------------
            // WAIT_ACTIVATIONS — same pattern as WAIT_WEIGHTS.
            // Reset weight_row_counter here so BROADCAST_WEIGHTS starts at row 0.
            // -----------------------------------------------------------------
            WAIT_ACTIVATIONS: begin
                if (activation_buffer_ready || activation_load_done) begin
                    weight_row_counter <= 8'd0;
                    state              <= BROADCAST_WEIGHTS;
                end
            end

            // -----------------------------------------------------------------
            // BROADCAST_WEIGHTS — feed N weight rows into the systolic array,
            // one row per cycle. weight_drain_row_done pulses each cycle to
            // advance weight_fifo to the next row.
            // -----------------------------------------------------------------
            BROADCAST_WEIGHTS: begin
                array_weight_load     <= 1'b1;
                weight_drain_enable   <= 1'b1;
                weight_drain_row_done <= 1'b1;
                weight_row_counter    <= weight_row_counter + 1'b1;

                if (weight_row_counter >= N - 1'b1) begin
                    state <= WAIT_BROADCAST;
                end
            end

            // -----------------------------------------------------------------
            // WAIT_BROADCAST — de-assert weight load signals; give PEs one cycle
            // to settle. Clear PE accumulators only on the first K-tile pass —
            // subsequent passes must accumulate into the existing partial sum.
            // -----------------------------------------------------------------
            WAIT_BROADCAST: begin
                array_weight_load   <= 1'b0;
                weight_drain_enable <= 1'b0;

                if (first_k_tile_reg && !accumulate_mode) begin
                    array_clear_acc <= 1'b1;
                end

                state <= STREAM_COMPUTE;
            end

            // -----------------------------------------------------------------
            // STREAM_COMPUTE — enable array and stream activations.
            // acc_accumulate_mode is HIGH on any pass after the first K-tile,
            // so the accumulator adds rather than overwrites.
            // Streaming runs autonomously inside activation_fifo for 2N-1 cycles.
            // -----------------------------------------------------------------
            STREAM_COMPUTE: begin
                array_enable             <= 1'b1;
                activation_stream_enable <= 1'b1;
                array_clear_acc          <= 1'b0;
                acc_results_enable       <= 1'b1;
                acc_accumulate_mode      <= !first_k_tile_reg || accumulate_mode;

                if (activation_stream_done) begin
                    activation_stream_enable <= 1'b0;
                    drain_cycle_counter      <= 8'd0;
                    state                    <= DRAIN_PIPELINE;
                end
            end

            // -----------------------------------------------------------------
            // DRAIN_PIPELINE — partial sums are still in flight after activations
            // stop. Keep array and accumulator capture active for 2N cycles to
            // flush the deepest pipeline path (N PE stages).
            // -----------------------------------------------------------------
            DRAIN_PIPELINE: begin
                drain_cycle_counter <= drain_cycle_counter + 1'b1;

                if (drain_cycle_counter >= 2 * N) begin
                    array_enable       <= 1'b0;
                    acc_results_enable <= 1'b0;
                    state              <= ACCUMULATE;
                end
            end

            // -----------------------------------------------------------------
            // ACCUMULATE — results are in the accumulator buffer.
            // If more K-tiles remain, loop back. Otherwise write back.
            // -----------------------------------------------------------------
            ACCUMULATE: begin
                if (last_k_tile_reg) begin
                    state <= WRITEBACK;
                end else begin
                    state <= NEXT_K_TILE;
                end
            end

            // -----------------------------------------------------------------
            // NEXT_K_TILE — advance the inner K loop via tiling_controller.
            // Clear first_k_tile_reg; re-register last_k_tile_reg from the
            // updated tiling_controller output for the upcoming tile.
            // -----------------------------------------------------------------
            NEXT_K_TILE: begin
                tile_advance     <= 1'b1;
                first_k_tile_reg <= 1'b0;
                last_k_tile_reg  <= last_k_tile;
                state            <= LOAD_WEIGHTS;
            end

            // -----------------------------------------------------------------
            // NEXT_N_TILE / NEXT_M_TILE — advance middle / outer loop.
            // Reset first_k_tile_reg to 1 since we're starting a fresh (m,n) tile.
            // Note: in the current FSM, WRITEBACK transitions directly to LOAD_WEIGHTS
            // via tile_advance; these states exist for explicit loop control if needed.
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
            // WRITEBACK — quantize INT32 partial sums → INT8, then check if
            // there are more tiles. tile_done = 0: advance and loop back.
            // tile_done = 1: all tiles complete, go to MATMUL_DONE.
            // -----------------------------------------------------------------
            WRITEBACK: begin
                if (!acc_quant_done) begin
                    acc_quant_enable <= 1'b1;
                end else begin
                    if (tile_done) begin
                        state <= MATMUL_DONE;
                    end else begin
                        tile_advance     <= 1'b1;
                        first_k_tile_reg <= first_k_tile;
                        last_k_tile_reg  <= last_k_tile;
                        state            <= LOAD_WEIGHTS;
                    end
                end
            end

            // -----------------------------------------------------------------
            // MATMUL_DONE — assert done for one cycle, drop busy, return to IDLE.
            // -----------------------------------------------------------------
            MATMUL_DONE: begin
                busy  <= 1'b0;
                done  <= 1'b1;
                state <= IDLE;
            end

            default: state <= IDLE;

        endcase
    end
end

// Mirror internal state to the external status port
always @(posedge clk) begin
    matmul_state <= state;
end

endmodule
