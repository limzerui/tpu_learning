`default_nettype none
`timescale 1ns/1ns

// =============================================================================
// TILING CONTROLLER
// =============================================================================
//
// WHAT PROBLEM DOES THIS SOLVE?
//   The systolic array is 8×8. Real matrices might be 1024×1024.
//   This module manages the THREE NESTED LOOPS that chop the big problem
//   into 8×8 tile-sized pieces and computes the SRAM address for each tile.
//
//   Think of it as a hardware "for loop counter" that also does address math:
//
//     for m in range(num_tiles_m):           ← outer loop  (rows of output C)
//       for n in range(num_tiles_n):         ← middle loop (cols of output C)
//         for k in range(num_tiles_k):       ← inner loop  (shared dimension)
//           matrix_controller.run_tile(A[m,k], B[k,n], C[m,n])
//
//   matrix_controller pulses `advance` when one tile is done.
//   We move the (m,n,k) counters and update addresses.
//   When all tiles are done, we assert `done`.
//
// ADDRESS FORMULA — most important concept:
//   All matrices stored as flat byte arrays in SRAM, row-major layout.
//   To find the word address of tile [tile_m][tile_k] of matrix A (size M×K):
//
//     byte_offset = tile_m * TILE_SIZE * matrix_k + tile_k * TILE_SIZE
//     word_addr   = base_addr_a + (byte_offset >> 2)   // divide by 4 (4 bytes/word)
//
//   Similarly for B (K×N) and C (M×N — but C is INT32 so no >>2 needed).
//
// THE THREE-STATE MACHINE:
//   IDLE     → wait for `start`
//   RUNNING  → hold the current tile, wait for `advance`, then move to next tile
//   COMPLETE → all done, reassert `done` until new `start`
//
// =============================================================================

module tiling_controller #(
    parameter TILE_SIZE  = 8,    // Must match the systolic array dimension N
    parameter MAX_DIM    = 4096, // Maximum supported matrix dimension
    parameter ADDR_WIDTH = 14    // Log2 of unified buffer depth
) (
    input wire clk,
    input wire reset,

    // -------------------------------------------------------------------------
    // Configuration (driven by the host/sequencer before asserting start)
    // -------------------------------------------------------------------------
    input wire [15:0]           matrix_m,      // Total matrix rows (A and C)
    input wire [15:0]           matrix_n,      // Total matrix columns (B and C)
    input wire [15:0]           matrix_k,      // Shared inner dimension (cols of A, rows of B)
    input wire [ADDR_WIDTH-1:0] base_addr_a,   // SRAM word address where A starts
    input wire [ADDR_WIDTH-1:0] base_addr_b,   // SRAM word address where B starts
    input wire [ADDR_WIDTH-1:0] base_addr_c,   // SRAM word address where C starts

    // -------------------------------------------------------------------------
    // Control Interface
    // -------------------------------------------------------------------------
    input wire  start,   // Pulse: begin (or restart) the tiling iteration
    input wire  advance, // Pulse from matrix_controller: current tile done, give me next
    output reg  done,    // Asserted when all tiles have been processed
    output reg  active,  // High while tiling is in progress

    // -------------------------------------------------------------------------
    // Current Tile Indices (read by matrix_controller for debugging/status)
    // -------------------------------------------------------------------------
    output reg [15:0] tile_m,  // Which row-tile of C are we processing now?
    output reg [15:0] tile_n,  // Which col-tile of C?
    output reg [15:0] tile_k,  // Which K-tile (inner dimension)?

    // -------------------------------------------------------------------------
    // Tile Counts (combinational — used externally to check loop bounds)
    // -------------------------------------------------------------------------
    output wire [15:0] num_tiles_m,  // ceil(matrix_m / TILE_SIZE)
    output wire [15:0] num_tiles_n,
    output wire [15:0] num_tiles_k,

    // -------------------------------------------------------------------------
    // Tile SRAM Addresses (updated every time advance is pulsed)
    // matrix_controller uses these to set up weight_fifo and activation_fifo
    // -------------------------------------------------------------------------
    output reg [ADDR_WIDTH-1:0] addr_a_tile,  // A tile start address (word-addressed)
    output reg [ADDR_WIDTH-1:0] addr_b_tile,  // B tile start address
    output reg [ADDR_WIDTH-1:0] addr_c_tile,  // C tile start address (writeback target)

    // -------------------------------------------------------------------------
    // Tile Dimensions — for edge tiles (tiles at the boundary of the matrix)
    // A full tile is TILE_SIZE × TILE_SIZE.
    // An edge tile may be smaller (e.g., last tile of a 9-element dimension = 1 element).
    // matrix_controller passes these to weight_fifo and activation_fifo so they
    // don't over-fetch memory.
    // -------------------------------------------------------------------------
    output reg [7:0] tile_rows_a,   // Valid rows in the current A tile
    output reg [7:0] tile_cols_a,   // Valid cols in the current A tile (= rows of B tile)
    output reg [7:0] tile_rows_b,
    output reg [7:0] tile_cols_b,
    output reg [7:0] tile_rows_c,
    output reg [7:0] tile_cols_c,

    // -------------------------------------------------------------------------
    // K-tile Flags (critical for accumulator_buffer behavior)
    // first_k_tile = 1 → clear accumulator before compute (overwrite mode)
    // last_k_tile  = 1 → run quantization after compute (result is complete)
    // -------------------------------------------------------------------------
    output reg first_k_tile,
    output reg last_k_tile,

    // -------------------------------------------------------------------------
    // Sequencer Loop Interface (connects to sequencer.sv for program flow)
    // -------------------------------------------------------------------------
    input wire         loop_check,
    input wire [1:0]   loop_level,           // 0=K, 1=N, 2=M
    output reg         loop_iteration_done,
    output reg [13:0]  loop_target_pc,
    input wire [13:0]  loop_start_pc,

    // Debug
    output wire [15:0] debug_total_tiles  // num_m × num_n × num_k (for monitoring)
);

    // =========================================================================
    // BLOCK A — TILE COUNT + OFFSET CALCULATIONS (combinational)
    // =========================================================================
    assign num_tiles_m = (matrix_m + TILE_SIZE - 1) / TILE_SIZE;
    assign num_tiles_n = (matrix_n + TILE_SIZE - 1) / TILE_SIZE;
    assign num_tiles_k = (matrix_k + TILE_SIZE - 1) / TILE_SIZE;

    assign debug_total_tiles = num_tiles_k * num_tiles_m * num_tiles_n;

    //address
    wire [31:0] a_offset;
    wire [31:0] b_offset;
    wire [31:0] c_offset;

    assign a_offset = tile_m * TILE_SIZE * matrix_k + tile_k * TILE_SIZE;
    assign b_offset = tile_k * TILE_SIZE * matrix_n + tile_n * TILE_SIZE;
    assign c_offset = tile_m * TILE_SIZE * matrix_n + tile_n * TILE_SIZE;

    // -------------------------------------------------------------------------
    // min_dim helper — returns the smaller of two 16-bit values.
    // -------------------------------------------------------------------------
    function [15:0] min_dim;
        input [15:0] a, b;
        begin
            min_dim = (a < b) ? a : b;
        end
    endfunction

    // =========================================================================
    // BLOCK B — STATE DEFINITIONS + REGISTERS
    // =========================================================================
    localparam IDLE = 2'b00;
    localparam RUNNING = 2'b01;
    localparam COMPLETE = 2'b10;
    
    reg [1:0] state;
    
    always @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            done <= 1'b0;
            active <= 1'b0;
            tile_m <= 16'd0;
            tile_n <= 16'd0;
            tile_k <= 16'd0;
            addr_a_tile <= {ADDR_WIDTH{1'b0}};
            addr_b_tile <= {ADDR_WIDTH{1'b0}};
            addr_c_tile <= {ADDR_WIDTH{1'b0}};
            tile_rows_a <= 8'd0;
            tile_cols_a <= 8'd0;
            tile_rows_b <= 8'd0;
            tile_cols_b <= 8'd0;
            tile_rows_c <= 8'd0;
            tile_cols_c <= 8'd0;
            first_k_tile <= 1'b0;
            last_k_tile <= 1'b0;
            loop_iteration_done <= 1'b0;
            loop_target_pc <= 14'd0;
        end else begin
            loop_iteration_done <= 1'b0;

            case (state) 
                IDLE: begin
                    done <= 1'b0;

                    if (start) begin
                        tile_m <= 16'd0;
                        tile_n <= 16'd0;
                        tile_k <= 16'd0;

                        addr_a_tile <= base_addr_a;
                        addr_b_tile <= base_addr_b;
                        addr_c_tile <= base_addr_c;

                        // Calculate initial tile dimensions
                        tile_rows_a <= min_dim(TILE_SIZE, matrix_m);
                        tile_cols_a <= min_dim(TILE_SIZE, matrix_k);
                        tile_rows_b <= min_dim(TILE_SIZE, matrix_k);
                        tile_cols_b <= min_dim(TILE_SIZE, matrix_n);
                        tile_rows_c <= min_dim(TILE_SIZE, matrix_m);
                        tile_cols_c <= min_dim(TILE_SIZE, matrix_n);

                        first_k_tile <= 1'b1;
                        last_k_tile <= (num_tiles_k == 1);

                        active <= 1'b1;
                        state <= RUNNING;
                    end 
                end

                RUNNING: begin
                    if (advance) begin
                        // ─── BRANCH 1: Inner K loop can still advance ─────────────────────
                        // We are computing more partial sums for the same output tile C[m,n].
                        // A moves right (same row-stripe, next k-column).
                        // B moves down (next k-row, same n-column).
                        // C address does NOT change — still writing the same output tile.
                        if (tile_k + 1 < num_tiles_k) begin
                            tile_k       <= tile_k + 1'b1;
                            first_k_tile <= 1'b0;  // Subsequent K passes use accumulate_mode
                            // last_k_tile: will the NEXT k-tile be the last one?
                            last_k_tile  <= (tile_k + 2 >= num_tiles_k);

                            // A[m, k+1]: same row-stripe, next k-column.
                            // byte offset = tile_m*TILE_SIZE*matrix_k + (tile_k+1)*TILE_SIZE
                            addr_a_tile <= base_addr_a + ((tile_m * TILE_SIZE * matrix_k + (tile_k + 1) * TILE_SIZE) >> 2);

                            // B[k+1, n]: next k-row, same n-column.
                            // byte offset = (tile_k+1)*TILE_SIZE*matrix_n + tile_n*TILE_SIZE
                            addr_b_tile <= base_addr_b + (((tile_k + 1) * TILE_SIZE * matrix_n + tile_n * TILE_SIZE) >> 2);

                            // Edge tile dimensions: the last k-tile may have fewer than TILE_SIZE elements.
                            tile_cols_a <= min_dim(TILE_SIZE, matrix_k - (tile_k + 1) * TILE_SIZE);
                            tile_rows_b <= min_dim(TILE_SIZE, matrix_k - (tile_k + 1) * TILE_SIZE);

                        // ─── BRANCH 2: K loop done, advance N ────────────────────────────
                        // C[m,n] is complete. Move to the next column-tile of C.
                        // A resets to the start of the same m-row (back to k=0).
                        // B jumps to the next n-column, starting from its top (k=0).
                        // C jumps one tile to the right.
                        end else if (tile_n + 1 < num_tiles_n) begin
                            tile_k       <= 16'd0;
                            tile_n       <= tile_n + 1'b1;
                            first_k_tile <= 1'b1;  // Fresh C tile — overwrite, don't accumulate
                            last_k_tile  <= (num_tiles_k == 1);

                            // A: reset to start of same m-row (k resets to 0, so no k offset)
                            addr_a_tile <= base_addr_a + ((tile_m * TILE_SIZE * matrix_k) >> 2);

                            // B: next n-column, k=0 (top of B), so no row offset.
                            // byte offset = (tile_n+1)*TILE_SIZE  (only col skip, no row skip)
                            addr_b_tile <= base_addr_b + (((tile_n + 1) * TILE_SIZE) >> 2);

                            // C: same m-row, next n-column.
                            addr_c_tile <= base_addr_c + ((tile_m * TILE_SIZE * matrix_n + (tile_n + 1) * TILE_SIZE) >> 2);

                            // Edge tile dimensions for the new N column.
                            tile_cols_a <= min_dim(TILE_SIZE, matrix_k);
                            tile_rows_b <= min_dim(TILE_SIZE, matrix_k);
                            tile_cols_b <= min_dim(TILE_SIZE, matrix_n - (tile_n + 1) * TILE_SIZE);
                            tile_cols_c <= min_dim(TILE_SIZE, matrix_n - (tile_n + 1) * TILE_SIZE);

                            if (loop_check && loop_level == 2'b00) begin
                                loop_iteration_done <= 1'b1;
                                loop_target_pc      <= loop_start_pc;
                            end

                        // ─── BRANCH 3: K and N done, advance M ───────────────────────────
                        // An entire row-stripe of C is complete. Move to the next row.
                        // A jumps down to the next m-row, n and k both reset.
                        // B resets to its base (all m-tiles use the same B).
                        // C jumps down to the next m-row.
                        end else if (tile_m + 1 < num_tiles_m) begin
                            tile_k       <= 16'd0;
                            tile_n       <= 16'd0;
                            tile_m       <= tile_m + 1'b1;
                            first_k_tile <= 1'b1;
                            last_k_tile  <= (num_tiles_k == 1);

                            // A[m+1, 0]: next m-row, k reset to 0.
                            addr_a_tile <= base_addr_a + (((tile_m + 1) * TILE_SIZE * matrix_k) >> 2);
                            // B: reset to base — all m-tiles loop through the same B.
                            addr_b_tile <= base_addr_b;
                            // C[m+1, 0]: next m-row, n reset to 0.
                            addr_c_tile <= base_addr_c + (((tile_m + 1) * TILE_SIZE * matrix_n) >> 2);

                            // Edge tile dimensions for the new M row.
                            tile_rows_a <= min_dim(TILE_SIZE, matrix_m - (tile_m + 1) * TILE_SIZE);
                            tile_cols_a <= min_dim(TILE_SIZE, matrix_k);
                            tile_rows_b <= min_dim(TILE_SIZE, matrix_k);
                            tile_cols_b <= min_dim(TILE_SIZE, matrix_n);
                            tile_rows_c <= min_dim(TILE_SIZE, matrix_m - (tile_m + 1) * TILE_SIZE);
                            tile_cols_c <= min_dim(TILE_SIZE, matrix_n);

                            if (loop_check && loop_level == 2'b01) begin
                                loop_iteration_done <= 1'b1;
                                loop_target_pc      <= loop_start_pc;
                            end

                        // ─── BRANCH 4: All tiles done ─────────────────────────────────────
                        end else begin
                            done   <= 1'b1;
                            active <= 1'b0;
                            state  <= COMPLETE;

                            if (loop_check && loop_level == 2'b10) begin
                                loop_iteration_done <= 1'b1;
                                loop_target_pc      <= loop_start_pc;
                            end
                        end
                    end
                end

                COMPLETE: begin
                    // Hold done=1 until a new `start` arrives.
                    // This allows back-to-back matrix multiplies without requiring a full reset.
                    if (start) begin
                        tile_m       <= 16'd0;
                        tile_n       <= 16'd0;
                        tile_k       <= 16'd0;
                        done         <= 1'b0;
                        active       <= 1'b1;
                        first_k_tile <= 1'b1;
                        last_k_tile  <= (num_tiles_k == 1);
                        state        <= RUNNING;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
