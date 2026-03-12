`default_nettype none  // Catch any undeclared wire names at compile time
`timescale 1ns/1ns

// =============================================================================
// ACTIVATION FIFO — Tiled Activation Buffer with Diagonal Skew Generation
// =============================================================================
//
// WHAT THIS MODULE DOES:
//   This module bridges the flat memory layout (unified buffer) and the
//   systolic array's unusual timing requirement: activations must arrive
//   diagonally skewed — NOT all at the same time.
//
// WHY SKEW?
//   In a systolic array doing C = A × B:
//     - Weights (B) sit stationary in each PE.
//     - Activations (A) stream in from the left, one element per PE per cycle.
//     - For PE[i][j] to correctly multiply A[i][k] × B[k][j], row i's activation
//       must arrive exactly i cycles AFTER row 0's activation.
//     - Without skew, all rows get the same element simultaneously → wrong answer.
//
// VISUAL EXAMPLE (4×4 tile):
//
//   Memory layout (row-major):       Skewed output on array input wires:
//     A[0,0] A[0,1] A[0,2] A[0,3]     Cycle 0:  A[0,0]   0       0       0
//     A[1,0] A[1,1] A[1,2] A[1,3]     Cycle 1:  A[0,1]  A[1,0]   0       0
//     A[2,0] A[2,1] A[2,2] A[2,3]     Cycle 2:  A[0,2]  A[1,1]  A[2,0]   0
//     A[3,0] A[3,1] A[3,2] A[3,3]     Cycle 3:  A[0,3]  A[1,2]  A[2,1]  A[3,0]
//                                      Cycle 4:   0      A[1,3]  A[2,2]  A[3,1]
//                                      Cycle 5:   0       0      A[2,3]  A[3,2]
//                                      Cycle 6:   0       0       0      A[3,3]
//
//   Total: 2N-1 = 7 cycles for N=4 (or 15 cycles for N=8)
//   Notice: row i is shifted i columns to the right → i cycles of delay.
//
// TWO JOBS:
//   1. LOAD  — Read an N×N activation tile from the unified buffer (memory).
//              Done with a 6-state FSM. One 32-bit word = 4 INT8 activations.
//   2. STREAM — Output the loaded tile with per-row diagonal delay using
//               a chain of shift registers (one chain per row).
//
// =============================================================================

module activation_fifo #(
    parameter N            = 8,   // Systolic array dimension (N×N)
    parameter DATA_WIDTH   = 8,   // INT8 activations
    parameter TILE_SIZE    = 8,   // NxN tile of activations (same as N by default)
    parameter BUFFER_DEPTH = 64   // N×N = 64 activations per buffer
) (
    input wire clk,
    input wire reset,   // Synchronous active-high reset

    // -------------------------------------------------------------------------
    // Memory Interface — connects to unified_buffer via memory_controller
    // -------------------------------------------------------------------------
    output reg          mem_req,            // 1 = requesting a read this cycle
    output reg [13:0]   mem_addr,           // Address of the word to read
    input wire [31:0]   mem_rdata,          // Data returned: 4 packed INT8 values
    input wire          mem_valid,          // 1 = mem_rdata is valid this cycle

    // -------------------------------------------------------------------------
    // Load Control — tells the module what tile to load
    // -------------------------------------------------------------------------
    input wire          load_start,         // Pulse high for 1 cycle to begin loading
    input wire [13:0]   load_base_addr,     // Starting word address in unified buffer
    input wire [7:0]    load_rows,          // How many rows to load (1 to N)
    input wire [7:0]    load_cols,          // How many columns per row (1 to N)
    input wire [13:0]   load_stride,        // Word-address gap between row starts
    output wire         load_done,          // Pulses high for 1 cycle when loading is complete
    output wire         load_busy,          // High while loading is in progress

    // -------------------------------------------------------------------------
    // Systolic Array Interface — skewed activation output
    // -------------------------------------------------------------------------
    input wire          stream_enable,                          // Pulse to start streaming
    output wire signed [DATA_WIDTH-1:0] activation_out [N-1:0], // One activation per row of array
    output wire [N-1:0] activation_valid,                       // Per-row valid bitmask
    output wire         stream_done,                            // High when all data has been sent
    output wire         buffer_ready,                           // High when buffer is loaded & idle

    // -------------------------------------------------------------------------
    // Tile Position — used for partial tile handling at matrix edges
    // -------------------------------------------------------------------------
    input wire [15:0]   tile_row,       // Which tile row (in a tiled matmul)
    input wire [15:0]   tile_col,       // Which tile column
    input wire [15:0]   matrix_rows,    // Total matrix height (for partial tile clipping)
    input wire [15:0]   matrix_cols     // Total matrix width  (for partial tile clipping)
);

    // =========================================================================
    // BLOCK A — INTERNAL STORAGE
    // =========================================================================

    // N×N buffer holds one complete activation tile.
    // After loading, buffer[i][j] = activation at row i, column j of the tile.
    reg signed [DATA_WIDTH-1:0] buffer [0:N-1][0:N-1];

    // Per-row skew shift registers.
    // Row i needs i cycles of delay before its data appears at the output.
    // We implement this with a chain of i registers per row:
    //   buffer[i][k] → skew_regs[i][0] → skew_regs[i][1] → ... → skew_regs[i][i-1] → output
    //
    // Why N-2 as the last index?
    //   Row 0 → 0 delay stages (reads buffer directly, no skew_regs used)
    //   Row 1 → 1 stage:  uses skew_regs[1][0]
    //   Row 7 → 7 stages: uses skew_regs[7][0..6]
    //   Maximum stages needed = N-1.  Index range: [0 : N-2].
    //
    reg signed [DATA_WIDTH-1:0] skew_regs [0:N-1][0:N-2];

    // =========================================================================
    // BLOCK A — STATE REGISTERS
    // =========================================================================

    // Buffer state
    reg buffer_valid;       // Set to 1 when a full tile has been loaded and is ready to stream
    reg [5:0] words_loaded; // Running count of 32-bit words fetched (diagnostic / debug)

    // Stream state
    reg streaming;          // 1 while the skew output is actively running
    reg [3:0] stream_cycle; // Counts 0 → 2N-2 during one stream. Drives all timing decisions.
    reg [3:0] stream_col;   // Column index that row 0 is currently outputting (see explanation below)

    // =========================================================================
    // BLOCK A — LOAD FSM STATE
    // =========================================================================
    //
    // State machine that acts like a tiny DMA engine:
    //   IDLE → REQUEST → WAIT → STORE → NEXT_ROW (if row done) → REQUEST (if more rows)
    //                                             → DONE        (if all rows done)
    //   DONE → IDLE
    //
    localparam LD_IDLE     = 3'b000; // Waiting for load_start
    localparam LD_REQUEST  = 3'b001; // Sending read request to memory controller
    localparam LD_WAIT     = 3'b010; // Waiting for mem_valid response
    localparam LD_STORE    = 3'b011; // Unpacking 32-bit word into 4 buffer slots
    localparam LD_NEXT_ROW = 3'b100; // Advancing to the next row (or finishing)
    localparam LD_DONE     = 3'b101; // All rows loaded; set buffer_valid

    reg [2:0]  ld_state;
    reg [13:0] ld_addr;         // Current word address being fetched
    reg [7:0]  ld_row_count;    // Rows still left to load (decrements each row)
    reg [7:0]  ld_col_count;    // 32-bit words still needed in the current row
    reg [2:0]  ld_current_row;  // Which row of buffer[] we are filling right now
    reg [2:0]  ld_current_col;  // Which column offset within that row (advances by 4 per word)
    reg [7:0]  ld_total_rows;   // Saved snapshot of load_rows input
    reg [7:0]  ld_total_cols;   // Saved snapshot of load_cols input
    reg [13:0] ld_row_stride;   // Saved snapshot of load_stride
    reg [13:0] ld_row_base;     // Word address of the start of the current row

    // =========================================================================
    // BLOCK A — COMBINATIONAL OUTPUT ASSIGNMENTS
    // =========================================================================
    //
    // These are purely combinational — no clock edge, just continuous wires.
    //
    assign load_done    = (ld_state == LD_DONE);
    assign load_busy    = (ld_state != LD_IDLE) && (ld_state != LD_DONE);
    assign buffer_ready = buffer_valid && !streaming;   // Ready iff loaded AND not mid-stream
    assign stream_done  = streaming && (stream_cycle >= 2*N - 1);

    // =========================================================================
    // BLOCK A — SKEWED OUTPUT GENERATION (generate block)
    // =========================================================================
    //
    // Each row's output wire is connected differently:
    //   Row 0 → reads buffer[0][stream_col] directly (zero delay)
    //   Row i → reads skew_regs[i][i-1]  (the LAST stage of row i's shift chain)
    //
    // The generate block lets us write "if row==0 do X, else do Y" at compile time
    // for each of the N rows simultaneously.
    genvar row;
    generate
        for (row = 0; row < N; row = row + 1) begin : skew_output
            if (row == 0) begin
                // Row 0: no delay at all — read the column directly from buffer.
                // stream_col tracks which column of row 0 is being output this cycle.
                assign activation_out[0] = streaming ? buffer[0][stream_col] : 8'sd0;
            end else begin
                // Row i: output comes from the last stage of its i-deep shift chain.
                // The data entered stage [0] i cycles ago, so it has been delayed by i cycles.
                assign activation_out[row] = streaming ? skew_regs[row][row-1] : 8'sd0; 
            end
        end
    endgenerate

    // =========================================================================
    // BLOCK A — VALID SIGNAL GENERATION (generate block)
    // =========================================================================
    //
    // Row i is valid for exactly N cycles, starting at stream_cycle == i.
    // Why? Because row i is delayed by i cycles, so its first output
    // emerges at cycle i. It then stays valid for N cycles (one per column).
    //
    //   Row 0: valid during stream_cycle 0, 1, 2, ..., N-1
    //   Row 1: valid during stream_cycle 1, 2, 3, ..., N
    //   Row i: valid during stream_cycle i, i+1, ..., i+N-1
    //
    generate
        for (row = 0; row < N; row = row + 1) begin : valid_gen
            assign activation_valid[row] = streaming &&
                                           (stream_cycle >= row) &&
                                           (stream_cycle < row + N);
        end
    endgenerate

    // =========================================================================
    // BLOCK B — LOAD FSM
    // =========================================================================
    //
    // GOAL: When load_start pulses, fetch load_rows rows of activations from
    //       memory (via the memory controller) and store them in buffer[][].
    //
    // MEMORY PACKING: One 32-bit word holds 4 INT8 values (packed little-endian):
    //   word[7:0]  → column k+0
    //   word[15:8] → column k+1
    //   word[23:16]→ column k+2
    //   word[31:24]→ column k+3
    //
    // For a row of 8 INT8 columns we need 8/4 = 2 words (ld_col_count tracks this).
    //
    always @(posedge clk) begin
        if (reset) begin
            ld_state       <= LD_IDLE;
            ld_addr        <= 14'd0;
            ld_row_count   <= 8'd0;
            ld_col_count   <= 8'd0;
            ld_current_row <= 3'd0;
            ld_current_col <= 3'd0;
            ld_total_rows  <= 8'd0;
            ld_total_cols  <= 8'd0;
            ld_row_stride  <= 14'd0;
            ld_row_base    <= 14'd0;
            mem_req        <= 1'b0;
            mem_addr       <= 14'd0;
            buffer_valid   <= 1'b0;
            words_loaded   <= 6'd0;
        end else begin
            case (ld_state)
                // ---------------------------------------------------------------
                LD_IDLE: begin
                // ---------------------------------------------------------------
                // Park here waiting for the controller to pulse load_start.
                // We also gate on !streaming to ensure we never overwrite a buffer
                // that is still being consumed by the systolic array.
                    mem_req <= 1'b0;
                    if (load_start && !streaming) begin
                        // Snapshot all input parameters (they may change next cycle)
                        ld_addr        <= load_base_addr;
                        ld_row_base    <= load_base_addr;
                        ld_row_count   <= load_rows;
                        ld_total_rows  <= load_rows;
                        ld_total_cols  <= load_cols;
                        ld_row_stride  <= load_stride;
                        // ceil(cols / 4): because each 32-bit word carries 4 INT8 values.
                        // Example: 8 cols → (8+3)>>2 = 2 words.
                        //          5 cols → (5+3)>>2 = 2 words (last word is partial).
                        ld_col_count   <= (load_cols + 3) >> 2; //words needed in the current row
                        ld_current_row <= 3'd0;
                        ld_current_col <= 3'd0;
                        words_loaded   <= 6'd0;
                        buffer_valid   <= 1'b0; // Buffer content is now stale
                        ld_state       <= LD_REQUEST;
                    end
                end

                // ---------------------------------------------------------------
                LD_REQUEST: begin
                // ---------------------------------------------------------------
                // Assert mem_req for one cycle to ask the memory controller for
                // the word at ld_addr.
                    mem_req  <= 1'b1;
                    mem_addr <= ld_addr;
                    ld_state <= LD_WAIT;
                end

                // ---------------------------------------------------------------
                LD_WAIT: begin
                // ---------------------------------------------------------------
                // De-assert mem_req and wait for mem_valid to pulse back.
                // The memory controller may take multiple cycles to respond.
                    mem_req <= 1'b0;
                    if (mem_valid) begin
                        ld_state <= LD_STORE;
                    end
                end

                // ---------------------------------------------------------------
                LD_STORE: begin
                // ---------------------------------------------------------------
                // Unpack the 32-bit word from mem_rdata into up to 4 buffer slots.
                // Guard each write with a bounds check (< N) to handle partial rows
                // at the edge of a matrix — columns beyond load_cols stay zero.
                    if (ld_current_col < N)     buffer[ld_current_row][ld_current_col]     <= mem_rdata[7:0];
                    if (ld_current_col + 1 < N) buffer[ld_current_row][ld_current_col + 1] <= mem_rdata[15:8];
                    if (ld_current_col + 2 < N) buffer[ld_current_row][ld_current_col + 2] <= mem_rdata[23:16];
                    if (ld_current_col + 3 < N) buffer[ld_current_row][ld_current_col + 3] <= mem_rdata[31:24];

                    words_loaded   <= words_loaded + 1;
                    ld_addr        <= ld_addr + 1;        // Advance to next word
                    ld_col_count   <= ld_col_count - 1;   // One fewer word needed
                    ld_current_col <= ld_current_col + 4; // 4 columns consumed

                    if (ld_col_count <= 1) begin
                        // This was the last word for the current row
                        ld_state <= LD_NEXT_ROW;
                    end else begin
                        ld_state <= LD_REQUEST; // Fetch next word in this row
                    end
                end

                // ---------------------------------------------------------------
                LD_NEXT_ROW: begin
                // ---------------------------------------------------------------
                // Current row is complete. Either move to the next row or finish.
                    ld_row_count   <= ld_row_count - 1;
                    ld_current_row <= ld_current_row + 1;
                    ld_current_col <= 3'd0; // Reset column position for the new row

                    if (ld_row_count <= 1) begin
                        // That was the last row
                        ld_state <= LD_DONE;
                    end else begin
                        // Jump to the next row's starting address.
                        // ld_row_stride is the word-address gap between rows.
                        // We update ld_row_base combinatorially here and reload it.
                        ld_row_base  <= ld_row_base + ld_row_stride;
                        ld_addr      <= ld_row_base + ld_row_stride;
                        ld_col_count <= (ld_total_cols + 3) >> 2; // Recalculate words/row
                        ld_state     <= LD_REQUEST;
                    end
                end

                // ---------------------------------------------------------------
                LD_DONE: begin
                // ---------------------------------------------------------------
                // Signal that the buffer is ready for streaming, then go idle.
                // The one-cycle LD_DONE state lets load_done pulse for exactly 1 cycle.
                    buffer_valid <= 1'b1;
                    ld_state     <= LD_IDLE;
                end

                default: ld_state <= LD_IDLE;

            endcase
        end
    end

    // =========================================================================
    // BLOCK C — STREAMING AND SKEW LOGIC
    // =========================================================================
    //
    // GOAL: When stream_enable goes high (and buffer is valid), drive 2N-1 cycles
    //       of diagonally-skewed activation output into the systolic array.
    //
    //   stream_col is the column index that ROW 0 is currently outputting.
    //   Row 0 has NO delay, so it outputs buffer[0][0], buffer[0][1], etc.
    //   every single cycle.  stream_col just tracks which one.
    //
    //   Why the two-branch update?
    //     In a full N=8 stream, cycles 0..14 happen. Row 0 is valid cycles 0..7.
    //     During cycles 0..6:  stream_col = stream_cycle + 1 NEXT cycle
    //     During cycles 7..13: stream_col just increments normally (capped at N-1)
    //     After cycle 7, row 0 is no longer valid, so stream_col value doesn't matter
    //     for correctness — but we still track it cleanly for debug.
    //
    //   Concretely for N=8:
    //     stream_cycle=0 → stream_col set to 0 at start → output buffer[0][0]
    //     stream_cycle=1 → stream_col=1              → output buffer[0][1]
    //     ...
    //     stream_cycle=7 → stream_col=7              → output buffer[0][7]
    //     (row 0 goes invalid after this cycle)
    //
    //   For row 1 (i=1), we want buffer[1][0] to appear at the output on cycle 1,
    //   buffer[1][1] on cycle 2, etc. (1 cycle behind row 0).
    //   We achieve this with ONE shift register stage:
    //     Cycle 0: skew_regs[1][0] ← buffer[1][0]
    //     Cycle 1: output = skew_regs[1][0] = buffer[1][0]  ← exactly 1 cycle delayed ✓
    //
    //   For row 2 (i=2), we want 2 cycles of delay. TWO stages:
    //     Cycle 0: skew_regs[2][0] ← buffer[2][0]
    //     Cycle 1: skew_regs[2][1] ← skew_regs[2][0] = buffer[2][0]
    //     Cycle 2: output = skew_regs[2][1] = buffer[2][0]  ← 2 cycles delayed ✓
    //
    //   General rule for row i:
    //     Stage 0 is loaded from buffer[i][stream_cycle - i + 1]
    //     (the "+1" schedules the load 1 cycle BEFORE the value is needed at the output)
    //     The load window is: stream_cycle in [i-1, i-1+N-1] = [i-1, i+N-2]
    //     i.e.  stream_cycle >= i-1  AND  stream_cycle < i-1+N = i+N-1
    //     Stages 1 through i-1 just shift the data right by one more cycle each.
    //
    integer i, j;
    always @(posedge clk) begin
        if (reset) begin
            streaming    <= 1'b0;
            stream_cycle <= 4'd0;
            stream_col   <= 4'd0;
            // Zero all shift register stages to prevent X propagation in simulation
            for (i = 0; i < N; i = i + 1) begin
                for (j = 0; j < N - 1; j = j + 1) begin
                    skew_regs[i][j] <= 8'sd0;
                end
            end
        end else begin

            if (stream_enable && buffer_valid && !streaming) begin
                // ---------------------------------------------------------------
                // START OF STREAM
                // The upstream controller (matrix_controller) pulses stream_enable.
                // We begin the 2N-1 cycle diagonal output sequence.
                // ---------------------------------------------------------------
                streaming    <= 1'b1;
                stream_cycle <= 4'd0;
                stream_col   <= 4'd0;

            end else if (streaming) begin
                // ---------------------------------------------------------------
                // MID-STREAM — advance cycle counter and update skew registers
                // ---------------------------------------------------------------
                stream_cycle <= stream_cycle + 1;

                // Update stream_col: tracks which column row 0 is outputting.
                // During the first N-1 cycles, it runs 1 ahead of stream_cycle
                // (because the buffer read happens combinationally, referenced by
                // the CURRENT stream_col, while stream_cycle shows what just happened).
                // After cycle N-1, it just keeps incrementing until capped at N-1.
                if (stream_cycle < N - 1) begin
                    stream_col <= stream_cycle + 1;
                end else if (stream_col < N - 1) begin
                    stream_col <= stream_col + 1;
                end

                // ---------------------------------------------------------------
                // Per-row shift register update — THE SKEW ENGINE
                // ---------------------------------------------------------------
                // Row 0 is handled by the generate block (direct buffer read, no regs).
                // Rows 1..N-1 each have a shift chain of depth i.
                for (i = 1; i < N; i = i + 1) begin

                    // Stage 0 — feed new data from buffer into the shift chain.
                    // The load window for row i starts at stream_cycle = i-1 and
                    // lasts N cycles (one per column).
                    if (stream_cycle >= i - 1 && stream_cycle < i + N - 1) begin
                        // Column index = how far into this row we have advanced.
                        // At stream_cycle = i-1, we load column 0 of row i.
                        // At stream_cycle = i,   we load column 1, etc.
                        skew_regs[i][0] <= buffer[i][stream_cycle - i + 1];
                    end else begin
                        skew_regs[i][0] <= 8'sd0; // Flush zeros outside valid window
                    end

                    // Stages 1 through i-1 — each stage shifts data one step forward.
                    // This is a simple pipeline: reg[j] <= reg[j-1].
                    // After i cycles, data placed in stage 0 reaches stage i-1 (the output).
                    for (j = 1; j < i; j = j + 1) begin
                        skew_regs[i][j] <= skew_regs[i][j-1];
                    end
                end

                // ---------------------------------------------------------------
                // END OF STREAM
                // 2N-2 is the last cycle (stream runs cycles 0 through 2N-2 inclusive).
                // After this we clear streaming and mark the buffer as consumed.
                // The tiling controller must call load_start again before the next tile.
                // ---------------------------------------------------------------
                if (stream_cycle >= 2*N - 2) begin
                    streaming    <= 1'b0;
                    buffer_valid <= 1'b0; // Buffer consumed; must be reloaded before next stream
                end

            end  // else if (streaming)
        end
    end // always

    // =========================================================================
    // BLOCK D — SIMULATION INITIALIZATION
    // =========================================================================
    // `initial` blocks are NOT synthesised (Xilinx BRAM initialises to zero
    // automatically via the INIT attribute).
    // This block exists solely so the simulator doesn't propagate X values
    // through the buffer before the first load completes.
    //
    integer r, c;
    initial begin
        for (r = 0; r < N; r = r + 1) begin
            for (c = 0; c < N; c = c + 1) begin
                buffer[r][c] = 8'sd0;
            end
        end
    end

endmodule
