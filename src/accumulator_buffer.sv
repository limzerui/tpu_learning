`default_nettype none
`timescale 1ns/1ns

// =============================================================================
// ACCUMULATOR BUFFER — Result Gathering & Quantization
// =============================================================================
//
// This module sits between the systolic array and memory. It has three jobs:
//
// 1. CAPTURE:
//    Each cycle, catches up to N INT32 partial sums draining from the south
//    edge of the systolic array. result_valid[col] flags which columns are live.
//
// 2. ACCUMULATE:
//    For tiled matrix multiply (K > N), the same output tile is computed in
//    multiple passes. accumulate_mode=1 adds each pass to the running total.
//    accumulate_mode=0 overwrites (used on the first K-tile pass).
//
// 3. QUANTIZE:
//    Converts INT32 sums → INT8 outputs using fixed-point scaling:
//      out = clamp( (acc * scale) >>> 8, -128, 127 )
//    scale is Q8.8 fixed-point (1.0 = 256). Serial FSM processes one element
//    per 2 clock cycles to reuse a single DSP multiplier across all 64 cells.
// =============================================================================

module accumulator_buffer #(
    parameter N          = 8,   // Array dimension
    parameter ACC_WIDTH  = 32,  // Accumulator precision
    parameter DATA_WIDTH = 8,   // Output precision (INT8) after quantization
    parameter TILE_SIZE  = 8    // Tile size
)(
    input wire clk,
    input wire reset,

    // -------------------------------------------------------------------------
    // Systolic Array Interface (Input)
    // -------------------------------------------------------------------------
    input wire signed [ACC_WIDTH-1:0] result_in [N-1:0],
    input wire [N-1:0] result_valid,
    input wire results_enable, //accept results from array

    // -------------------------------------------------------------------------
    // Accumulation Control
    // -------------------------------------------------------------------------
    input wire accumulate_mode, //1=add, 0=overwrite
    input wire clear_buffer, //clear accumulators

    // tile_row_idx: matrix_controller tells us which output row is arriving this cycle
    input wire [7:0] tile_row_idx,
    input wire [7:0] k_tile_idx, //K dimension tile index

    // -------------------------------------------------------------------------
    // Quantization & Readback (Output)
    // -------------------------------------------------------------------------
    input wire read_enable,
    input wire [7:0] read_row,
    output reg signed [ACC_WIDTH-1:0] read_data [N-1:0],
    output reg read_valid,

    input wire quant_enable, //start quantization
    input wire [15:0] quant_scale, //fixed point scale Q8.8
    input wire quant_zero_point, //output zero point offset
    output reg signed [DATA_WIDTH-1:0] quant_out [N-1:0], //quantized output
    output reg quant_valid,
    output reg quant_done,

    //write back to unified buffer
    output reg wb_request,
    output reg [13:0] wb_addr,
    output reg [31:0] wb_data,
    input wire wb_ack,

    //status
    output wire buffer_busy,
    output wire [7:0] rows_accumulated,
    output wire [ACC_WIDTH-1:0] debug_acc [N-1:0][N-1:0]
);

    // =========================================================================
    // STORAGE
    // =========================================================================

    // Primary accumulator: NxN INT32 grid, one cell per output element of C.
    // Indexed [row][col] — same spatial layout as the systolic array.
    reg signed [ACC_WIDTH-1:0] accumulators [0:N-1][0:N-1];

    // Row tracking — records which rows have received at least one valid result.
    reg [N-1:0] row_valid;
    reg [7:0]   accumulated_rows;

    // Capture control — used by buffer_busy to signal activity.
    reg [2:0] capture_row;
    reg       capturing;

    // =========================================================================
    // QUANTIZATION STATE MACHINE
    // Iterates through all NxN cells: Q_COMPUTE (multiply) → Q_CLIP (shift+clamp)
    // Repeats 64 times, then streams results row-by-row in Q_OUTPUT.
    // =========================================================================
    localparam Q_IDLE      = 3'b000;
    localparam Q_COMPUTE   = 3'b001;  // Issue multiply for one element
    localparam Q_CLIP      = 3'b010;  // Shift and saturate-clip to INT8
    localparam Q_OUTPUT    = 3'b011;  // Stream quantized rows out
    localparam Q_WRITEBACK = 3'b100;  // (reserved for DMA writeback)
    localparam Q_DONE      = 3'b101;  // Assert quant_done, return to idle

    reg [2:0]  quant_state;
    reg [3:0]  quant_row;
    reg [3:0]  quant_col;

    // 48-bit product holds INT32 × INT16 without overflow before the >>> 8 shift.
    reg signed [47:0] quant_product;

    // Shifted result — must be at module scope; regs cannot be declared inside case branches.
    reg signed [31:0] quant_shifted;

    // Staging buffer: holds INT8 results after quantization, before streaming output.
    // Separate from accumulators so INT32 data is not corrupted mid-computation.
    reg signed [DATA_WIDTH-1:0] quant_buffer [0:N-1][0:N-1];

    // Writeback position trackers (used if Q_WRITEBACK DMA is implemented).
    reg [3:0] wb_row;
    reg [3:0] wb_word;

    // =========================================================================
    // COMBINATIONAL OUTPUT ASSIGNMENTS
    // =========================================================================
    assign buffer_busy      = capturing || (quant_state != Q_IDLE);
    assign rows_accumulated = accumulated_rows;

    // Expose full accumulator array for external debug visibility.
    genvar di, dj;
    generate
        for (di = 0; di < N; di = di + 1) begin : debug_row
            for (dj = 0; dj < N; dj = dj + 1) begin : debug_col
                assign debug_acc[di][dj] = accumulators[di][dj];
            end
        end
    endgenerate
    
    // =========================================================================
    // CAPTURE LOGIC
    // Watches result_valid each cycle. When result_valid[col] is high, the
    // systolic array column `col` has a valid partial sum. It is written into
    // accumulators[tile_row_idx][col] — matrix_controller drives tile_row_idx
    // in sync with which activation row is currently being processed.
    // =========================================================================
    integer col;
    always @(posedge clk) begin
        if (reset || clear_buffer) begin
            // Reset or clear_buffer: initialize capture state and accumulators
            capturing        <= 1'b0;
            capture_row      <= 3'd0;         // 3-bit reg — must use 3'd0, not 1'b0
            row_valid        <= {N{1'b0}};    // N-bit bitmask — clear all bits
            accumulated_rows <= 8'd0;         // 8-bit counter
            
            // Clear all accumulator cells
            for (col = 0; col < N; col = col + 1) begin
                accumulators[0][col] <= 32'sd0;
                accumulators[1][col] <= 32'sd0;
                accumulators[2][col] <= 32'sd0;
                accumulators[3][col] <= 32'sd0;
                accumulators[4][col] <= 32'sd0;
                accumulators[5][col] <= 32'sd0;
                accumulators[6][col] <= 32'sd0;
                accumulators[7][col] <= 32'sd0;
            end
        end else if (results_enable) begin
            // Results arrive skewed — column i exits the array i cycles after column 0.
            // result_valid[col] is high exactly when column `col` has valid data.
            for (col = 0; col < N; col = col + 1) begin
                if (result_valid[col]) begin
                    if (accumulate_mode)
                        // K-tile accumulation: add this pass's partial sum to the running total.
                        accumulators[tile_row_idx][col] <= accumulators[tile_row_idx][col] + result_in[col];
                    else
                        // First pass (or single-tile): overwrite with fresh result.
                        accumulators[tile_row_idx][col] <= result_in[col];
                end
            end

            // |result_valid is a one-line OR-reduction: true if any column arrived this cycle.
            // row_valid prevents double-counting the same row across multiple cycles.
            if (|result_valid && !row_valid[tile_row_idx]) begin
                row_valid[tile_row_idx] <= 1'b1;
                accumulated_rows        <= accumulated_rows + 1'b1;
            end
        end
    end
    
    // =========================================================================
    // READBACK PATH
    // Allows the matrix_controller or host to inspect raw INT32 accumulator
    // values before quantization (e.g. for debugging or layer fusion).
    // Outputs one full row per cycle when read_enable is asserted.
    // =========================================================================
    always @(posedge clk) begin
        if (reset) begin
            read_valid <= 1'b0;
            for (col = 0; col < N; col = col + 1) begin
                read_data[col] <= 32'sd0;
            end
        end else begin
            read_valid <= 1'b0;
            if (read_enable && read_row < N) begin
                for (col = 0; col < N; col = col + 1) begin
                    read_data[col] <= accumulators[read_row][col];
                end
                read_valid <= 1'b1;
            end
        end
    end
    // =========================================================================
    // QUANTIZATION ENGINE
    // Serial FSM: processes one accumulator cell per 2 cycles (COMPUTE → CLIP),
    // scanning all NxN=64 cells. Results staged in quant_buffer, then streamed
    // out row-by-row in Q_OUTPUT. Total latency: 2*N*N + N cycles.
    // =========================================================================
    // WHY SERIAL? One DSP multiplier reused 64 times saves 63 multipliers vs
    // a fully parallel implementation, at the cost of ~128 cycles of latency.
    integer qi, qj;
    always @(posedge clk) begin
        if (reset) begin
            quant_state <= Q_IDLE;
            quant_row <= 4'd0;
            quant_col <= 4'd0;
            quant_valid <= 1'b0;
            quant_done <= 1'b0;
            quant_product <= 48'sd0;
            wb_request <= 1'b0;
            wb_addr <= 14'd0;
            wb_data <= 32'd0;
            wb_row <= 4'd0;
            wb_word <= 4'd0;
            for (qi = 0; qi < N; qi = qi + 1) begin
                quant_out[qi] <= 8'sd0;
                for (qj = 0; qj < N; qj = qj + 1) begin
                    quant_buffer[qi][qj] <= 8'sd0;
                end
            end
        end else begin
            case (quant_state) 
                Q_IDLE: begin
                    quant_valid <= 1'b0;
                    quant_done <= 1'b0;
                    wb_request <= 1'b0;
                    if (quant_enable) begin
                        quant_state <= Q_COMPUTE;
                        quant_row <= 4'd0;
                        quant_col <= 4'd0;
                    end
                end
                Q_COMPUTE: begin
                    // Multiply accumulator by the Q8.8 scale factor.
                    // $signed({1'b0, quant_scale}) zero-extends quant_scale into a positive
                    // signed value so the signed × unsigned multiply is correct.
                    // quant_product is 48 bits to hold the full result without overflow.
                    quant_product <= accumulators[quant_row][quant_col] * $signed({1'b0, quant_scale});
                    quant_state   <= Q_CLIP;
                end
                Q_CLIP: begin
                    // >>> is arithmetic right shift — fills upper bits with the sign bit,
                    // preserving the correct sign for negative values (unlike >> which fills with 0).
                    quant_shifted = quant_product >>> 8;

                    // Saturating clip to INT8 range [-128, 127].
                    // Without saturation, 130 would silently wrap to -126.
                    if (quant_shifted > 127)
                        quant_buffer[quant_row][quant_col] <= 8'sd127;
                    else if (quant_shifted < -128)
                        quant_buffer[quant_row][quant_col] <= -8'sd128;
                    else
                        // $signed() is CRITICAL: without it quant_shifted[7:0] is treated as
                        // unsigned, silently corrupting any negative quantized output.
                        quant_buffer[quant_row][quant_col] <= $signed(quant_shifted[7:0]);

                    // Advance the [row][col] scan cursor across all NxN elements.
                    if (quant_col >= N-1) begin
                        quant_col <= 4'd0;
                        if (quant_row >= N-1) begin
                            quant_state <= Q_OUTPUT;          // All 64 elements done
                        end else begin
                            quant_row   <= quant_row + 1;
                            quant_state <= Q_COMPUTE;         // Next row
                        end
                    end else begin
                        quant_col   <= quant_col + 1;
                        quant_state <= Q_COMPUTE;             // Next column
                    end
                end

                Q_OUTPUT: begin
                    // Stream one full row of INT8 values per cycle.
                    // Downstream (memory controller) latches quant_out while quant_valid is high.
                    for (qi = 0; qi < N; qi = qi + 1) begin
                        quant_out[qi] <= quant_buffer[quant_row][qi];
                    end
                    quant_valid <= 1'b1;

                    if (quant_row >= N-1) begin
                        quant_state <= Q_DONE;
                    end else begin
                        quant_row   <= quant_row + 1;
                        quant_state <= Q_OUTPUT;  // Continue streaming next row
                    end
                end
                Q_DONE: begin
                    quant_valid <= 1'b0;
                    quant_done <= 1'b1;
                    quant_state <= Q_IDLE;
                end

                default: quant_state <= Q_IDLE;
            endcase
        end
    end


    // =========================================================================
    // SIMULATION INITIALISATION
    // Prevents X-propagation in simulation. Not synthesised on FPGA.
    // =========================================================================
    integer ri, ci;
    initial begin
        for (ri = 0; ri < N; ri = ri + 1) begin
            for (ci = 0; ci < N; ci = ci + 1) begin
                accumulators[ri][ci] = 32'sd0;
                quant_buffer[ri][ci] = 8'sd0;
            end
        end
    end


endmodule
