`default_nettype none
`timescale 1ns/1ns

// =============================================================================
// ACCUMULATOR BUFFER — Result Gathering & Quantization
// =============================================================================
// This module captures the 32-bit results from the systolic array and 
// prepares them for the next layer.
//
// 1. CAPTURE & ACCUMULATE:
//    Captures N 32-bit values from the south edge of the array.
//    If accumulate_mode=1, adds them to existing values (for tiled matmul).
//
// 2. QUANTIZATION:
//    Converts 32-bit sums → 8-bit activations using: 
//    out = clamp( (in * scale) >> 8, -128, 127 )
//    memory is stored in INT8, and we dont want to lose precision for small values. so we need a scale factor that maps the full dynamic range of INT32 to [-128,127]
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
    // BLOCK A — STORAGE
    // =========================================================================
    reg signed [ACC_WIDTH-1:0] accumulators [0:N-1][0:N-1]; //NxN rows of INT32. same shape as PE array. this is where the running sum lives across K-tile passes

    //track which rows are valid
    reg [N-1:0] row_valid;
    reg [7:0] accumulated_rows; //how many distinct rows we have seen

    reg [2:0] capture_row; //current row being captured 
    reg capturing;

    //quantization state machine
    localparam Q_IDLE = 3'b000;
    localparam Q_COMPUTE = 3'b001;
    localparam Q_CLIP = 3'b010;
    localparam Q_OUTPUT = 3'b011;
    localparam Q_WRITEBACK = 3'b100;
    localparam Q_DONE = 3'b101;    

    reg [2:0] quant_state;
    reg [3:0] quant_row;
    reg [3:0]         quant_col;
    reg signed [47:0] quant_product;  // INT32 × INT16 = 48-bit max, must hold full product before clipping
    reg signed [31:0] quant_shifted;  // Shifted result after >>> 8 — declared at module scope (cannot declare regs inside case branches)
    //special buffer for after quantization so you don't overwrite the INT32 accumulator 
    reg signed [DATA_WIDTH-1:0] quant_buffer [0:N-1][0:N-1]; //quantization iterates one element at a time, so would be unstable mid-computation

    //writeback state
    reg [3:0] wb_row;
    reg [3:0] wb_word; //4 INT8s per 32 bit word

    //output assignments
    assign buffer_busy = capturing || (quant_state != Q_IDLE);
    assign rows_accumulated = accumulated_rows;

    //debug output
    genvar di, dj;
    generate
        for (di = 0; di < N; di = di + 1) begin : debug_row
            for (dj = 0; dj < N; dj = dj + 1) begin : debug_col
                assign debug_acc[di][dj] = accumulators[di][dj];
            end
        end
    endgenerate
    
    // =========================================================================
    // BLOCK B — CAPTURE LOGIC
    // =========================================================================
    // [TODO] Implement an 'always' block that listens to result_valid mask
    // [TODO] If result_valid[j] is high, update accumulators[tile_row_idx][j]
    // [TODO] Handle the accumulate_mode toggle (add vs overwrite)
    integer col;
    always @(posedge clk) begin
        if (reset || clear_buffer) begin
            capturing <= 1'b0;
            capture_row <= 1'b0;
            row_valid <= 1'b0;
            accumulated_rows <= 1'b0;
            //clear accumulators
            
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
            //capture results as they arrive (skewed manner)
            for (col = 0; col < N; col = col + 1) begin
                if (result_valid[col]) begin
                    //determine which row this result belongs to 
                    //based on systolic timing, results for row r arrive at cycles r to r+N-1
                    if (accumulate_mode) begin
                        accumulators[tile_row_idx][col] <= accumulators[tile_row_idx][col] + result_in[col];
                    end else begin
                        accumulators[tile_row_idx][col] <= result_in[col];
                    end
                end
            end

            //track accumulated rows
            if (|result_valid && !row_valid[tile_row_idx]) begin //| means "is any bit of result_valid high?" its how detect that at least one column arrived this cycle without writing 8 way OR
                row_valid[tile_row_idx] <= 1'b1;
                accumulated_rows <= accumulated_rows + 1'b1;
            end
        end
    end
    
    //add readback path
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
    // BLOCK C — QUANTIZATION ENGINE
    // =========================================================================
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
                    // Multiply INT32 accumulator by Q8.8 scale factor.
                    // $signed({1'b0, quant_scale}) zero-extends quant_scale to a positive
                    // signed number — without this, the signed×unsigned multiply gives wrong results.
                    // Result is 48 bits to hold the full product without overflow.
                    quant_product <= accumulators[quant_row][quant_col] * $signed({1'b0, quant_scale});
                    quant_state   <= Q_CLIP;
                end
                Q_CLIP: begin
                    // Arithmetic right shift by 8 removes the Q8.8 denominator (divides by 256).
                    // >>> preserves the sign bit (unlike >> which fills with zeros).
                    quant_shifted = quant_product >>> 8;

                    // Saturating clip: if result overflows INT8 range, clamp to boundary.
                    // Without this, a value of 130 would silently wrap to -126 — catastrophic.
                    if (quant_shifted > 127)
                        quant_buffer[quant_row][quant_col] <= 8'sd127;
                    else if (quant_shifted < -128)
                        quant_buffer[quant_row][quant_col] <= -8'sd128;
                    else
                        quant_buffer[quant_row][quant_col] <= quant_shifted[7:0];
                    //advanced to next state
                    if (quant_col >= N-1) begin
                        quant_col <= 4'd0;
                        if (quant_row >= N - 1) begin
                            quant_state <= Q_OUTPUT;
                        end else begin
                            quant_row <= quant_row + 1;
                            quant_state <= Q_COMPUTE;
                        end
                    end else begin
                        quant_col <= quant_col + 1;
                        quant_state <= Q_COMPUTE;
                    end
                end
                Q_OUTPUT: begin
                    //output quantized results row by row
                    for (qi = 0; qi < N; qi = qi + 1) begin
                        quant_out[qi] <= quant_buffer[quant_row][qi];
                    end
                    quant_valid <= 1'b1;

                    if (quant_row >= N-1) begin
                        quant_state <= Q_DONE;       // All N rows streamed — done
                    end else begin
                        quant_row   <= quant_row + 1;
                        quant_state <= Q_OUTPUT;     // BUG FIX: stay in OUTPUT to stream next row
                                                     // (was Q_COMPUTE which would re-quantize)
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


    // Initialize accumulators to zero
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
