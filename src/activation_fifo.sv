//weight fifo just holds weights in a 2d arrays and send one row at a time
//activation must arrive in systolic array diagonally skewed, not all at same cycle

//tiled activation before with automatic skew generation
//converts row-major mem layout to skewed systolic input. 
//handles partial tiles at matrix edges

//this module 1)loads NxN tile of activation from the unified buffer via the mem controller 2) streams the acivation with per row diagnoal skew into the systolic array

module activation_fifo #(
    parameter N = 8,                        // Systolic array dimension
    parameter DATA_WIDTH = 8,               // INT8 activations
    parameter TILE_SIZE = 8,                // NxN tile of activations
    parameter BUFFER_DEPTH = 64             // 8x8 = 64 activations per buffer
) (
    input wire clk,
    input wire reset,

    // Memory interface (to unified buffer)
    output reg                      mem_req,
    output reg [13:0]               mem_addr,
    input wire [31:0]               mem_rdata,      // 4 INT8 values per word
    input wire                      mem_valid,

    // Load control
    input wire                      load_start,
    input wire [13:0]               load_base_addr, // Starting address in unified buffer
    input wire [7:0]                load_rows,      // Number of rows to load (1-N)
    input wire [7:0]                load_cols,      // Number of columns per row (1-N)
    input wire [13:0]               load_stride,    // Address stride between rows
    output wire                     load_done,
    output wire                     load_busy,

    // Systolic array interface (skewed output)
    input wire                      stream_enable,  // Start streaming to array
    output wire signed [DATA_WIDTH-1:0] activation_out [N-1:0],  // N activations, one per row
    output wire [N-1:0]             activation_valid,            // Per-row valid signals
    output wire                     stream_done,                 // All data streamed
    output wire                     buffer_ready,                // Data loaded and ready

    // Tile position (for partial tile handling)
    input wire [15:0]               tile_row,       // Current tile row index
    input wire [15:0]               tile_col,       // Current tile column index
    input wire [15:0]               matrix_rows,    // Total matrix rows
    input wire [15:0]               matrix_cols     // Total matrix columns
);

    //activation storage buffer
    reg signed [DATA_WIDTH-1:0] buffer [0:N-1][0:N-1];

    //row i needs i cycles of delay. implement with a chain of register
    //row 0 has 0 delay stages, row 1 has 1 delay stage etc
    reg signed [DATA_WIDTH-1:0] skew_regs [0:N-1][0:N-1];

    //buffer state
    reg buffer_valid;
    reg [5:0] words_loaded;

    //stream state
    reg streaming; //high while streaming data to sys array
    reg [3:0] stream_cycle; //couns 0 to 2N-1
    reg [3:0] stream_col; //current column index for row 0's output

    //load FSM state
    localparam LD_IDLE = 3'b000;
    localparam LD_REQUEST = 3'b001;
    localparam LD_WAIT = 3'b010;
    localparam LD_STORE = 3'b011;
    localparam LD_DONE = 3'b100;

    reg [2:0] ld_state;
    reg [13:0] ld_addr;
    reg [7:0] ld_row_count;
    reg [2:0] ld_current_row;           // Current row index
    reg [2:0] ld_current_col;           // Current column position
    reg [7:0] ld_total_rows;
    reg [7:0] ld_total_cols;
    reg [13:0] ld_row_stride;
    reg [13:0] ld_row_base;             // Base address of current row   reg [7:0] ld_col_count;

    //combination ouput assignment
    assign load_done = (ld_state == LD_DONE);
    assign load_busy = (ld_state != LD_IDLE) && (ld_state != LD_DONE);
    assign buffer_ready = buffer_valid && !streaming;
    assign stream_done = streaming && (stream_cycle >= 2*N - 1);

    //skewed output generation block
    //generate block to wire each rows output differently
    genvar row;
    generate
        for (row = 0; row < N; row = row + 1) begin : skew_output
            if (row == 0) begin
                assign activation_out[0] = streaming ? buffer[0][stream_col] : 0;
            end else begin
                assign activation_out[row] = streaming ? skew_regs[row][row-1] : 8'sd0;
            end
        end
    endgenerate

    //valid signal generation block
    generate
        for (row = 0; row < N; row = row + 1) begin : valid_gen
            assign activation_valid[row] = streaming && (stream_cycle >= row) && (stream_cycle < row + N);
        end
    endgenerate

    //load FSM
    //goal: when load_start pulses, fetch N rows of activation data from memory and unpack them into the buffer

    alwasy @(posedge clk) begin
                if (reset) begin
            ld_state <= LD_IDLE;
            ld_addr <= 14'd0;
            ld_row_count <= 8'd0;
            ld_col_count <= 8'd0;
            ld_current_row <= 3'd0;
            ld_current_col <= 3'd0;
            ld_total_rows <= 8'd0;
            ld_total_cols <= 8'd0;
            ld_row_stride <= 14'd0;
            ld_row_base <= 14'd0;
            mem_req <= 1'b0;
            mem_addr <= 14'd0;
            buffer_valid <= 1'b0;
            words_loaded <= 6'd0;
        end else begin
            case (ld_state)
                LD_IDLE: begin
                    mem_req <= 1'b0;
                    if (load_start && !streaming) begin
                        ld_addr <= load_base_addr;
                        ld_row_base <= load_base_addr;
                        ld_row_count <= load_rows;
                        ld_total_rows <= load_rows;
                        ld_total_cols <= load_cols;
                        ld_row_stride <= load_stride;
                        // Words per row = ceil(cols / 4)
                        ld_col_count <= (load_cols + 3) >> 2;
                        ld_current_row <= 3'd0;
                        ld_current_col <= 3'd0;
                        words_loaded <= 6'd0;
                        buffer_valid <= 1'b0;
                        ld_state <= LD_REQUEST;
                    end
                end

                LD_REQUEST: begin
                    mem_req <= 1'b1;
                    mem_addr <= ld_addr;
                    ld_state <= LD_WAIT;
                end

                LD_WAIT: begin
                    mem_req <= 1'b0;
                    if (mem_valid) begin
                        ld_state <= LD_STORE;
                    end
                end

                LD_STORE: begin
                    //one 32 bit memory word contains 4 packed INT8 values
                    if (ld_current_col < N) buffer[ld_current_row][ld_current_col] <= mem_rdata[7:0];
                    if (ld_current_col + 1 < N) buffer[ld_current_row][ld_current_col + 1] <= mem_rdata[15:8];
                    if (ld_current_col + 2 < N) buffer[ld_current_row][ld_current_col + 2] <= mem_rdata[23:16];
                    if (ld_current_col + 3 < N) buffer[ld_current_row][ld_current_col + 3] <= mem_rdata[31:24];

                    words_loaded <= words_loaded + 1;
                    ld_addr <= ld_addr + 1;
                    ld_col_count <= ld_col_count - 1;

                    if (ld_col_count <= 1) begin
                        //done with this row
                        ld_state <= LD_NEXT_ROW;
                    end else begin
                        ld_state <= LD_REQUEST;
                    end
                end

                LD_NEXT_ROW: begin
                    ld_row_count <= ld_row_count - 1;
                    ld_current_row <= ld_current_row + 1;
                    ld_current_col <= 3'd0;

                    if (ld_row_count <= 1) begin
                        // All rows done
                        ld_state <= LD_DONE;
                    end else begin
                        // Move to next row
                        ld_row_base <= ld_row_base + ld_row_stride;
                        ld_addr <= ld_row_base + ld_row_stride;
                        ld_col_count <= (ld_total_cols + 3) >> 2;
                        ld_state <= LD_REQUEST;
                    end    
                end
                        
                LD_DONE: begin
                    buffer_valid <= 1'b1;
                    ld_state <= LD_IDLE;
                end

                default: begin
                    ld_state <= LD_IDLE;
                end
            endcase
        end
    end

    


    
endmodule
