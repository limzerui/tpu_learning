//weight fifo just holds weights in a 2d arrays and send one row at a time
//activation must arrive in systolic array diagonally skewed, not all at same cycle

//tiled activation before with automatic skew generation
//converts row-major mem layout to skewed systolic input. 
//handles partial tiles at matrix edges

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


