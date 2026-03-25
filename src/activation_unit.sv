

module activation_unit #(
    parameter N = 8,                    // Number of parallel lanes
    parameter DATA_WIDTH = 8,           // Input/output width (INT8)
    parameter LUT_ADDR_WIDTH = 8        // 256-entry LUTs
) (
    input wire clk,
    input wire reset,

    input wire enable, //start processing
    input wire [2:0] func_sel,
    output reg busy,
    output reg done,

    input wire [N*DATAWIDTH-1:0] data_in,
    input wire data_in_valid,
    output reg [N*DATAWIDTH-1:0] data_out,
    output reg data_out_valid,

    output wire [3:0] debug_state
);

    localparam FUNC_RELU = 3'b000;
    localparam FUNC_GELU = 3'b001;
    localparam FUNC_SILU = 3'b010;
    localparam FUNC_SIGMOID = 3'b011;
    localparam FUNC_TANH = 3'b100;

    localparam IDLE = 2'b00;
    localparam COMPUTE = 2'b01;
    localparam OUTPUT = 2'b010;

    reg[1:0] state;
    assign debug_state = state;

    reg signed [DATA_WIDTH-1:0] input_regs [0:N-1];
    reg signed [DATA_WIDTH-1:0] output_regs [0:N-1];

    //sigmoid LUT (256 entries, Q0.7 format )
    //input is INT8, output is INT8 representing [0,1)
    //map to input range
    reg [DATA_WIDTH-1:0] sigmoid_lut [0:255];
    
    //Tanh LUT (256 entries, Q0.7 format )
    //input is INT8, output is INT8 representing [-1,1)
    //map to input range
    reg signed [DATA_WIDTH-1:0] tanh_lut [0:255];
    
    //GELU approximation using tanh
    // Using approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x^3)))
    reg signed [DATA_WIDTH-1:0] gelu_lut [0:255];
    
    

endmodule
