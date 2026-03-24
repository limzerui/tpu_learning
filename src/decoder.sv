
// [31:24] Opcode (8 bits) | [23:20] Flags (4 bits) | [19:16] Dst (4 bits) | [15:8] Src1 (8 bits) | [7:0] Src2/Imm (8 bits)
module decoder (
    input wire clk,
    input wire reset,

    input wire [2:0]            core_state,         // Current FSM state
    input wire                  decode_enable,      // Enable decoding
    input wire [31:0]           instruction,        // 32-bit instruction

    output reg [7:0]            decoded_opcode,
    output reg [3:0]            decoded_flags,
    output reg [3:0]            decoded_dst,
    output reg [7:0]            decoded_src1,
    output reg [7:0]            decoded_src2,

    output wire flag_accumulate,
    output wire flag_async,
    output wire flag_broadcast,
    output wire flag_transpose,

    output reg mem_read_enable,
    output reg mem_write_enable,
    output reg [1:0] mem_target, // 0=weights, 1=activations, 2=outputs, 3=scratch

    output reg array_enable,
    output reg array_weight_load,
    output reg array_clear_acc,

    output reg activation_enable,
    output reg [2:0] activation_func, // 0=ReLU, 1=GELU, 2=SiLU, 3=Sigmoid, 4=Tanh

    output reg matmul_start,
    output reg softmax_start,
    output reg layernorm_start,
    output reg transpose_start,
    output reg add_start,
    output reg scale_start,

    output reg sync_wait,
    output reg loop_start,
    output reg loop_end,
    output reg halt,

    output reg is_memory_op,
    output reg is_compute_op,
    output reg is_control_op,

    output wire [31:0] debug_instruction
);

    localparam OP_NOP       = 8'h00;
    localparam OP_LOAD_W    = 8'h01;
    localparam OP_LOAD_A    = 8'h02;
    localparam OP_MATMUL    = 8'h03;
    localparam OP_STORE     = 8'h04;
    localparam OP_ACT_RELU  = 8'h05;
    localparam OP_ACT_GELU  = 8'h06;
    localparam OP_ACT_SILU  = 8'h07;
    localparam OP_SOFTMAX   = 8'h08;
    localparam OP_ADD       = 8'h09;
    localparam OP_LAYERNORM = 8'h0A;
    localparam OP_TRANSPOSE = 8'h0B;
    localparam OP_SCALE     = 8'h0C;
    localparam OP_SYNC      = 8'h0D;
    localparam OP_LOOP      = 8'h0E;
    localparam OP_HALT      = 8'h0F;

    localparam STATE_DECODE = 3'b010;

    //flag 
    assign flag_accumulate = decoded_flags[0];
    assign flag_async = decoded_flags[1];
    assign flag_broadcast = decoded_flags[2];
    assign flag_transpose = decoded_flags[3];

    assign debug_instruction = instruction;

    always @(posedge clk) begin
        if (reset) begin
            // Clear all decoded fields
            decoded_opcode <= 8'h00;
            decoded_flags <= 4'h0;
            decoded_dst <= 4'h0;
            decoded_src1 <= 8'h00;
            decoded_src2 <= 8'h00;

            // Clear all control signals
            mem_read_enable <= 1'b0;
            mem_write_enable <= 1'b0;
            mem_target <= 2'b00;

            array_enable <= 1'b0;
            array_weight_load <= 1'b0;
            array_clear_acc <= 1'b0;

            activation_enable <= 1'b0;
            activation_func <= 3'b000;

            matmul_start <= 1'b0;
            softmax_start <= 1'b0;
            layernorm_start <= 1'b0;
            transpose_start <= 1'b0;
            add_start <= 1'b0;
            scale_start <= 1'b0;

            sync_wait <= 1'b0;
            loop_start <= 1'b0;
            loop_end <= 1'b0;
            halt <= 1'b0;

            is_memory_op <= 1'b0;
            is_compute_op <= 1'b0;
            is_control_op <= 1'b0;
        end else if (decode_enable || core_state == STATE_DECODE) begin
            decoded_opcode <= instruction[31:24];
            decoded_flags <= instruction[23:20];
            decoded_dst <= instruction[19:16];
            decoded_src1 <= instruction[15:8];
            decoded_src2 <= instruction[7:0];

            // Reset all control signals
            mem_read_enable <= 1'b0;
            mem_write_enable <= 1'b0;
            mem_target <= 2'b00;

            array_enable <= 1'b0;
            array_weight_load <= 1'b0;
            array_clear_acc <= 1'b0;

            activation_enable <= 1'b0;
            activation_func <= 3'b000;

            matmul_start <= 1'b0;
            softmax_start <= 1'b0;
            layernorm_start <= 1'b0;
            transpose_start <= 1'b0;
            add_start <= 1'b0;
            scale_start <= 1'b0;

            sync_wait <= 1'b0;
            loop_start <= 1'b0;
            loop_end <= 1'b0;
            halt <= 1'b0;

            is_memory_op <= 1'b0;
            is_compute_op <= 1'b0;
            is_control_op <= 1'b0;
            // Decode based on opcode
            case (instruction[31:24])
                OP_NOP: begin
                    is_control_op <= 1'b1;
                end

                OP_LOAD_W: begin
                    mem_read_enable <= 1'b1;
                    mem_target <= 2'b00;  // Weights region
                    array_weight_load <= 1'b1;
                    is_memory_op <= 1'b1;
                end

                OP_LOAD_A: begin
                    // Load activations from unified buffer to activation FIFO
                    mem_read_enable <= 1'b1;
                    mem_target <= 2'b01;  // Activations region
                    is_memory_op <= 1'b1;
                end

                OP_MATMUL: begin
                    // Execute matrix multiplication on systolic array
                    array_enable <= 1'b1;
                    array_clear_acc <= ~instruction[20];  // Clear unless accumulate flag set
                    matmul_start <= 1'b1;
                    is_compute_op <= 1'b1;
                end

                OP_STORE: begin
                    // Store results from accumulator to unified buffer
                    mem_write_enable <= 1'b1;
                    mem_target <= 2'b10;  // Outputs region
                    is_memory_op <= 1'b1;
                end

                OP_ACT_RELU: begin
                    // Apply ReLU activation
                    activation_enable <= 1'b1;
                    activation_func <= 3'b000;  // ReLU
                    is_compute_op <= 1'b1;
                end

                OP_ACT_GELU: begin
                    // Apply GELU activation
                    activation_enable <= 1'b1;
                    activation_func <= 3'b001;  // GELU
                    is_compute_op <= 1'b1;
                end

                OP_ACT_SILU: begin
                    // Apply SiLU/Swish activation
                    activation_enable <= 1'b1;
                    activation_func <= 3'b010;  // SiLU
                    is_compute_op <= 1'b1;
                end

                OP_SOFTMAX: begin
                    // Apply softmax (multi-pass)
                    softmax_start <= 1'b1;
                    is_compute_op <= 1'b1;
                end

                OP_ADD: begin
                    // Element-wise addition
                    add_start <= 1'b1;
                    is_compute_op <= 1'b1;
                end

                OP_LAYERNORM: begin
                    layernorm_start <= 1'b1;
                    is_compute_op <= 1'b1;
                end

                OP_TRANSPOSE: begin
                    // Transpose matrix in buffer
                    transpose_start <= 1'b1;
                    is_compute_op <= 1'b1;
                end

                OP_SCALE: begin
                    // Scale by constant (src2 = scale factor)
                    scale_start <= 1'b1;
                    is_compute_op <= 1'b1;
                end

                OP_SYNC: begin
                    // Wait for all pending operations
                    sync_wait <= 1'b1;
                    is_control_op <= 1'b1;
                end

                OP_LOOP: begin
                    // Loop control
                    // dst[3:0] = loop index (0-2 for M, N, K)
                    // src1 = loop count
                    // src2 = target PC for loop start
                    loop_start <= 1'b1;
                    is_control_op <= 1'b1;
                end

                OP_HALT: begin
                    // Stop execution
                    halt <= 1'b1;
                    is_control_op <= 1'b1;
                end

                default: begin
                    // Unknown opcode - treat as NOP
                    is_control_op <= 1'b1;
                end
            endcase
        end
    end
    endmodule

module instruction_builder (
    input wire [7:0]    opcode,
    input wire [3:0]    flags,
    input wire [3:0]    dst,
    input wire [7:0]    src1,
    input wire [7:0]    src2,
    output wire [31:0]  instruction
)
    assign instruction = {opcode, flags, dst, src1, src2};
endmodule