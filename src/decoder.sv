`default_nettype none
`timescale 1ns/1ns

// =============================================================================
// DECODER
// =============================================================================
//
// WHAT PROBLEM DOES THIS SOLVE?
//   The sequencer fetches 32-bit instructions from the program store and drives
//   this module with them. The decoder's job is to crack each instruction into
//   control signals that the rest of the TPU understands — without the sequencer
//   needing to know the opcode encoding.
//
// INSTRUCTION FORMAT (32 bits):
//   [31:24] opcode   — 8-bit operation code
//   [23:20] flags    — 4 modifier bits (accumulate / async / broadcast / transpose)
//   [19:16] dst      — 4-bit destination register / tile index
//   [15:8]  src1     — 8-bit source operand 1 (e.g. address)
//   [7:0]   src2/imm — 8-bit source operand 2 or immediate (e.g. scale factor)
//
// FLAG BITS (decoded_flags):
//   [0] accumulate — MATMUL adds into existing accumulator instead of clearing
//   [1] async      — operation runs without stalling the sequencer
//   [2] broadcast  — weight tile is broadcast to all rows
//   [3] transpose  — operand is read transposed
//
// OPCODE TABLE:
//   0x00 NOP       — no operation
//   0x01 LOAD_W    — load weight tile from unified buffer → weight FIFO
//   0x02 LOAD_A    — load activation tile → activation FIFO
//   0x03 MATMUL    — run systolic array (flag[0]=accumulate)
//   0x04 STORE     — write accumulator result → unified buffer
//   0x05 ACT_RELU  — apply ReLU activation
//   0x06 ACT_GELU  — apply GELU activation
//   0x07 ACT_SILU  — apply SiLU/Swish activation
//   0x08 SOFTMAX   — softmax (multi-pass)
//   0x09 ADD       — element-wise addition
//   0x0A LAYERNORM — layer normalisation
//   0x0B TRANSPOSE — matrix transpose in buffer
//   0x0C SCALE     — multiply by scalar (src2 = scale factor)
//   0x0D SYNC      — stall until all pending ops complete
//   0x0E LOOP      — loop control (dst=loop index, src1=count, src2=target PC)
//   0x0F HALT      — stop execution
//
// =============================================================================

module decoder (
    input wire clk,
    input wire reset,

    // -------------------------------------------------------------------------
    // Sequencer Interface
    // -------------------------------------------------------------------------
    input wire [2:0]  core_state,     // Sequencer FSM state — decode fires in STATE_DECODE
    input wire        decode_enable,  // Explicit enable (alternative to core_state check)
    input wire [31:0] instruction,    // 32-bit instruction word from program store

    // -------------------------------------------------------------------------
    // Decoded Instruction Fields
    // -------------------------------------------------------------------------
    output reg [7:0] decoded_opcode,
    output reg [3:0] decoded_flags,
    output reg [3:0] decoded_dst,
    output reg [7:0] decoded_src1,
    output reg [7:0] decoded_src2,

    // Flag aliases — combinational shortcuts into decoded_flags
    output wire flag_accumulate,  // flags[0]: add into accumulator (don't clear)
    output wire flag_async,       // flags[1]: fire and continue (non-blocking)
    output wire flag_broadcast,   // flags[2]: broadcast weight row to all PEs
    output wire flag_transpose,   // flags[3]: read operand transposed

    // -------------------------------------------------------------------------
    // Memory Control Outputs
    // -------------------------------------------------------------------------
    output reg        mem_read_enable,
    output reg        mem_write_enable,
    output reg [1:0]  mem_target,  // 0=weights, 1=activations, 2=outputs, 3=scratch

    // -------------------------------------------------------------------------
    // Systolic Array Control Outputs
    // -------------------------------------------------------------------------
    output reg array_enable,
    output reg array_weight_load,
    output reg array_clear_acc,

    // -------------------------------------------------------------------------
    // Activation Unit Control Outputs
    // -------------------------------------------------------------------------
    output reg        activation_enable,
    output reg [2:0]  activation_func,  // 0=ReLU, 1=GELU, 2=SiLU, 3=Sigmoid, 4=Tanh

    // -------------------------------------------------------------------------
    // Operation Start Pulses
    // -------------------------------------------------------------------------
    output reg matmul_start,
    output reg softmax_start,
    output reg layernorm_start,
    output reg transpose_start,
    output reg add_start,
    output reg scale_start,

    // -------------------------------------------------------------------------
    // Control Flow Outputs
    // -------------------------------------------------------------------------
    output reg sync_wait,   // Stall sequencer until in-flight ops drain
    output reg loop_start,  // Enter loop (dst=level, src1=count, src2=PC)
    output reg loop_end,    // Exit loop
    output reg halt,        // Stop sequencer

    // -------------------------------------------------------------------------
    // Instruction Class Flags (for sequencer scheduling)
    // -------------------------------------------------------------------------
    output reg is_memory_op,
    output reg is_compute_op,
    output reg is_control_op,

    // -------------------------------------------------------------------------
    // Debug
    // -------------------------------------------------------------------------
    output wire [31:0] debug_instruction
);

// =============================================================================
// OPCODE DEFINITIONS
// =============================================================================

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

// Sequencer state that triggers decode
localparam STATE_DECODE = 3'b010;

// =============================================================================
// COMBINATIONAL OUTPUTS
// =============================================================================

// Alias individual flag bits for readability downstream
assign flag_accumulate = decoded_flags[0];
assign flag_async      = decoded_flags[1];
assign flag_broadcast  = decoded_flags[2];
assign flag_transpose  = decoded_flags[3];

assign debug_instruction = instruction;

// =============================================================================
// DECODE LOGIC
// =============================================================================
//
// On every decode cycle:
//   1. Latch the raw instruction fields into decoded_* registers.
//   2. Default all control outputs to 0 (prevents stale signals).
//   3. Assert only the signals relevant to the current opcode.
//
// WHY default-to-zero each cycle?
//   Control signals like matmul_start are one-cycle pulses consumed by the
//   sequencer. If we don't clear them, a stalled pipeline sees the pulse again.

always @(posedge clk) begin
    if (reset) begin
        decoded_opcode   <= 8'h00;
        decoded_flags    <= 4'h0;
        decoded_dst      <= 4'h0;
        decoded_src1     <= 8'h00;
        decoded_src2     <= 8'h00;

        mem_read_enable  <= 1'b0;
        mem_write_enable <= 1'b0;
        mem_target       <= 2'b00;

        array_enable     <= 1'b0;
        array_weight_load <= 1'b0;
        array_clear_acc  <= 1'b0;

        activation_enable <= 1'b0;
        activation_func   <= 3'b000;

        matmul_start    <= 1'b0;
        softmax_start   <= 1'b0;
        layernorm_start <= 1'b0;
        transpose_start <= 1'b0;
        add_start       <= 1'b0;
        scale_start     <= 1'b0;

        sync_wait  <= 1'b0;
        loop_start <= 1'b0;
        loop_end   <= 1'b0;
        halt       <= 1'b0;

        is_memory_op  <= 1'b0;
        is_compute_op <= 1'b0;
        is_control_op <= 1'b0;

    end else if (decode_enable || core_state == STATE_DECODE) begin
        // Latch raw fields
        decoded_opcode <= instruction[31:24];
        decoded_flags  <= instruction[23:20];
        decoded_dst    <= instruction[19:16];
        decoded_src1   <= instruction[15:8];
        decoded_src2   <= instruction[7:0];

        // Default all outputs to 0 before asserting opcode-specific ones
        mem_read_enable  <= 1'b0;
        mem_write_enable <= 1'b0;
        mem_target       <= 2'b00;

        array_enable      <= 1'b0;
        array_weight_load <= 1'b0;
        array_clear_acc   <= 1'b0;

        activation_enable <= 1'b0;
        activation_func   <= 3'b000;

        matmul_start    <= 1'b0;
        softmax_start   <= 1'b0;
        layernorm_start <= 1'b0;
        transpose_start <= 1'b0;
        add_start       <= 1'b0;
        scale_start     <= 1'b0;

        sync_wait  <= 1'b0;
        loop_start <= 1'b0;
        loop_end   <= 1'b0;
        halt       <= 1'b0;

        is_memory_op  <= 1'b0;
        is_compute_op <= 1'b0;
        is_control_op <= 1'b0;

        // Assert opcode-specific signals
        case (instruction[31:24])

            OP_NOP: begin
                is_control_op <= 1'b1;
            end

            OP_LOAD_W: begin
                mem_read_enable   <= 1'b1;
                mem_target        <= 2'b00;   // Weight region
                array_weight_load <= 1'b1;
                is_memory_op      <= 1'b1;
            end

            OP_LOAD_A: begin
                mem_read_enable <= 1'b1;
                mem_target      <= 2'b01;     // Activation region
                is_memory_op    <= 1'b1;
            end

            OP_MATMUL: begin
                array_enable    <= 1'b1;
                array_clear_acc <= ~instruction[20];  // flag[0]=accumulate → skip clear
                matmul_start    <= 1'b1;
                is_compute_op   <= 1'b1;
            end

            OP_STORE: begin
                mem_write_enable <= 1'b1;
                mem_target       <= 2'b10;    // Output region
                is_memory_op     <= 1'b1;
            end

            OP_ACT_RELU: begin
                activation_enable <= 1'b1;
                activation_func   <= 3'b000;  // ReLU
                is_compute_op     <= 1'b1;
            end

            OP_ACT_GELU: begin
                activation_enable <= 1'b1;
                activation_func   <= 3'b001;  // GELU
                is_compute_op     <= 1'b1;
            end

            OP_ACT_SILU: begin
                activation_enable <= 1'b1;
                activation_func   <= 3'b010;  // SiLU/Swish
                is_compute_op     <= 1'b1;
            end

            OP_SOFTMAX: begin
                softmax_start <= 1'b1;
                is_compute_op <= 1'b1;
            end

            OP_ADD: begin
                add_start     <= 1'b1;
                is_compute_op <= 1'b1;
            end

            OP_LAYERNORM: begin
                layernorm_start <= 1'b1;
                is_compute_op   <= 1'b1;
            end

            OP_TRANSPOSE: begin
                transpose_start <= 1'b1;
                is_compute_op   <= 1'b1;
            end

            OP_SCALE: begin
                // src2 carries the scale factor
                scale_start   <= 1'b1;
                is_compute_op <= 1'b1;
            end

            OP_SYNC: begin
                sync_wait     <= 1'b1;
                is_control_op <= 1'b1;
            end

            OP_LOOP: begin
                // dst  [3:0] = loop level (0=K, 1=N, 2=M)
                // src1 [7:0] = iteration count
                // src2 [7:0] = target PC (jump-back address)
                loop_start    <= 1'b1;
                is_control_op <= 1'b1;
            end

            OP_HALT: begin
                halt          <= 1'b1;
                is_control_op <= 1'b1;
            end

            default: begin
                // Unknown opcode — treat as NOP, flag as control so sequencer advances
                is_control_op <= 1'b1;
            end

        endcase
    end
end

endmodule

// =============================================================================
// INSTRUCTION BUILDER (combinational helper)
// =============================================================================
//
// Packs individual instruction fields into the canonical 32-bit encoding.
// Useful in testbenches and the assembler shim — no clock required.

module instruction_builder (
    input wire [7:0]  opcode,
    input wire [3:0]  flags,
    input wire [3:0]  dst,
    input wire [7:0]  src1,
    input wire [7:0]  src2,
    output wire [31:0] instruction
);
    assign instruction = {opcode, flags, dst, src1, src2};
endmodule
