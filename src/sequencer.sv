`default_nettype none
`timescale 1ns/1ns

// TPU SEQUENCER
// > Master control FSM for TPU execution pipeline
// > 11-state machine managing instruction flow and operation sequencing
// > Coordinates fetcher, decoder, memory, and compute units
//
// State Machine:
//   IDLE        - Waiting for start signal
//   FETCH       - Fetch instruction from memory
//   DECODE      - Decode instruction into control signals
//   LOAD_SETUP  - Configure memory loads (weights/activations)
//   LOAD_WAIT   - Wait for load operations to complete
//   EXECUTE     - Start compute operation
//   EXEC_WAIT   - Wait for compute to complete
//   STORE_SETUP - Configure result store
//   STORE_WAIT  - Wait for store to complete
//   SYNC        - Wait for all operations (sync instruction)
//   UPDATE      - Update PC and loop counters
//   DONE        - Execution complete (halt)

module sequencer (
    input wire clk,
    input wire reset,

    // Control interface
    input wire                  start,
    input wire                  stop,
    output reg                  running,
    output reg                  done,

    // Fetcher interface
    output reg                  fetch_enable,
    input wire [2:0]            fetcher_state,
    input wire                  instruction_valid,

    // Decoder interface
    output reg                  decode_enable,
    input wire                  is_memory_op,
    input wire                  is_compute_op,
    input wire                  is_control_op,
    input wire                  halt_decoded,
    input wire                  sync_decoded,
    input wire                  loop_decoded,
    input wire                  matmul_decoded,

    // Memory operation status
    input wire                  load_busy,
    input wire                  load_done,
    input wire                  store_busy,
    input wire                  store_done,

    // Compute operation status
    input wire                  compute_busy,
    input wire                  compute_done,
    input wire                  matmul_busy,
    input wire                  matmul_done,

    // Loop controller interface
    output reg                  loop_check,
    input wire                  loop_active,
    input wire                  loop_iteration_done,
    input wire [13:0]           loop_target_pc,

    // PC control
    output reg                  pc_branch,
    output reg [13:0]           pc_branch_target,

    // State output
    output reg [3:0]            seq_state,

    // Debug
    output wire [3:0]           debug_state,
    output wire [31:0]          debug_cycle_count
);

    // State definitions
    localparam IDLE        = 4'b0000;
    localparam FETCH       = 4'b0001;
    localparam DECODE      = 4'b0010;
    localparam LOAD_SETUP  = 4'b0011;
    localparam LOAD_WAIT   = 4'b0100;
    localparam EXECUTE     = 4'b0101;
    localparam EXEC_WAIT   = 4'b0110;
    localparam STORE_SETUP = 4'b0111;
    localparam STORE_WAIT  = 4'b1000;
    localparam SYNC        = 4'b1001;
    localparam UPDATE      = 4'b1010;
    localparam DONE_STATE  = 4'b1011;

    // Fetcher states (from fetcher.sv)
    localparam FETCHER_IDLE = 3'b000;
    localparam FETCHER_FETCHED = 3'b011;

    // Internal state
    reg [3:0] state;
    reg [31:0] cycle_count;
    reg instruction_type_memory;
    reg instruction_type_compute;
    reg instruction_type_control;

    // Debug outputs
    assign debug_state = state;
    assign debug_cycle_count = cycle_count;
    assign seq_state = state;

    // Main state machine
    always @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            running <= 1'b0;
            done <= 1'b0;
            fetch_enable <= 1'b0;
            decode_enable <= 1'b0;
            loop_check <= 1'b0;
            pc_branch <= 1'b0;
            pc_branch_target <= 14'd0;
            cycle_count <= 32'd0;
            instruction_type_memory <= 1'b0;
            instruction_type_compute <= 1'b0;
            instruction_type_control <= 1'b0;
        end else begin
            // Default signal values
            fetch_enable <= 1'b0;
            decode_enable <= 1'b0;
            loop_check <= 1'b0;
            pc_branch <= 1'b0;

            // Cycle counter
            if (running) begin
                cycle_count <= cycle_count + 1;
            end

            case (state)
                IDLE: begin
                    done <= 1'b0;
                    cycle_count <= 32'd0;

                    if (start && !stop) begin
                        running <= 1'b1;
                        fetch_enable <= 1'b1;
                        state <= FETCH;
                    end
                end

                FETCH: begin
                    if (stop) begin
                        running <= 1'b0;
                        state <= DONE_STATE;
                    end else if (instruction_valid || fetcher_state == FETCHER_FETCHED) begin
                        decode_enable <= 1'b1;
                        state <= DECODE;
                    end else begin
                        // Keep fetching
                        fetch_enable <= 1'b1;
                    end
                end

                DECODE: begin
                    if (stop) begin
                        running <= 1'b0;
                        state <= DONE_STATE;
                    end else begin
                        // Capture instruction type for routing
                        instruction_type_memory <= is_memory_op;
                        instruction_type_compute <= is_compute_op;
                        instruction_type_control <= is_control_op;

                        // Route based on instruction type
                        if (halt_decoded) begin
                            // HALT instruction
                            state <= DONE_STATE;
                        end else if (sync_decoded) begin
                            // SYNC instruction - wait for all pending ops
                            state <= SYNC;
                        end else if (loop_decoded) begin
                            // LOOP instruction - check loop status
                            loop_check <= 1'b1;
                            state <= UPDATE;
                        end else if (is_memory_op) begin
                            // Memory operation (LOAD_W, LOAD_A, STORE)
                            state <= LOAD_SETUP;
                        end else if (is_compute_op) begin
                            // Compute operation (MATMUL, activations, etc.)
                            state <= EXECUTE;
                        end else begin
                            // NOP or unknown - go to UPDATE
                            state <= UPDATE;
                        end
                    end
                end

                LOAD_SETUP: begin
                    // Memory load/store setup happens in one cycle
                    // The decoder already set the control signals
                    state <= LOAD_WAIT;
                end

                LOAD_WAIT: begin
                    if (stop) begin
                        running <= 1'b0;
                        state <= DONE_STATE;
                    end else if (!load_busy && !store_busy) begin
                        // Load/store complete
                        if (load_done || store_done) begin
                            state <= UPDATE;
                        end else begin
                            // No operation was actually started (edge case)
                            state <= UPDATE;
                        end
                    end
                    // Else keep waiting
                end

                EXECUTE: begin
                    // Compute execution starts
                    state <= EXEC_WAIT;
                end

                EXEC_WAIT: begin
                    if (stop) begin
                        running <= 1'b0;
                        state <= DONE_STATE;
                    end else if (!compute_busy && !matmul_busy) begin
                        // Check if we need to store results
                        if (matmul_decoded) begin
                            // Matmul complete, may need to store
                            state <= STORE_SETUP;
                        end else begin
                            // Other compute ops - go to update
                            state <= UPDATE;
                        end
                    end
                    // Else keep waiting
                end

                STORE_SETUP: begin
                    // Store setup
                    state <= STORE_WAIT;
                end

                STORE_WAIT: begin
                    if (stop) begin
                        running <= 1'b0;
                        state <= DONE_STATE;
                    end else if (!store_busy) begin
                        state <= UPDATE;
                    end
                    // Else keep waiting
                end

                SYNC: begin
                    // Wait for all pending operations
                    if (stop) begin
                        running <= 1'b0;
                        state <= DONE_STATE;
                    end else if (!load_busy && !store_busy && !compute_busy && !matmul_busy) begin
                        state <= UPDATE;
                    end
                    // Else keep waiting
                end

                UPDATE: begin
                    // Update PC and check loops
                    if (loop_active && loop_iteration_done) begin
                        // Loop back
                        pc_branch <= 1'b1;
                        pc_branch_target <= loop_target_pc;
                    end
                    // Else PC increments naturally (handled by fetcher)

                    // Start next instruction fetch
                    fetch_enable <= 1'b1;
                    state <= FETCH;
                end

                DONE_STATE: begin
                    running <= 1'b0;
                    done <= 1'b1;

                    // Can restart if start asserted again
                    if (start && !stop) begin
                        done <= 1'b0;
                        running <= 1'b1;
                        fetch_enable <= 1'b1;
                        cycle_count <= 32'd0;
                        state <= FETCH;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
