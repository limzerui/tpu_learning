// =========================================================================
// TPU SEQUENCER (MASTER CONTROLLER)
// =========================================================================
// > The central "Brain" of the TPU execution pipeline.
// > Orchestrates instruction fetching, decoding, data loading, computation,
//   and storing results back to memory.
//
// FSM Execution Flow:
//   IDLE         -> Waiting for the start signal
//   FETCH        -> Grab the next 32-bit instruction from instruction memory
//   DECODE       -> Parse instruction to determine operation type and signals
//   LOAD_SETUP   -> Initiate weight/activation load from Unified Buffer
//   LOAD_WAIT    -> Waiting for load operation to complete
//   EXECUTE      -> Trigger functional units (Systolic Array, Activation Unit)
//   EXEC_WAIT    -> Waiting for computation to finish
//   STORE_SETUP  -> Initiate storing output results to Unified Buffer
//   STORE_WAIT   -> Waiting for store operation to complete
//   SYNC         -> Halt and wait for all executing operations to synchronize
//   UPDATE       -> Increment Program Counter (PC) or branch if looping
//   DONE_STATE   -> Program finished execution
// =========================================================================
module sequencer (
    input wire                  start,              // Start execution
    input wire                  stop,               // Stop execution
    output reg                  running,            // Currently executing
    output reg                  done,               // Execution complete

    // Fetcher interface
    output reg                  fetch_enable, //tells fetcher module to grab the next 32 bit instruction
    input wire [2:0]            fetcher_state,
    input wire                  instruction_valid,

    // Decoder interface
    output reg                  decode_enable, //tells decoder module to "disect" instruction after fetching
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
    output reg                  loop_check, //pulse to tiling controller to ask if we should loop/any more left
    input wire                  loop_active,
    input wire                  loop_iteration_done,
    input wire [13:0]           loop_target_pc,

    // PC control
    output reg                  pc_branch, //tell fetcher to jump to specific addr or just next line
    output reg [13:0]           pc_branch_target,

    // State output
    output reg [3:0]            seq_state,

    // Debug
    output wire [3:0]           debug_state,
    output wire [31:0]          debug_cycle_count
);

    localparam IDLE = 4'b0000;
    localparam FETCH = 4'b0001;
    localparam DECODE = 4'b0010;
    localparam LOAD_SETUP = 4'b0011;
    localparam LOAD_WAIT = 4'b0100;
    localparam EXECUTE = 4'b0101;
    localparam EXEC_WAIT = 4'b0110;
    localparam STORE_SETUP = 4'b0111;
    localparam STORE_WAIT = 4'b1000;
    localparam SYNC = 4'b1001;
    localparam UPDATE = 4'b1010;
    localparam DONE_STATE = 4'b1011;

    localparam FETCHER_IDLE = 3'b000;
    localparam FETCHER_FETCHED = 3'b011;

    reg [3:0] state;
    reg [31:0] cycle_count;
    reg instruction_type_memory;
    reg instruction_type_compute;
    reg instruction_type_control;

    //debugs
    assign debug_state = state;
    assign debug_cycle_count = cycle_count;
    assign seq_state = state;

    always @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            running <= 1'b0;
            done <= 1'b0;
            fetch_enable <= 1'b0;
            decode_enable <= 1'b0;
            loop_check <= 1'b0;
            pc_branch <= 1'b0;
            pc_branch_target <= 14'b0;
            cycle_count <= 32'b0;
            instruction_type_memory <= 1'b0;
            instruction_type_compute <= 1'b0;
            instruction_type_control <= 1'b0;
        end else begin
            fetch_enable <= 1'b0;
            decode_enable <= 1'b0;
            loop_check <= 1'b0;
            pc_branch <= 1'b0;

            if (running) begin
                cycle_count <= cycle_count + 1'b1;

            end

            case (state)
                // -----------------------------------------------------------------
                // STATE 0: IDLE
                // > Wait for the host to send the 'start' signal to begin execution
                // -----------------------------------------------------------------
                IDLE: begin
                    done <= 1'b0;
                    cycle_count <= 32'b0;

                    if (start && !stop) begin
                        running <= 1'b1;
                        fetch_enable <= 1'b1;
                        state <= FETCH;
                    end
                end
                // -----------------------------------------------------------------
                // STATE 1: FETCH
                // > Pulse fetch_enable and wait until Fetcher returns a valid instruction
                // -----------------------------------------------------------------
                FETCH: begin
                    if (stop) begin
                        running <= 1'b0;
                        state <= DONE_STATE;
                    end else if (instruction_valid || fetcher_state == FETCHER_FETCHED) begin
                        decode_enable <= 1'b1; // Trigger the Decoder
                        state <= DECODE;
                    end else begin
                        fetch_enable <= 1'b1; // Keep asking for instruction
                    end
                end

                // -----------------------------------------------------------------
                // STATE 2: DECODE
                // > Route the execution based on what the Decoder found
                // -----------------------------------------------------------------
                DECODE: begin
                    if (stop) begin
                        running <= 1'b0;
                        state <= DONE_STATE;
                    end else begin
                        // Save the instruction categorization for the current run
                        instruction_type_memory <= is_memory_op;
                        instruction_type_compute <= is_compute_op;
                        instruction_type_control <= is_control_op;

                        // Priority-based state transitions
                        if (halt_decoded) begin
                            running <= 1'b0;
                            state <= DONE_STATE;
                        end else if (sync_decoded) begin
                            state <= SYNC;
                        end else if (loop_decoded) begin
                            state <= UPDATE;
                        end else if (is_memory_op) begin // LOAD_W, LOAD_A, STORE
                            state <= LOAD_SETUP;
                        end else if (is_compute_op) begin // MATMUL, ACT_*
                            state <= EXECUTE;
                        end else begin // NOP or Unknown
                            state <= UPDATE;
                        end
                    end
                end

                // -----------------------------------------------------------------
                // STATE 3 & 4: LOAD OPERATIONS
                // > LOAD_SETUP initiates. LOAD_WAIT monitors bus status continuously.
                // -----------------------------------------------------------------
                LOAD_SETUP: begin
                    state <= LOAD_WAIT; // Memory requests happen over 1 cycle; Decoder sets signals
                end

                LOAD_WAIT: begin
                    if (stop) begin
                        running <= 1'b0;
                        state <= DONE_STATE;
                    end else if (!load_busy && !store_busy) begin // Wait until Memory bus is idle
                        if (load_done || store_done) begin
                            state <= UPDATE;
                        end
                    end
                end

                // -----------------------------------------------------------------
                // STATE 5 & 6: EXECUTE COMPUTE BLOCKS
                // > EXECUTE initiates. EXEC_WAIT waits for Math logic to finish.
                // -----------------------------------------------------------------
                EXECUTE: begin
                    state <= EXEC_WAIT;
                end

                EXEC_WAIT: begin
                    if (stop) begin
                        running <= 1'b0;
                        state <= DONE_STATE;
                    end else if (!compute_busy && !matmul_busy) begin
                        if (matmul_decoded) begin
                            // Matmul done, route automatically to STORE setup
                            state <= STORE_SETUP;
                        end else begin
                            state <= UPDATE;
                        end
                    end
                end

                // -----------------------------------------------------------------
                // STATE 7 & 8: STORE OPERATIONS
                // > Setup to store results into the unified memory
                // -----------------------------------------------------------------
                STORE_SETUP: begin
                    state <= STORE_WAIT;
                end

                STORE_WAIT: begin
                    if (stop) begin
                        running <= 1'b0;
                        state <= DONE_STATE;
                    end else if (!store_busy) begin
                        state <= UPDATE;
                    end
                end

                // -----------------------------------------------------------------
                // STATE 9: SYNC
                // > Pause until everything finishes (Barrier synchronization)
                // -----------------------------------------------------------------
                SYNC: begin
                    if (stop) begin
                        running <= 1'b0;
                        state <= DONE_STATE;
                    end else if (!load_busy && !store_busy && !compute_busy && !matmul_busy) begin
                        state <= UPDATE;
                    end
                end

                // -----------------------------------------------------------------
                // STATE 10: UPDATE LOGIC
                // > Determine where to point the PC next (Increment or Loop branch)
                // -----------------------------------------------------------------
                UPDATE: begin
                    if (loop_active && loop_iteration_done) begin
                        // Need another loop iteration, so branch backward
                        pc_branch <= 1'b1;
                        pc_branch_target <= loop_target_pc;
                    end
                    // Trigger fetch for the upcoming instruction cycle
                    fetch_enable <= 1'b1;
                    state <= FETCH;
                end

                // -----------------------------------------------------------------
                // STATE 11: DONE
                // > Raise the done flag and halt execution loop
                // -----------------------------------------------------------------
                DONE_STATE: begin
                    done <= 1'b1;
                    running <= 1'b0;

                    // Allows host to restart directly from the DONE_STATE
                    if (start && !stop) begin
                        done <= 1'b0;
                        running <= 1'b1;
                        fetch_enable <= 1'b1;
                        cycle_count <= 32'd0;
                        state <= FETCH;
                    end
                end

                default: begin
                    state <= IDLE;
                end

            endcase
        end
    end

endmodule