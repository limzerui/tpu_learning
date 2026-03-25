//FSM for the whole system

//IDLE->wait for start
//fetch->fetch instruction from mem
//decoder->for instruction and get important signals
//load_setup->load weights, activation from mem
//load_wait->
//execut->start the comput
//exec_wait->
//store_setup->store results to mem
//store_wait->
//sync->wait for all operations
//update->update PC and loop counter
//done

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
    localparam DONE = 4'b1011;

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

            case (state) begin
                IDLE: begin
                    done <= 1'b0;
                    cycle_count <= 32'b0;

                    if (start && !stop) begin
                        running <= 1'b1;
                        fetch_enable <= 1'b1;
                        state <= FETCH;
                    end
                
                end
                FETCH: begin
                    if (stop) begin
                        running <= 1'b0;
                        state <= DONE;
                    end else if (instruction_valid || fetcher_state == FETCHER_FETCHED) begin
                        decode_enable <= 1'b1;
                        state <= DECODE;
                    end else begin
                        fetch_enable <= 1'b1;
                    end
                end

                DECODE: begin
                    if (stop) begin
                        running <= 1'b0;
                        state <= DONE;
                    end else begin
                        instruction_type_memory <= is_memory_op;
                        instruction_type_compute <= is_compute_op;
                        instruction_type_control <= is_control_op;

                        if (halt_decoded) begin
                            running <= 1'b0;
                            state <= DONE;
                        end else if (sync_decoded) begin
                            state <= SYNC;
                        end else if (loop_decoded) begin
                            state <= UPDATE;
                        end else if (is_memory_op) begin //load or store
                            state <= LOAD_SETUP;
                        end else if (is_compute_op) begin //matmul, activation
                            state <= EXECUTE;
                        end else begin //NOP or unknown
                            state <= Update;
                        end
                    end
                end

                LOAD_SETUP: begin
                    state <= LOAD_WAIT; //mem load and store happens in one cycle. decoder also set signals
                end

                LOAD_WAIT: begin
                    if (stop) begin
                        running <= 1'b0;
                        state <= DONE;
                    end else if (!load_busy && !store_busy) begin
                        if (load_done || store_done) begin
                            state <= UPDATE;
                        end
                    end
                end

                EXECUTION: begin
                    state <= EXEC_WAIT;
                end

                EXEC_WAIT: begin
                    if (stop) begin
                        running <= 1'b0;
                        state <= DONE;
                    end else if (!compute_busy && !matmul_busy) begin
                        if (matmul_decoded) begin
                            //matmul done, need to store
                            state <= STORE_SETUP;
                        end else begin
                            state <= UPDATE;
                        end
                    end
                end

                STORE_SETUP: begin
                    state <= STORE_WAIT;
                end

                STORE_WAIT: begin
                    if (stop) begin
                        running <= 1'b0;
                        state <= DONE;
                    end else if (!store_begin) begin
                        state <= UPDATE;
                    end
                end

                SYNC: begin
                    //wait for all operations to finish
                    if (stop) begin
                        running <= 1'b0;
                        state <= DONE;
                    end else if (!load_busy && !store_busy && !compute_busy && !matmul_busy) begin
                        state <= UPDATE;
                    end
                end

                UPDATE: begin
                    //update PC and check for loops
                    if (loop_active && loop_itertation_done) begin
                        //loop back
                        pc_branch <= 1'b1;
                        pc_branch_target <= loop_target_pc;
                    end
                    //next instruction
                    fetch_enable <= 1'b1;
                    state <= FETCH;
                end

                DONE: begin
                    done <= 1'b1;
                    running <= 1'b0;

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

            end
            endcase
        end
    end

endmodule