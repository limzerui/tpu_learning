`default_nettype none
`timescale 1ns/1ns

// TPU instruction fetcher.
//
// This block owns the program counter and converts sequencer fetch requests into
// read requests against an instruction source. The instruction source can be the
// simple instruction_memory module below or a later top-level memory path.
module fetcher #(
    parameter ADDR_WIDTH = 14,
    parameter INSTR_WIDTH = 32,
    parameter MAX_PROGRAM_SIZE = 256
) (
    input wire clk,
    input wire reset,

    // Control interface
    input wire                      fetch_enable,
    input wire                      branch_taken,
    input wire [ADDR_WIDTH-1:0]     branch_target,
    input wire                      halt,

    // Program counter interface
    output reg [ADDR_WIDTH-1:0]     pc,
    output reg [ADDR_WIDTH-1:0]     next_pc,
    input wire [ADDR_WIDTH-1:0]     pc_override,
    input wire                      pc_override_valid,

    // Instruction memory interface
    output reg                      mem_req,
    output reg [ADDR_WIDTH-1:0]     mem_addr,
    input wire [INSTR_WIDTH-1:0]    mem_rdata,
    input wire                      mem_valid,

    // Fetched instruction output
    output reg [INSTR_WIDTH-1:0]    instruction,
    output reg                      instruction_valid,

    // State output
    output wire [2:0]               fetcher_state,

    // Debug
    output wire [ADDR_WIDTH-1:0]    debug_pc,
    output wire                     debug_fetching
);

    localparam IDLE    = 3'b000;
    localparam REQUEST = 3'b001;
    localparam WAIT    = 3'b010;
    localparam FETCHED = 3'b011;
    localparam HALTED  = 3'b100;

    reg [2:0] state;

    assign debug_pc = pc;
    assign debug_fetching = (state == REQUEST) || (state == WAIT);
    assign fetcher_state = state;

    always @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            pc <= {ADDR_WIDTH{1'b0}};
            next_pc <= {{ADDR_WIDTH-1{1'b0}}, 1'b1};
            mem_req <= 1'b0;
            mem_addr <= {ADDR_WIDTH{1'b0}};
            instruction <= {INSTR_WIDTH{1'b0}};
            instruction_valid <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    instruction_valid <= 1'b0;
                    mem_req <= 1'b0;

                    if (halt) begin
                        state <= HALTED;
                    end else if (fetch_enable) begin
                        if (pc_override_valid) begin
                            pc <= pc_override;
                            next_pc <= pc_override + 1'b1;
                        end else if (branch_taken) begin
                            pc <= branch_target;
                            next_pc <= branch_target + 1'b1;
                        end

                        state <= REQUEST;
                    end
                end

                REQUEST: begin
                    // One-cycle read request. WAIT observes mem_valid.
                    mem_req <= 1'b1;
                    mem_addr <= pc;
                    state <= WAIT;
                end

                WAIT: begin
                    mem_req <= 1'b0;

                    if (mem_valid) begin
                        instruction <= mem_rdata;
                        instruction_valid <= 1'b1;
                        next_pc <= pc + 1'b1;
                        state <= FETCHED;
                    end
                end

                FETCHED: begin
                    if (halt) begin
                        instruction_valid <= 1'b0;
                        state <= HALTED;
                    end else if (fetch_enable) begin
                        instruction_valid <= 1'b0;

                        if (pc_override_valid) begin
                            pc <= pc_override;
                            next_pc <= pc_override + 1'b1;
                        end else if (branch_taken) begin
                            pc <= branch_target;
                            next_pc <= branch_target + 1'b1;
                        end else begin
                            pc <= next_pc;
                            next_pc <= next_pc + 1'b1;
                        end

                        state <= REQUEST;
                    end else begin
                        // Valid is a pulse; return to idle until sequencer asks again.
                        instruction_valid <= 1'b0;
                        state <= IDLE;
                    end
                end

                HALTED: begin
                    mem_req <= 1'b0;
                    instruction_valid <= 1'b0;

                    if (!halt && fetch_enable) begin
                        state <= REQUEST;
                    end
                end

                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule

// Simple program store used by the fetcher.
//
// Keeping this separate from fetcher is useful because fetcher should only care
// about PC sequencing and memory handshakes. The top level can later replace
// this memory with another instruction source without changing fetch control.
module instruction_memory #(
    parameter ADDR_WIDTH = 14,
    parameter DATA_WIDTH = 32,
    parameter DEPTH = 256
) (
    input wire clk,
    input wire reset,

    // Read interface
    input wire                      read_en,
    input wire [ADDR_WIDTH-1:0]     read_addr,
    output reg [DATA_WIDTH-1:0]     read_data,
    output reg                      read_valid,

    // Program load interface
    input wire                      write_en,
    input wire [ADDR_WIDTH-1:0]     write_addr,
    input wire [DATA_WIDTH-1:0]     write_data
);

    (* ram_style = "block" *)
    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (reset) begin
            read_data <= {DATA_WIDTH{1'b0}};
            read_valid <= 1'b0;
        end else begin
            read_valid <= 1'b0;

            if (read_en) begin
                if (read_addr < DEPTH) begin
                    read_data <= mem[read_addr];
                end else begin
                    read_data <= {DATA_WIDTH{1'b0}};
                end

                read_valid <= 1'b1;
            end
        end
    end

    always @(posedge clk) begin
        if (write_en && write_addr < DEPTH) begin
            mem[write_addr] <= write_data;
        end
    end

    integer i;
    initial begin
        for (i = 0; i < DEPTH; i = i + 1) begin
            mem[i] = {DATA_WIDTH{1'b0}};
        end
    end

endmodule
