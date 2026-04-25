module fetcher #(
    parameter ADDR_WIDTH = 14,              
    parameter DATA_WIDTH = 32,
    parameter MAX_PROGRAM_SIZE = 256; 
) (
    input wire clk,
    input wire reset,

    input wire fetch_enable,
    input wire branch_taken,
    input wire [ADDR_WIDTH-1:0] branch_target,
    input wire halt, // Signal to halt fetching (e.g., on error or special instruction)

    output reg [ADDR_WIDTH-1:0] pc,
    output reg [ADDR_WIDTH-1:0] next_pc,
    input wire [ADDR_WIDTH-1:0] pc_override,
    input wire pc_override_valid,

    output reg mem_req,
    output reg [ADDR_WIDTH-1:0] mem_addr,
    input wire [INSTR_WIDTH-1:0] mem_rdata,
    input wire mem_valid,
    
    ,



)
