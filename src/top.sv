//fetcher->decoder->sequencer->

module top #(
    parameter N = 8
    parameter DATA_WIDTH = 9;
    parameter ADDR_WIDTH = 32;
    parameter ACC_WIDTH = 32;
    parameter MEM_DEPTH = 65536;
) (
    input wire clk,
    input wire reset,

    input wire host_start
    input wire host_stop,
    output wire host_busy,
    output wire host_done,
    output wire [3:0] host_state,

    //host mem interfact for loading program and data
    input wire host_mem_req,
    input wire host_mem_write,
    input wire [ADDR_WIDTH-1:0] host_mem_addr,
    input wire [DATA_WIDTH-1:0] host_mem_wdata,
    output wire [DATA_WIDTH-1:0] host_mem_rdata,
    output wire host_mem_ready,


    //



    
)

endmodule