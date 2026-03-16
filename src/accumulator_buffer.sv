`default_nettype none
`timescale 1ns/1ns

// =============================================================================
// ACCUMULATOR BUFFER — Result Gathering & Quantization
// =============================================================================
// This module captures the 32-bit results from the systolic array and 
// prepares them for the next layer.
//
// 1. CAPTURE & ACCUMULATE:
//    Captures N 32-bit values from the south edge of the array.
//    If accumulate_mode=1, adds them to existing values (for tiled matmul).
//
// 2. QUANTIZATION:
//    Converts 32-bit sums → 8-bit activations using: 
//    out = clamp( (in * scale) >> 8, -128, 127 )
// =============================================================================

module accumulator_buffer #(
    parameter N          = 8,   // Array dimension
    parameter ACC_WIDTH  = 32,  // Precision of partial sums
    parameter DATA_WIDTH = 8    // Output precision (INT8)
)(
    input wire clk,
    input wire reset,

    // -------------------------------------------------------------------------
    // Systolic Array Interface (Input)
    // -------------------------------------------------------------------------
    // [TODO] Declare result_in [N-1:0] and result_valid [N-1:0]
    input wire signed [ACC_WIDTH-1:0] result_in [N-1:0],
    input wire [N-1:0] result_valid,
    // [TODO] Add results_enable pulse to start capturing
    input wire results_enable, //accept results from array

    // -------------------------------------------------------------------------
    // Accumulation Control
    // -------------------------------------------------------------------------
    // [TODO] Add accumulate_mode (1=add, 0=overwrite)
    input wire accumulate_mode, //1=add, 0=overwrite
    // [TODO] Add clear_buffer (sets all 64 accumulators to 0)
    input wire clear_buffer, //clear accumulators

    // [TODO] Add tile_row_idx to know which row is being captured
    

    // -------------------------------------------------------------------------
    // Quantization & Readback (Output)
    // -------------------------------------------------------------------------
    // [TODO] Add quant_enable, quant_scale (16-bit Q8.8)
    // [TODO] Add quant_out [N-1:0] and quant_valid/quant_done signals
    
    // -------------------------------------------------------------------------
    // Status
    // -------------------------------------------------------------------------
    output wire busy
);

    // =========================================================================
    // BLOCK A — STORAGE
    // =========================================================================
    // [TODO] Define a 2D array of signed [ACC_WIDTH-1:0] to hold NxN values
    
    // [TODO] Add a register to track which rows have been filled


    // =========================================================================
    // BLOCK B — CAPTURE LOGIC
    // =========================================================================
    // [TODO] Implement an 'always' block that listens to result_valid mask
    // [TODO] If result_valid[j] is high, update accumulators[tile_row_idx][j]
    // [TODO] Handle the accumulate_mode toggle (add vs overwrite)


    // =========================================================================
    // BLOCK C — QUANTIZATION ENGINE
    // =========================================================================
    // [TODO] Create a state machine (IDLE -> COMPUTE -> DONE)
    // [TODO] Iterate through the NxN grid
    // [TODO] Multiply (acc * scale), then shift right by 8
    // [TODO] Apply saturating clip (if > 127 then 127; if < -128 then -128)


    // =========================================================================
    // BLOCK D — SIMULATION INIT
    // =========================================================================
    // [TODO] Clear everything to 0 in an 'initial' block

endmodule
