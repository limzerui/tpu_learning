`default_nettype none  // Catch undeclared wires at compile time — good RTL hygiene
`timescale 1ns/1ns    // Simulation time unit / precision

// =============================================================================
// PROCESSING ELEMENT (PE)
// =============================================================================
// The PE is the atomic compute unit in the systolic array.
// 64 of these form an 8×8 grid and together compute one matrix multiply tile.
//
// DATAFLOW — Weight-Stationary:
//   - Weights are loaded ONCE and held in weight_reg for the entire computation
//   - Activations stream IN from the west, MAC, then pass OUT to the east
//   - Partial sums accumulate flowing from north to south
//
// DATA DIRECTIONS:
//   West  → East  : activation data passes through (8-bit, 1 cycle delay)
//   North → South : partial sums accumulate downward (32-bit)
//   West input    : also used to receive weight during weight_load phase
//
// TWO OPERATING MODES:
//   1. weight_load = 1  →  Store data_in_west into weight_reg (no MAC)
//   2. weight_load = 0  →  MAC: psum_out = psum_in + (activation × weight)
//
// TIMING (1-cycle registered pipeline):
//   - All outputs register on posedge clk
//   - 1-cycle latency through each PE — this creates the systolic wavefront
// =============================================================================

module pe #(
    parameter DATA_WIDTH = 8,   // Operand width: INT8 (matches Google TPU v1)
    parameter ACC_WIDTH  = 32   // Accumulator width: INT32 prevents overflow
                                // (up to 8 INT8×INT8 products accumulate per column)
) (
    input wire clk,
    input wire reset,   // Synchronous active-high reset
    input wire enable,  // When low, PE freezes all outputs (used during weight load of OTHER rows)

    // --- Control Signals ---
    input wire weight_load, // 1 = store data_in_west as new weight; 0 = compute mode
    input wire clear_acc,   // 1 = reset local accumulator (start of new tile); 0 = keep accumulating

    // --- Data Inputs ---
    // data_in_west: dual-purpose
    //   - During weight_load: carries the weight value to store
    //   - During compute   : carries the activation to multiply
    input wire signed [DATA_WIDTH-1:0] data_in_west,

    // psum_in_north: partial sum flowing in from the PE directly above.
    // Row 0 receives 0 (no PE above it — injected at the array boundary).
    input wire signed [ACC_WIDTH-1:0] psum_in_north,

    // --- Data Outputs ---
    // data_out_east: activation forwarded to the PE to the right.
    // Registered (1 cycle delay) — this delay is what creates the systolic wavefront.
    output reg signed [DATA_WIDTH-1:0] data_out_east,

    // psum_out_south: accumulated partial sum passed down to the PE below.
    // Last row's psum_out_south is the final dot-product result for that column.
    output reg signed [ACC_WIDTH-1:0] psum_out_south,

    // --- Debug Outputs (combinational, no registers) ---
    output wire signed [DATA_WIDTH-1:0] weight_debug, // Live view of weight_reg
    output wire signed [ACC_WIDTH-1:0]  acc_debug     // Live view of accumulator
);

    // -------------------------------------------------------------------------
    // Internal Registers
    // -------------------------------------------------------------------------

    // Holds the stationary weight for the duration of a tile computation.
    // Written once during weight_load phase, read every cycle during compute.
    reg signed [DATA_WIDTH-1:0] weight_reg;

    // Local accumulator — tracks the running dot-product for this PE.
    // Note: this is SEPARATE from psum_out_south (see explanation below).
    reg signed [ACC_WIDTH-1:0] accumulator;

    // -------------------------------------------------------------------------
    // Combinational Multiply
    // -------------------------------------------------------------------------

    // INT8 × INT8 = 16-bit product. We use 2×DATA_WIDTH to hold it without overflow.
    // SystemVerilog automatically sign-extends because both operands are 'signed'.
    wire signed [2*DATA_WIDTH-1:0] mult_result;
    assign mult_result = data_in_west * weight_reg;

    // Wire debug ports directly to internal registers (no extra logic)
    assign weight_debug = weight_reg;
    assign acc_debug    = accumulator;

    // -------------------------------------------------------------------------
    // Sequential Logic — Registered on every posedge clk
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (reset) begin
            // Clear everything to known zero state
            weight_reg     <= {DATA_WIDTH{1'b0}};
            accumulator    <= {ACC_WIDTH{1'b0}};
            data_out_east  <= {DATA_WIDTH{1'b0}};
            psum_out_south <= {ACC_WIDTH{1'b0}};

        end else if (enable) begin

            if (weight_load) begin
                // -------------------------------------------------------
                // MODE 1: WEIGHT LOADING
                // -------------------------------------------------------
                // The matrix controller is broadcasting weights into the array.
                // data_in_west carries the weight for THIS PE's column.
                // We latch it and stop computation for this cycle.
                weight_reg <= data_in_west;

                // Zero out east output — we don't want garbage activations
                // propagating east while weights are being loaded
                data_out_east <= {DATA_WIDTH{1'b0}};

                // Still pass psum through so the chain isn't broken
                psum_out_south <= psum_in_north;

            end else begin
                // -------------------------------------------------------
                // MODE 2: COMPUTE (MAC) MULTIPLY and ACCUMULATE
                // -------------------------------------------------------

                // STEP 1: Forward activation east (the systolic pass-through).
                // The 1-cycle register delay here is CRITICAL — it's what
                // staggers activations diagonally across the array.
                // Without it, all PEs would see the same activation simultaneously
                // and you'd get incorrect results.
                data_out_east <= data_in_west;

                // STEP 2: MAC — accumulate into the partial sum chain.
                // psum_out_south = (psum from PE above) + (my activation × my weight)
                // This builds up a column dot-product as data flows south.
                psum_out_south <= psum_in_north + mult_result;

                // STEP 3: Update LOCAL accumulator.
                // This is NOT the same as psum_out_south!
                // psum_out_south is a pipeline register (passes value south each cycle).
                // accumulator is a sticky register used for output-stationary mode
                // or for reading out a final result without forwarding.
                if (clear_acc) begin
                    // Start fresh for a new output tile
                    accumulator <= mult_result;
                end else begin
                    // Keep adding to the running total
                    accumulator <= accumulator + mult_result;
                end
            end

        end else begin
            // -------------------------------------------------------
            // PE DISABLED — hold outputs stable (no change)
            // -------------------------------------------------------
            // This matters during weight loading of OTHER rows:
            // only one row's PEs have weight_load active at a time,
            // but all PEs are still 'connected' in the pipeline.
            //NOTE important for synthesis, FPGA tools use clock enables to save power, and explicit assignments make intent unambiguous
            data_out_east  <= data_out_east;
            psum_out_south <= psum_out_south;
        end
    end

endmodule