`default_nettype none  // Catch undeclared wires at compile time — good RTL hygiene
`timescale 1ns/1ns    // Simulation time unit / precision

// =============================================================================
// PROCESSING ELEMENT (PE)
// =============================================================================
// The PE is the ATOMIC COMPUTE UNIT of the TPU. 64 of these (in an 8×8 grid)
// compute one matrix-multiply tile together.
//
// KEY CONCEPT — Weight-Stationary Dataflow:
//   Instead of moving weights each cycle, we load them into the PE ONCE and
//   hold them fixed ("stationary") for the entire computation. Then we stream
//   activations through and let partial sums flow downward and accumulate.
//
//   Think of each PE as a tiny MAC (Multiply-Accumulate) unit that "remembers"
//   one weight and keeps adding (activation × weight) to a running total.
//
// DATA DIRECTIONS:
//   ┌───────────────────────────────────┐
//   │        psum_in_north (INT32)      │  ← partial sum flowing IN from PE above
//   │               │                  │
//   │   data_in_west (INT8) ──→──[×]──→ data_out_east      │  ← activation passes east
//   │                          │        │
//   │                         [+]       │
//   │                          │        │
//   │        psum_out_south (INT32)     │  → partial sum flowing OUT to PE below
//   └───────────────────────────────────┘
//
// TWO OPERATING MODES:
//   1. weight_load = 1  →  Latch data_in_west into weight_reg. No computation.
//   2. weight_load = 0  →  Compute: psum_out = psum_in + (data_in_west × weight_reg)
//
// TIMING — 1-cycle registered pipeline:
//   All outputs are registered on posedge clk. This 1-cycle delay through each PE
//   is what creates the "systolic wavefront" — activations arrive at PE[row][col]
//   exactly col cycles after entering the left edge of the array.
//
// WORD ON `clear_acc`:
//   There are TWO "accumulations" happening here:
//   a) `psum_out_south`: a PIPELINE register that passes partial sums south each cycle.
//      This is NOT the final result—it's a relay.
//   b) `accumulator` : a STICKY register that accumulates the local dot-product over time.
//      This is used for output-stationary mode or direct result readout.
//   `clear_acc` resets only the sticky `accumulator`, not the psum pipeline.
// =============================================================================

module pe #(
    parameter DATA_WIDTH = 8,   // Operand width: INT8 (matches Google TPU v1)
    parameter ACC_WIDTH  = 32   // Accumulator width: INT32 prevents overflow.
                                // Worst case: 8 × (127 × 127) = 128,898 — fits easily in INT32.
) (
    input wire clk,
    input wire reset,   // Synchronous, active-high reset
    input wire enable,  // When low: PE freezes outputs. Used when OTHER rows are loading weights.

    // --- Control ---
    input wire weight_load, // 1 = latch data_in_west as new weight; 0 = compute mode
    input wire clear_acc,   // 1 = reset sticky accumulator (start of new tile)

    // --- Data Inputs ---
    // data_in_west is DUAL PURPOSE:
    //   During weight_load → carries the weight value to be stored
    //   During compute    → carries the activation to be multiplied
    input wire signed [DATA_WIDTH-1:0] data_in_west,

    // psum_in_north: partial sum flowing in from the PE directly above.
    // Row 0 receives 32'sd0 (injected at the array top boundary — no PE above it).
    input wire signed [ACC_WIDTH-1:0] psum_in_north,

    // --- Data Outputs ---
    // data_out_east: activation forwarded to the PE on the right.
    // CRITICAL: registered (1-cycle delay). This delay staggers activations diagonally,
    // which is what makes the systolic array work correctly.
    output reg signed [DATA_WIDTH-1:0] data_out_east,

    // psum_out_south: accumulated partial sum passed DOWN to the PE below.
    // The last row's psum_out_south is the final dot-product result for that column.
    output reg signed [ACC_WIDTH-1:0] psum_out_south,

    // --- Debug Outputs (purely combinational — no delay, no registers) ---
    output wire signed [DATA_WIDTH-1:0] weight_debug, // Directly exposes weight_reg
    output wire signed [ACC_WIDTH-1:0]  acc_debug     // Directly exposes accumulator
);

    // -------------------------------------------------------------------------
    // Internal Registers
    // -------------------------------------------------------------------------

    // Holds the stationary weight for the duration of a tile computation.
    // Written once during weight_load phase. Read every cycle during compute.
    reg signed [DATA_WIDTH-1:0] weight_reg;

    // Local sticky accumulator — stores the running dot-product FOR THIS PE.
    // NOTE: This is different from psum_out_south. See module header for explanation.
    reg signed [ACC_WIDTH-1:0] accumulator;

    // -------------------------------------------------------------------------
    // Combinational — Multiply
    // INT8 × INT8 = 16-bit product. 2×DATA_WIDTH avoids overflow.
    // SystemVerilog handles sign extension automatically when both operands are `signed`.
    // -------------------------------------------------------------------------
    wire signed [2*DATA_WIDTH-1:0] mult_result;
    assign mult_result = data_in_west * weight_reg;

    // Debug ports wire directly to internal registers (zero latency, no logic)
    assign weight_debug = weight_reg;
    assign acc_debug    = accumulator;

    // -------------------------------------------------------------------------
    // Sequential Logic
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (reset) begin
            // All outputs → 0. Using replication syntax {N{bit}} is idiomatic RTL.
            weight_reg     <= {DATA_WIDTH{1'b0}};
            accumulator    <= {ACC_WIDTH{1'b0}};
            data_out_east  <= {DATA_WIDTH{1'b0}};
            psum_out_south <= {ACC_WIDTH{1'b0}};

        end else if (enable) begin

            if (weight_load) begin
                // ==============================================================
                // MODE 1 — WEIGHT LOADING
                // ==============================================================
                // The weight_fifo is broadcasting a row of weights into the array.
                // Only THIS PE's row has weight_load asserted; other rows' PEs are disabled.
                // We capture the weight and freeze computation this cycle.
                weight_reg <= data_in_west;

                // Zero east output so garbage doesn't propagate east while weights load.
                data_out_east <= {DATA_WIDTH{1'b0}};

                // Still relay psum south so the pipeline chain isn't broken.
                psum_out_south <= psum_in_north;

            end else begin
                // ==============================================================
                // MODE 2 — COMPUTE (MAC)
                // ==============================================================

                // STEP 1: Forward activation east.
                // The 1-cycle register delay IS intentional and CRITICAL.
                // It ensures that PE[row][col] sees the activation 'col' cycles after
                // it enters the left edge — this creates the diagonal wavefront.
                data_out_east  <= data_in_west;

                // STEP 2: MAC — add (activation × weight) to the incoming partial sum.
                // This builds up a column's dot product as data flows from north → south.
                // mult_result is automatically sign-extended from 16-bit to ACC_WIDTH here.
                psum_out_south <= psum_in_north + mult_result;

                // STEP 3: Update sticky accumulator.
                // `clear_acc` resets it for a new output tile, then MACs continue.
                if (clear_acc) begin
                    accumulator <= mult_result; // Start fresh with current product
                end else begin
                    accumulator <= accumulator + mult_result;
                end
            end

        end else begin
            // ==================================================================
            // PE DISABLED — hold outputs stable
            // ==================================================================
            // While another row is loading weights (weight_load & weight_row_select[i]=1),
            // THIS row's PEs have enable=0. We explicitly hold outputs to prevent
            // latches. FPGA tools (Vivado) use clock enables for power savings,
            // and explicit assignments like these make the intent unambiguous for synthesis.
            data_out_east  <= data_out_east;
            psum_out_south <= psum_out_south;
        end
    end

