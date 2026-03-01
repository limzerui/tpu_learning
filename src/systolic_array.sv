`default_nettype none
`timescale 1ns/1ns

// =============================================================================
// SYSTOLIC ARRAY
// =============================================================================
// An NxN grid of Processing Elements (PEs) that computes C = A × B
// using weight-stationary dataflow.
//
// HOW IT WORKS — 3 phases for one 8×8 tile:
//
//   Phase 1 — WEIGHT LOAD (8 cycles):
//     Weights (matrix B) are broadcast column-by-column into the PE grid.
//     Each cycle one row of PEs latches 8 weights (one per column).
//     weight_row_select is one-hot, stepping through rows 0→7.
//
//   Phase 2 — COMPUTE (2N−1 = 15 cycles):
//     Activations (matrix A) stream in from the WEST edge, one row per cycle,
//     but diagonally staggered (row 0 enters cycle 0, row 1 enters cycle 1, etc.)
//     — this staggering is handled by the activation_fifo upstream, NOT here.
//     Each PE does: psum_out = psum_in + (activation × weight), and forwards
//     the activation east. Partial sums flow south, accumulating the dot product.
//
//   Phase 3 — DRAIN:
//     After 2N−1 cycles, all results have exited the south edge.
//     result_valid[i] goes high when column i's result is ready.
//
// INTERNAL WIRE GRID (the key data structure):
//
//   data_h[row][col]  — horizontal activation wire entering PE[row][col] from the west
//   psum_v[row][col]  — vertical partial-sum wire entering PE[row][col] from the north
//
//   These are (N+1)-wide to include the array boundary (input or output edge):
//     data_h[i][0]    ← activation_in[i]           (west boundary)
//     data_h[i][N]    → not used (east boundary)
//     psum_v[0][j]    ← psum_in[j]  (usually 0)    (north boundary)
//     psum_v[N][j]    → result_out[j]               (south boundary)
//
// TIMING SUMMARY:
//   Weight load  :  N     cycles
//   Compute      :  2N−1  cycles
//   Total/tile   :  3N−1  cycles   (23 cycles for N=8)
//   Peak compute :  N²    MACs/cycle
// =============================================================================

module systolic_array #(
    parameter N          = 8,   // Array dimension: N×N PEs
    parameter DATA_WIDTH = 8,   // Operand width: INT8
    parameter ACC_WIDTH  = 32   // Accumulator width: INT32
) (
    input wire clk,
    input wire reset,
    input wire enable,       // Enable computation (gates all PE activity)
    input wire weight_load,  // High during weight load phase; low during compute
    input wire clear_acc,    // Clear all PE local accumulators (start of new tile)

    // -------------------------------------------------------------------------
    // Weight Loading Interface
    // -------------------------------------------------------------------------
    // weight_data[j]      : j-th column's weight value (all columns receive weights
    //                       simultaneously each cycle)
    // weight_row_select[i]: one-hot — which row of PEs latches weight_data this cycle
    //                       Sequence: 00000001 → 00000010 → ... → 10000000 over 8 cycles
    input wire signed [DATA_WIDTH-1:0] weight_data [N-1:0],
    input wire [N-1:0]                 weight_row_select,

    // -------------------------------------------------------------------------
    // Activation Input Interface (west edge)
    // -------------------------------------------------------------------------
    // activation_in[i]  : activation for row i entering from the west
    // activation_valid  : bitmask — which rows currently have valid activation data
    //                     (used by the validity pipeline to track when results emerge)
    input wire signed [DATA_WIDTH-1:0] activation_in [N-1:0],
    input wire [N-1:0]                 activation_valid,

    // -------------------------------------------------------------------------
    // Partial Sum Input (north edge — typically fed with zeros)
    // -------------------------------------------------------------------------
    // psum_in[j]: initial partial sum injected into the top of column j.
    // Usually zero for a fresh matmul tile. Non-zero enables accumulation
    // across multiple tiles (tiled matmul).
    input wire signed [ACC_WIDTH-1:0] psum_in [N-1:0],

    // -------------------------------------------------------------------------
    // Result Output (south edge)
    // -------------------------------------------------------------------------
    // result_out[j]  : fully accumulated dot product for column j
    //                  Valid when result_valid[j] is high
    // result_valid[j]: goes high when valid data has propagated through column j
    output wire signed [ACC_WIDTH-1:0] result_out [N-1:0],
    output wire [N-1:0]                result_valid,

    // -------------------------------------------------------------------------
    // Debug — direct access into every PE's internal registers
    // -------------------------------------------------------------------------
    output wire signed [DATA_WIDTH-1:0] debug_weights      [N-1:0][N-1:0],
    output wire signed [ACC_WIDTH-1:0]  debug_accumulators [N-1:0][N-1:0]
);

    // =========================================================================
    // INTERNAL WIRE GRID
    // =========================================================================

    // data_h[row][col]: activation value on the horizontal wire ENTERING PE[row][col]
    //   col=0 is the west boundary (fed from activation_in)
    //   col=N is the east boundary (unused, discarded)
    //   Size: N rows × (N+1) columns
    wire signed [DATA_WIDTH-1:0] data_h [N-1:0][N:0];

    // psum_v[row][col]: partial sum on the vertical wire ENTERING PE[row][col]
    //   row=0 is the north boundary (fed from psum_in, usually 0)
    //   row=N is the south boundary (output — final dot product per column)
    //   Size: (N+1) rows × N columns
    wire signed [ACC_WIDTH-1:0] psum_v [N:0][N-1:0];

    // Per-PE control wires (2D arrays matching the NxN grid)
    wire pe_enable      [N-1:0][N-1:0]; // Enable signal routed to each PE
    wire pe_weight_load [N-1:0][N-1:0]; // weight_load gated by row select

    // =========================================================================
    // BOUNDARY CONNECTIONS
    // Connect module ports to the edges of the internal wire grid
    // =========================================================================
    genvar i, j;
    generate
        for (i = 0; i < N; i = i + 1) begin : input_connections

            // WEST edge: row i's activation input → first column of data_h
            assign data_h[i][0] = activation_in[i];

            // NORTH edge: column i's initial psum → first row of psum_v
            // (i is used as column index here — N rows, N columns, same N)
            assign psum_v[0][i] = psum_in[i];

            // SOUTH edge: last row of psum_v → result output for column i
            assign result_out[i] = psum_v[N][i];

        end
    endgenerate

    // =========================================================================
    // VALIDITY PIPELINE
    // Tracks when valid result data has propagated through the array
    // =========================================================================
    //
    // WHY a pipeline? Because activations don't all arrive at once — they're
    // staggered diagonally. Row 0 enters on cycle 0, row 1 on cycle 1, etc.
    // The result for column j exits the bottom on cycle N + j.
    //
    // This is a shift register: each cycle, the current activation_valid bitmask
    // is shifted forward by one stage. After 2N-1 stages, column j's validity
    // bit has propagated N-1+j stages and is used to gate result_valid[j].
    //
    // valid_pipeline[stage][bit]:
    //   stage goes 0 → 2N-1 (15 stages for N=8)
    //   bit j corresponds to column j
    reg [N-1:0] valid_pipeline [2*N-1:0]; // [stage][column bitmask]

    integer p;
    always @(posedge clk) begin
        if (reset) begin
            // Clear all pipeline stages
            for (p = 0; p < 2*N; p = p + 1) begin
                valid_pipeline[p] <= {N{1'b0}};
            end
        end else if (enable && !weight_load) begin
            // During compute phase: shift validity forward one stage per cycle
            valid_pipeline[0] <= activation_valid;         // Inject current valid bits
            for (p = 1; p < 2*N; p = p + 1) begin
                valid_pipeline[p] <= valid_pipeline[p-1]; // Propagate forward
            end
        end
        // During weight_load: freeze the pipeline (don't shift)
    end

    // result_valid[i]: column i's result is valid when its bit has propagated
    // N-1+i stages through the pipeline (accounts for diagonal stagger)
    generate
        for (i = 0; i < N; i = i + 1) begin : result_valid_gen
            // valid_pipeline[N-1+i] is the stage arrived at after N-1+i shifts
            // Bit [i] of that stage corresponds to column i's data
            assign result_valid[i] = valid_pipeline[N-1+i][i];
        end
    endgenerate

    // =========================================================================
    // PE INSTANTIATION — NxN grid
    // =========================================================================
    generate
        for (i = 0; i < N; i = i + 1) begin : pe_rows
            for (j = 0; j < N; j = j + 1) begin : pe_cols

                // Global enable fans out to every PE
                assign pe_enable[i][j] = enable;

                // Weight load is gated per-row:
                // PE[i][j] only latches a weight when its row i is selected
                assign pe_weight_load[i][j] = weight_load && weight_row_select[i];

                // West input MUX:
                // During weight_load : inject weight_data[j] (the j-th column's weight)
                // During compute     : pass data_h[i][j]     (activation from west neighbor)
                // This mux is the key that makes data_in_west dual-purpose in pe.sv
                wire signed [DATA_WIDTH-1:0] pe_west_input;
                assign pe_west_input = pe_weight_load[i][j] ? weight_data[j] : data_h[i][j];

                // Instantiate PE — port names must match pe.sv exactly
                pe #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH (ACC_WIDTH)
                ) pe_inst (
                    .clk           (clk),
                    .reset         (reset),
                    .enable        (pe_enable[i][j]),
                    .weight_load   (pe_weight_load[i][j]),
                    .clear_acc     (clear_acc),

                    // Data in: the muxed west input (weight or activation)
                    .data_in_west  (pe_west_input),

                    // Partial sum in: from PE directly above (or 0 for top row)
                    .psum_in_north (psum_v[i][j]),

                    // Data out: activation forwarded to PE to the right
                    // Feeds data_h[i][j+1], which is PE[i][j+1]'s west input
                    .data_out_east (data_h[i][j+1]),

                    // Partial sum out: accumulated result flowing to PE below
                    // Feeds psum_v[i+1][j], which is PE[i+1][j]'s north input
                    .psum_out_south(psum_v[i+1][j]),

                    // Debug: directly exposes internal registers of this PE
                    .weight_debug  (debug_weights[i][j]),
                    .acc_debug     (debug_accumulators[i][j])
                );

            end
        end
    endgenerate

endmodule




