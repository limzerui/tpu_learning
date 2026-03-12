`default_nettype none
`timescale 1ns/1ns

// =============================================================================
// SYSTOLIC ARRAY — NxN Matrix Multiply Accelerator
// =============================================================================
// Computes C = A × B using weight-stationary dataflow over three sequential phases.
//
// ─── PHASE 1: WEIGHT LOAD (N cycles) ───────────────────────────────────────
//   The weight_fifo broadcasts ONE ROW of the weight matrix per cycle.
//   `weight_row_select` is a one-hot bitmask that tells WHICH row of PEs should
//   capture the weights this cycle. Sequence: 8'b00000001 → 8'b00000010 → ... → 8'b10000000
//
//   WHAT HAPPENS INSIDE EACH PE [row][col]:
//     - pe_weight_load[row][col] = (weight_load & weight_row_select[row])
//     - When asserted, data_in_west = weight_data[col] (the j-th column's weight value)
//     - PE stores it in weight_reg and holds it there for the ENTIRE compute phase.
//
// ─── PHASE 2: COMPUTE (2N−1 cycles) ────────────────────────────────────────
//   Activations stream in from the WEST edge. The activation_fifo applies diagonal
//   skew BEFORE this module — so row i's activation arrives i cycles after row 0.
//
//   WHY skew? Without it, all PEs in a row would see the same activation value at
//   once, but they each need to see a DIFFERENT activation from a different time step.
//   The skew ensures PE[row][col] multiplies the CORRECT activation with its stored weight.
//
//   Each PE:  psum_out = psum_in + (activation × weight)  — the MAC operation.
//   Partial sums flow SOUTH, accumulating a dot-product as they go down each column.
//   After 2N-1 cycles, the full result for each column exits the south edge.
//
// ─── PHASE 3: DRAIN ────────────────────────────────────────────────────────
//   Results exit at psum_v[N][j] (= result_out[j]).
//   result_valid[j] goes high when column j's data is valid.
//
// ─── INTERNAL WIRE GRID ────────────────────────────────────────────────────
//
//                  psum_v[0][j] = psum_in[j] (usually 0)
//                       │
//   data_h[i][0] ──→ [PE[0][0]] ──→ [PE[0][1]] ──→ ... ─→ data_h[0][N] (discarded)
//   (= activation_in[i])   │               │
//                    psum_v[1][0]    psum_v[1][1]
//                           │               │
//   data_h[i][0] ──→ [PE[1][0]] ──→ [PE[1][1]] ──→ ...
//                           │
//                    psum_v[N][j] = result_out[j]  ← final dot-product
//
//   data_h[i][col] : activation on the horizontal wire ENTERING PE[i][col] from west.
//   psum_v[row][j]  : partial sum on the vertical wire ENTERING PE[row][j] from north.
//
// ─── TIMING SUMMARY ────────────────────────────────────────────────────────
//   Weight load  :  N       cycles
//   Compute      :  2N − 1  cycles
//   Full tile    :  3N − 1  cycles  (23 cycles for N=8)
//   Throughput   :  N²      MACs/cycle during compute phase
// =============================================================================

module systolic_array #(
    parameter N          = 8,   // Array dimension: N×N PEs
    parameter DATA_WIDTH = 8,   // Operand width: INT8
    parameter ACC_WIDTH  = 32   // Accumulator width: INT32
) (
    input wire clk,
    input wire reset,
    input wire enable,       // Gates ALL PE activity. De-assert during idle/configuration.
    input wire weight_load,  // High during Phase 1 (weight load); low during Phase 2 (compute)
    input wire clear_acc,    // Clears the sticky local accumulator inside every PE

    // ─── Weight Loading Interface ────────────────────────────────────────────
    // weight_data[j]       : all N columns receive their weight values simultaneously
    // weight_row_select[i] : one-hot — only the selected row's PEs latch the weight
    input wire signed [DATA_WIDTH-1:0] weight_data [N-1:0],
    input wire [N-1:0]                 weight_row_select,

    // ─── Activation Input (west edge) ────────────────────────────────────────
    // activation_in[i]  : activation for row i, pre-skewed by activation_fifo
    // activation_valid  : bitmask indicating which rows have live, valid data this cycle
    input wire signed [DATA_WIDTH-1:0] activation_in [N-1:0],
    input wire [N-1:0]                 activation_valid,

    // ─── Partial Sum Input (north edge) ──────────────────────────────────────
    // Normally all zeros for a fresh tile.
    // Driving a non-zero value here enables accumulation across multiple tiles (tiled matmul).
    input wire signed [ACC_WIDTH-1:0] psum_in [N-1:0],

    // ─── Results (south edge) ────────────────────────────────────────────────
    // result_out[j]  : fully accumulated dot-product for column j
    // result_valid[j]: goes high once valid data has propagated through column j
    output wire signed [ACC_WIDTH-1:0] result_out [N-1:0],
    output wire [N-1:0]                result_valid,

    // ─── Debug ───────────────────────────────────────────────────────────────
    output wire signed [DATA_WIDTH-1:0] debug_weights      [N-1:0][N-1:0],
    output wire signed [ACC_WIDTH-1:0]  debug_accumulators [N-1:0][N-1:0]
);

    // =========================================================================
    // INTERNAL WIRE GRID
    // =========================================================================
    //
    // Size: N rows × (N+1) columns
    //   data_h[i][0]   ← activation_in[i]          (west boundary input)
    //   data_h[i][N]   → discarded                  (east boundary, nothing connects here)
    wire signed [DATA_WIDTH-1:0] data_h [N-1:0][N:0];

    // Size: (N+1) rows × N columns
    //   psum_v[0][j]   ← psum_in[j]                 (north boundary, usually 0)
    //   psum_v[N][j]   → result_out[j]               (south boundary output)
    wire signed [ACC_WIDTH-1:0] psum_v [N:0][N-1:0];

    // Per-PE control wires (2D arrays matching the NxN grid)
    wire pe_enable      [N-1:0][N-1:0];
    wire pe_weight_load [N-1:0][N-1:0];

    // =========================================================================
    // BOUNDARY CONNECTIONS
    // =========================================================================
    genvar i, j;
    generate
        for (i = 0; i < N; i = i + 1) begin : boundary_connections

            // WEST boundary: feed each row's activation into the first column of data_h
            assign data_h[i][0] = activation_in[i];

            // NORTH boundary: feed each column's initial psum (usually 0) into top of psum_v
            assign psum_v[0][i] = psum_in[i];

            // SOUTH boundary: expose each column's final accumulated psum as result_out
            assign result_out[i] = psum_v[N][i];
        end
    endgenerate

    // =========================================================================
    // VALIDITY PIPELINE — Tracking when results become valid
    // =========================================================================
    //
    // WHY IS THIS NEEDED?
    //   Activations are skewed — row 0 enters on cycle 0, row 1 on cycle 1, etc.
    //   So column j's final result isn't ready until cycle 2N-1+j (roughly).
    //   We need a mechanism to tell downstream logic EXACTLY when result_out[j] is valid.
    //
    // HOW IT WORKS (a shift register):
    //   Each cycle during compute, we inject the current `activation_valid` bitmask into
    //   stage 0 of the pipeline and shift everything forward by one stage.
    //
    //   After N-1+j shifts, stage N-1+j holds the validity bits that originally entered
    //   when row 0's activation was injected — meaning the data has now had time to
    //   propagate N-1+j PE stages (i.e., through the full column j).
    //
    //   result_valid[j] = valid_pipeline[N-1+j][j]
    //
    // Example (N=4, column 2):
    //   valid_pipeline[N-1+2] = valid_pipeline[5] = the valid mask from 5 cycles ago.
    //   Bit [2] of that mask tells us if column 2's data was valid when it started.
    //
    reg [N-1:0] valid_pipeline [2*N-1:0]; // [stage][column bitmask]

    integer p;
    always @(posedge clk) begin
        if (reset) begin
            for (p = 0; p < 2*N; p = p + 1)
                valid_pipeline[p] <= {N{1'b0}};
        end else if (enable && !weight_load) begin
            // During compute: inject new valid bits at stage 0, shift everything up
            valid_pipeline[0] <= activation_valid;
            for (p = 1; p < 2*N; p = p + 1)
                valid_pipeline[p] <= valid_pipeline[p-1];
        end
        // During weight_load: freeze the pipeline (valid bits don't shift)
    end

    // result_valid[j]: asserted when column j's accumulated result is ready
    generate
        for (i = 0; i < N; i = i + 1) begin : result_valid_gen
            assign result_valid[i] = valid_pipeline[N-1+i][i];
        end
    endgenerate

    // =========================================================================
    // PE GRID INSTANTIATION — NxN
    // =========================================================================
    generate
        for (i = 0; i < N; i = i + 1) begin : pe_rows
            for (j = 0; j < N; j = j + 1) begin : pe_cols

                // Global enable fans out to every PE in the grid
                assign pe_enable[i][j] = enable;

                // Weight load is GATED PER ROW:
                // Only PE[i][j] where weight_row_select[i]=1 will latch a weight this cycle.
                // This is what makes the one-hot row selector work.
                assign pe_weight_load[i][j] = weight_load && weight_row_select[i];

                // WEST INPUT MUX — the key to making data_in_west dual-purpose in pe.sv:
                //   During weight_load : inject weight_data[j] (j-th column's weight)
                //   During compute     : pass data_h[i][j]     (activation from left neighbor)
                wire signed [DATA_WIDTH-1:0] pe_west_input;
                assign pe_west_input = pe_weight_load[i][j] ? weight_data[j] : data_h[i][j];

                // Instantiate the PE — port names must match pe.sv exactly
                pe #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .ACC_WIDTH (ACC_WIDTH)
                ) pe_inst (
                    .clk            (clk),
                    .reset          (reset),
                    .enable         (pe_enable[i][j]),
                    .weight_load    (pe_weight_load[i][j]),
                    .clear_acc      (clear_acc),

                    // data_in_west = muxed: weight during load, activation during compute
                    .data_in_west   (pe_west_input),

                    // Partial sum in: from PE directly above (row 0 gets psum_in, usually 0)
                    .psum_in_north  (psum_v[i][j]),

                    // Activation out: forwarded east to PE[i][j+1]
                    // This feeds data_h[i][j+1], which is PE[i][j+1]'s west input
                    .data_out_east  (data_h[i][j+1]),

                    // Partial sum out: flows down to PE[i+1][j]
                    // Feeds psum_v[i+1][j], which is PE[i+1][j]'s north input
                    .psum_out_south (psum_v[i+1][j]),

                    // Debug: directly exposes this PE's internal weight and accumulator
                    .weight_debug   (debug_weights[i][j]),
                    .acc_debug      (debug_accumulators[i][j])
                );

            end
        end
    endgenerate

endmodule
