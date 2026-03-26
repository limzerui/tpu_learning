// =========================================================================
// TPU ACTIVATION UNIT
// =========================================================================
// > Applies non-linear activation functions to output data from the
//   Accumulator Buffer before storing results back to the Unified Buffer.
//
// Supported Functions (func_sel):
//   3'b000 FUNC_RELU    -> max(0, x)                      [most common]
//   3'b001 FUNC_GELU    -> x * sigmoid(1.702 * x)         [Transformers]
//   3'b010 FUNC_SILU    -> x * sigmoid(x)  (Swish)        [EfficientNet]
//   3'b011 FUNC_SIGMOID -> 1 / (1 + exp(-x))              [gate functions]
//   3'b100 FUNC_TANH    -> (exp(x) - exp(-x)) / (exp(x) + exp(-x))
//
// Architecture:
//   - N parallel "lanes", each processing one INT8 value simultaneously
//   - GELU, SiLU, Sigmoid, Tanh use pre-computed LUTs (Look-Up Tables)
//     to avoid expensive hardware multipliers/dividers
//   - ReLU is pure combinational logic (no LUT needed)
//
// LUT Details:
//   - 256 entries (covers all 8-bit inputs)
//   - Piecewise linear approximation (8 segments per function)
//   - Input index = raw INT8 value + 128 (converts signed [-128,127]
//     to unsigned address [0,255])
//
// FSM:
//   IDLE    -> Wait for enable + data_in_valid
//   COMPUTE -> Latch inputs, select from pre-computed combinational results
//   OUTPUT  -> Drive data_out and assert data_out_valid
// =========================================================================

module activation_unit #(
    parameter N = 8,                    // Number of parallel lanes (matches systolic array width)
    parameter DATA_WIDTH = 8,           // Input/output width in bits (INT8)
    parameter LUT_ADDR_WIDTH = 8        // 256-entry LUTs (2^8 = 256)
) (
    input wire clk,
    input wire reset,

    // Control interface
    input wire              enable,         // Pulse high to start processing a batch
    input wire [2:0]        func_sel,       // Selects activation function (see localparams above)
    output reg              busy,           // High while processing, low when idle
    output reg              done,           // Pulses high for 1 cycle when output is ready

    // Data interface (N parallel lanes packed into a single wide bus)
    // Each lane occupies DATA_WIDTH bits: lane[0] = data[7:0], lane[1] = data[15:8], etc.
    input wire  [N*DATA_WIDTH-1:0]  data_in,       // Packed input from Accumulator Buffer
    input wire                      data_in_valid,  // Must be high when asserting enable
    output reg  [N*DATA_WIDTH-1:0]  data_out,       // Packed output to Unified Buffer
    output reg                      data_out_valid,  // High for 1 cycle when data_out is valid

    // Debug
    output wire [3:0]       debug_state
);

    // -------------------------------------------------------------------------
    // Function select constants
    // -------------------------------------------------------------------------
    localparam FUNC_RELU    = 3'b000;
    localparam FUNC_GELU    = 3'b001;
    localparam FUNC_SILU    = 3'b010;
    localparam FUNC_SIGMOID = 3'b011;
    localparam FUNC_TANH    = 3'b100;

    // -------------------------------------------------------------------------
    // FSM state encoding
    // -------------------------------------------------------------------------
    localparam IDLE    = 2'b00;
    localparam COMPUTE = 2'b01;
    localparam OUTPUT  = 2'b10; // Must be 2-bit to fit in reg [1:0] state

    reg [1:0] state;
    assign debug_state = {2'b00, state}; // Zero-extend to 4 bits for debug port

    // -------------------------------------------------------------------------
    // Internal data registers (N lanes of INT8)
    // -------------------------------------------------------------------------
    reg signed [DATA_WIDTH-1:0] input_regs  [0:N-1]; // Latched inputs
    reg signed [DATA_WIDTH-1:0] output_regs [0:N-1]; // Computed results

    // -------------------------------------------------------------------------
    // Look-Up Tables (LUTs)
    // =========================================================================
    // Why LUTs? Computing exp(), division, and tanh() in hardware requires
    // large floating-point units that are slow and area-expensive. Since we
    // use INT8 (only 256 possible input values), we can pre-calculate ALL
    // possible answers at initialization and store them in a tiny SRAM block.
    // At runtime: result = lut[input + 128]  (just a memory read = 1 cycle)
    //
    // Index conversion: input_reg is signed INT8 [-128, 127]
    //   Adding 128 shifts the range to unsigned [0, 255] for array indexing.
    // =========================================================================

    // Sigmoid LUT: output range [0, 127] representing [0.0, 1.0)
    // (unsigned because sigmoid never goes negative)
    reg [DATA_WIDTH-1:0]        sigmoid_lut [0:255];

    // Tanh LUT: output range [-127, 127] representing [-1.0, 1.0)
    reg signed [DATA_WIDTH-1:0] tanh_lut    [0:255];

    // GELU LUT: GELU(x) ≈ x * sigmoid(1.702 * x)
    // Output range [-128, 127] (can be slightly negative near x=-0.17)
    reg signed [DATA_WIDTH-1:0] gelu_lut    [0:255];

    // -------------------------------------------------------------------------
    // LUT Initialization (runs ONCE before simulation/synthesis)
    // Uses piecewise linear approximation: y = y_start + slope * (i - x_start)
    // -------------------------------------------------------------------------
    integer i;
    initial begin

        // =====================================================================
        // SIGMOID LUT: f(x) = 1 / (1 + exp(-x))
        // Input domain: x in [-8, 8] (i=0 maps to x=-8, i=255 maps to x=+8)
        // Output domain: [0, 127] (127 represents 1.0 in Q0.7 fixed-point)
        // =====================================================================
        for (i = 0; i < 256; i = i + 1) begin
            if (i < 32) begin
                // x in [-8, -6]: sigmoid is essentially flat at 0
                sigmoid_lut[i] = 8'd0;
            end else if (i < 64) begin
                // x in [-6, -4]: gentle linear rise, slope = 0.5
                // y = 0 + 0.5*(i-32)  =>  (i-32) >> 1
                sigmoid_lut[i] = (i - 32) >> 1;
            end else if (i < 96) begin
                // x in [-4, -2]: steeper rise, slope = 1.0
                // y = 16 + 1.0*(i-64)
                sigmoid_lut[i] = 16 + (i - 64);
            end else if (i < 128) begin
                // x in [-2, 0]: steep rise to center (0.5 = value 64)
                // y = 48 + 0.75*(i-96)  =>  48 + (i-96)*3 >> 2
                sigmoid_lut[i] = 48 + ((i - 96) * 3 >> 2);
            end else if (i < 160) begin
                // x in [0, 2]: steep rise from center (0.5 = value 64)
                sigmoid_lut[i] = 64 + ((i - 128) * 3 >> 2);
            end else if (i < 192) begin
                // x in [2, 4]: shallower, slope = 1.0
                sigmoid_lut[i] = 88 + (i - 160);
            end else if (i < 224) begin
                // x in [4, 6]: gentle approach to ceiling, slope = 0.5
                sigmoid_lut[i] = 120 + ((i - 192) >> 1);
            end else begin
                // x in [6, 8]: sigmoid is essentially flat at 1
                sigmoid_lut[i] = 8'd127;
            end
        end

        // =====================================================================
        // TANH LUT: f(x) = (e^x - e^-x) / (e^x + e^-x)
        // Input domain: x in [-4, 4] (tighter range; saturates faster than sigmoid)
        // Output domain: [-127, 127] representing [-1.0, 1.0)
        // =====================================================================
        for (i = 0; i < 256; i = i + 1) begin
            if (i < 32) begin
                // x < -3: tanh ≈ -1
                tanh_lut[i] = -8'sd127;
            end else if (i < 64) begin
                // x in [-3, -2], slope = 2.0
                tanh_lut[i] = -8'sd127 + $signed((i - 32) << 1);
            end else if (i < 96) begin
                // x in [-2, -1], slope = 2.0
                tanh_lut[i] = -8'sd63 + $signed((i - 64) << 1);
            end else if (i < 128) begin
                // x in [-1, 0]: steep rise to 0, slope = 2.0
                tanh_lut[i] = $signed((i - 128) << 1);
            end else if (i < 160) begin
                // x in [0, 1]: steep rise from 0, slope = 2.0
                tanh_lut[i] = $signed((i - 128) << 1);
            end else if (i < 192) begin
                // x in [1, 2], slope = 2.0
                tanh_lut[i] = 8'sd63 + $signed((i - 160) << 1);
            end else if (i < 224) begin
                // x in [2, 3], slope = 2.0
                tanh_lut[i] = 8'sd127 - $signed((224 - i) << 1);
            end else begin
                // x > 3: tanh ≈ +1
                tanh_lut[i] = 8'sd127;
            end
        end

        // =====================================================================
        // GELU LUT: f(x) ≈ x * sigmoid(1.702 * x)
        // Piecewise linear approximation of the smooth GELU curve
        // Output can be slightly negative (unlike ReLU), hence 'signed'
        // =====================================================================
        for (i = 0; i < 256; i = i + 1) begin
            if (i < 64) begin
                // x < -2: GELU ≈ 0 (unlike ReLU, GELU doesn't hard-clip)
                gelu_lut[i] = 8'd0;
            end else if (i < 96) begin
                // x in [-2, -1]: slight dip below 0
                gelu_lut[i] = $signed(i - 80) >>> 2;
            end else if (i < 128) begin
                // x in [-1, 0]: transition region
                gelu_lut[i] = $signed(i - 128) >>> 1;
            end else if (i < 160) begin
                // x in [0, 1]: approximately linear, slope ≈ 1.0
                gelu_lut[i] = (i - 128);
            end else if (i < 192) begin
                // x in [1, 2]: slope ≈ 1.5 (approaching linear)
                gelu_lut[i] = 32 + ((i - 160) * 3 >> 1);
            end else begin
                // x > 2: approximately linear with slope 1 (GELU ≈ x here)
                gelu_lut[i] = 80 + (i - 192);
            end
        end
    end

    // -------------------------------------------------------------------------
    // Combinational Activation Logic
    // =========================================================================
    // This always @(*) block is PURELY COMBINATIONAL — no clocks, no registers.
    // For every lane simultaneously, it pre-calculates all activation results.
    // The FSM (below) then selects which result to keep based on func_sel.
    //
    // Note: the 'for' loop here generates N identical parallel hardware blocks,
    //       NOT sequential computation. All lanes compute at the same time.
    // -------------------------------------------------------------------------
    reg signed [DATA_WIDTH-1:0] relu_result    [0:N-1];
    reg signed [DATA_WIDTH-1:0] gelu_result    [0:N-1];
    reg signed [DATA_WIDTH-1:0] silu_result    [0:N-1];
    reg        [DATA_WIDTH-1:0] sigmoid_result [0:N-1]; // unsigned: [0,127]
    reg signed [DATA_WIDTH-1:0] tanh_result    [0:N-1];

    // SiLU intermediate: x * sigmoid(x)
    // x is Q7.0 (7-bit integer), sigmoid output is Q0.7 (7-bit fraction)
    // Their product is Q7.7 (16-bit). We then shift right by 7 to get Q7.0.
    reg signed [15:0] silu_product [0:N-1];

    integer lane;

    always @(*) begin
        for (lane = 0; lane < N; lane = lane + 1) begin

            // RELU: clamp negatives to 0
            if (input_regs[lane] < 0)
                relu_result[lane] = 8'd0;
            else
                relu_result[lane] = input_regs[lane];

            // LUT lookups: add 128 to convert signed index [-128,127] -> [0,255]
            sigmoid_result[lane] = sigmoid_lut[input_regs[lane] + 128];
            tanh_result[lane]    = tanh_lut   [input_regs[lane] + 128];
            gelu_result[lane]    = gelu_lut   [input_regs[lane] + 128];

            // SILU: x * sigmoid(x)
            // {1'b0, sigmoid_result} zero-extends the unsigned sigmoid to a signed 9-bit value
            // so the multiplication keeps the sign of x correctly
            silu_product[lane] = $signed(input_regs[lane]) * $signed({1'b0, sigmoid_result[lane]});
            silu_result[lane]  = silu_product[lane][14:7]; // Take upper 8 bits (drop the 7 fractional bits)
        end
    end

    // -------------------------------------------------------------------------
    // Main FSM (Sequential)
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (reset) begin
            state          <= IDLE;
            busy           <= 1'b0;
            done           <= 1'b0;
            data_out       <= {(N*DATA_WIDTH){1'b0}};
            data_out_valid <= 1'b0;
            for (lane = 0; lane < N; lane = lane + 1) begin
                input_regs [lane] <= {DATA_WIDTH{1'b0}};
                output_regs[lane] <= {DATA_WIDTH{1'b0}};
            end
        end else begin
            // Default: clear pulse signals so they only fire for 1 cycle
            done           <= 1'b0;
            data_out_valid <= 1'b0;

            case (state)
                // ---------------------------------------------------------------
                // IDLE: Wait for host to assert enable with valid data
                // ---------------------------------------------------------------
                IDLE: begin
                    if (enable && data_in_valid) begin
                        busy <= 1'b1;
                        // Unpack the flat data_in bus into N individual input registers
                        // [lane*DATA_WIDTH +: DATA_WIDTH] = slice DATA_WIDTH bits starting at bit lane*DATA_WIDTH
                        for (lane = 0; lane < N; lane = lane + 1) begin
                            input_regs[lane] <= data_in[lane*DATA_WIDTH +: DATA_WIDTH];
                        end
                        state <= COMPUTE;
                    end
                end

                // ---------------------------------------------------------------
                // COMPUTE: Combinational results are already computed above.
                // Just latch the correct one based on func_sel into output_regs.
                // This takes exactly 1 clock cycle.
                // ---------------------------------------------------------------
                COMPUTE: begin
                    for (lane = 0; lane < N; lane = lane + 1) begin
                        case (func_sel)
                            FUNC_RELU:    output_regs[lane] <= relu_result[lane];
                            FUNC_GELU:    output_regs[lane] <= gelu_result[lane];
                            FUNC_SILU:    output_regs[lane] <= silu_result[lane];
                            FUNC_SIGMOID: output_regs[lane] <= $signed(sigmoid_result[lane]); // cast unsigned lut output to signed
                            FUNC_TANH:    output_regs[lane] <= tanh_result[lane];
                            default:      output_regs[lane] <= {DATA_WIDTH{1'b0}}; // Pass zeros for unknown func
                        endcase
                    end
                    state <= OUTPUT;
                end

                // ---------------------------------------------------------------
                // OUTPUT: Pack the N output registers back into a flat bus and
                // signal to the downstream module that the data is valid.
                // ---------------------------------------------------------------
                OUTPUT: begin
                    for (lane = 0; lane < N; lane = lane + 1) begin
                        data_out[lane*DATA_WIDTH +: DATA_WIDTH] <= output_regs[lane];
                    end
                    data_out_valid <= 1'b1; // Valid for 1 cycle
                    done           <= 1'b1; // Signal completion to Sequencer
                    busy           <= 1'b0;
                    state          <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule