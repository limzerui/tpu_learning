`default_nettype none
`timescale 1ns/1ns

// =============================================================================
// MEMORY CONTROLLER — Dual-Channel Round-Robin Arbiter
// =============================================================================
//
// WHAT THIS MODULE DOES:
//   Multiple hardware units (host, weight FIFO, activation FIFO, accumulator)
//   all need to access the same unified buffer SRAM.
//   The memory controller sits between them and the SRAM, granting one
//   transaction at a time per channel, using round-robin fairness.
//
// YES — CHANNEL 0 AND 1 MAP DIRECTLY TO PORT A AND PORT B OF UNIFIED_BUFFER:
//
//   ┌──────────────┐       ┌────────────────────┐       ┌──────────────────┐
//   │  Consumer 0  │──req──│                    │──ch0──│  Port A          │
//   │  Consumer 1  │──req──│  Memory Controller │       │                  │
//   │  Consumer 2  │──req──│  (this module)     │──ch1──│  Port B          │ unified_buffer
//   │  Consumer 3  │──req──│                    │       │                  │
//   └──────────────┘       └────────────────────┘       └──────────────────┘
//
//   Channel 0 (ch0) → driven into port_a_en/we/addr/wdata of unified_buffer
//   Channel 1 (ch1) → driven into port_b_en/we/addr/wdata of unified_buffer
//
// In a typical TPU configuration:
//   ch0 (Port A): arbitrated between HOST, WEIGHT_FIFO, ACTIVATION_FIFO
//   ch1 (Port B): dedicated to ACCUMULATOR WRITEBACK (results from systolic array)
//
// ROUND-ROBIN ARBITRATION:
//   Each channel tracks a `rr_priority` pointer. On each IDLE cycle, it scans
//   consumers starting from `rr_priority`, picks the first pending one, serves it,
//   then advances `rr_priority` to one past the served consumer.
//   This prevents any single consumer from starving others.
//
// FSM PER CHANNEL (5 states):
//   IDLE         → scan for a pending consumer
//   READ_WAITING → memory request sent, waiting for mem_chX_valid pulse
//   WRITE_WAITING→ write sent (1-cycle SRAM), immediately proceed to relay
//   READ_RELAYING→ data latched, hold it until consumer de-asserts request
//   WRITE_RELAYING→ ack held until consumer de-asserts request
// =============================================================================

module memory_controller #(
    parameter ADDR_WIDTH    = 14,
    parameter DATA_WIDTH    = 32,
    parameter NUM_CONSUMERS = 4,
    parameter NUM_CHANNELS  = 2
) (
    input wire clk,
    input wire reset,

    // =========================================================================
    // Consumer Interface — N consumers, each can request reads or writes
    // =========================================================================

    // READ: consumer asserts read_valid + read_address, holds until read_ready
    input wire  [NUM_CONSUMERS-1:0]      consumer_read_valid,
    input wire  [ADDR_WIDTH-1:0]         consumer_read_address [NUM_CONSUMERS-1:0],
    output reg  [NUM_CONSUMERS-1:0]      consumer_read_ready,    // Pulsed when data ready
    output reg  [DATA_WIDTH-1:0]         consumer_read_data [NUM_CONSUMERS-1:0], // Read result

    // WRITE: consumer asserts write_valid + address + data, holds until write_ready
    input wire  [NUM_CONSUMERS-1:0]      consumer_write_valid,
    input wire  [ADDR_WIDTH-1:0]         consumer_write_address [NUM_CONSUMERS-1:0],
    input wire  [DATA_WIDTH-1:0]         consumer_write_data [NUM_CONSUMERS-1:0],
    output reg  [NUM_CONSUMERS-1:0]      consumer_write_ready,   // Pulsed when write complete

    // =========================================================================
    // Memory Channel Interfaces — connect directly to unified_buffer ports
    // ch0 → unified_buffer Port A
    // ch1 → unified_buffer Port B
    // =========================================================================
    output reg                   mem_ch0_en,
    output reg                   mem_ch0_we,
    output reg  [ADDR_WIDTH-1:0] mem_ch0_addr,
    output reg  [DATA_WIDTH-1:0] mem_ch0_wdata,
    input wire  [DATA_WIDTH-1:0] mem_ch0_rdata,
    input wire                   mem_ch0_valid,   // 1 cycle after read request

    output reg                   mem_ch1_en,
    output reg                   mem_ch1_we,
    output reg  [ADDR_WIDTH-1:0] mem_ch1_addr,
    output reg  [DATA_WIDTH-1:0] mem_ch1_wdata,
    input wire  [DATA_WIDTH-1:0] mem_ch1_rdata,
    input wire                   mem_ch1_valid,

    // =========================================================================
    // Status / Debug
    // =========================================================================
    output wire                      busy,             // Either channel is active
    output wire [NUM_CONSUMERS-1:0]  consumer_pending, // Consumers waiting but not yet served
    output wire [2:0]                debug_state_ch0,
    output wire [2:0]                debug_state_ch1
);

    // =========================================================================
    // State Encoding
    // =========================================================================
    localparam IDLE           = 3'b000;
    localparam READ_WAITING   = 3'b001; // Request sent to SRAM, waiting for valid pulse
    localparam READ_RELAYING  = 3'b010; // Data latched, holding until consumer de-asserts
    localparam WRITE_WAITING  = 3'b011; // Write sent to SRAM (1-cycle, no valid pulse)
    localparam WRITE_RELAYING = 3'b100; // ACK held until consumer de-asserts

    // =========================================================================
    // Internal Registers
    // =========================================================================
    // Per-channel FSM state
    reg [2:0] channel_state [0:NUM_CHANNELS-1];

    // Which consumer is currently being served by each channel
    reg [$clog2(NUM_CONSUMERS)-1:0] current_consumer [0:NUM_CHANNELS-1];

    // Prevents a consumer from being granted on BOTH channels simultaneously
    reg [NUM_CONSUMERS-1:0] consumer_being_served;

    // Round-robin pointer — shared across channels (one global fairness tracker)
    reg [$clog2(NUM_CONSUMERS)-1:0] rr_priority;

    integer arb_j;                                        // Loop variable for arbitration
    reg [$clog2(NUM_CONSUMERS)-1:0] arb_idx;              // Candidate consumer index

    // =========================================================================
    // Output Assignments
    // =========================================================================
    assign busy             = (channel_state[0] != IDLE) || (channel_state[1] != IDLE);
    assign consumer_pending = (consumer_read_valid | consumer_write_valid) & ~consumer_being_served;
    assign debug_state_ch0  = channel_state[0];
    assign debug_state_ch1  = channel_state[1];

    // =========================================================================
    // CHANNEL 0 FSM — Arbitrates Port A of unified_buffer
    // =========================================================================
    always @(posedge clk) begin
        if (reset) begin
            channel_state[0]    <= IDLE;
            current_consumer[0] <= 0;
            mem_ch0_en          <= 1'b0;
            mem_ch0_we          <= 1'b0;
            mem_ch0_addr        <= {ADDR_WIDTH{1'b0}};
            mem_ch0_wdata       <= {DATA_WIDTH{1'b0}};
            consumer_being_served <= {NUM_CONSUMERS{1'b0}};
            rr_priority         <= 0;
        end else begin
            case (channel_state[0])

                IDLE: begin
                    mem_ch0_en <= 1'b0;
                    mem_ch0_we <= 1'b0;
                    // Round-robin scan: start from rr_priority, wrap around
                    for (arb_j = 0; arb_j < NUM_CONSUMERS; arb_j = arb_j + 1) begin
                        arb_idx = (rr_priority + arb_j) % NUM_CONSUMERS;

                        if (consumer_read_valid[arb_idx] && !consumer_being_served[arb_idx]) begin
                            // Grant read to this consumer
                            consumer_being_served[arb_idx] <= 1'b1;
                            current_consumer[0]            <= arb_idx;
                            mem_ch0_en                     <= 1'b1;
                            mem_ch0_we                     <= 1'b0;
                            mem_ch0_addr                   <= consumer_read_address[arb_idx];
                            rr_priority                    <= (arb_idx + 1) % NUM_CONSUMERS;
                            channel_state[0]               <= READ_WAITING;
                            arb_j                          = NUM_CONSUMERS; // Synthesizable break

                        end else if (consumer_write_valid[arb_idx] && !consumer_being_served[arb_idx]) begin
                            // Grant write to this consumer
                            consumer_being_served[arb_idx] <= 1'b1;
                            current_consumer[0]            <= arb_idx;
                            mem_ch0_en                     <= 1'b1;
                            mem_ch0_we                     <= 1'b1;
                            mem_ch0_addr                   <= consumer_write_address[arb_idx];
                            mem_ch0_wdata                  <= consumer_write_data[arb_idx];
                            rr_priority                    <= (arb_idx + 1) % NUM_CONSUMERS;
                            channel_state[0]               <= WRITE_WAITING;
                            arb_j                          = NUM_CONSUMERS; // Synthesizable break
                        end
                    end
                end

                READ_WAITING: begin
                    // SRAM has 1-cycle read latency — wait for the valid pulse
                    if (mem_ch0_valid) begin
                        mem_ch0_en                                     <= 1'b0;
                        consumer_read_ready[current_consumer[0]]       <= 1'b1;
                        consumer_read_data[current_consumer[0]]        <= mem_ch0_rdata;
                        channel_state[0]                               <= READ_RELAYING;
                    end
                end

                WRITE_WAITING: begin
                    // SRAM write takes effect this cycle — no valid pulse needed
                    // Immediately de-assert and notify consumer
                    mem_ch0_en                                         <= 1'b0;
                    mem_ch0_we                                         <= 1'b0;
                    consumer_write_ready[current_consumer[0]]          <= 1'b1;
                    channel_state[0]                                   <= WRITE_RELAYING;
                end

                READ_RELAYING: begin
                    // Hold read_ready + data until consumer de-asserts their request
                    // (ACK handshake — consumer controls when to release)
                    if (!consumer_read_valid[current_consumer[0]]) begin
                        consumer_being_served[current_consumer[0]] <= 1'b0;
                        consumer_read_ready[current_consumer[0]]   <= 1'b0;
                        channel_state[0]                           <= IDLE;
                    end
                end

                WRITE_RELAYING: begin
                    // Hold write_ready until consumer de-asserts their request
                    if (!consumer_write_valid[current_consumer[0]]) begin
                        consumer_being_served[current_consumer[0]] <= 1'b0;
                        consumer_write_ready[current_consumer[0]]  <= 1'b0;
                        channel_state[0]                           <= IDLE;
                    end
                end

                default: channel_state[0] <= IDLE;
            endcase
        end
    end

    // Channel 1 state machine (similar to channel 0, starts from different priority)
    always @(posedge clk) begin
        if (reset) begin
            channel_state[1] <= IDLE;
            current_consumer[1] <= 0;
            mem_ch1_en <= 1'b0;
            mem_ch1_we <= 1'b0;
            mem_ch1_addr <= {ADDR_WIDTH{1'b0}};
            mem_ch1_wdata <= {DATA_WIDTH{1'b0}};
        end else begin
            case (channel_state[1])
                IDLE: begin
                    mem_ch1_en <= 1'b0;
                    mem_ch1_we <= 1'b0;

                    // Look for pending requests (start from opposite end for fairness)
                    begin
                        integer j;
                        for (j = 0; j < NUM_CONSUMERS; j = j + 1) begin
                            reg [$clog2(NUM_CONSUMERS)-1:0] idx;
                            // Start from opposite direction for channel 1
                            idx = (rr_priority + NUM_CONSUMERS/2 + j) % NUM_CONSUMERS;

                            if (consumer_read_valid[idx] && !consumer_being_served[idx]) begin
                                consumer_being_served[idx] <= 1'b1;
                                current_consumer[1] <= idx;

                                mem_ch1_en <= 1'b1;
                                mem_ch1_we <= 1'b0;
                                mem_ch1_addr <= consumer_read_address[idx];
                                channel_state[1] <= READ_WAITING;
                                j = NUM_CONSUMERS; // Break
                            end else if (consumer_write_valid[idx] && !consumer_being_served[idx]) begin
                                consumer_being_served[idx] <= 1'b1;
                                current_consumer[1] <= idx;

                                mem_ch1_en <= 1'b1;
                                mem_ch1_we <= 1'b1;
                                mem_ch1_addr <= consumer_write_address[idx];
                                mem_ch1_wdata <= consumer_write_data[idx];
                                channel_state[1] <= WRITE_WAITING;
                                j = NUM_CONSUMERS; // Break
                            end
                        end
                    end
                end

                READ_WAITING: begin
                    if (mem_ch1_valid) begin
                        mem_ch1_en <= 1'b0;
                        consumer_read_ready[current_consumer[1]] <= 1'b1;
                        consumer_read_data[current_consumer[1]] <= mem_ch1_rdata;
                        channel_state[1] <= READ_RELAYING;
                    end
                end

                WRITE_WAITING: begin
                    mem_ch1_en <= 1'b0;
                    mem_ch1_we <= 1'b0;
                    consumer_write_ready[current_consumer[1]] <= 1'b1;
                    channel_state[1] <= WRITE_RELAYING;
                end

                READ_RELAYING: begin
                    if (!consumer_read_valid[current_consumer[1]]) begin
                        consumer_being_served[current_consumer[1]] <= 1'b0;
                        consumer_read_ready[current_consumer[1]] <= 1'b0;
                        channel_state[1] <= IDLE;
                    end
                end

                WRITE_RELAYING: begin
                    if (!consumer_write_valid[current_consumer[1]]) begin
                        consumer_being_served[current_consumer[1]] <= 1'b0;
                        consumer_write_ready[current_consumer[1]] <= 1'b0;
                        channel_state[1] <= IDLE;
                    end
                end

                default: channel_state[1] <= IDLE;
            endcase
        end
    end

    // =========================================================================
    // Simulation Initialisation
    // =========================================================================
    integer init_i;
    initial begin
        for (init_i = 0; init_i < NUM_CONSUMERS; init_i = init_i + 1) begin  // was: ini_i (typo)
            consumer_read_ready[init_i]  = 1'b0;   // = not <= in initial blocks
            consumer_write_ready[init_i] = 1'b0;
            consumer_read_data[init_i]   = {DATA_WIDTH{1'b0}};
        end
        consumer_being_served = {NUM_CONSUMERS{1'b0}};
        rr_priority           = 0;
    end

endmodule


// =============================================================================
// SIMPLIFIED SINGLE-CHANNEL MEMORY CONTROLLER
// =============================================================================
// Stripped-down version: one arbitrated channel, no dual-port.
// Useful when you only have one memory port to share (e.g. a simple FIFO or
// register-based buffer rather than a dual-port BRAM).
//
// 3-state FSM:
//   IDLE → pick winning consumer, send request to memory
//   WAIT → wait for memory to respond (read: wait for valid; write: 1 cycle)
//   ACK  → hold ack until consumer de-asserts request
// =============================================================================

module memory_controller_simple #(
    parameter ADDR_WIDTH    = 14,
    parameter DATA_WIDTH    = 32,
    parameter NUM_CONSUMERS = 4
) (
    input wire clk,
    input wire reset,

    // Consumer interface: request + optional write data
    // consumer_req[i]: consumer i wants a transaction
    // consumer_we[i] : 1 = write, 0 = read
    input wire  [NUM_CONSUMERS-1:0]  consumer_req,
    input wire  [NUM_CONSUMERS-1:0]  consumer_we,
    input wire  [ADDR_WIDTH-1:0]     consumer_addr  [NUM_CONSUMERS-1:0],
    input wire  [DATA_WIDTH-1:0]     consumer_wdata [NUM_CONSUMERS-1:0],
    output reg  [NUM_CONSUMERS-1:0]  consumer_ack,    // One-hot: which consumer is acked
    output reg  [DATA_WIDTH-1:0]     consumer_rdata,  // Shared read bus (all consumers see it)

    // Single memory channel
    output reg                       mem_en,
    output reg                       mem_we,
    output reg  [ADDR_WIDTH-1:0]     mem_addr,
    output reg  [DATA_WIDTH-1:0]     mem_wdata,
    input wire  [DATA_WIDTH-1:0]     mem_rdata,
    input wire                       mem_valid   // SRAM pulses this 1 cycle after read
);

    localparam IDLE = 2'b00;
    localparam WAIT = 2'b01;  // Waiting for memory response
    localparam ACK  = 2'b10;  // Holding ack until consumer releases

    reg [1:0]                        state;
    reg [$clog2(NUM_CONSUMERS)-1:0]  current;      // Which consumer is being served
    reg [$clog2(NUM_CONSUMERS)-1:0]  priority_ptr; // Round-robin pointer

    integer                          rr_i;
    reg [$clog2(NUM_CONSUMERS)-1:0]  rr_idx;

    always @(posedge clk) begin
        if (reset) begin
            state          <= IDLE;
            current        <= 0;
            priority_ptr   <= 0;
            mem_en         <= 1'b0;
            mem_we         <= 1'b0;
            mem_addr       <= {ADDR_WIDTH{1'b0}};
            mem_wdata      <= {DATA_WIDTH{1'b0}};
            consumer_ack   <= {NUM_CONSUMERS{1'b0}};
            consumer_rdata <= {DATA_WIDTH{1'b0}};
        end else begin
            case (state)

                IDLE: begin
                    consumer_ack <= {NUM_CONSUMERS{1'b0}}; // Clear all acks
                    // Round-robin: scan from priority_ptr, wrap modulo NUM_CONSUMERS
                    for (rr_i = 0; rr_i < NUM_CONSUMERS; rr_i = rr_i + 1) begin
                        rr_idx = (priority_ptr + rr_i) % NUM_CONSUMERS;
                        if (consumer_req[rr_idx]) begin
                            current      <= rr_idx;
                            mem_en       <= 1'b1;
                            mem_we       <= consumer_we[rr_idx];
                            mem_addr     <= consumer_addr[rr_idx];
                            mem_wdata    <= consumer_wdata[rr_idx];
                            priority_ptr <= (rr_idx + 1) % NUM_CONSUMERS;
                            state        <= WAIT;
                            rr_i         = NUM_CONSUMERS; // Break out of loop
                        end
                    end
                end

                WAIT: begin
                    // For READS : wait for mem_valid (1-cycle SRAM latency)
                    // For WRITES: SRAM writes immediately — mem_we is already registered
                    //             so we can proceed after 1 cycle stall (mem_we is still 1 here)
                    if (mem_valid || mem_we) begin
                        mem_en              <= 1'b0;
                        mem_we              <= 1'b0;
                        consumer_rdata      <= mem_rdata;   // Latch read data (0 for writes)
                        consumer_ack[current] <= 1'b1;
                        state               <= ACK;
                    end
                end

                ACK: begin
                    // Hold ack until consumer de-asserts its request
                    // (consumer sees ack=1, captures data, then lowers req)
                    if (!consumer_req[current]) begin
                        consumer_ack[current] <= 1'b0;
                        state                 <= IDLE;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
