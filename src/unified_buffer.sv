`default_nettype none
`timescale 1ns/1ns

// =============================================================================
// UNIFIED BUFFER — Dual-Port SRAM
// =============================================================================
// A single shared memory block (64KB) accessible through two independent ports:
//
//   Port A — Read/Write, used by the HOST (CPU-side control, weight/activation loads)
//   Port B — Read/Write, used by the SYSTOLIC ARRAY (accumulator writeback, result reads)
//
// WHY dual-port?
//   The host needs to load weights while the array might be storing results.
//   A single-port SRAM would require strict sequencing — they'd block each other.
//   Two ports let host and array operate in parallel (with a caveat: same-address
//   write collision is undefined — the memory controller must prevent this).
//
// MEMORY MAP (word-addressed, DATA_WIDTH=32 bits per word):
//   [0x0000 - 0x13FF]  Activations  (5120 words = 20KB)
//   [0x1400 - 0x2FFF]  Weights      (6144 words = 24KB)
//   [0x3000 - 0x3FFF]  Outputs      (4096 words = 16KB)
//   [0x4000 - 0x4FFF]  Scratch      (4096 words = 16KB) — your addition
//
// FPGA SYNTHESIS NOTE:
//   (* ram_style = "block" *) tells Xilinx tools to use BRAM primitives,
//   not LUT-based distributed RAM. Always use this for large memories.
// LATENCY: 1 cycle read latency (registered output — synchronous BRAM behaviour)
//   Write: data appears in mem on the same posedge clk as write strobe
//   Read : port_x_rdata is valid ONE cycle AFTER port_x_en is asserted
//          port_x_valid pulses for ONE cycle to signal the read data is ready
//
// BYTE ENABLES:
//   port_x_byte_en[3:0] lets you write individual bytes of a 32-bit word.
//   bit 0 → bytes [7:0], bit 1 → bytes [15:8], etc.
//   Useful for packing 4 INT8 values into one 32-bit word.
// =============================================================================

module unified_buffer #(
    parameter DEPTH      = 16384, // Number of words (16K × 32 bits = 64KB total)
    parameter DATA_WIDTH = 32,    // Word width: 32 bits (holds 4 INT8 values packed)
    parameter ADDR_WIDTH = 14     // Address width: 2^14 = 16384 words
) (
    input wire clk,
    input wire reset,

    // =========================================================================
    // PORT A — Host Interface (read/write)
    // =========================================================================
    input wire                   port_a_en,      // Enable Port A transaction this cycle
    input wire                   port_a_we,      // 1=write, 0=read
    input wire  [ADDR_WIDTH-1:0] port_a_addr,    // Word address to access
    input wire  [DATA_WIDTH-1:0] port_a_wdata,   // Data to write (ignored on reads)
    output reg  [DATA_WIDTH-1:0] port_a_rdata,   // Data read (valid one cycle after en)
    output reg                   port_a_valid,   // Pulses 1 cycle after a read completes
    input wire  [3:0]            port_a_byte_en, // Byte-level write mask [3:0]

    // =========================================================================
    // PORT B — Systolic Array / Accumulator Interface (read/write)
    // =========================================================================
    input wire                   port_b_en,      // Enable Port B transaction this cycle
    input wire                   port_b_we,      // 1=write, 0=read
    input wire  [ADDR_WIDTH-1:0] port_b_addr,    // Word address to access
    input wire  [DATA_WIDTH-1:0] port_b_wdata,   // Data to write
    output reg  [DATA_WIDTH-1:0] port_b_rdata,   // Data read
    output reg                   port_b_valid,   // Read valid pulse
    input wire  [3:0]            port_b_byte_en, // Byte-level write mask [3:0]

    // =========================================================================
    // Status / Debug
    // =========================================================================
    output wire                  busy,               // Always 0 (SRAM never stalls)
    output wire [ADDR_WIDTH-1:0] debug_last_addr_a,  // Last address accessed on Port A
    output wire [ADDR_WIDTH-1:0] debug_last_addr_b   // Last address accessed on Port B
);

    // =========================================================================
    // Memory Array
    // (* ram_style = "block" *) — FPGA synthesis attribute, forces BRAM inference.
    // Without this, the tool may try to build this from LUTs (very inefficient for 64KB).
    // =========================================================================
    (* ram_style = "block" *)
    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    // Track last accessed address per port (debug only)
    reg [ADDR_WIDTH-1:0] last_addr_a;
    reg [ADDR_WIDTH-1:0] last_addr_b;

    assign debug_last_addr_a = last_addr_a;
    assign debug_last_addr_b = last_addr_b;
    assign busy              = 1'b0; // SRAM is always ready (no pipeline stalls)

    // =========================================================================
    // Address Region Constants
    // These document the memory map — used by the memory controller to route
    // requests to the correct region. Defined here as single source of truth.
    // =========================================================================
    localparam ADDR_ACTIVATION_START = 14'h0000; // 0x0000 — activation region starts
    localparam ADDR_ACTIVATION_END   = 14'h13FF; // 0x13FF — 5120 words (20KB)
    localparam ADDR_WEIGHT_START     = 14'h1400; // 0x1400 — weight region starts
    localparam ADDR_WEIGHT_END       = 14'h2FFF; // 0x2FFF — 6144 words (24KB)
    localparam ADDR_OUTPUT_START     = 14'h3000; // 0x3000 — output region starts
    localparam ADDR_OUTPUT_END       = 14'h3FFF; // 0x3FFF — 4096 words (16KB)
    localparam ADDR_SCRATCH_START    = 14'h4000; // 0x4000 — scratch space (your addition)
    localparam ADDR_SCRATCH_END      = 14'h4FFF; // 0x4FFF — 4096 words (16KB)

    // =========================================================================
    // PORT A — Sequential Logic
    // =========================================================================
    always @(posedge clk) begin
        if (reset) begin
            port_a_rdata <= {DATA_WIDTH{1'b0}};
            port_a_valid <= 1'b0;
            last_addr_a  <= {ADDR_WIDTH{1'b0}};
        end else begin
            // Default: de-assert valid each cycle (it's a one-cycle pulse, not sticky)
            port_a_valid <= 1'b0;

            if (port_a_en) begin
                last_addr_a <= port_a_addr; // Track for debug

                if (port_a_we) begin
                    // WRITE: update only the bytes indicated by byte_en
                    // This is how you write individual INT8 values into a 32-bit word
                    if (port_a_byte_en[0]) mem[port_a_addr][ 7: 0] <= port_a_wdata[ 7: 0];
                    if (port_a_byte_en[1]) mem[port_a_addr][15: 8] <= port_a_wdata[15: 8];
                    if (port_a_byte_en[2]) mem[port_a_addr][23:16] <= port_a_wdata[23:16];
                    if (port_a_byte_en[3]) mem[port_a_addr][31:24] <= port_a_wdata[31:24];
                    // No valid pulse on write — write is fire-and-forget
                end else begin
                    // READ: latch memory contents into output register
                    // Result is available NEXT cycle (1-cycle latency, synchronous BRAM)
                    port_a_rdata <= mem[port_a_addr];
                    port_a_valid <= 1'b1; // Pulse valid for exactly one cycle
                end
            end
        end
    end

    // =========================================================================
    // PORT B — Sequential Logic (mirrors Port A, independent access)
    // =========================================================================
    always @(posedge clk) begin
        if (reset) begin
            port_b_rdata <= {DATA_WIDTH{1'b0}};
            port_b_valid <= 1'b0;
            last_addr_b  <= {ADDR_WIDTH{1'b0}};
        end else begin
            port_b_valid <= 1'b0; // De-assert each cycle by default

            if (port_b_en) begin
                last_addr_b <= port_b_addr;

                if (port_b_we) begin
                    if (port_b_byte_en[0]) mem[port_b_addr][ 7: 0] <= port_b_wdata[ 7: 0];
                    if (port_b_byte_en[1]) mem[port_b_addr][15: 8] <= port_b_wdata[15: 8];
                    if (port_b_byte_en[2]) mem[port_b_addr][23:16] <= port_b_wdata[23:16];
                    if (port_b_byte_en[3]) mem[port_b_addr][31:24] <= port_b_wdata[31:24];
                end else begin
                    port_b_rdata <= mem[port_b_addr];
                    port_b_valid <= 1'b1;
                end
            end
        end
    end

    // =========================================================================
    // Simulation Initialisation
    // FPGA NOTE: `initial` blocks are NOT synthesizable on some targets.
    // Xilinx BRAM supports an INIT attribute for this — but for simulation,
    // this forces the memory to a known zero state so you don't get X propagation.
    // =========================================================================
    integer k;
    initial begin
        for (k = 0; k < DEPTH; k = k + 1) begin
            mem[k] = {DATA_WIDTH{1'b0}};
        end
    end

endmodule


// =============================================================================
// UNIFIED BUFFER WITH BURST SUPPORT (wrapper module)
// =============================================================================
// Wraps unified_buffer to add sequential address auto-increment for burst ops.
//
// "PASS THROUGH TO UNDERLYING BUFFER" means:
//   This wrapper module is NOT a new memory — it has NO SRAM of its own.
//   It simply instantiates unified_buffer internally and wires its own ports
//   to that instance's ports. When you access the burst controller, you are
//   actually reading/writing the same unified_buffer SRAM inside.
//   The burst controller just auto-increments the address for you each cycle
//   so you don't have to drive it manually from outside.
//
// TWO ACCESS PATHS:
//   1. BURST (Port A wrapper): Trigger burst_start → the controller reads/writes
//      N consecutive words automatically, signalling burst_valid/burst_done.
//      Example: read 8 weight values starting at address 0x1400.
//
//   2. DIRECT (Port B passthrough): Bypasses the burst controller entirely.
//      direct_* signals wire straight through to Port B of unified_buffer.
//      Used for single-cycle random reads/writes (e.g. host loading data).
//
// STATE MACHINE:
//   IDLE       → wait for burst_start
//   BURST_READ → read burst_len words, auto-incrementing address each cycle
//   BURST_WRITE→ write burst_len words, auto-incrementing address each cycle
// =============================================================================

module unified_buffer_burst #(
    parameter DEPTH      = 16384, // Must match underlying unified_buffer
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 14,
    parameter BURST_LEN  = 8      // Default burst length (override with burst_len input)
) (
    input wire clk,
    input wire reset,

    // -------------------------------------------------------------------------
    // Burst Interface — controls Port A of the underlying unified_buffer
    // -------------------------------------------------------------------------
    input wire                   burst_start,      // Pulse high to start a burst
    input wire                   burst_write,      // 1=burst write, 0=burst read
    input wire  [ADDR_WIDTH-1:0] burst_base_addr,  // Starting word address
    input wire  [3:0]            burst_len,        // Number of words to transfer
    input wire  [DATA_WIDTH-1:0] burst_wdata,      // Write data (valid when burst_wdata_valid=1)
    input wire                   burst_wdata_valid, // Handshake: burst_wdata is valid this cycle
    output wire [DATA_WIDTH-1:0] burst_rdata,      // Read data from underlying buffer
    output wire                  burst_valid,      // burst_rdata is valid this cycle
    output wire                  burst_done,       // Pulses when last word transferred

    // -------------------------------------------------------------------------
    // Direct Interface — passes straight through to Port B of unified_buffer
    // No state machine, no sequencing — raw single-cycle access
    // -------------------------------------------------------------------------
    input wire                   direct_en,
    input wire                   direct_we,
    input wire  [ADDR_WIDTH-1:0] direct_addr,
    input wire  [DATA_WIDTH-1:0] direct_wdata,
    output wire [DATA_WIDTH-1:0] direct_rdata,
    output wire                  direct_valid
);

    // State machine encoding
    localparam IDLE        = 2'b00;
    localparam BURST_READ  = 2'b01;
    localparam BURST_WRITE = 2'b10;

    reg [1:0]            state;
    reg [3:0]            burst_counter; // Counts down from burst_len to 0
    reg [ADDR_WIDTH-1:0] current_addr;  // Auto-increments each cycle during burst
    reg                  burst_active;  // High while a burst is in progress

    // =========================================================================
    // Port A wires — connect burst controller outputs to underlying buffer Port A
    // =========================================================================
    wire                   port_a_en;
    wire                   port_a_we;
    wire [ADDR_WIDTH-1:0]  port_a_addr;
    wire [DATA_WIDTH-1:0]  port_a_wdata_w;
    wire [DATA_WIDTH-1:0]  port_a_rdata_w;
    wire                   port_a_valid_w;

    // Port A is driven by the burst controller logic:
    assign port_a_en       = burst_active || burst_start;
    assign port_a_we       = (state == BURST_WRITE) && burst_wdata_valid;
    assign port_a_addr     = burst_active ? current_addr : burst_base_addr;
    assign port_a_wdata_w  = burst_wdata;

    // Feed back results from underlying buffer to burst outputs
    assign burst_rdata = port_a_rdata_w;
    assign burst_valid = port_a_valid_w && (state == BURST_READ);
    assign burst_done  = (burst_counter == 0) && burst_active;

    // =========================================================================
    // Burst State Machine
    // =========================================================================
    always @(posedge clk) begin
        if (reset) begin
            state         <= IDLE;
            burst_active  <= 1'b0;
            burst_counter <= 4'b0;
            current_addr  <= {ADDR_WIDTH{1'b0}};
        end else begin
            case (state)
                IDLE: begin
                    if (burst_start) begin
                        current_addr  <= burst_base_addr;
                        burst_counter <= burst_len;
                        burst_active  <= 1'b1;
                        // Route to correct state based on read/write direction
                        state <= burst_write ? BURST_WRITE : BURST_READ;
                    end
                end

                BURST_READ: begin
                    if (burst_counter == 1) begin
                        // Last word — return to idle
                        state        <= IDLE;
                        burst_active <= 1'b0;
                    end else begin
                        // More words to go — advance address and count down
                        burst_counter <= burst_counter - 1'b1;
                        current_addr  <= current_addr + 1'b1;
                    end
                end

                BURST_WRITE: begin
                    if (burst_counter == 1) begin
                        state        <= IDLE;
                        burst_active <= 1'b0;
                    end else begin
                        burst_counter <= burst_counter - 1'b1;
                        current_addr  <= current_addr + 1'b1;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

    // =========================================================================
    // Instantiate Underlying Buffer
    // This is what "pass through to underlying buffer" means:
    // unified_buffer_burst has NO memory of its own. It creates one
    // unified_buffer instance inside, then connects its own ports to it.
    // Port A is controlled by the burst state machine above.
    // Port B is wired DIRECTLY to the direct_* ports — pure passthrough, no logic.
    // =========================================================================
    unified_buffer #(
        .DEPTH      (DEPTH),
        .DATA_WIDTH (DATA_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH)
    ) u_buffer (
        .clk   (clk),
        .reset (reset),

        // Port A → driven by burst controller
        .port_a_en      (port_a_en),
        .port_a_we      (port_a_we),
        .port_a_addr    (port_a_addr),
        .port_a_wdata   (port_a_wdata_w),
        .port_a_rdata   (port_a_rdata_w),
        .port_a_valid   (port_a_valid_w),
        .port_a_byte_en (4'b1111),       // Burst always writes full 32-bit words

        // Port B → direct passthrough, no state machine involvement
        .port_b_en      (direct_en),
        .port_b_we      (direct_we),
        .port_b_addr    (direct_addr),
        .port_b_wdata   (direct_wdata),
        .port_b_rdata   (direct_rdata),
        .port_b_valid   (direct_valid),
        .port_b_byte_en (4'b1111),

        // Unused outputs — explicitly left unconnected
        .busy              (),
        .debug_last_addr_a (),
        .debug_last_addr_b ()
    );

endmodule