`default_nettype none
`timescale 1ns/1ns

// =============================================================================
// UNIFIED BUFFER — True Dual-Port SRAM (64KB)
// =============================================================================
//
// WHY "UNIFIED"?
//   A single shared block of memory replaces separate weight memory and
//   activation memory. This lets the software decide how to partition the
//   space dynamically, and lets both the host and the systolic array
//   (via the memory controller) access the same data.
//
// WHY DUAL-PORT?
//   The host (CPU/controller) needs to load weights into one region while the
//   systolic array might be writing back results to another region.
//   A single-port SRAM would force strict sequencing — one would have to wait.
//   Two independent ports allow TRUE PARALLELISM, with the only restriction
//   being that writing to the EXACT SAME address on both ports simultaneously
//   is undefined (the memory controller prevents this).
//
//   Port A → used by the Host (weight/activation loads from upstream)
//   Port B → used by the Systolic Array accumulator (result writeback)
//
// MEMORY MAP (word-addressed, 32 bits per word):
//   [0x0000 – 0x13FF]  Activations   5120 words = 20KB
//   [0x1400 – 0x2FFF]  Weights       6144 words = 24KB
//   [0x3000 – 0x3FFF]  Outputs       4096 words = 16KB
//   [0x4000 – 0x4FFF]  Scratch       4096 words = 16KB
//
// READ LATENCY: 1 clock cycle (synchronous / registered output — BRAM behaviour)
//   → Write: data appears in `mem` on the SAME posedge as write strobe
//   → Read : port_x_rdata is valid ONE CYCLE AFTER port_x_en is asserted
//   → port_x_valid pulses for EXACTLY ONE cycle to signal the read is ready
//
// BYTE ENABLES:
//   port_x_byte_en[3:0] lets you write individual bytes of a 32-bit word.
//   bit 0 → bytes [7:0], bit 1 → bytes [15:8], bit 2 → bytes [23:16], bit 3 → bytes [31:24]
//   This is how you pack and unpack four INT8 values into/from one 32-bit word.
//
// FPGA NOTE:
//   (* ram_style = "block" *) tells Xilinx that this MUST be mapped to BRAM primitives,
//   NOT to LUT-based distributed RAM. Always use this for large memories (> ~256 words).
//   Without it, a 64KB LUT RAM would consume almost the entire FPGA fabric.
// =============================================================================

module unified_buffer #(
    parameter DEPTH      = 16384, // 16K words × 32 bits = 64KB total
    parameter DATA_WIDTH = 32,    // Each word = 32 bits = four packed INT8 values
    parameter ADDR_WIDTH = 14     // 2^14 = 16384 addressable words
) (
    input wire clk,
    input wire reset,

    // =========================================================================
    // PORT A — Host Interface (read/write)
    // =========================================================================
    input wire                   port_a_en,       // Initiate a Port A transaction this cycle
    input wire                   port_a_we,        // 1 = write, 0 = read
    input wire  [ADDR_WIDTH-1:0] port_a_addr,      // Word address
    input wire  [DATA_WIDTH-1:0] port_a_wdata,     // Write data (ignored if port_a_we=0)
    output reg  [DATA_WIDTH-1:0] port_a_rdata,     // Read result (valid one cycle after read request)
    output reg                   port_a_valid,     // Pulses HIGH for exactly one cycle after read
    input wire  [3:0]            port_a_byte_en,   // Byte write mask — which bytes to update

    // =========================================================================
    // PORT B — Systolic Array / Accumulator Interface (read/write)
    // =========================================================================
    input wire                   port_b_en,
    input wire                   port_b_we,
    input wire  [ADDR_WIDTH-1:0] port_b_addr,
    input wire  [DATA_WIDTH-1:0] port_b_wdata,
    output reg  [DATA_WIDTH-1:0] port_b_rdata,
    output reg                   port_b_valid,
    input wire  [3:0]            port_b_byte_en,

    // =========================================================================
    // Status / Debug
    // =========================================================================
    output wire                  busy,               // Always 0 — SRAM never stalls the requester
    output wire [ADDR_WIDTH-1:0] debug_last_addr_a,  // Last address accessed on Port A (debug)
    output wire [ADDR_WIDTH-1:0] debug_last_addr_b   // Last address accessed on Port B (debug)
);

    // =========================================================================
    // Memory Array Declaration
    // =========================================================================
    // (* ram_style = "block" *) is a Xilinx synthesis attribute.
    // It FORCES BRAM inference regardless of what heuristics the tool would normally use.
    // At 64KB this is non-negotiable — LUT RAM would be absurdly expensive.
    (* ram_style = "block" *)
    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

    // Per-port last-accessed address (for debug/monitoring only)
    reg [ADDR_WIDTH-1:0] last_addr_a;
    reg [ADDR_WIDTH-1:0] last_addr_b;

    assign debug_last_addr_a = last_addr_a;
    assign debug_last_addr_b = last_addr_b;
    assign busy              = 1'b0; // SRAM is always ready — no pipeline stalls

    // =========================================================================
    // Address Region Constants (single source of truth for the memory map)
    // =========================================================================
    // The memory controller uses these to know which region a request targets.
    // Keeping them here centralises the memory map definition.
    localparam ADDR_ACTIVATION_START = 14'h0000;
    localparam ADDR_ACTIVATION_END   = 14'h13FF; // 5120 words (20KB)
    localparam ADDR_WEIGHT_START     = 14'h1400;
    localparam ADDR_WEIGHT_END       = 14'h2FFF; // 6144 words (24KB)
    localparam ADDR_OUTPUT_START     = 14'h3000;
    localparam ADDR_OUTPUT_END       = 14'h3FFF; // 4096 words (16KB)
    localparam ADDR_SCRATCH_START    = 14'h4000;
    localparam ADDR_SCRATCH_END      = 14'h4FFF; // 4096 words (16KB)

    // =========================================================================
    // PORT A — Sequential Logic
    // =========================================================================
    always @(posedge clk) begin
        if (reset) begin
            port_a_rdata <= {DATA_WIDTH{1'b0}};
            port_a_valid <= 1'b0;
            last_addr_a  <= {ADDR_WIDTH{1'b0}};
        end else begin
            // De-assert valid by default — it is a one-cycle pulse, not a sticky flag.
            // If we don't do this, valid would stay high until the next read.
            port_a_valid <= 1'b0;

            if (port_a_en) begin
                last_addr_a <= port_a_addr;

                if (port_a_we) begin
                    // ─ WRITE PATH ─────────────────────────────────────────
                    // byte_en lets you surgically write individual bytes of a 32-bit word.
                    // Example: to write only the third byte (bits [23:16]), set byte_en = 4'b0100.
                    // This is how the prefetch DMA packs four INT8 weights into one word.
                    if (port_a_byte_en[0]) mem[port_a_addr][ 7: 0] <= port_a_wdata[ 7: 0];
                    if (port_a_byte_en[1]) mem[port_a_addr][15: 8] <= port_a_wdata[15: 8];
                    if (port_a_byte_en[2]) mem[port_a_addr][23:16] <= port_a_wdata[23:16];
                    if (port_a_byte_en[3]) mem[port_a_addr][31:24] <= port_a_wdata[31:24];
                    // No valid pulse on write — write is "fire and forget"
                end else begin
                    // ─ READ PATH ──────────────────────────────────────────
                    // BRAM has 1-cycle read latency (registered output).
                    // Data is latched into port_a_rdata and valid pulses for one cycle NEXT cycle.
                    port_a_rdata <= mem[port_a_addr];
                    port_a_valid <= 1'b1;
                end
            end
        end
    end

    // =========================================================================
    // PORT B — Sequential Logic (mirrors Port A, fully independent)
    // =========================================================================
    always @(posedge clk) begin
        if (reset) begin
            port_b_rdata <= {DATA_WIDTH{1'b0}};
            port_b_valid <= 1'b0;
            last_addr_b  <= {ADDR_WIDTH{1'b0}};
        end else begin
            port_b_valid <= 1'b0;

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
    // =========================================================================
    // `initial` blocks are NOT synthesized on most targets.
    // On Xilinx FPGAs, BRAM has an INIT attribute for pre-loading — but for
    // simulation purposes, this zeroes the memory to eliminate X-propagation.
    // Without this, an uninitialised read would return 'X' and propagate X
    // through your entire datapath, making simulation results unreadable.
    integer k;
    initial begin
        for (k = 0; k < DEPTH; k = k + 1) begin
            mem[k] = {DATA_WIDTH{1'b0}}; // = not <= in initial blocks (blocking)
        end
    end

endmodule


// =============================================================================
// UNIFIED BUFFER WITH BURST SUPPORT (Wrapper Module)
// =============================================================================
//
// WHAT THIS ADDS:
//   The base `unified_buffer` requires the caller to manually drive the address
//   every single cycle. For sequential DMA-style reads/writes (e.g. fetching
//   all 8 weights for a row), this is verbose and error-prone.
//
//   This wrapper adds a BURST CONTROLLER on Port A that auto-increments the
//   address each cycle, so a caller only needs to provide:
//     - A base address
//     - A length (number of words)
//     - A start pulse
//   ...and the controller handles the rest.
//
// THIS MODULE HAS NO OWN MEMORY:
//   It provides NO extra SRAM. It simply instantiates `unified_buffer` internally
//   and wires its control logic into Port A. Port B is a direct passthrough.
//
// TWO ACCESS PATHS:
//   1. BURST (via Port A controller):
//      Set burst_start → burst_len words auto-transferred → burst_done pulses.
//   2. DIRECT (via Port B passthrough):
//      Single-cycle random access — raw reads/writes, no state machine.
//      Used by the host for loading individual data values.
//
// BURST STATE MACHINE:
//   IDLE        → wait for burst_start
//   BURST_READ  → read burst_len words, address auto-increments each cycle
//   BURST_WRITE → write burst_len words, address auto-increments each cycle
// =============================================================================

module unified_buffer_burst #(
    parameter DEPTH      = 16384, // Must match underlying unified_buffer
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 14,
    parameter BURST_LEN  = 8      // Default burst length (overrideable via burst_len input)
) (
    input wire clk,
    input wire reset,

    // ─────────────────────────────────────────────────────────────────────────
    // Burst Interface — controls Port A of unified_buffer
    // ─────────────────────────────────────────────────────────────────────────
    input wire                   burst_start,       // Pulse once to start a burst
    input wire                   burst_write,       // 1 = burst write; 0 = burst read
    input wire  [ADDR_WIDTH-1:0] burst_base_addr,   // Start address for burst
    input wire  [3:0]            burst_len,         // Number of words to transfer
    input wire  [DATA_WIDTH-1:0] burst_wdata,       // Write data word (for write bursts)
    input wire                   burst_wdata_valid,  // Handshake: burst_wdata is valid this cycle
    output wire [DATA_WIDTH-1:0] burst_rdata,       // Read data from underlying buffer
    output wire                  burst_valid,       // burst_rdata is valid this cycle
    output wire                  burst_done,        // Pulses HIGH when last word transferred

    // ─────────────────────────────────────────────────────────────────────────
    // Direct Interface — passes straight through to Port B of unified_buffer
    // No state machine, no sequencing. Single-cycle raw access.
    // ─────────────────────────────────────────────────────────────────────────
    input wire                   direct_en,
    input wire                   direct_we,
    input wire  [ADDR_WIDTH-1:0] direct_addr,
    input wire  [DATA_WIDTH-1:0] direct_wdata,
    output wire [DATA_WIDTH-1:0] direct_rdata,
    output wire                  direct_valid
);

    localparam IDLE        = 2'b00;
    localparam BURST_READ  = 2'b01;
    localparam BURST_WRITE = 2'b10;

    reg [1:0]            state;
    reg [3:0]            burst_counter; // Counts down from burst_len to 0
    reg [ADDR_WIDTH-1:0] current_addr;  // Auto-increments each cycle during a burst
    reg                  burst_active;  // High while any burst is in progress

    // ─────────────────────────────────────────────────────────────────────────
    // Port A wires — driven by the burst controller
    // ─────────────────────────────────────────────────────────────────────────
    wire                   port_a_en;
    wire                   port_a_we;
    wire [ADDR_WIDTH-1:0]  port_a_addr;
    wire [DATA_WIDTH-1:0]  port_a_wdata_w;
    wire [DATA_WIDTH-1:0]  port_a_rdata_w;
    wire                   port_a_valid_w;

    // When idle, we can trigger on burst_start directly to save one cycle of latency.
    assign port_a_en      = burst_active || burst_start;
    // Write enable only during BURST_WRITE and when caller says its data is valid
    assign port_a_we      = (state == BURST_WRITE) && burst_wdata_valid;
    // During burst: use auto-incremented address; before burst starts: use base addr
    assign port_a_addr    = burst_active ? current_addr : burst_base_addr;
    assign port_a_wdata_w = burst_wdata;

    // Feed results back to the burst interface
    assign burst_rdata = port_a_rdata_w;
    assign burst_valid = port_a_valid_w && (state == BURST_READ);
    assign burst_done  = (burst_counter == 0) && burst_active;

    // ─────────────────────────────────────────────────────────────────────────
    // Burst State Machine
    // ─────────────────────────────────────────────────────────────────────────
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
                        state <= burst_write ? BURST_WRITE : BURST_READ;
                    end
                end

                BURST_READ: begin
                    if (burst_counter == 1) begin
                        // Last word — return to IDLE
                        state        <= IDLE;
                        burst_active <= 1'b0;
                    end else begin
                        // Advance address and count down remaining words
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

    // ─────────────────────────────────────────────────────────────────────────
    // Underlying Memory — this is the ONLY place SRAM actually lives
    //
    // Port A → driven by the burst state machine above
    // Port B → wired DIRECTLY to the direct_* ports (pure passthrough)
    //          The direct interface bypasses all state logic entirely.
    // ─────────────────────────────────────────────────────────────────────────
    unified_buffer #(
        .DEPTH      (DEPTH),
        .DATA_WIDTH (DATA_WIDTH),
        .ADDR_WIDTH (ADDR_WIDTH)
    ) u_buffer (
        .clk   (clk),
        .reset (reset),

        // Port A → burst controller
        .port_a_en      (port_a_en),
        .port_a_we      (port_a_we),
        .port_a_addr    (port_a_addr),
        .port_a_wdata   (port_a_wdata_w),
        .port_a_rdata   (port_a_rdata_w),
        .port_a_valid   (port_a_valid_w),
        .port_a_byte_en (4'b1111), // Burst always works on full 32-bit words

        // Port B → direct passthrough (no burst controller involvement)
        .port_b_en      (direct_en),
        .port_b_we      (direct_we),
        .port_b_addr    (direct_addr),
        .port_b_wdata   (direct_wdata),
        .port_b_rdata   (direct_rdata),
        .port_b_valid   (direct_valid),
        .port_b_byte_en (4'b1111),

        // Unused debug/status outputs — explicitly left unconnected
        .busy              (),
        .debug_last_addr_a (),
        .debug_last_addr_b ()
    );

endmodule