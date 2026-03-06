//weight fifo, double buffered weight staging
//needs weights loaded before computing
//lioading fform sram and computing cant overlap in single buffer
//two buffers, A drains into systolic array
//BB is filled from SRAm. when A is empty, instantly swap

//independent processes running in parallel, 
    //prefetch fsm: reads from unified buffer-> fills the inactive buffer
    //drain, sends rows from aactive buffer to systolic array

module weight_fifo #(
    parameter N = 8; //systolic array dimension
    parameter DATA_WIDTH = 8 //INT8 weights
    parameter BUFFER_DEPTH = 64 //NxN weights
)(
    input wire clk,
    input wire reset,
    // -----------------------------------------------------------------------
    // MEMORY INTERFACE — connects to unified_buffer via memory_controller
    // -----------------------------------------------------------------------
    output reg          mem_req,    // Pulse high to request one 32-bit word
    output reg  [13:0]  mem_addr,   // Word address in unified buffer
    input wire  [31:0]  mem_rdata,  // 32-bit response (holds 4 INT8 weights)
    input wire          mem_valid,  // Memory controller: "data is ready this cycle"
    output wire         mem_busy,   // We are actively prefetching (tells controller)
    // -----------------------------------------------------------------------
    // PREFETCH CONTROL — host tells us when and where to fetch next tile
    // -----------------------------------------------------------------------
    input wire          prefetch_start,      // Pulse: begin fetching a new tile
    input wire  [13:0]  prefetch_base_addr,  // Start address of tile in unified buffer
    input wire  [7:0]   prefetch_rows,       // How many rows to fetch (up to N=8)
    output wire         prefetch_done,       // High for one cycle when fetch complete
    output wire         prefetch_busy,       // High while fetch is in progress
    // -----------------------------------------------------------------------
    // SYSTOLIC ARRAY INTERFACE — feeds weights row by row into the PE grid
    // -----------------------------------------------------------------------
    input wire          drain_enable,    // Array is ready to receive weights
    input wire          drain_row_done,  // Array: "finished loading current row, next please"
    output wire signed [DATA_WIDTH-1:0] weight_out [N-1:0], // N weights for current row
    output wire [N-1:0] weight_row_select, // One-hot: which row of PEs to load
    output wire         weight_valid,      // weight_out is valid this cycle
    output wire         buffer_empty,      // Active buffer has been fully drained
    output wire         buffer_ready,      // Active buffer has data to send
    // -----------------------------------------------------------------------
    // STATUS / DEBUG
    // -----------------------------------------------------------------------
    output wire [1:0]   active_buffer,  // 0=A active, 1=B active
    output wire [5:0]   fill_level_a,   // How many words are loaded in Buffer A
    output wire [5:0]   fill_level_b
);

    //what needs to exist between clock cycles?
    //double buffer storage'
    //each biuffer is an NxN 2D array with INT8 weights
    reg signed [DATA_WIDTH-1:0] buffer_a [0:N-1][0:N-1];
    reg signed [DATA_WIDTH-1:0] buffer_b [0:N-1][0:N-1];

    //buffer control state
    reg active_buf //0= sys array reads from A, else B
    reg [2:0] drain_row //which row currently outputting from the buffer

    reg buffer_a_valid //A contains fully loaded tile
    reg buffer_b_valid //B contains fully loaded tile
    reg [5:0] words_in_a //words loaded into A
    reg [5:0] words_in_b //words loaded into B

    //pre fetch fsm state
    //5 state because one mem read takes multiple cycles
    //request-wait for grant-wait for data-store - loop
    localparam PF_IDLE = 3'b000;
    localparam PF_REQUEST = 3'b001;
    localparam PF_WAIT = 3'b010;
    localparam PF_STORE = 3'b011;
    localparam PF_DONE = 3'b100;

    reg [2:0] pf_state;
    reg [13:0] pf_addr;
    reg [5:0] pf_word_count; //number of words fetched so far
    reg [5:0] pf_total_words; //total words toi fetch for this tile
    reg       pf_target_buffer; //which buffer we're filling always ~active_buf

    //position tracking within the buffer being filled:
    //each 32 bit fills 4 consecutive columns on one row
    //pf_col advances by 4 per word; when it wraps, pf_row increments
    reg [2:0] pf_row;
    reg [2:0] pf_col; //0,4,8,12,16,20,24,28

    //combinational outputs
    //what do i need to comput wihtout a state machine
    assign active_buffer = {1'b0, active_buf};
    assign fill_level_a = words_in_a;
    assign fill_level_b = words_in_b;

    //is the active buffer empty/ready?
    assign buffer_ready = (active_buf == 0) ? buffer_a_valid : buffer_b_valid;
    assign buffer_empty = ~buffer_ready;

    //prefetch status
    assign prefetch_done = (pf_state == PF_DONE);
    assign prefetch_busy = (pf_state != PF_IDLE) && (pf_state != PF_DONE);
    assign mem_busy = prefetch_busy;

    //weight output multiplexing
    //output N weights from current row of active buffer
    //the generate loop creates N instances of the weight_output module -> more of a automation tool
    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : weight_output
            assign weight_out[i] = (active_buf == 0) ? buffer_a[drain_row][i] : buffer_b[drain_row][i];
        end
    endgenerate

    //row select one-hot encoding
    //tell systolic array which row of PE to load weight into
    //one hot-> if drain_row 2, weight row select = 8'b00000100
    assign weight_row_select = (drain_enable && buffer_ready) ? (8'b1 << drain_row) : 8'b0;
    assign weight_valid = drain_enable && buffer_ready;

    //what changes drain_row, and when do we swap buffers?
    //drain lgic, stream weights from active buffer into the systolic array
    //advance drain row each time array siognals
    //when all N rows are done, swap to the other buffer
    //actual outputs are combinational
    always @(posedge clk) begin
        if (reset) begin
            drain_row <= 3'b0;
            active_buf <= 1'b0;
            buffer_a_valid <= 1'b0;
            buffer_b_valid <= 1'b0;
            words_in_a <= 6'd0;
            words_in_b <= 6'd0;
        end else begin
            if (drain_enable && buffer_ready) begin
                if (drain_row_done) begin
                    if (drain_row >= N - 1'b1) begin
                        drain_row <= 3'b0;
                        if (active_buf == 0) begin
                            buffer_a_valid <= 0;
                            words_in_a <= 0;
                        end else begin
                            buffer_b_valid <= 0;
                            words_in_b <= 0;
                        end
                        if ((active_buf == 0 && buffer_b_valid) || (active_buf == 1 && buffer_a_valid)) begin
                            active_buf <= ~active_buf;
                        end
                    end else begin
                        drain_row <= drain_row + 1'b1;
                    end
                end
            end
        end
    end


    //what are the exact steps to load one 32 bit word from SRAM into the bffer
    //one word-> 4 weights where each weight is 8 bits
    //repeat it 16 times track position with pf_row and pf_col
    //pf is pre fetch

    always @(posedge clk) begin
        if (reset) begin
           pf_state <= PF_IDLE;
           pf_addr <= 14'd0;
           pf_word_count <= 6'd0;
           pf_total_words <= 6'd0;
           pf_target_buffer <= 1'b0;
           pf_row <= 3'b0;
           pf_col <= 3'b0; 
           mem_req <= 1'b0;
           mem_addr <= 14'd0;
        end else begin
            case (pf_state)
                PF_IDLE: begin
                   mem_req <= 1'b0;
                   if (prefetch_start) begin
                        pf_addr <= prefetch_base_addr;
                        //each row needs N/4 words
                        pf_total_words <= prefetch_rows * (N/4);
                        pf_word_count <= 6'd0;
                        pf_row <= 3'b0;
                        pf_col <= 3'b0;
                        //set target buffer to the inactive buffer
                        pf_target_buffer <= ~active_buf;
                        pf_state <= PF_REQUEST;
                   end
                end
                PF_REQUEST: begin
                    mem_req <= 1'b1;
                    mem_addr <= pf_addr;
                    pf_state <= PF_WAIT;
                end
                PF_WAIT: begin
                    mem_req <= 1'b0;
                    if (mem_grant) begin
                        pf_state <= PF_STORE;
                    end
                end
                PF_STORE: begin
                    //mem_rdata is 32 bits with 4 INT8 weights packed int 32 bits -> we need to unpack
                    if (pf_target_buffer == 0) begin
                        buffer_a[pf_row][pf_col] <= mem_rdata[7:0];
                        buffer_a[pf_row][pf_col+1] <= mem_rdata[15:8];
                        buffer_a[pf_row][pf_col+2] <= mem_rdata[23:16];
                        buffer_a[pf_row][pf_col+3] <= mem_rdata[31:24];
                    end else begin
                        buffer_b[pf_row][pf_col] <= mem_rdata[7:0];
                        buffer_b[pf_row][pf_col+1] <= mem_rdata[15:8];
                        buffer_b[pf_row][pf_col+2] <= mem_rdata[23:16];
                        buffer_b[pf_row][pf_col+3] <= mem_rdata[31:24];
                    end
                    //advance the address counter
                    pf_addr <= pf_addr + 1'b1;
                    pf_word_count <= pf_word_count + 1'b1;
                    
                    //advance position-> col steps 0->4->0
                    if (pf_col + 4>= N) begin
                        pf_col <= 3'b0;
                        pf_row <= pf_row + 1'b1;
                    end else begin
                        pf_col <= pf_col + 4;
                    end

                    //check if done with all words
                    if (pf_word_count + 1 >= pf_total_words) begin
                        pf_state <= PF_DONE;
                    end else begin
                        pf_state <= PF_REQUEST; //fetch next round of word
                    end
                end
                PF_DONE: begin
                    //mark the buffer we just filled as valid
                    //drain logic will see this and allow swap on next tile end
                    if (pf_target_buffer == 0) begin
                        buffer_a_valid <= 1'b1;
                        words_in_a <= pf_total_words;
                    end else begin
                        buffer_b_valid <= 1'b1;
                        words_in_b <= pf_total_words;
                    end
                    pf_state <= PF_IDLE; //ready for next prefetch
                end
                default: begin
                    pf_state <= PF_IDLE;
                end
            endcase            
        end
    end

    //simulation init - zero all weights to avoid X propogation in sim
    integer r,c;
    initial begin
        for (r = 0; r < N; r = r + 1) begin
            for (c = 0; c < N; c = c + 1) begin
                buffer_a[r][c] = 8'b0;
                buffer_b[r][c] = 8'b0;
            end
        end
    end








    



endmodule