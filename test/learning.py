import cocotb
from cocotb.triggers import Timer

@cocotb.test()
async def my_first_test(dut):

    for _ in range(10):
        dut.clk.value = 0
        await Timer(1, unit = "ns")
        dut.clk.value = 1
        await Timer(1, unit = "ns")
    
    cocotb.log.info("my signal 1 is %s", dut.signal_1.value)
    assert dut.
