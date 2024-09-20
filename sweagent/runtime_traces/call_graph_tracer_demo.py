import sys
import asyncio
from sweagent.runtime_traces.call_graph_tracer import register_runtime_trace

register_runtime_trace()

def function_a():
    print("Function A")
    function_b()

def function_b():
    print("Function B")
    function_c()

def function_c():
    print("Function C")
    raise ValueError("Simulated error")

async def async_function_1():
    print("Async function 1")
    await asyncio.sleep(0.1)
    function_c()

async def async_function_2():
    print("Async function 2")
    await asyncio.sleep(0.2)
    function_c()

def test_example():
    print("Test example")
    function_a()

async def test_async_example1():
    print("Test async example 1")
    await asyncio.gather(
        async_function_1(),
        async_function_2(),
        return_exceptions=True
    )

async def test_async_example2():
    print("Test async example 2")
    await asyncio.gather(
        async_function_1(),
        async_function_2(),
        return_exceptions=True
    )

if __name__ == "__main__":
    try:
        test_example()
    except ValueError:
        print("Caught ValueError")

    if sys.version_info >= (3, 7):
        asyncio.run(test_async_example1())
        asyncio.run(test_async_example2())
    else:
        # For Python 3.6, use alternative to asyncio.run()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(test_async_example1())
        loop.run_until_complete(test_async_example2())
        loop.close()

    # Demonstrate uncaught exception
    test_example()
