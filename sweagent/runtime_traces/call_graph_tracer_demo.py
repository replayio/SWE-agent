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
    raise RuntimeError("Async error 1")

async def async_function_2():
    print("Async function 2")
    await asyncio.sleep(0.2)
    raise ValueError("Async error 2")

def test_example():
    print("Test example")
    function_a()

async def test_async_example1():
    print("Test async example")
    await asyncio.gather(
        async_function_1(),
        async_function_2(),
        return_exceptions=True
    )

async def test_async_example2():
    print("Test async example")
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

    asyncio.run(test_async_example1())
    asyncio.run(test_async_example2())

    # Demonstrate uncaught exception
    test_example()

# Expected output:
"""
Test example
Function A
Function B
Function C
Partial call graph for failed test test_example:
test_example
  function_a
    function_b
      function_c (exception: ValueError)
Caught ValueError

Test async example
Async function 1
Async function 2
Partial call graph for failed test test_async_example:
test_async_example
  async_function_1 (exception: RuntimeError)
Partial call graph for failed test test_async_example:
test_async_example
  async_function_2 (exception: ValueError)

Test example
Function A
Function B
Function C
Partial call graph for failed test test_example:
test_example
  function_a
    function_b
      function_c (exception: ValueError)
Partial call graph for uncaught exception in test test_example:
test_example
  function_a
    function_b
      function_c (exception: ValueError)
Traceback (most recent call last):
  ...
ValueError: Simulated error
"""