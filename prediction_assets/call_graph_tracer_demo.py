import asyncio  # noqa: I002
import importlib.util
import os
import sys

os.environ["TDD_TRACE_TARGET_CONFIG"] = """
{
    "target_file": "prediction_assets/call_graph_tracer_demo.py",
    "target_function_name": "function_that_throws",
    "decl_lineno": 48
}
""".strip()

# Add this directory to sys.path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Dynamically import ./call_graph_tracer.py and inject it:
module = importlib.import_module("call_graph_tracer")
module.register_runtime_trace()


class A:
    class B:
        @classmethod
        def my_method(cls):
            print("A.B.my_method")

def function_a():
    print("Function A")
    A().B.my_method()
    function_b()
    function_b2()

def function_b(x = 2):
    print("Function B")

def function_b2(x = 2):
    print("Function B2")
    function_c(x)
    function_that_throws(x)

def function_c(x = 3):
    print("Function C")
    function_d()

def function_d(x = 3):
    print("Function D")

def function_that_throws(x = 3):
    print("Function that throws")
    raise ValueError("Simulated error")

async def async_function_1():
    print("Async function 1")
    await asyncio.sleep(0.1)
    function_that_throws()

async def async_function_2():
    print("Async function 2")
    await asyncio.sleep(0.2)
    function_that_throws()

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
