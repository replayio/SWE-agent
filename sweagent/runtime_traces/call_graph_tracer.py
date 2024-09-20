import sys
import asyncio
from types import FrameType
from typing import Any, Callable, Dict, List, Optional

# Check if contextvars is available (Python 3.7+)
try:
    from contextvars import ContextVar
except ImportError:
    # For Python 3.6 and earlier, use a simple thread-local storage
    import threading
    class ContextVar:
        def __init__(self, name, default=None):
            self.name = name
            self.default = default
            self.local = threading.local()

        def get(self):
            return getattr(self.local, 'value', self.default)

        def set(self, value):
            self.local.value = value

supports_async_tracing = sys.version_info >= (3, 7)

class CallTreeNode:
    def __init__(self, name: str):
        self.name = name
        self.children: List['CallTreeNode'] = []
        self.exception: Optional[str] = None

    def add_child(self, child: 'CallTreeNode'):
        self.children.append(child)

    def set_exception(self, exc_name: str):
        self.exception = exc_name

    def __str__(self, level: int = 0) -> str:
        result = "  " * level + self.name
        if self.exception:
            result += f" (exception: {self.exception})"
        result += "\n"
        for child in self.children:
            result += child.__str__(level + 1)
        return result

class CallGraphTracer:
    def __init__(self):
        self.call_stack = ContextVar('call_stack', default=[])
        self.current_test = ContextVar('current_test', default=None)
        self.failed_tests: Dict[str, CallTreeNode] = {}

    def trace_calls(self, frame: FrameType, event: str, arg: Any) -> Optional[Callable]:
        func_name = frame.f_code.co_name

        if event == 'call':
            node = CallTreeNode(func_name)
            if func_name.startswith('test_'):
                self.current_test.set(node)
            call_stack = self.call_stack.get().copy()
            if call_stack:
                call_stack[-1].add_child(node)
            call_stack.append(node)
            self.call_stack.set(call_stack)

        elif event == 'return':
            call_stack = self.call_stack.get().copy()
            if call_stack:
                returned_node = call_stack.pop()
                self.call_stack.set(call_stack)
                if returned_node == self.current_test.get():
                    self.current_test.set(None)

        elif event == 'exception':
            call_stack = self.call_stack.get().copy()
            if call_stack:
                call_stack[-1].set_exception(arg[0].__name__)
                current_test = self.current_test.get()
                if current_test and current_test.name not in self.failed_tests:
                    self.failed_tests[current_test.name] = current_test
                    print(f"Partial call graph for failed test {current_test.name}:")
                    print(str(current_test))
                    if not supports_async_tracing:
                        print(f"Warning: Async call frames may be missing due to Python version used ({sys.version.split()[0]}).")

        return self.trace_calls

_original_excepthook = sys.excepthook
_tracer = None

def exception_handler(exc_type, exc_value, exc_traceback):
    global _tracer
    if _tracer:
        current_test = _tracer.current_test.get()
        if current_test:
            print(f"Partial call graph for uncaught exception in test {current_test.name}:")
            print(str(current_test))
            if not supports_async_tracing:
                print(f"Warning: Async call frames may be missing due to Python version used ({sys.version.split()[0]}).")
    _original_excepthook(exc_type, exc_value, exc_traceback)

def register_runtime_trace():
    global _tracer
    _tracer = CallGraphTracer()
    sys.settrace(_tracer.trace_calls)
    sys.excepthook = exception_handler

    if supports_async_tracing:
        # For Python 3.7 and above, set custom event loop policy
        asyncio.set_event_loop_policy(TracingEventLoopPolicy())

if supports_async_tracing:
    class TracingEventLoopPolicy(asyncio.DefaultEventLoopPolicy):
        def new_event_loop(self):
            loop = super().new_event_loop()
            loop.set_task_factory(self.tracing_task_factory)
            return loop

        @staticmethod
        def tracing_task_factory(loop, coro):
            task = asyncio.tasks.Task(coro, loop=loop)
            if hasattr(coro, 'cr_frame') and coro.cr_frame is not None:
                coro.cr_frame.f_trace = _tracer.trace_calls
            return task
