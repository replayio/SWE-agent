import sys
import asyncio
from types import FrameType
from typing import Any, Callable, Dict, List, Optional
from contextvars import ContextVar

class CallTreeNode:
    def __init__(self, name: str):
        self.name = name
        self.children: List[CallTreeNode] = []
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
        self.call_stack: ContextVar[List[CallTreeNode]] = ContextVar('call_stack', default=[])
        self.current_test: ContextVar[Optional[CallTreeNode]] = ContextVar('current_test', default=None)
        self.failed_tests: Dict[str, CallTreeNode] = {}

    def trace_calls(self, frame: FrameType, event: str, arg: Any) -> Callable:
        func_name = frame.f_code.co_name
        
        if event == 'call':
            node = CallTreeNode(func_name)
            if func_name.startswith('test_'):
                self.current_test.set(node)
            call_stack = self.call_stack.get()
            if call_stack:
                call_stack[-1].add_child(node)
            call_stack.append(node)
            self.call_stack.set(call_stack)
        
        elif event == 'return':
            call_stack = self.call_stack.get()
            if call_stack:
                returned_node = call_stack.pop()
                self.call_stack.set(call_stack)
                if returned_node == self.current_test.get():
                    self.current_test.set(None)
        
        elif event == 'exception':
            call_stack = self.call_stack.get()
            if call_stack:
                call_stack[-1].set_exception(arg[0].__name__)
                current_test = self.current_test.get()
                if current_test and current_test.name not in self.failed_tests:
                    self.failed_tests[current_test.name] = current_test
                    print(f"Partial call graph for failed test {current_test.name}:")
                    print(str(current_test))
        
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
    _original_excepthook(exc_type, exc_value, exc_traceback)

def register_runtime_trace():
    global _tracer
    _tracer = CallGraphTracer()
    sys.settrace(_tracer.trace_calls)
    sys.excepthook = exception_handler

# Monkey-patch asyncio to use our tracer
def _patch_asyncio():
    original_run = asyncio.events.AbstractEventLoop.run_in_executor

    def patched_run_in_executor(self, *args, **kwargs):
        sys.settrace(_tracer.trace_calls)
        return original_run(self, *args, **kwargs)

    asyncio.events.AbstractEventLoop.run_in_executor = patched_run_in_executor

_patch_asyncio()
