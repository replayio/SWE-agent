import sys
import asyncio
import threading
import os
from types import FrameType
from typing import Any, Callable, List, Optional
import traceback

# Compatibility for ContextVar
try:
    from contextvars import ContextVar
except ImportError:
    class ContextVar:
        def __init__(self, name, default=None):
            self.local = threading.local()
            self.default = default

        def get(self):
            return getattr(self.local, 'value', self.default)

        def set(self, value):
            setattr(self.local, 'value', value)

supports_async_tracing = sys.version_info >= (3, 7)

class CallTreeNode:
    def __init__(self, name: str, filename: str, lineno: int):
        self.name = name
        self.filename = os.path.relpath(filename)
        self.lineno = lineno
        self.children: List['CallTreeNode'] = []
        self.exception: Optional[str] = None

    def add_child(self, child: 'CallTreeNode'):
        self.children.append(child)

    def set_exception(self, exc_name: str):
        self.exception = exc_name

    def __str__(self, level=0, visited=None):
        if visited is None:
            visited = set()
        if id(self) in visited:
            return "  " * level + "[Recursion]\n"
        visited.add(id(self))
        indent = '  ' * level
        result = f"{indent}{self.name} ({self.filename}:{self.lineno})"
        if self.exception:
            result += f" (exception: {self.exception})"
        result += '\n'
        for child in self.children:
            result += child.__str__(level + 1, visited)
        return result

class CallGraphTracer:
    def __init__(self):
        self.call_stack = ContextVar('call_stack', default=[])

    def trace_calls(self, frame: FrameType, event: str, arg: Any) -> Optional[Callable]:
        if event == 'call':
            call_stack = self.call_stack.get()
            call_stack = call_stack.copy()
            func_name = frame.f_code.co_name
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
            node = CallTreeNode(func_name, filename, lineno)
            if call_stack:
                call_stack[-1].add_child(node)
            call_stack.append(node)
            self.call_stack.set(call_stack)
            frame.f_trace = self.trace_calls
        elif event == 'return':
            call_stack = self.call_stack.get()
            call_stack = call_stack.copy()
            if call_stack:
                call_stack.pop()
                self.call_stack.set(call_stack)
        elif event == 'exception':
            call_stack = self.call_stack.get()
            call_stack = call_stack.copy()
            if call_stack:
                exc_type, _, _ = arg
                call_stack[-1].set_exception(exc_type.__name__)
                if frame.f_back is None:
                    root_node = call_stack[0]
                    print("Call graph for uncaught exception:")
                    print(str(root_node))
        return self.trace_calls

    def task_done_callback(self, task: asyncio.Task):
        exc = task.exception()
        if exc:
            tb = exc.__traceback__
            nodes = []
            while tb:
                frame = tb.tb_frame
                func_name = frame.f_code.co_name
                filename = frame.f_code.co_filename
                lineno = frame.f_lineno
                node = CallTreeNode(func_name, filename, lineno)
                node.set_exception(type(exc).__name__)
                nodes.append(node)
                tb = tb.tb_next
            nodes.reverse()
            # Include task creation stack
            creation_stack = getattr(task, '_creation_stack', [])
            full_stack = creation_stack + nodes
            for i in range(len(full_stack) - 1):
                full_stack[i].add_child(full_stack[i + 1])
            root_node = full_stack[0]
            print("Call graph for task exception:")
            print(str(root_node))

_tracer = None

def exception_handler(exc_type, exc_value, exc_traceback):
    if _tracer:
        tb = exc_traceback
        nodes = []
        while tb:
            frame = tb.tb_frame
            func_name = frame.f_code.co_name
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno
            node = CallTreeNode(func_name, filename, lineno)
            node.set_exception(exc_type.__name__)
            nodes.append(node)
            tb = tb.tb_next
        nodes.reverse()
        for i in range(len(nodes) - 1):
            nodes[i].add_child(nodes[i + 1])
        print("Call graph for uncaught exception:")
        print(str(nodes[0]))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

def register_runtime_trace():
    global _tracer
    _tracer = CallGraphTracer()
    sys.settrace(_tracer.trace_calls)
    threading.settrace(_tracer.trace_calls)
    sys.excepthook = exception_handler

    if supports_async_tracing:
        asyncio.set_event_loop_policy(TracingEventLoopPolicy())

if supports_async_tracing:
    class TracingEventLoopPolicy(asyncio.DefaultEventLoopPolicy):
        def new_event_loop(self):
            loop = super().new_event_loop()
            loop.set_task_factory(self.tracing_task_factory)
            loop.set_exception_handler(_tracer.task_done_callback)
            return loop

        def tracing_task_factory(self, loop, coro):
            task = asyncio.Task(coro, loop=loop)
            # Capture task creation stack
            creation_stack = []
            for frame_info in traceback.extract_stack()[:-1]:
                node = CallTreeNode(frame_info.name, frame_info.filename, frame_info.lineno)
                creation_stack.append(node)
            task._creation_stack = creation_stack
            task.add_done_callback(_tracer.task_done_callback)
            return task
