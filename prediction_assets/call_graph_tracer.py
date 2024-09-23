import sys
import threading
import traceback
import os
from types import FrameType
from typing import Any, Callable, List, Optional, Tuple

# Determine Python version
python_version = sys.version_info

# Compatibility for ContextVar
if python_version >= (3, 7):
    from contextvars import ContextVar
else:
    # For Python 3.6, use threading.local() (doesn't handle async context)
    class ContextVar:
        def __init__(self, name, default=None):
            self.local = threading.local()
            self.default = default

        def get(self):
            return getattr(self.local, 'value', self.default)

        def set(self, value):
            setattr(self.local, 'value', value)

class CallTreeNode:
    def __init__(self, name: str, filename: str, lineno: int, decl_lineno: int):
        self.name = name
        try:
            rel_path = os.path.relpath(filename)
            if rel_path.startswith('..') or os.path.isabs(rel_path):
                self.filename = f"EXTERNAL/{os.path.basename(filename)}"
            else:
                self.filename = rel_path
        except Exception:
            self.filename = filename  # Fallback if os functions are unavailable
        self.lineno = lineno
        self.decl_lineno = decl_lineno
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
        result = f"{indent}{self.name} ({self.filename}:{self.lineno}"
        if self.lineno != self.decl_lineno:
            result += f", declared at {self.decl_lineno}"
        result += ")"
        if self.exception:
            result += f" (exception: {self.exception})"
        result += '\n'
        for child in self.children:
            result += child.__str__(level + 1, visited)
        return result

class CallGraphTracer:
    def __init__(self):
        self.call_stack = ContextVar('call_stack', default=[])
        try:
            self.cwd = os.getcwd()
        except Exception:
            self.cwd = None

    def should_trace(self, frame: FrameType) -> bool:
        if frame is None or frame.f_code is None or frame.f_code.co_filename is None:
            return False
        try:
            filename = frame.f_code.co_filename
            abs_filename = os.path.abspath(filename)
            if self.cwd:
                return abs_filename.startswith(self.cwd)
            else:
                return True
        except Exception as err:
            print(f"DDBG error should_trace {err}")
            return True

    def trace_calls(self, frame: FrameType, event: str, arg: Any) -> Optional[Callable]:
        try:
            if not self.should_trace(frame):
                return None
            if event == 'call':
                call_stack = self.call_stack.get()
                call_stack = call_stack.copy()
                func_name = frame.f_code.co_name
                filename = frame.f_code.co_filename
                decl_lineno = frame.f_code.co_firstlineno
                
                if frame.f_back:
                    lineno = frame.f_back.f_lineno
                else:
                    lineno = decl_lineno
                
                node = CallTreeNode(func_name, filename, lineno, decl_lineno)
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
                exc_type, _, _ = arg
                if exc_type is GeneratorExit:
                    return None
                call_stack = self.call_stack.get()
                call_stack = call_stack.copy()
                if call_stack:
                    call_stack[-1].set_exception(exc_type.__name__)
                    if frame.f_back is None or frame.f_code.co_name.startswith('test_'):
                        current_node = next((node for node in reversed(call_stack) 
                                            if node.name == frame.f_code.co_name 
                                            and node.filename == frame.f_code.co_filename), None) or call_stack[-1]
                        print("Call graph for exception in current frame:", file=sys.stderr)
                        # TODO: # This causes a duplicate report. Also, the order of frames is not always the same.
                        print(str(current_node), file=sys.stderr)
            return self.trace_calls
        except Exception as err:
            print(f"DDBG error trace_calls {err}")
            return None

    if python_version >= (3, 7):
        import asyncio

        def task_done_callback(self, task):
            try:
                exc = task.exception()
                if exc:
                    tb = exc.__traceback__
                    nodes: List[CallTreeNode] = []
                    while tb:
                        frame = tb.tb_frame
                        if not self.should_trace(frame):
                            tb = tb.tb_next
                            continue
                        func_name = frame.f_code.co_name
                        filename = frame.f_code.co_filename
                        lineno = tb.tb_lineno
                        decl_lineno = frame.f_code.co_firstlineno
                        node = CallTreeNode(func_name, filename, lineno, decl_lineno)
                        node.set_exception(type(exc).__name__)
                        nodes.append(node)
                        tb = tb.tb_next
                    creation_stack: List[CallTreeNode] = getattr(task, '_creation_stack', [])
                    full_stack = creation_stack + nodes
                    for i in range(len(full_stack) - 1):
                        full_stack[i].add_child(full_stack[i + 1])
                    if full_stack:
                        root_node = full_stack[0]
                        print("Call graph for task exception:")
                        print(str(root_node))
            except Exception as err:
                print(f"DDBG error task_done_callback {err}")
                pass

_tracer = None

def exception_handler(exc_type, exc_value, exc_traceback):
    try:
        if _tracer and exc_type is not GeneratorExit:
            tb = exc_traceback
            nodes: List[CallTreeNode] = []
            while tb:
                frame = tb.tb_frame
                if not _tracer.should_trace(frame):
                    tb = tb.tb_next
                    continue
                func_name = frame.f_code.co_name
                filename = frame.f_code.co_filename
                lineno = tb.tb_lineno
                decl_lineno = frame.f_code.co_firstlineno
                node = CallTreeNode(func_name, filename, lineno, decl_lineno)
                node.set_exception(exc_type.__name__)
                nodes.append(node)
                tb = tb.tb_next
            nodes.reverse()
            for i in range(len(nodes) - 1):
                nodes[i].add_child(nodes[i + 1])
            if nodes:
                print("Call graph for uncaught exception:")
                print(str(nodes[0]))
    except Exception as err:
        pass
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

def register_runtime_trace():
    
    print(f"DDBG register_runtime_trace")
    global _tracer
    _tracer = CallGraphTracer()
    sys.settrace(_tracer.trace_calls)
    threading.settrace(_tracer.trace_calls)
    sys.excepthook = exception_handler

    # TODO: Trees reported in different places have different order of nodes.

    if python_version >= (3, 7):
        import asyncio

        class TracingEventLoopPolicy(asyncio.DefaultEventLoopPolicy):
            def new_event_loop(self):
                loop = super().new_event_loop()
                loop.set_task_factory(self.tracing_task_factory)
                loop.set_exception_handler(self.exception_handler)
                return loop

            def tracing_task_factory(self, loop, coro):
                task = asyncio.Task(coro, loop=loop)
                creation_stack: List[CallTreeNode] = []
                for frame_info in traceback.extract_stack()[:-1]:
                    if frame_info.filename is None:
                        continue
                    try:
                        abs_filename = os.path.abspath(frame_info.filename)
                        if _tracer.cwd and not abs_filename.startswith(_tracer.cwd):
                            continue
                    except Exception:
                        continue
                    node = CallTreeNode(frame_info.name, frame_info.filename, frame_info.lineno, frame_info.lineno)
                    creation_stack.append(node)
                task._creation_stack = creation_stack
                task.add_done_callback(_tracer.task_done_callback)
                return task

            def exception_handler(self, loop, context):
                if 'exception' in context and 'task' in context:
                    _tracer.task_done_callback(context['task'])
                loop.default_exception_handler(context)

        asyncio.set_event_loop_policy(TracingEventLoopPolicy())