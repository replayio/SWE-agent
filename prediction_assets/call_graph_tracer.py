# noqa: I002
# ruff: noqa: UP006

# NOTE: We ignore some warnings in this file to allow for backward compatability.

import inspect
import json  # noqa: I002
import os
import sys
import threading
import traceback
from json.decoder import JSONDecodeError
from types import FrameType, TracebackType
from typing import Callable, Dict, List, Optional, Union  # noqa: UP035

# REPO_ROOT = os.environ.get("REPO_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# INSTANCE_NAME = os.environ.get("TDD_INSTANCE_NAME")

TargetConfig = Dict[str, Union[Optional[str], Optional[int]]]

def parse_json(json_string):
    try:
        return json.loads(json_string)
    except JSONDecodeError as e:
        # Get the position of the error
        pos = e.pos
        
        # Get the line and column of the error
        lineno = json_string.count('\n', 0, pos) + 1
        colno = pos - json_string.rfind('\n', 0, pos)
        
        # Get the problematic lines (including context)
        lines = json_string.splitlines()
        context_range = 2  # Number of lines to show before and after the error
        start = max(0, lineno - context_range - 1)
        end = min(len(lines), lineno + context_range)
        context_lines = lines[start:end]
        
        # Create the context string with line numbers
        context = ""
        for i, line in enumerate(context_lines, start=start+1):
            if i == lineno:
                context += f"{i:4d} > {line}\n"
                context += "       " + " " * (colno - 1) + "^\n"
            else:
                context += f"{i:4d}   {line}\n"
        
        # Construct and raise a new error with more information
        error_msg = f"JSON parsing failed at line {lineno}, column {colno}:\n\n{context.rstrip()}\nError: {str(e)}"
        raise ValueError(error_msg) from e

# Parse config.
TRACE_TARGET_CONFIG_STR = os.environ.get("TDD_TRACE_TARGET_CONFIG")
TRACE_TARGET_CONFIG: Optional[TargetConfig] = None
if TRACE_TARGET_CONFIG_STR:
    TRACE_TARGET_CONFIG = parse_json(TRACE_TARGET_CONFIG_STR)
    if TRACE_TARGET_CONFIG:
        if "target_file" not in TRACE_TARGET_CONFIG:
            raise ValueError("TDD_TRACE_TARGET_CONFIG must provide 'target_file'.")
        if "target_function_name" not in TRACE_TARGET_CONFIG:
            raise ValueError("TDD_TRACE_TARGET_CONFIG must provide 'target_function_name' if 'target_file' is provided.")

# Record parameter values and return values, only if target region is sufficiently scoped.
RECORD_VALUES = not not TRACE_TARGET_CONFIG



class FrameInfo:
    def __init__(
        self,
        frame: Optional[FrameType] = None,
        decl_filename: Optional[str] = None,
        decl_lineno: Optional[int] = None,
        call_filename: Optional[str] = None,
        call_lineno: Optional[int] = None,
        function_name: Optional[str] = None,
        code_context: Optional[List[str]] = None,
        index: Optional[int] = None,
    ):
        self.frame = frame
        self.decl_filename = decl_filename or (frame.f_code.co_filename if frame else None)
        self.decl_lineno = decl_lineno or (frame.f_code.co_firstlineno if frame else None)

        self.call_filename = call_filename or (frame.f_back.f_code.co_filename if frame and frame.f_back else None)
        self.call_lineno = call_lineno or (frame.f_back.f_lineno if frame else None)

        self.function_name = function_name or (frame.f_code.co_name if frame else None)
        self.code_context = code_context
        self.index = index


    @classmethod
    def from_frame(cls, frame: FrameType) -> "FrameInfo":
        return cls(frame=frame)

    @classmethod
    def from_traceback(cls, tb: TracebackType) -> "FrameInfo":
        return cls(frame=tb.tb_frame, call_lineno=tb.tb_lineno)

    @classmethod
    def from_frame_summary(cls, summary: traceback.FrameSummary) -> "FrameInfo":
        return cls(
            call_filename=summary.filename,
            call_lineno=summary.lineno,
            
            function=summary.name,
            code_context=summary.line and [summary.line] or None,
        )

    def get_name(self) -> str:
        """
        This does NOT actually provide the *qualified* name. It looks like we can't do that reliably across versions.
        """
        frame = self.frame
        if frame is None:
            return self.function_name or ""

        code = frame.f_code
        return code.co_name


    def get_locals(self) -> Dict[str, any]:
        return self.frame.f_locals


python_version = sys.version_info

# Compatibility for ContextVar
if python_version >= (3, 7):
    from contextvars import ContextVar
else:

    class ContextVar:
        def __init__(self, name, default=None):
            self.local = threading.local()
            self.default = default

        def get(self):
            return getattr(self.local, "value", self.default)

        def set(self, value):
            setattr(self.local, "value", value)


def get_relative_filename(filename: str) -> str:
    try:
        rel_path = os.path.relpath(filename)
        if rel_path.startswith("..") or os.path.isabs(rel_path):
            return f"EXTERNAL/{os.path.basename(filename)}"
        else:
            return rel_path
    except Exception:
        return filename

class BaseNode:
    def __init__(self):
        self.children: List[BaseNode] = []

    def add_child(self, child: "BaseNode"):
        self.children.append(child)
        child.parent = self


class OmittedNode(BaseNode):
    def __init__(self):
        super().__init__()
        self.name = "(omitted child)"

    def __str__(self, level=0, visited=None):
        indent = "  " * level
        return f"{indent}{self.name}\n"

    def to_dict(self):
        return {"name": self.name, "type": "OmittedNode"}

class CallGraphNode(BaseNode):
    is_partial: bool = False

    def __init__(self, frame_info: FrameInfo):
        super().__init__()
        self.frame_info = frame_info
        self.children: list[CallGraphNode] = []
        self.exception: Optional[str] = None
        self.parameters: Dict[str, any] = {}
        self.return_value: any = None
        self.parent: Optional[CallGraphNode] = None

    @property
    def name(self) -> str:
        return self.frame_info.get_name()

    @property
    def decl_filename(self) -> str:
        return get_relative_filename(self.frame_info.decl_filename)

    @property
    def call_filename(self) -> str:
        return get_relative_filename(self.frame_info.call_filename)

    def set_exception(self, exc_name: str):
        self.exception = exc_name

    def set_parameters(self, params: Dict[str, any]):
        if RECORD_VALUES:
            self.parameters = params

    def set_return_value(self, value: any):
        if RECORD_VALUES:
            self.return_value = value

    def __str__(self, level=0, visited=None):
        if visited is None:
            visited = set()
        if id(self) in visited:
            return "  " * level + "[Recursion]\n"
        visited.add(id(self))
        indent = "  " * level
        result = f"{indent}{self.name}" + f" [decl: {self.decl_filename or '?'}:{self.frame_info.decl_lineno or '?'}"
        if self.frame_info.call_lineno != self.frame_info.decl_lineno:
            result += f", called_from: {self.call_filename or '?'}:{self.frame_info.call_lineno or '?'}"
        result += "]"
        if RECORD_VALUES:
            if self.parameters:
                result += f", Parameters:{self.parameters}"
            if self.return_value is not None:
                result += f", ReturnValue:{self.return_value}"
        result += "\n"
        for child in self.children:
            result += child.__str__(level + 1, visited)
        return result

class CallGraph:
    def __init__(self):
        self.call_stack: List[CallGraphNode] = ContextVar("call_stack", default=[])
        self.root: Optional[CallGraphNode] = None
        self.is_partial: bool = False
        try:
            self.cwd = os.getcwd()
        except Exception:
            self.cwd = None

    def should_trace(self, frame: FrameType) -> bool:
        if frame is None or frame.f_code is None or frame.f_code.co_filename is None:
            # Ignore code without code or filename.
            # TODO: Not sure why frames can have no code or filename. Might be some builtins?
            return False
        if frame.f_code.co_filename == __file__:
            # Ignore the trace code itself.
            return False
        filename = frame.f_code.co_filename
        abs_filename = os.path.abspath(filename)
        if self.cwd:
            # Ignore external code.
            return abs_filename.startswith(self.cwd)
        else:
            return True
        
    def access_call_stack(self) -> List[CallGraphNode]:
        call_stack = self.call_stack.get()
        res: List[CallGraphNode] = call_stack.copy()
        return res

    def trace_calls(self, event_frame: FrameType, event: str, arg: any) -> Optional[Callable]:
        try:
            frame_info = FrameInfo.from_frame(event_frame)

            if not self.should_trace(event_frame):
                return None
            if event == "call":
                call_stack = self.access_call_stack()

                node = CallGraphNode(frame_info)

                # Store parameter values
                node.set_parameters(frame_info.get_locals())

                if call_stack:
                    call_stack[-1].add_child(node)
                else:
                    self.root = node
                call_stack.append(node)
                self.call_stack.set(call_stack)
                event_frame.f_trace = self.trace_calls
            elif event == "return":
                call_stack = self.access_call_stack()
                if call_stack:
                    # Store return value
                    call_stack[-1].set_return_value(arg)
                    call_stack.pop()
                    self.call_stack.set(call_stack)
            elif event == "exception":
                exc_type, _, _ = arg
                if exc_type is GeneratorExit:
                    return None
                call_stack = self.access_call_stack()
                if call_stack:
                    call_stack[-1].set_exception(exc_type.__name__)
                    test_node = (
                        next(
                            (
                                node
                                for node in reversed(call_stack)
                                if node.name.startswith("test_")
                            ),
                            None,
                        )
                    )
                    if test_node:
                        self.print_graph_on_exception("EXCEPTION", test_node)
            return self.trace_calls
        except Exception:
            return None

    def find_node(self, target_config: TargetConfig) -> Optional[CallGraphNode]:
        root = self.root
        stack = [root]
        while stack:
            node = stack.pop()
            if (node.name == target_config.get('target_function_name') and
                node.decl_filename == target_config.get('target_file') and
                (target_config.get('decl_lineno') is None or 
                node.frame_info.decl_lineno == target_config['decl_lineno'])):
                return node
            stack.extend(reversed(node.children))
        return None

    def get_partial_graph(self, target_config: TargetConfig) -> Optional[BaseNode]:
        """
        Create a partial graph of the first function call of given name.
        Should only contain the flat call graph surrounding that node, i.e. its parent and its children.
        """

        def create_partial_node(node: CallGraphNode) -> CallGraphNode:
            partial_node = CallGraphNode(node.frame_info)
            partial_node.parameters = node.parameters
            partial_node.return_value = node.return_value
            partial_node.exception = node.exception
            partial_node.is_partial = True
            return partial_node

        if not self.root:
            return None

        target_node = self.find_node(target_config)
        if not target_node:
            return None

        partial_node = create_partial_node(target_node)

        if target_node.parent:
            # Pick parent of target node, if available.
            root = create_partial_node(target_node.parent)
            root.add_child(partial_node)  # Add the target node
            if len(target_node.parent.children) > 1:
                # Add OmittedNode to represent siblings.
                root.add_child(OmittedNode())
        else:
            root = partial_node

        # Add children
        for child in target_node.children:
            child_partial = create_partial_node(child)
            partial_node.add_child(child_partial)
            if child.children:
                # Add OmittedNode to represent children.
                child_partial.add_child(OmittedNode())

        return root

    def print_graph_on_exception(self, cause: str, node: BaseNode):
        result: str = None
        if TRACE_TARGET_CONFIG:
            partial_graph = self.get_partial_graph(TRACE_TARGET_CONFIG)
            partial_info = f" PARTIAL='{str(TRACE_TARGET_CONFIG)}'"
            if partial_graph:
                result = str(partial_graph)
            else:
                # Hackfix: Stringify without values, if we could not target the function.
                global RECORD_VALUES
                RECORD_VALUES = False
                result = "(❌ ERROR: Could not find target function. Providing high-level call graph instead. ❌)\n" + str(node)
                RECORD_VALUES = True
        else:
            partial_info = ""
            result = str(node)
        print("\n\n" + f"<CALL_GRAPH_ON_EXCEPTION cause='{cause}'{partial_info}>", file=sys.stderr)
        print(result, file=sys.stderr)
        print("\n</CALL_GRAPH_ON_EXCEPTION>", file=sys.stderr)

    if python_version >= (3, 7):

        def task_done_callback(self, task):
            exc = task.exception()
            if exc:
                try:
                    tb = exc.__traceback__
                    nodes: list[CallGraphNode] = []
                    while tb:
                        frame_info = FrameInfo.from_traceback(tb)
                        if not self.should_trace(frame_info.frame):
                            tb = tb.tb_next
                            continue
                        node = CallGraphNode(frame_info)
                        node.set_parameters(frame_info.get_locals())
                        node.set_exception(type(exc).__name__)
                        nodes.append(node)
                        tb = tb.tb_next
                    creation_stack: list[CallGraphNode] = getattr(task, "_creation_stack", [])
                    full_stack = creation_stack + nodes
                    for i in range(len(full_stack) - 1):
                        full_stack[i].add_child(full_stack[i + 1])
                    if full_stack:
                        self.print_graph_on_exception("FUTURE_DONE_CALLBACK", full_stack[0])
                except Exception:
                    print("\n\n\nERROR IN exception_handler:\n\n\n")
                    traceback.print_exc()


def register_runtime_trace():
    global _current_graph
    _current_graph = CallGraph()
    sys.settrace(_current_graph.trace_calls)
    threading.settrace(_current_graph.trace_calls)

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
                creation_stack: list[CallGraphNode] = []

                # Get the current stack with frame objects
                stack = inspect.stack()

                # Skip the last frame (this method)
                for stack_frame in stack[1:]:
                    if stack_frame.filename is None:
                        continue
                    abs_filename = os.path.abspath(stack_frame.filename)
                    if _current_graph.cwd and not abs_filename.startswith(_current_graph.cwd):
                        continue

                    # Create FrameInfo object directly from the frame
                    frame_info = FrameInfo.from_frame(stack_frame.frame)

                    node = CallGraphNode(frame_info)
                    creation_stack.append(node)

                task._creation_stack = creation_stack
                task.add_done_callback(_current_graph.task_done_callback)
                return task

            def exception_handler(self, loop, context):
                if "exception" in context and "task" in context:
                    _current_graph.task_done_callback(context["task"])
                loop.default_exception_handler(context)

        asyncio.set_event_loop_policy(TracingEventLoopPolicy())


_current_graph = None


def exception_handler(exc_type, exc_value, exc_traceback):
    try:
        if _current_graph and exc_type is not GeneratorExit:
            tb = exc_traceback
            nodes: list[CallGraphNode] = []
            while tb:
                frame_info = FrameInfo.from_traceback(tb)
                if not _current_graph.should_trace(frame_info.frame):
                    tb = tb.tb_next
                    continue
                node = CallGraphNode(frame_info)
                node.set_exception(exc_type.__name__)
                node.set_parameters(frame_info.get_locals())
                nodes.append(node)
                tb = tb.tb_next
            nodes.reverse()
            for i in range(len(nodes) - 1):
                nodes[i].add_child(nodes[i + 1])
            if nodes:
                _current_graph.print_graph_on_exception("UNCAUGHT_EXCEPTION_HANDLER", nodes[0])
    except Exception:
        print("\n\n\nERROR IN exception_handler:\n\n\n")
        traceback.print_exc()
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


sys.excepthook = exception_handler

if __name__ == "__main__":
    register_runtime_trace()
