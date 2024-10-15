_current_call_graph = None

def trace_line(s):
    global _current_call_graph
    if _current_call_graph:
        _current_call_graph.append_to_current_node(f"TRACE_LINE: {s}")


def register_call_graph(call_graph):
    global _current_call_graph
    _current_call_graph = call_graph
