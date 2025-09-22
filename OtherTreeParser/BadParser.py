import re
import os
from treelib import Tree
import graphviz

CALL_RE = re.compile(r"^\s*Call:(.+)$")
EXIT_RE = re.compile(r"^\s*Exit:(.+)$")
FAIL_RE = re.compile(r"^\s*Fail:(.+)$")
REDO_RE = re.compile(r"^\s*Redo:(.+)$")

def parse_trace(trace_lines):
    tree = Tree()
    tree.create_node("root", "root")
    stack = ["root"]
    node_counter = 0

    for line in trace_lines:
        line = line.strip()
        if not line:
            continue

        if m := CALL_RE.match(line):
            node_counter += 1
            label = f"Call: {m.group(1).strip()}"
            node_id = f"n{node_counter}"
            tree.create_node(label, node_id, parent=stack[-1])
            stack.append(node_id)

        elif m := EXIT_RE.match(line):
            node_counter += 1
            label = f"Exit: {m.group(1).strip()} ✅"
            node_id = f"n{node_counter}"
            tree.create_node(label, node_id, parent=stack[-1])
            if len(stack) > 1:
                stack.pop()

        elif m := FAIL_RE.match(line):
            node_counter += 1
            label = f"Fail: {m.group(1).strip()} ❌"
            node_id = f"n{node_counter}"
            tree.create_node(label, node_id, parent=stack[-1])
            if len(stack) > 1:
                stack.pop()

        elif m := REDO_RE.match(line):
            node_counter += 1
            label = f"Redo: {m.group(1).strip()} 🔄"
            node_id = f"n{node_counter}"
            tree.create_node(label, node_id, parent=stack[-1])
            # geen pop → slang

    return tree


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    trace_path = os.path.join(base_dir, "trace.txt")

    with open(trace_path, "r") as f:
        trace_lines = f.readlines()

    tree = parse_trace(trace_lines)

    print("=== Console output van de boom ===")
    tree.show(line_type="ascii")

    dot_path = os.path.join(base_dir, "search_tree.dot")
    tree.to_graphviz(dot_path)
    print(f"DOT-bestand opgeslagen: {dot_path}")

    # verticale slang (default Graphviz)
    graph = graphviz.Source.from_file(dot_path)
    png_path = os.path.join(base_dir, "search_tree")
    graph.render(filename=png_path, format="png", cleanup=True)
    print(f"PNG-bestand opgeslagen: {png_path}.png")
