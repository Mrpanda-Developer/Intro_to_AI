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
    stack = [("root", "root")]  # (node_id, full_context)
    node_counter = 0

    for line in trace_lines:
        line = line.strip()
        if not line:
            continue

        if m := CALL_RE.match(line):
            node_counter += 1
            call_content = m.group(1).strip()
            label = f"Call: {call_content}"
            node_id = f"n{node_counter}"
            
            # Parent is de top van de stack
            parent_id, _ = stack[-1]
            tree.create_node(label, node_id, parent=parent_id)
            
            # Push naar stack met nieuwe context
            stack.append((node_id, call_content))

        elif m := EXIT_RE.match(line):
            node_counter += 1
            exit_content = m.group(1).strip()
            label = f"Exit: {exit_content} ✅"
            node_id = f"n{node_counter}"
            
            # Parent is de bijbehorende call (top van stack)
            if len(stack) > 1:
                parent_id, call_context = stack[-1]
                tree.create_node(label, node_id, parent=parent_id)
                stack.pop()  # Verwijder de voltooide call van stack

        elif m := FAIL_RE.match(line):
            node_counter += 1
            fail_content = m.group(1).strip()
            label = f"Fail: {fail_content} ❌"
            node_id = f"n{node_counter}"
            
            # Parent is de bijbehorende call (top van stack)
            if len(stack) > 1:
                parent_id, call_context = stack[-1]
                tree.create_node(label, node_id, parent=parent_id)
                stack.pop()  # Verwijder de gefaalde call van stack

        elif m := REDO_RE.match(line):
            node_counter += 1
            redo_content = m.group(1).strip()
            label = f"Redo: {redo_content} 🔄"
            node_id = f"n{node_counter}"
            
            # Parent is de huidige call (top van stack)
            if len(stack) > 1:
                parent_id, call_context = stack[-1]
                tree.create_node(label, node_id, parent=parent_id)
                # Bij Redo blijven we op de stack voor mogelijke volgende calls

    return tree


def create_compact_dot(tree, filename):
    """Maak een compacte DOT weergave met verticale layout"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("digraph tree {\n")
        f.write("  rankdir=TB;\n")  # Top to Bottom
        f.write("  node [shape=rectangle, fontname=\"Arial\", fontsize=10];\n")
        f.write("  edge [arrowhead=vee];\n")
        f.write("  splines=false;\n")
        f.write("  nodesep=0.2;\n")
        f.write("  ranksep=0.3;\n\n")
        
        # Schrijf alle nodes
        for node in tree.all_nodes():
            # Maak labels compacter
            label = node.tag
            if len(label) > 40:
                # Split lange predicaat calls
                if "(" in label and ")" in label:
                    parts = label.split("(")
                    if len(parts) > 1:
                        pred_name = parts[0].replace("Call: ", "").replace("Exit: ", "").replace("Fail: ", "").replace("Redo: ", "")
                        args = "(" + parts[1]
                        if len(pred_name) > 15:
                            pred_name = pred_name[:15] + "..."
                        label = label.split(":")[0] + ": " + pred_name + args
            
            f.write(f'  "{node.identifier}" [label="{label}"];\n')
        
        f.write("\n")
        
        # Schrijf hierarchische edges
        for node in tree.all_nodes():
            for child in tree.children(node.identifier):
                f.write(f'  "{node.identifier}" -> "{child.identifier}";\n')
        
        f.write("}\n")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    trace_path = os.path.join(base_dir, "trace.txt")

    with open(trace_path, "r", encoding="utf-8") as f:
        trace_lines = f.readlines()

    tree = parse_trace(trace_lines)

    print("=== Boomstructuur (tekst) ===")
    tree.show(line_type="ascii")

    # Toon de diepte van de boom
    print(f"\nBoom diepte: {tree.depth()}")
    print(f"Aantal nodes: {tree.size()}")
    
    # Toon de structuur per level
    for level in range(tree.depth() + 1):
        nodes_at_level = [node for node in tree.all_nodes() if tree.depth(node.identifier) == level]
        print(f"Level {level}: {len(nodes_at_level)} nodes")

    # DOT bestand maken
    dot_path = os.path.join(base_dir, "search_tree.dot")
    create_compact_dot(tree, dot_path)
    print(f"\nDOT-bestand opgeslagen: {dot_path}")

    # PNG genereren
    try:
        graph = graphviz.Source.from_file(dot_path)
        png_path = os.path.join(base_dir, "search_tree")
        graph.render(filename=png_path, format="png", cleanup=True)
        print(f"PNG-bestand opgeslagen: {png_path}.png")
    except Exception as e:
        print(f"Fout bij het genereren van PNG: {e}")
        print("Probeer handmatig: dot -Tpng search_tree.dot -o search_tree.png")

    # Debug: toon de eerste 20 lines van de trace om te zien wat er geparsed wordt
    print("\n=== Eerste 20 regels van trace ===")
    for i, line in enumerate(trace_lines[:20]):
        print(f"{i+1:2d}: {line.strip()}")