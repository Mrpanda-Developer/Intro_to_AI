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
    
    # Stack voor parent nodes en diepte tracking
    stack = [("root", 0)]  # (parent_id, depth)
    node_counter = 0
    current_depth = 0

    for line in trace_lines:
        line = line.strip()
        if not line:
            continue

        if m := CALL_RE.match(line):
            node_counter += 1
            label = f"Call: {m.group(1).strip()}"
            node_id = f"n{node_counter}"
            
            # Bepaal de juiste parent based on depth
            current_parent, parent_depth = stack[-1]
            tree.create_node(label, node_id, parent=current_parent)
            
            # Verhoog diepte en push naar stack
            current_depth = parent_depth + 1
            stack.append((node_id, current_depth))

        elif m := EXIT_RE.match(line):
            node_counter += 1
            label = f"Exit: {m.group(1).strip()} ✅"
            node_id = f"n{node_counter}"
            
            if len(stack) > 1:
                current_parent, parent_depth = stack[-1]
                tree.create_node(label, node_id, parent=current_parent)
                stack.pop()  # Ga terug naar vorige niveau
                current_depth = parent_depth

        elif m := FAIL_RE.match(line):
            node_counter += 1
            label = f"Fail: {m.group(1).strip()} ❌"
            node_id = f"n{node_counter}"
            
            if len(stack) > 1:
                current_parent, parent_depth = stack[-1]
                tree.create_node(label, node_id, parent=current_parent)
                stack.pop()  # Ga terug naar vorige niveau
                current_depth = parent_depth

        elif m := REDO_RE.match(line):
            node_counter += 1
            label = f"Redo: {m.group(1).strip()} 🔄"
            node_id = f"n{node_counter}"
            
            if len(stack) > 1:
                current_parent, parent_depth = stack[-1]
                tree.create_node(label, node_id, parent=current_parent)
                # Bij Redo blijven we opzelfde diepte

    return tree

def parse_trace_deep(trace_lines):
    """Alternative parser die diepere nesting forceert"""
    tree = Tree()
    tree.create_node("root", "root")
    
    call_stack = []  # Stack voor calls
    parent_stack = ["root"]  # Stack voor parents
    node_counter = 0
    pending_calls = []  # Calls die wachten op nesting

    for line in trace_lines:
        line = line.strip()
        if not line:
            continue

        if m := CALL_RE.match(line):
            node_counter += 1
            label = f"Call: {m.group(1).strip()}"
            node_id = f"n{node_counter}"
            
            # Als er pending calls zijn, nest deze
            if pending_calls:
                prev_call_id, prev_label = pending_calls.pop()
                tree.create_node(prev_label, prev_call_id, parent=parent_stack[-1])
                parent_stack.append(prev_call_id)
            
            call_stack.append((node_id, label))
            pending_calls.append((node_id, label))

        elif m := EXIT_RE.match(line):
            node_counter += 1
            label = f"Exit: {m.group(1).strip()} ✅"
            node_id = f"n{node_counter}"
            
            if call_stack:
                call_id, call_label = call_stack.pop()
                
                # Creëer de call node als deze nog niet bestaat
                if not tree.contains(call_id):
                    tree.create_node(call_label, call_id, parent=parent_stack[-1] if parent_stack else "root")
                
                tree.create_node(label, node_id, parent=call_id)
                
                # Als we teruggaan, update parent stack
                if parent_stack and parent_stack[-1] == call_id:
                    parent_stack.pop()

        elif m := FAIL_RE.match(line):
            node_counter += 1
            label = f"Fail: {m.group(1).strip()} ❌"
            node_id = f"n{node_counter}"
            
            if call_stack:
                call_id, call_label = call_stack.pop()
                
                if not tree.contains(call_id):
                    tree.create_node(call_label, call_id, parent=parent_stack[-1] if parent_stack else "root")
                
                tree.create_node(label, node_id, parent=call_id)
                
                if parent_stack and parent_stack[-1] == call_id:
                    parent_stack.pop()

        elif m := REDO_RE.match(line):
            node_counter += 1
            label = f"Redo: {m.group(1).strip()} 🔄"
            node_id = f"n{node_counter}"
            
            if call_stack:
                call_id, call_label = call_stack[-1]  # Peek, niet pop
                if tree.contains(call_id):
                    tree.create_node(label, node_id, parent=call_id)

    # Verwerk resterende pending calls
    for call_id, call_label in pending_calls:
        if not tree.contains(call_id):
            tree.create_node(call_label, call_id, parent=parent_stack[-1] if parent_stack else "root")

    return tree

def analyze_trace_structure(trace_lines):
    """Analyzeer de trace om de nesting te begrijpen"""
    depth = 0
    max_depth = 0
    call_pattern = []
    
    print("=== Trace structuur analyse ===")
    for i, line in enumerate(trace_lines[:50]):  # Bekijk eerste 50 regels
        line = line.strip()
        if not line:
            continue
            
        if CALL_RE.match(line):
            depth += 1
            max_depth = max(max_depth, depth)
            call_pattern.append(("Call", depth, line))
            print(f"{i+1:3d}: {'  ' * depth}Call (depth: {depth})")
            
        elif EXIT_RE.match(line) or FAIL_RE.match(line):
            call_pattern.append(("Exit/Fail", depth, line))
            print(f"{i+1:3d}: {'  ' * depth}Exit/Fail (depth: {depth})")
            if depth > 0:
                depth -= 1
                
        elif REDO_RE.match(line):
            call_pattern.append(("Redo", depth, line))
            print(f"{i+1:3d}: {'  ' * depth}Redo (depth: {depth})")
    
    print(f"\nMaximale diepte in trace: {max_depth}")
    return max_depth, call_pattern


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    trace_path = os.path.join(base_dir, "trace.txt")

    with open(trace_path, "r", encoding="utf-8") as f:
        trace_lines = f.readlines()

    # Analyseer eerst de trace structuur
    max_depth, pattern = analyze_trace_structure(trace_lines)
    
    # Kies de juiste parser based on diepte
    if max_depth >= 3:
        print("\nGebruik deep parser voor geneste structuur")
        tree = parse_trace_deep(trace_lines)
    else:
        print("\nGebruik standaard parser")
        tree = parse_trace(trace_lines)

    print("\n=== Boomstructuur (tekst) ===")
    tree.show(line_type="ascii")

    # Toon diepte informatie
    print(f"\nBoom diepte: {tree.depth()}")
    print(f"Aantal nodes: {tree.size()}")
    
    for level in range(tree.depth() + 1):
        nodes_at_level = [node for node in tree.all_nodes() if tree.depth(node.identifier) == level]
        if nodes_at_level:
            print(f"Level {level}: {len(nodes_at_level)} nodes")
            if level <= 3:  # Toon eerste few levels voor debug
                for node in nodes_at_level[:3]:  # Toon eerste 3 nodes per level
                    print(f"     - {node.tag}")

    # Maak DOT file met geforceerde diepte
    dot_path = os.path.join(base_dir, "search_tree.dot")
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write("digraph tree {\n")
        f.write("  rankdir=TB;\n")
        f.write("  node [shape=rectangle, fontsize=10];\n")
        f.write("  edge [arrowhead=vee];\n")
        
        # Forceer minimale diepte
        f.write("  minlen=2;\n")  # Minimale edge length voor diepte
        f.write("  nodesep=0.5;\n")
        f.write("  ranksep=0.8;\n\n")
        
        for node in tree.all_nodes():
            f.write(f'  "{node.identifier}" [label="{node.tag}"];\n')
        
        f.write("\n")
        
        # Groepeer nodes per level voor betere hierarchie
        for level in range(tree.depth() + 1):
            nodes_at_level = [f'"{node.identifier}"' for node in tree.all_nodes() if tree.depth(node.identifier) == level]
            if nodes_at_level:
                f.write(f"  {{rank=same; {'; '.join(nodes_at_level)}; }}\n")
        
        f.write("\n")
        
        for node in tree.all_nodes():
            for child in tree.children(node.identifier):
                f.write(f'  "{node.identifier}" -> "{child.identifier}";\n')
        
        f.write("}\n")

    print(f"\nDOT-bestand opgeslagen: {dot_path}")

    # Genereer PNG
    try:
        graph = graphviz.Source.from_file(dot_path)
        png_path = os.path.join(base_dir, "search_tree")
        graph.render(filename=png_path, format="png", cleanup=True)
        print(f"PNG-bestand opgeslagen: {png_path}.png")
    except Exception as e:
        print(f"Fout bij PNG generatie: {e}")