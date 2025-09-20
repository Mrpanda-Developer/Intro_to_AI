import re

def extract_parts(predicate: str):
    """
    Haalt filmnaam en actor/director naam uit een predicate.
    Bijv. 'actor(the_hudsucker_proxy,tim_robbins,norville_barnes)'
      -> ('the_hudsucker_proxy', 'tim_robbins')
    """
    if predicate.startswith("actor(") or predicate.startswith("director("):
        inner = predicate[predicate.find("(")+1 : predicate.rfind(")")]
        parts = inner.split(",")
        if len(parts) >= 2:
            movie = parts[0].strip()
            name = parts[1].strip()
            return movie, name
    return None, predicate

def parse_prolog_trace(trace_lines, filename="tree.dot", root_label="Finding an actor who is also a director"):
    with open(filename, "w", encoding="utf-8") as f:   # utf-8 for ❌ kinda vibes I guess LOL xDDD
        f.write("digraph G {\n")
        f.write('  node [shape=ellipse, style=filled, fontname="Arial"];\n')

        # Root node
        f.write(f'  "root" [label="{root_label}", shape=ellipse, fillcolor=lightgrey, style=filled];\n')

        last_exit = "root"  # start bij root

        for line in trace_lines:
            line = line.strip()

            if line.startswith("Exit:"):
                predicate = line[len("Exit:"):].strip()
                movie, name = extract_parts(predicate)
                label = f"{name}" if movie is None else f"{movie}\\n{name}"  # film + naam
                node_id = f"exit_{name}_{len(line)}"
                f.write('  node [shape=ellipse, style=filled, fontname="Arial", fontsize=10, width=0.3, height=0.3, fixedsize=false];\n')
                f.write(f'  "{last_exit}" -> "{node_id}";\n')
                last_exit = node_id

            elif line.startswith("Fail:"):
                predicate = line[len("Fail:"):].strip()
                movie, name = extract_parts(predicate)
                label = f"{name} ❌"
                node_id = f"fail_{name}_{len(line)}"
                f.write(f'  "{node_id}" [label="{label}", fillcolor=red, fontcolor=white, color=black];\n')
                f.write(f'  "{last_exit}" -> "{node_id}";\n')

        f.write("}\n")

if __name__ == "__main__":
    with open("trace.txt", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    parse_prolog_trace(lines, "tree.dot", root_label="Finding an actor who is also a director")
    print("DOT-bestand opgeslagen als tree.dot — open dit in https://dreampuf.github.io/GraphvizOnline/")
