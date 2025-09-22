from treelib import Tree
from pathlib import Path
import re
from typing import List, Dict, Tuple, Optional, Set

# ------------------ Regex patterns for parsing Prolog trace ------------------ #
PAIR_CALL_RE = re.compile(r"^Call:\s*([^@]+)@>(.+)$")
PAIR_EXIT_RE = re.compile(r"^Exit:\s*([^@]+)@>(.+)$")
PAIR_FAIL_RE = re.compile(r"^Fail:\s*([^@]+)@>(.+)$")

EXIT_DIR_RE  = re.compile(r"^Exit:director\(([^,]+),\s*([^)]+)\)")
EXIT_ACT_RE  = re.compile(r"^Exit:actor\(([^,]+),\s*([^,]+),\s*([^)]+)\)")
FAIL_ACT_RE  = re.compile(r"^Fail:actor\(([^,]+),\s*([^,]+),\s*([^)]+)\)")
FAIL_DIR_RE  = re.compile(r"^Fail:director\(([^,]+),\s*([^)]+)\)")

BIND_A1_RE   = re.compile(r"A1\s*=\s*(\S+)")
BIND_A2_RE   = re.compile(r"A2\s*=\s*(\S+)")
BIND_M_RE    = re.compile(r"M\s*=\s*(\S+)")

def normalize_name(s: str) -> str:
    """Normalize names by removing extra whitespace"""
    return " ".join(s.strip().split())

def create_actor_pair_key(actor1: str, actor2: str) -> Tuple[str, str]:
    """Create a consistent key for actor pairs (alphabetical order)"""
    names = sorted([actor1, actor2])
    return tuple(names) if names[0] != names[1] else (actor1, actor2)

# ------------------ Main parser function ------------------ #
def parse_prolog_trace_group_by_pair(
    trace_lines: List[str],
    output_file: str = "tree.dot",
    root_label: str = "Actor Comparisons"
) -> None:
    """
    Parse Prolog trace output and create a readable tree visualization
    
    The tree groups actor comparisons logically and provides clear
    visual indicators for success, failure, and additional information
    """
    tree = Tree()
    tree.create_node(root_label, "root")

    # Track unique actors and their comparison pairs
    all_actors: Set[str] = set()
    actor_pairs: Dict[Tuple[str, str], Dict[str, object]] = {}
    current_pair_context: Optional[Tuple[str, str]] = None
    node_id_counter = 0

    def create_actor_node(actor_name: str) -> str:
        """Create or retrieve an actor parent node"""
        nonlocal node_id_counter
        actor_node_id = f"actor_{actor_name.replace(' ', '_').replace(',', '')}"
        if not tree.contains(actor_node_id):
            tree.create_node(actor_name, actor_node_id, parent="root")
        return actor_node_id

    def create_pair_node(actor1: str, actor2: str) -> Tuple[str, str]:
        """Create or retrieve a comparison pair node"""
        nonlocal node_id_counter
        
        # Skip self-comparisons
        if actor1 == actor2:
            return None
            
        # Use consistent key ordering
        pair_key = create_actor_pair_key(actor1, actor2)
        
        if pair_key not in actor_pairs:
            node_id_counter += 1
            pair_node_id = f"pair_{node_id_counter}"
            
            # Place under first actor alphabetically
            parent_actor_id = create_actor_node(pair_key[0])
            node_label = f"vs {pair_key[1]}"
            
            tree.create_node(node_label, pair_node_id, parent=parent_actor_id)
            actor_pairs[pair_key] = {
                "id": pair_node_id,
                "actors": pair_key,
                "has_success": False,
                "has_failure": False,
                "added_directors": set(),
                "movie_info_added": False,
                "failure_explanation": None
            }
            
            all_actors.add(pair_key[0])
            all_actors.add(pair_key[1])
            
        return pair_key

    # Buffer for multi-line binding information
    binding_buffer: List[str] = []

    def process_binding_buffer():
        """Extract and process A1/A2/M binding information"""
        nonlocal binding_buffer, node_id_counter
        if not binding_buffer: return
        
        combined_text = " ".join(line.strip() for line in binding_buffer)
        
        a1_match = BIND_A1_RE.search(combined_text)
        a2_match = BIND_A2_RE.search(combined_text)
        movie_match = BIND_M_RE.search(combined_text)
        
        if a1_match and a2_match and movie_match:
            actor1 = a1_match.group(1)
            actor2 = a2_match.group(1)
            movie = movie_match.group(1)
            pair_key = create_pair_node(actor1, actor2)
            if pair_key:
                pair_data = actor_pairs[pair_key]
                if not pair_data["movie_info_added"]:
                    pair_id = pair_data["id"]
                    # Create movie information subtree
                    movie_node_id = f"{pair_id}_movie"
                    tree.create_node(f"Movie: {movie}", movie_node_id, parent=pair_id)
                    tree.create_node(f"Actor: {actor1}", f"{movie_node_id}_a1", parent=movie_node_id)
                    tree.create_node(f"Actor: {actor2}", f"{movie_node_id}_a2", parent=movie_node_id)
                    tree.create_node(f"✅ Match in {movie}", f"{pair_id}_ok", parent=pair_id)
                    pair_data["movie_info_added"] = True
        binding_buffer = []

    # --------------- Main processing loop --------------- #
    for line_raw in trace_lines:
        line_clean = line_raw.strip()
        if not line_clean:
            continue

        # Collect binding lines
        if any(keyword in line_clean for keyword in ("A1 =", "A2 =", "M =")) and not line_clean.startswith(("Call:", "Exit:", "Fail:", "Redo:")):
            binding_buffer.append(line_clean)
            continue

        process_binding_buffer()

        # --- New comparison pair ---
        call_match = PAIR_CALL_RE.match(line_clean)
        if call_match:
            actor1 = normalize_name(call_match.group(1))
            actor2 = normalize_name(call_match.group(2))
            current_pair_context = create_pair_node(actor1, actor2)
            continue

        # --- Successful comparison ---
        exit_match = PAIR_EXIT_RE.match(line_clean)
        if exit_match:
            actor1 = normalize_name(exit_match.group(1))
            actor2 = normalize_name(exit_match.group(2))
            pair_key = create_pair_node(actor1, actor2)
            if pair_key:
                pair_data = actor_pairs[pair_key]
                if not pair_data["has_success"] and not pair_data["has_failure"]:
                    tree.create_node("✅ Match found", f"{pair_data['id']}_success", parent=pair_data["id"])
                    pair_data["has_success"] = True
                current_pair_context = pair_key
            continue

        # --- Failed comparison ---
        fail_match = PAIR_FAIL_RE.match(line_clean)
        if fail_match:
            actor1 = normalize_name(fail_match.group(1))
            actor2 = normalize_name(fail_match.group(2))
            pair_key = create_pair_node(actor1, actor2)
            if pair_key:
                pair_data = actor_pairs[pair_key]
                if not pair_data["has_success"] and not pair_data["has_failure"]:
                    tree.create_node("❌ No match", f"{pair_data['id']}_fail", parent=pair_data["id"])
                    pair_data["has_failure"] = True
            current_pair_context = None
            continue

        # --- Director information ---
        director_exit_match = EXIT_DIR_RE.match(line_clean)
        if director_exit_match and current_pair_context:
            movie = normalize_name(director_exit_match.group(1))
            director = normalize_name(director_exit_match.group(2))
            actor1, actor2 = current_pair_context
            if director in [actor1, actor2]:
                pair_data = actor_pairs[current_pair_context]
                director_key = "a1" if director == actor1 else "a2"
                if director_key not in pair_data["added_directors"]:
                    tree.create_node(f"🎬 {director} directed {movie}", f"{pair_data['id']}_dir_{director_key}", parent=pair_data["id"])
                    pair_data["added_directors"].add(director_key)
            continue

        # --- Actor failure details ---
        actor_fail_match = FAIL_ACT_RE.match(line_clean)
        if actor_fail_match and current_pair_context:
            movie = normalize_name(actor_fail_match.group(1))
            actor = normalize_name(actor_fail_match.group(2))
            actor1, actor2 = current_pair_context
            if actor in [actor1, actor2]:
                pair_data = actor_pairs[current_pair_context]
                if not pair_data["failure_explanation"]:
                    pair_data["failure_explanation"] = f"⚠️ {actor} not in {movie}"
                    tree.create_node(pair_data["failure_explanation"], f"{pair_data['id']}_fail_reason", parent=pair_data["id"])
            continue

        # --- Director failure details ---
        director_fail_match = FAIL_DIR_RE.match(line_clean)
        if director_fail_match and current_pair_context:
            movie = normalize_name(director_fail_match.group(1))
            director = normalize_name(director_fail_match.group(2))
            actor1, actor2 = current_pair_context
            if director in [actor1, actor2]:
                pair_data = actor_pairs[current_pair_context]
                if not pair_data["failure_explanation"]:
                    pair_data["failure_explanation"] = f"⚠️ {director} didn't direct {movie}"
                    tree.create_node(pair_data["failure_explanation"], f"{pair_data['id']}_fail_reason", parent=pair_data["id"])
            continue

    # Process any remaining bindings
    process_binding_buffer()

    # Add generic failure explanations where needed
    for pair_key, pair_data in actor_pairs.items():
        if pair_data["has_failure"] and not pair_data["failure_explanation"] and not pair_data["has_success"]:
            tree.create_node(f"⚠️ No connection", f"{pair_data['id']}_fail_reason", parent=pair_data["id"])

    # Clean up empty actor nodes
    for actor in list(all_actors):
        actor_node_id = f"actor_{actor.replace(' ', '_').replace(',', '')}"
        if tree.get_node(actor_node_id) and not tree.children(actor_node_id):
            tree.remove_node(actor_node_id)

    # Generate final output
    export_to_styled_graphviz(tree, output_file)
    print(f"Visualization saved to {output_file}")
    print("Open with: GraphvizOnline")

def export_to_styled_graphviz(tree: Tree, output_path: str):
    """
    Export tree to Graphviz format with clean, readable styling
    
    Uses consistent white background with black text for all nodes,
    maintaining visual clarity without color coding
    """
    # Generate basic Graphviz output
    temp_file = output_path + ".temp"
    tree.to_graphviz(filename=temp_file)
    raw_output = Path(temp_file).read_text(encoding="utf-8").splitlines()

    # Apply clean styling
    styled_output = [
        "digraph tree {",
        "  rankdir=TB;",
        "  ranksep=1.0; nodesep=0.8;",
        "  splines=ortho;",
        "  concentrate=true;",
        '  node [shape=box, style="rounded,filled", fontname="Arial", fontsize=16, width=2.5, height=0.8, fillcolor="#ffffff", fontcolor="#000000"];',
        '  edge [penwidth=3.0, arrowsize=1.5];',
        '  graph [fontsize=18, fontname="Arial Bold"];',
    ]

    for line in raw_output[1:]:
        if 'label="' in line:
            # Ensure all nodes use box shape and consistent styling
            line = line.replace('shape=circle', 'shape=box')
            line = line.replace('fillcolor="[^"]*"', 'fillcolor="#ffffff"')
            line = line.replace('fontcolor="[^"]*"', 'fontcolor="#000000"')
            styled_output.append(line)
        else:
            line = line.replace('shape=circle', 'shape=box')
            styled_output.append(line)

    styled_output.append("}")
    
    # Write final output
    Path(output_path).write_text("\n".join(styled_output), encoding="utf-8")
    try:
        Path(temp_file).unlink(missing_ok=True)
    except Exception:
        pass

# --------------- Main execution --------------- #
if __name__ == "__main__":
    # Read input and generate visualization
    try:
        with open("trace.txt", encoding="utf-8") as file:
            trace_content = file.readlines()
        parse_prolog_trace_group_by_pair(trace_content, "tree.dot")
    except FileNotFoundError:
        print("Error: trace.txt file not found")
    except Exception as e:
        print(f"Error processing file: {e}")