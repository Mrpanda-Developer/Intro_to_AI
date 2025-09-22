from treelib import Tree
from pathlib import Path
import re
import graphviz

def parse_prolog_backtracking_tree(trace_lines):
    """
    Parse Prolog trace output and create a hierarchical search tree showing the complete backtracking process.
    
    The tree demonstrates Prolog's depth-first search with backtracking, showing:
    - Call entries as new nodes in the search tree
    - Exit entries as successful completions  
    - Fail entries as dead ends causing backtracking
    - Redo entries as backtracking points
    
    Args:
        trace_lines (list): Lines from the Prolog trace output
        
    Returns:
        tuple: (Tree object, list of successful matches)
    """
    tree = Tree()
    tree.create_node("Prolog Actor-Director Search", "root")
    
    # Stack management for tracking search context
    # Each stack element: (parent_node_id, search_context, depth_level)
    # Used to maintain parent-child relationships during backtracking
    stack = [("root", "Start", 0)]
    node_counter = 0
    current_actor_search = None
    
    # Track successful actor-director matches for result reporting
    successful_matches = []
    
    # Track variable bindings to show how search progresses
    variable_bindings = {}  # Maps variables to their current values
    
    for i, line in enumerate(trace_lines):
        line = line.strip()
        if not line:
            continue
            
        node_counter += 1
        node_id = f"n{node_counter}"
        
        # --- VARIABLE BINDING TRACKING (IMPROVEMENT #1) ---
        # Track how Prolog binds variables during unification
        if "=" in line and not line.startswith(("Call:", "Exit:", "Fail:", "Redo:")):
            # This is a variable binding line like "A1 = sam_raimi"
            parts = line.split("=")
            if len(parts) == 2:
                var_name = parts[0].strip()
                var_value = parts[1].strip()
                variable_bindings[var_name] = var_value
                # Create a special node to show variable binding
                binding_node_id = f"bind_{node_counter}"
                parent_id, context, depth = stack[-1]
                label = f"Binding: {var_name} = {var_value}"
                tree.create_node(label, binding_node_id, parent=parent_id)
            continue
        
        # --- ACTOR SEARCH PATTERN ---
        # Handle actor/3 predicate calls and exits
        if line.startswith("Call:actor("):
            # New actor search starting - match all three parameters including variables
            match = re.match(r"Call:actor\(([^,]+),\s*([^,]+),\s*([^)]+)\)", line)
            if match:
                movie_var, actor_var, role_var = match.groups()
                parent_id, context, depth = stack[-1]
                
                # Show current variable bindings in the label
                bindings_text = ""
                if actor_var in variable_bindings:
                    bindings_text = f" ({variable_bindings.get(actor_var, actor_var)})"
                
                label = f"Call: Find actor{bindings_text}"
                tree.create_node(label, node_id, parent=parent_id)
                stack.append((node_id, f"ActorSearch:{actor_var}", depth + 1))
                
        elif line.startswith("Exit:actor("):
            # Actor found successfully - extract concrete values
            match = re.match(r"Exit:actor\(([^,]+),\s*([^,]+),\s*([^)]+)\)", line)
            if match:
                movie, actor, role = match.groups()
                parent_id, context, depth = stack[-1]
                
                # Update variable bindings with concrete values
                variable_bindings["M"] = movie  # Movie variable binding
                variable_bindings["A"] = actor  # Actor variable binding
                
                label = f"Exit: {actor} in '{movie}' as {role}"
                tree.create_node(label, node_id, parent=parent_id)
                # Keep stack position for director search
                
        # --- DIRECTOR SEARCH PATTERN ---  
        # Handle director/2 predicate calls and exits
        elif line.startswith("Call:director("):
            # Starting director search - match both parameters
            match = re.match(r"Call:director\(([^,]+),\s*([^)]+)\)", line)
            if match:
                movie_var, director_var = match.groups()
                parent_id, context, depth = stack[-1]
                
                # Show current bindings for better context
                bindings_text = ""
                if movie_var in variable_bindings:
                    bindings_text = f" for {variable_bindings[movie_var]}"
                
                label = f"Call: Find director{bindings_text}"
                tree.create_node(label, node_id, parent=parent_id)
                stack.append((node_id, f"DirectorSearch:{director_var}", depth + 1))
                
        elif line.startswith("Exit:director("):
            # Director found successfully
            match = re.match(r"Exit:director\(([^,]+),\s*([^)]+)\)", line)
            if match:
                movie, director = match.groups()
                parent_id, context, depth = stack[-1]
                
                # Update director variable binding
                variable_bindings["D"] = director
                
                label = f"Exit: Director {director} for '{movie}'"
                tree.create_node(label, node_id, parent=parent_id)
                
        # --- COMPARISON PATTERN (Actor vs Director) ---
        # Handle unification tests (A1 @> A2 pattern)
        elif line.startswith("Call:") and "\\=" in line:
            # Actor-Director comparison using @> operator
            match = re.match(r"Call:([^\\]+)\\([^=]+)=([^\\]+)", line)
            if match:
                left, op, right = match.groups()
                parent_id, context, depth = stack[-1]
                
                # Show current values for better understanding
                left_val = variable_bindings.get(left, left)
                right_val = variable_bindings.get(right, right)
                
                label = f"Call: Compare {left_val} @> {right_val}"
                tree.create_node(label, node_id, parent=parent_id)
                stack.append((node_id, f"Comparison:{left_val}_{right_val}", depth + 1))
                
        elif line.startswith("Exit:") and "\\=" in line:
            # Successful comparison
            match = re.match(r"Exit:([^\\]+)\\([^=]+)=([^\\]+)", line)
            if match:
                left, op, right = match.groups()
                parent_id, context, depth = stack[-1]
                
                left_val = variable_bindings.get(left, left)
                right_val = variable_bindings.get(right, right)
                
                label = f"Exit: MATCH {left_val} @> {right_val}"
                tree.create_node(label, node_id, parent=parent_id)
                successful_matches.append((left_val, right_val))
                
                # Pop comparison level from stack
                if len(stack) > 1:
                    stack.pop()
                    
        # --- FAILURE AND BACKTRACKING ---
        # Handle search failures that trigger backtracking
        elif line.startswith("Fail:director("):
            # Director search failed
            parent_id, context, depth = stack[-1]
            
            # Determine failure context for better labeling
            if "_944" in line:
                label = "Fail: No matching director found for actor"
                # Backtrack from director search level
                if len(stack) > 1 and "DirectorSearch:" in context:
                    stack.pop()
            else:
                label = "Fail: Director search failed"
                
            tree.create_node(label, node_id, parent=parent_id)
            
        elif line.startswith("Redo:director("):
            # Backtracking to try alternative directors
            parent_id, context, depth = stack[-1]
            label = "Redo: Backtrack - try alternative director"
            tree.create_node(label, node_id, parent=parent_id)
            
        elif line.startswith("Redo:actor("):
            # Backtracking to try alternative actors
            parent_id, context, depth = stack[-1]
            label = "Redo: Backtrack - try alternative actor"
            tree.create_node(label, node_id, parent=parent_id)
            
            # Pop back to actor search level, maintaining proper search order
            while len(stack) > 1 and not stack[-1][1].startswith("ActorSearch:"):
                stack.pop()
            if len(stack) > 1:
                stack.pop()
                
        # --- SUCCESSFUL ACTOR-DIRECTOR MATCHES ---
        elif line.startswith("Exit:director(") and "_944" in line:
            # Found a director that matches our criteria
            match = re.match(r"Exit:director\(([^,]+),\s*([^)]+)\)", line)
            if match:
                movie, director = match.groups()
                parent_id, context, depth = stack[-1]
                label = f"Exit: Candidate {director} directed '{movie}'"
                tree.create_node(label, node_id, parent=parent_id)
    
    return tree, successful_matches


def create_movie_relationship_tree(trace_lines):
    """
    Create an alternative tree view showing movie-actor-director relationships.
    
    This provides a cleaner, more focused visualization of the successful relationships
    found during the search, grouping by movie for better readability.
    
    Args:
        trace_lines (list): Lines from the Prolog trace output
        
    Returns:
        Tree: Tree object with movie-centric relationships
    """
    tree = Tree()
    tree.create_node("Movie Relationships Found", "root")
    
    movies = {}
    node_counter = 0
    
    for line in trace_lines:
        # Extract successful actor facts
        if line.startswith("Exit:actor("):
            match = re.match(r"Exit:actor\(([^,]+),\s*([^,]+),\s*([^)]+)\)", line)
            if match:
                movie, actor, role = match.groups()
                
                # Create movie node if it doesn't exist
                if movie not in movies:
                    node_counter += 1
                    movies[movie] = f"movie_{node_counter}"
                    tree.create_node(f"Movie: {movie}", movies[movie], parent="root")
                
                # Add actor to movie
                node_counter += 1
                tree.create_node(f"Actor: {actor} as {role}", f"actor_{node_counter}", parent=movies[movie])
        
        # Extract successful director facts
        elif line.startswith("Exit:director(") and not "_944" in line:
            match = re.match(r"Exit:director\(([^,]+),\s*([^)]+)\)", line)
            if match:
                movie, director = match.groups()
                
                # Add director to existing movie node
                if movie in movies:
                    node_counter += 1
                    tree.create_node(f"Director: {director}", f"dir_{node_counter}", parent=movies[movie])
    
    return tree


def export_tree_to_dot(tree, filename, title):
    """
    Export tree to Graphviz DOT format with enhanced styling and proper search order.
    
    IMPROVEMENT #2: Enhanced visualization with:
    - Color coding based on node type (call/exit/fail/redo)
    - Proper search order maintenance
    - Professional styling for readability
    
    Args:
        tree (Tree): treelib Tree object to export
        filename (str): Output DOT filename
        title (str): Graph title
    """
    dot_content = [
        f'digraph "{title}" {{',
        '  rankdir=TB;',  # Top-to-bottom layout
        '  nodesep=0.4; ranksep=0.6;',  # Spacing between nodes and ranks
        '  node [shape=box, style="rounded,filled", fontname="Arial", fontsize=10];',
        '  edge [arrowhead=vee, arrowsize=0.8];',
        '',
    ]
    
    # --- ENHANCED NODE STYLING WITH COLOR CODING ---
    for node in tree.all_nodes():
        label = node.tag.replace('"', '\\"')
        
        # Color coding based on node content type
        color = "#e1f5fe"  # Default blue for neutral nodes
        
        if "Call:" in label:
            color = "#f3e5f5"  # Purple for search calls
        elif "Exit:" in label and "MATCH" in label:
            color = "#e8f5e8"  # Green for successful matches
        elif "Exit:" in label:
            color = "#dcedc8"  # Light green for successful exits
        elif "Fail:" in label:
            color = "#ffebee"  # Red for failures
        elif "Redo:" in label:
            color = "#fff3e0"  # Orange for backtracking
        elif "Binding:" in label:
            color = "#e0f2f1"  # Teal for variable bindings
        
        dot_content.append(f'  "{node.identifier}" [label="{label}", fillcolor="{color}"];')
    
    dot_content.append('')
    
    # --- MAINTAIN PROLOG SEARCH ORDER IN EDGES ---
    # Group children by their creation order to maintain search sequence
    for node in tree.all_nodes():
        children = tree.children(node.identifier)
        if children:
            # Add ordering constraint to maintain left-to-right search order
            child_ids = [f'"{child.identifier}"' for child in children]
            dot_content.append(f'  {{rank=same; {", ".join(child_ids)}}}')
            
            for child in children:
                dot_content.append(f'  "{node.identifier}" -> "{child.identifier}";')
    
    dot_content.append('}')
    
    # Write to file with UTF-8 encoding for special characters
    Path(filename).write_text('\n'.join(dot_content), encoding='utf-8')


def create_simple_text_tree(tree, filename):
    """
    Create a simple text representation of the tree for quick reference.
    
    Args:
        tree (Tree): treelib Tree object
        filename (str): Output text filename
    """
    content = ["Search Tree Text Representation:", "=" * 40]
    
    for node in tree.expand_tree():
        depth = tree.depth(node)
        node_obj = tree.get_node(node)
        indent = "  " * depth
        content.append(f"{indent}{node_obj.tag}")
    
    Path(filename).write_text('\n'.join(content), encoding='utf-8')


def main():
    """
    Main execution function that coordinates the tree parsing and visualization.
    
    Reads the Prolog trace, creates multiple tree visualizations, and generates
    both DOT files and PNG images for analysis.
    """
    try:
        # Read and validate input file
        with open("trace.txt", encoding="utf-8") as f:
            lines = f.readlines()
        
        print("Processing Prolog trace file...")
        print(f"Read {len(lines)} lines from trace.txt")
        
        # Display sample of trace for verification
        print("\nSample of trace content (first 10 lines):")
        for i, line in enumerate(lines[:10]):
            print(f"{i+1:2d}: {line.strip()}")
        
        # --- CREATE BACKTRACKING SEARCH TREE ---
        print("\n" + "="*50)
        print("Creating backtracking search tree visualization...")
        search_tree, matches = parse_prolog_backtracking_tree(lines)
        
        # Export to multiple formats
        export_tree_to_dot(search_tree, "backtracking_search.dot", "Prolog Backtracking Search Tree")
        create_simple_text_tree(search_tree, "backtracking_search.txt")
        
        # --- CREATE MOVIE RELATIONSHIP TREE ---
        print("Creating movie relationship tree...")
        movie_tree = create_movie_relationship_tree(lines)
        export_tree_to_dot(movie_tree, "movie_relationships.dot", "Movie-Actor-Director Relationships")
        create_simple_text_tree(movie_tree, "movie_relationships.txt")
        
        # --- GENERATE PNG VISUALIZATIONS ---
        print("\nGenerating PNG images from DOT files...")
        for dot_file in ["backtracking_search.dot", "movie_relationships.dot"]:
            try:
                with open(dot_file, 'r', encoding='utf-8') as f:
                    dot_content = f.read()
                
                # Create graphviz graph and render as PNG
                graph = graphviz.Source(dot_content)
                output_base = dot_file.replace('.dot', '')
                graph.render(filename=output_base, format='png', cleanup=True)
                print(f"✅ Generated: {output_base}.png")
                
            except Exception as e:
                print(f"❌ Error generating PNG for {dot_file}: {e}")
        
        # --- DISPLAY RESULTS SUMMARY ---
        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)
        
        print(f"\nSearch Results Summary:")
        print(f"• Found {len(matches)} successful actor-director matches")
        for i, (left, right) in enumerate(matches, 1):
            print(f"  {i}. {left} @> {right}")
        
        print(f"\nTree Statistics:")
        print(f"• Backtracking tree: {search_tree.size()} nodes, depth {search_tree.depth()}")
        print(f"• Movie relationship tree: {movie_tree.size()} nodes")
        
        print(f"\nGenerated Files:")
        print(f"• backtracking_search.dot - Graphviz file with search process")
        print(f"• backtracking_search.png - Visual search tree diagram") 
        print(f"• backtracking_search.txt - Text representation")
        print(f"• movie_relationships.dot - Graphviz file with relationships")
        print(f"• movie_relationships.png - Visual relationship diagram")
        print(f"• movie_relationships.txt - Text representation")
        
        # Display text tree for quick verification
        print(f"\nBacktracking Tree Structure (text):")
        search_tree.show(line_type="ascii")
        
    except FileNotFoundError:
        print("❌ Error: trace.txt file not found in current directory")
        print("Please ensure trace.txt exists in the same folder as this script")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()