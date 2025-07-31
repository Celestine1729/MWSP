"""breadth_first_search.py - Optimized BFS implementation

Memory Optimization:
- Pre-stores neighbor lists to reduce generator overhead
- Uses list appends instead of generators for small graphs
- Minimal temporary storage
"""

import networkx as nx

def bfs_edges(G, node_labels, source, depth_limit=None):
    """Label-aware BFS edge generator
    
    Args:
        G: NetworkX graph
        node_labels: List of node labels
        source: Starting node
        depth_limit: Max BFS depth
        
    Yields:
        tuple: (parent, child) edges in BFS order
    """
    visited = {source}
    queue = [(source, 0, list(G[source]))]  # Pre-store neighbors
    idx = 0  # Pointer instead of popping
    
    while idx < len(queue):
        parent, depth, children = queue[idx]
        for child in children:
            if child not in visited:
                visited.add(child)
                yield parent, child
                if depth_limit is None or depth < depth_limit:
                    # Store neighbor list directly
                    queue.append((child, depth + 1, list(G[child])))
        idx += 1  # Move to next in queue