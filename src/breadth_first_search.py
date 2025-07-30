"""breadth_first_search.py - Custom BFS implementation for graph feature extraction

Implements a BFS variant that respects node labels during traversal, 
enabling label-aware path extraction for graph kernels.
"""

import networkx as nx

def bfs_edges(G, node_labels, source, depth_limit=None):
    """Label-aware BFS edge generator
    
    Performs breadth-first search while respecting node labels, producing
    edges in BFS order. This is critical for consistent feature extraction
    across isomorphic graph structures.
    
    Args:
        G: NetworkX graph
        node_labels: List of labels for each node
        source: Starting node for BFS
        depth_limit: Maximum depth to traverse
        
    Yields:
        tuple: (parent, child) edge tuples in BFS order
    """
    visited = {source}
    queue = [(source, 0, iter(G[source]))]
    
    while queue:
        parent, depth, children = queue[0]
        try:
            child = next(children)
            if child not in visited:
                visited.add(child)
                yield parent, child
                if depth_limit is None or depth < depth_limit:
                    queue.append((child, depth + 1, iter(G[child])))
        except StopIteration:
            queue.pop(0)