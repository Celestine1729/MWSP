# breadth_first_search.py
import networkx as nx

def bfs_edges(G, node_labels, source, depth_limit=None):
    """
    Custom BFS implementation that respects node labels
    
    Args:
        G: NetworkX graph
        node_labels: List of node labels
        source: Starting node
        depth_limit: Maximum BFS depth
        
    Yields:
        (u, v) edge tuples
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