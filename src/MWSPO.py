"""MWSPO.py - Optimized Multiscale Wasserstein Shortest Path Graph Kernel

This module implements a GPU-accelerated graph kernel for graph classification tasks.
It combines multiscale structural features with optimal transport distances to compute
similarities between graphs, enabling high-accuracy classification with SVM.

Key components:
1. Multiscale BFS-based feature extraction
2. Shortest path representation
3. Wasserstein distance computation
4. Heat kernel transformation
5. SVM classification

Designed for:
- Large-scale graph datasets (1000+ graphs)
- GPU cluster environments (Tesla V100, EPYC CPUs)
- Memory-constrained execution

Author: Celestine
Contact: https://t.me/celestine_1729
Date: July 2025
"""

import os
import sys
import time
import gc
import argparse
import numpy as np
import torch
import cugraph
import cudf
import networkx as nx
from gensim import corpora
import gensim
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import multiprocessing
import psutil
from geomloss import SamplesLoss

# Global configuration (tuned for 1000-1200 graph datasets)
USE_GPU = torch.cuda.is_available()
"""bool: Automatically detect GPU availability"""

GPU_DEVICES = list(range(torch.cuda.device_count())) if USE_GPU else []
"""list: Available GPU device IDs"""

MAX_NODES = 180
"""int: Maximum nodes per graph for subsampling (memory control)"""

SLICED_PROJECTIONS = 120
"""int: Number of projections for sliced Wasserstein approximation"""

GPU_BATCH_SIZE = 96
"""int: Number of distance computations per GPU batch"""

MAXH_LIMIT = 3
"""int: Maximum BFS depth for feature extraction"""

DEPTH_LIMIT = 2
"""int: Maximum shortest path depth"""

class ResourceMonitor:
    """Context manager for precise resource tracking
    
    Tracks execution time, RAM usage, and peak GPU memory consumption
    within its context block.
    
    Attributes:
        name (str): Identifier for the monitored block
        start_time (float): Timestamp when monitoring began
        start_mem (int): Initial RAM usage in bytes
        duration (float): Total execution time in seconds
        ram_used (float): RAM consumption in GB
    """
    
    def __init__(self, name):
        """
        Args:
            name: Descriptive name for the monitored code block
        """
        self.name = name
        
    def __enter__(self):
        """Start resource monitoring"""
        self.start_time = time.time()
        self.start_mem = psutil.virtual_memory().used
        if USE_GPU:
            torch.cuda.reset_peak_memory_stats()
        return self
        
    def __exit__(self, *args):
        """Finalize measurements and print report"""
        self.duration = time.time() - self.start_time
        self.ram_used = (psutil.virtual_memory().used - self.start_mem) / (1024**3)
        
        gpu_mem = 0
        if USE_GPU:
            for device in GPU_DEVICES:
                mem = torch.cuda.max_memory_allocated(device) / (1024**3)
                gpu_mem = max(gpu_mem, mem)
                torch.cuda.reset_peak_memory_stats(device)
        
        print(f"[RESOURCE] {self.name:<22} | {self.duration:7.1f}s | "
              f"RAM: {self.ram_used:5.1f}GB | GPU: {gpu_mem:5.1f}GB")

class GraphContainer:
    """Memory-efficient graph representation
    
    Optimized for large-scale processing with minimal overhead.
    
    Attributes:
        label (int): Graph class label
        g (nx.Graph): NetworkX graph object
        node_tags (list): Node labels or attributes
        node_features (np.ndarray): Feature matrix (nodes x features)
    """
    __slots__ = ['label', 'g', 'node_tags', 'node_features']
    
    def __init__(self, g, label, node_tags=None):
        """
        Args:
            g: NetworkX graph object
            label: Graph class label
            node_tags: List of node labels/tags
        """
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.node_features = None

def load_data(dataset, degree_as_tag=False):
    """Load and preprocess graph dataset
    
    Parses dataset from text format, builds feature matrices, and handles
    memory-efficient graph representations.
    
    Args:
        dataset: Name of dataset directory (must exist in datasets/)
        degree_as_tag: Whether to use node degree as tags
        
    Returns:
        tuple: (list of GraphContainer objects, number of classes)
        
    Raises:
        SystemExit: If dataset file not found
    """
    graphs = []
    label_dict = {}
    path = f'datasets/{dataset}/{dataset}.txt'
    
    if not os.path.exists(path):
        sys.exit(f"Error: Missing dataset at {path}")
    
    print(f"Loading {dataset} from {path}")
    with open(path, 'r') as f:
        n_graphs = int(f.readline().strip())
        for _ in range(n_graphs):
            n_nodes, label = map(int, f.readline().split())
            if label not in label_dict:
                label_dict[label] = len(label_dict)
            
            G = nx.Graph()
            tags = []
            for j in range(n_nodes):
                G.add_node(j)
                row = f.readline().split()
                n_edges = int(row[1])
                tags.append(int(row[0]))
                for k in range(2, 2 + n_edges):
                    G.add_edge(j, int(row[k]))
            
            graphs.append(GraphContainer(G, label, tags))
    
    # Feature engineering
    all_tags = set(tag for g in graphs for tag in g.node_tags)
    tag2idx = {tag: i for i, tag in enumerate(sorted(all_tags))}
    
    for graph in graphs:
        if degree_as_tag:
            graph.node_tags = [d for _, d in graph.g.degree()]
        
        # Build efficient one-hot encoded features
        n_nodes = len(graph.node_tags)
        n_features = len(all_tags)
        graph.node_features = np.zeros((n_nodes, n_features))
        for i, tag in enumerate(graph.node_tags):
            graph.node_features[i, tag2idx[tag]] = 1
    
    print(f"Loaded {len(graphs)} graphs | Classes: {len(label_dict)} | "
          f"Node features: {len(all_tags)}")
    return graphs, len(label_dict)

def sliced_wasserstein(X, Y, projections=100):
    """Compute sliced Wasserstein distance approximation
    
    GPU-accelerated linear-time approximation of Wasserstein distance
    using random projections.
    
    Args:
        X: Tensor of shape (n, d)
        Y: Tensor of shape (m, d)
        projections: Number of random projections
        
    Returns:
        float: Approximate Wasserstein distance
    """
    device = X.device
    projs = torch.randn(projections, X.shape[1], device=device)
    projs /= torch.norm(projs, dim=1, keepdim=True)
    
    X_proj = torch.sort(X @ projs.T, dim=0)[0]
    Y_proj = torch.sort(Y @ projs.T, dim=0)[0]
    return torch.mean(torch.abs(X_proj - Y_proj))

def compute_wasserstein(embeddings):
    """Compute pairwise Wasserstein distances between graph embeddings
    
    Hybrid computation strategy:
    - Uses exact Sinkhorn for small graph pairs
    - Uses sliced Wasserstein approximation for large graph pairs
    - Automatic subsampling for graphs exceeding MAX_NODES
    - Parallelized across multiple GPUs
    
    Args:
        embeddings: List of embedding matrices per graph
        
    Returns:
        np.ndarray: Symmetric distance matrix (n_graphs x n_graphs)
    """
    n = len(embeddings)
    M = np.zeros((n, n))
    
    # Node subsampling - critical for memory management
    emb_subsampled = []
    for emb in embeddings:
        if len(emb) > MAX_NODES:
            idx = np.random.choice(len(emb), MAX_NODES, replace=False)
            emb_subsampled.append(emb[idx])
        else:
            emb_subsampled.append(emb)
    
    # Create comparison pairs (upper triangle)
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def process_batch(batch, device_id):
        """Process batch of distance computations on specified GPU
        
        Args:
            batch: List of (i,j) index pairs
            device_id: GPU device ID
            
        Returns:
            list: (i, j, distance) tuples
        """
        device = torch.device(f'cuda:{device_id}')
        results = []
        for i, j in batch:
            X = torch.tensor(emb_subsampled[i], dtype=torch.float32, device=device)
            Y = torch.tensor(emb_subsampled[j], dtype=torch.float32, device=device)
            
            # Algorithm selection heuristic
            if X.shape[0] * Y.shape[0] > 1e6:  # Large matrix
                dist = sliced_wasserstein(X, Y, SLICED_PROJECTIONS).item()
            else:
                dist = SamplesLoss("sinkhorn", p=2, blur=0.01)(X, Y).item()
            results.append((i, j, dist))
        return results
    
    # GPU batch processing configuration
    batch_size = min(GPU_BATCH_SIZE, len(pairs) // max(1, len(GPU_DEVICES)))
    batches = [pairs[i:i+batch_size] for i in range(0, len(pairs), batch_size)]
    
    print(f"Computing {len(pairs)} distances on {len(GPU_DEVICES)} GPUs "
          f"(batch_size={batch_size})")
    
    all_results = []
    if USE_GPU and GPU_DEVICES:
        # Distribute batches across available GPUs
        batches_per_gpu = (len(batches) + len(GPU_DEVICES) - 1) // len(GPU_DEVICES)
        with Parallel(n_jobs=len(GPU_DEVICES), backend="threading") as parallel:
            results = parallel(
                delayed(process_batch)(
                    batches[i:i+batches_per_gpu],
                    GPU_DEVICES[idx % len(GPU_DEVICES)]
                ) for idx, i in enumerate(range(0, len(batches), batches_per_gpu))
            all_results = [res for batch in results for res in batch]
    else:
        # CPU fallback mode
        device = torch.device('cpu')
        for i, j in pairs:
            X = torch.tensor(emb_subsampled[i], dtype=torch.float32, device=device)
            Y = torch.tensor(emb_subsampled[j], dtype=torch.float32, device=device)
            dist = SamplesLoss("sinkhorn", p=2, blur=0.01)(X, Y).item()
            all_results.append((i, j, dist))
    
    # Build symmetric distance matrix
    for i, j, dist in all_results:
        M[i, j] = dist
        M[j, i] = dist
    
    # Ensure numerical stability
    np.fill_diagonal(M, 0)
    M = (M + M.T) / 2  # Force symmetry
    return M

def gpu_bfs(G, source, depth):
    """GPU-accelerated BFS tree extraction
    
    Uses cuGraph for parallel breadth-first search on GPU.
    
    Args:
        G: NetworkX graph
        source: Starting node
        depth: Maximum BFS depth
        
    Returns:
        list: Edge list of BFS tree
    """
    edges = list(G.edges())
    df = cudf.DataFrame({
        'src': [e[0] for e in edges],
        'dst': [e[1] for e in edges]
    })
    cug = cugraph.Graph()
    cug.from_cudf_edgelist(df, 'src', 'dst')
    try:
        tree = cugraph.bfs(cug, source, depth_limit=depth)
        return [(int(row['predecessor']), int(row['vertex'])) 
                for _, row in tree.iterrows() if row['predecessor'] >= 0]
    except:
        return []  # Fallback for isolated nodes

def build_embeddings(graphs, maxh, depth):
    """Construct multiscale graph embeddings
    
    Pipeline:
    1. Hierarchical feature extraction via BFS
    2. Shortest path enumeration
    3. Bag-of-Words embedding generation
    4. Multiscale feature concatenation
    
    Args:
        graphs: List of GraphContainer objects
        maxh: Maximum BFS depth
        depth: Maximum path depth
        
    Returns:
        list: Embedding matrices per graph
    """
    n_graphs = len(graphs)
    all_labels = [g.node_tags for g in graphs]
    
    # Hierarchical feature extraction
    for level in range(1, maxh):
        new_labels = []
        all_trees = []
        tree_dict = {}
        
        with ResourceMonitor(f"BFS Level {level}"):
            for gidx, graph in enumerate(graphs):
                G = graph.g
                labels = all_labels[gidx]
                trees = []
                
                # GPU acceleration for large graphs
                if USE_GPU and len(G) > 100:
                    for node in range(len(G)):
                        edges = gpu_bfs(G, node, level)
                        tree_str = ",".join(f"{labels[u]},{labels[v]}" for u,v in edges)
                        trees.append(tree_str)
                else:
                    for node in range(len(G)):
                        edges = list(bfs.bfs_edges(G, labels, source=node, depth_limit=level))
                        tree_str = ",".join(f"{labels[u]},{labels[v]}" for u,v in edges)
                        trees.append(tree_str)
                
                all_trees.extend(trees)
                tree_dict.update({t: i for i, t in enumerate(set(trees))})
        
        # Update labels
        with ResourceMonitor(f"Label Mapping {level}"):
            for gidx in range(n_graphs):
                start = sum(len(graphs[i].g) for i in range(gidx))
                end = start + len(graphs[gidx].g)
                new_labels.append([tree_dict[t] for t in all_trees[start:end]])
        
        all_labels = new_labels
        gc.collect()
    
    # Path extraction
    all_paths = []
    with ResourceMonitor("Path Extraction"):
        for graph, labels in zip(graphs, all_labels):
            G = graph.g
            paths = []
            
            if USE_GPU and len(G) > 50:
                try:
                    edges = list(G.edges())
                    df = cudf.DataFrame({
                        'src': [e[0] for e in edges],
                        'dst': [e[1] for e in edges]
                    })
                    cug = cugraph.Graph()
                    cug.from_cudf_edgelist(df, 'src', 'dst')
                    
                    for node in range(len(G)):
                        sssp = cugraph.sssp(cug, node, depth)
                        node_paths = [str(node)]
                        for target in sssp['vertex'].to_array():
                            if target == node: continue
                            path = cugraph.utils.get_traversed_path_list(sssp, target)
                            if path:
                                path_str = ",".join(map(str, path))
                                node_paths.append(path_str)
                        paths.append(node_paths)
                except:
                    USE_GPU = False  # Fallback to CPU
            else:
                for node in range(len(G)):
                    node_paths = [str(node)]
                    try:
                        for target in nx.single_source_shortest_path_length(G, node, depth):
                            if target == node: continue
                            path = nx.shortest_path(G, node, target)
                            path_str = ",".join(map(str, path))
                            node_paths.append(path_str)
                    except:
                        pass
                    paths.append(node_paths)
            
            all_paths.append(paths)
    
    # Embedding generation
    embeddings = []
    with ResourceMonitor("Embedding Generation"):
        for level in range(maxh):
            path_reprs = []
            for gidx in range(n_graphs):
                labels = all_labels[gidx]
                for i, paths in enumerate(all_paths[gidx]):
                    reprs = []
                    for path in paths:
                        nodes = path.split(",")
                        reprs.append(",".join(str(labels[int(n)]) for n in nodes))
                    path_reprs.append(reprs)
            
            # Build Bag-of-Words embeddings
            dictionary = corpora.Dictionary(path_reprs)
            corpus = [dictionary.doc2bow(paths) for paths in path_reprs]
            bow = gensim.matutils.corpus2dense(corpus, len(dictionary)).T
            embeddings.append(bow)
        
        # Concatenate multiscale features
        full_emb = np.hstack(embeddings)
        
        # Split by graph
        graph_embs = []
        idx = 0
        for graph in graphs:
            n = len(graph.g)
            graph_embs.append(full_emb[idx:idx+n])
            idx += n
        
        print(f"Embedding dimension: {full_emb.shape[1]}")
        return graph_embs

def run_experiment(dataset, maxh, depth):
    """End-to-end experiment pipeline
    
    Workflow:
    1. Load and preprocess data
    2. Generate multiscale embeddings
    3. Compute Wasserstein distance matrix
    4. Build kernel matrix
    5. Train and evaluate SVM classifier
    
    Args:
        dataset: Dataset name
        maxh: Maximum BFS depth
        depth: Maximum path depth
        
    Returns:
        tuple: (mean accuracy, standard deviation)
    """
    # Load and preprocess data
    with ResourceMonitor("Data Loading"):
        degree_tag = dataset in ['IMDB-BINARY', 'REDDIT-BINARY']
        graphs, n_classes = load_data(dataset, degree_tag)
        labels = np.array([g.label for g in graphs])
    
    # Feature extraction pipeline
    with ResourceMonitor("Feature Extraction"):
        embeddings = build_embeddings(graphs, maxh, depth)
    
    # Distance computation
    with ResourceMonitor("Wasserstein Distance"):
        D = compute_wasserstein(embeddings)
    
    # Kernel matrix construction
    with ResourceMonitor("Kernel Construction"):
        gamma = 0.1 / np.median(D[D > 0])  # Adaptive kernel scaling
        K = np.exp(-gamma * D)
    
    # Classification and evaluation
    with ResourceMonitor("SVM Training"):
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        accs = []
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(K, labels)):
            model = SVC(C=10, kernel='precomputed', max_iter=10000)
            model.fit(K[train_idx][:, train_idx], labels[train_idx])
            preds = model.predict(K[test_idx][:, train_idx])
            acc = accuracy_score(labels[test_idx], preds)
            accs.append(acc)
            print(f"Fold {fold+1}: Accuracy = {acc:.4f}")
        
        mean_acc = np.mean(accs) * 100
        std_acc = np.std(accs) * 100
        print(f"\nFinal Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    
    return mean_acc, std_acc

if __name__ == "__main__":
    # Command-line interface
    parser = argparse.ArgumentParser(
        description="Multiscale Wasserstein Shortest Path Graph Kernel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("--maxh", type=int, default=MAXH_LIMIT, 
                       help="Max BFS depth for feature extraction")
    parser.add_argument("--depth", type=int, default=DEPTH_LIMIT,
                       help="Max shortest path depth")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(f"MWSPO Graph Kernel Experiment: {args.dataset}")
    print(f"Configuration: maxh={args.maxh}, depth={args.depth}")
    print(f"Hardware: {len(GPU_DEVICES)} GPUs, {multiprocessing.cpu_count()} CPUs")
    print(f"Memory: {psutil.virtual_memory().total/(1024**3):.1f}GB RAM")
    print("="*70)
    
    # Safety checks
    if len(GPU_DEVICES) == 0:
        print("WARNING: No GPUs detected - falling back to CPU mode")
    
    # Enforce safety limits
    run_experiment(
        dataset=args.dataset,
        maxh=min(args.maxh, MAXH_LIMIT),
        depth=min(args.depth, DEPTH_LIMIT)
    )