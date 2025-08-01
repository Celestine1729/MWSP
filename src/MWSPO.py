"""MWSPO.py - Memory-Optimized Multiscale Wasserstein Shortest Path Graph Kernel
    July 2025 , by Celestine 
    contact : Celestine1729@proton.me

    "" One day I'll give you my heart ,When it's not in two" - PVRIS, Old Wounds

Designed for cluster execution with:
- 256GB RAM
- 2x Tesla V100 GPUs
- Large datasets (1000+ graphs) both for labelled and unlabeled graphs
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
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from breadth_first_search import bfs_edges
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import multiprocessing
import psutil
from geomloss import SamplesLoss
import hashlib

# ======================== GLOBAL CONFIGURATION ========================
# This section defines global constants and configurations for the MWSPO kernel , something like defines in C.
USE_GPU = torch.cuda.is_available()
"""bool: Automatically detect GPU availability for acceleration"""

GPU_DEVICES = list(range(torch.cuda.device_count())) if USE_GPU else []
"""list: Available GPU device IDs for parallel processing"""

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

FEATURE_DIM = 2**20  # 1,048,576 features 
"""int: Fixed feature dimension for hashing trick (prevents memory explosion)"""

# ========================== RESOURCE MONITOR ==========================
class ResourceMonitor:
    """Tracks execution resources within context block
    
    Usage:
        with ResourceMonitor("Process Name"):
            # code block to monitor
        # Automatically prints resource report
    
    Measures:
    - Execution time
    - RAM consumption (GB)
    - Peak GPU memory (GB)
    """
    
    def __init__(self, name):
        self.name = name
        
    def __enter__(self):
        self.start_time = time.time()
        self.start_mem = psutil.virtual_memory().used
        if USE_GPU:
            torch.cuda.reset_peak_memory_stats()
        return self
        
    def __exit__(self, *args):
        self.duration = time.time() - self.start_time
        self.ram_used = (psutil.virtual_memory().used - self.start_mem) / (1024**3)
        
        gpu_mem = 0
        if USE_GPU:
            for device in GPU_DEVICES:
                mem = torch.cuda.max_memory_allocated(device) / (1024**3)
                gpu_mem = max(gpu_mem, mem)
        
        # Monitor current process memory
        proc = psutil.Process()
        current_ram = proc.memory_info().rss / (1024**3)
        
        print(f"[RESOURCE] {self.name:<22} | {self.duration:7.1f}s | "
              f"RAM: +{self.ram_used:5.1f}GB (Total: {current_ram:5.1f}GB) | "
              # confess to your crush ,wrost thing she can say is no.
              f"GPU: {gpu_mem:5.1f}GB")

# ========================== GRAPH CONTAINER ==========================
# This class provides an efficient representation of a graph with minimal memory usage. 
class GraphContainer:
    """Efficient graph representation with minimal overhead
    
    Attributes:
        label (int): Graph class label
        g (nx.Graph): NetworkX graph object
        node_tags (list): Node labels/attributes
        node_features (np.ndarray): Feature matrix (nodes x features)
    """
    __slots__ = ['label', 'g', 'node_tags', 'node_features']
    
    def __init__(self, g, label, node_tags=None):
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.node_features = None

# ========================== DATA LOADING ==========================
def load_data(dataset, degree_as_tag=False):
    """Load and preprocess graph dataset
    
    Memory Optimizations:
    - Slots-based GraphContainer
    - Efficient one-hot encoding
    - Degree-based tags for specific datasets
    
    Args:
        dataset: Name of dataset directory
        degree_as_tag: Use node degree as tags (for REDDIT/IMDB)
        
    Returns:
        tuple: (graphs, n_classes)
    """
    graphs = []
    label_dict = {}
    path = f'datasets/{dataset}/{dataset}.txt' # Adjust path as needed here , we work with .txt files. 
    
    if not os.path.exists(path):
        sys.exit(f"Error: Missing dataset at {path}") # Ensure path is correct
    
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
    
    all_tags = set(tag for g in graphs for tag in g.node_tags)
    tag2idx = {tag: i for i, tag in enumerate(sorted(all_tags))}
    
    for graph in graphs:
        if degree_as_tag:
            graph.node_tags = [d for _, d in graph.g.degree()]
        
        n_nodes = len(graph.node_tags)
        n_features = len(all_tags)
        graph.node_features = np.zeros((n_nodes, n_features))
        for i, tag in enumerate(graph.node_tags):
            graph.node_features[i, tag2idx[tag]] = 1
    
    print(f"Loaded {len(graphs)} graphs | Classes: {len(label_dict)} | "
          f"Node features: {len(all_tags)}")
    return graphs, len(label_dict)

# ====================== WASSERSTEIN COMPUTATION ======================
# for tough graphs, we use the sliced Wasserstein distance, and for small graphs, we use the Sinkhorn distance.
# This is a heuristic to balance speed and accuracy.
def sliced_wasserstein(X, Y, projections=100): 
    """GPU-accelerated sliced Wasserstein distance
    
    Approximates Wasserstein distance in O(d(n+m)) time using random projections
    
    Args:
        X: (n, d) tensor
        Y: (m, d) tensor
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
    # """Compute pairwise Wasserstein distances between graph embeddings of the graph of various sizes
    # either using the sliced Wasserstein distance for large graphs or the Sinkhorn distance for small graphs.
    """
    
    Computes the pairwise Wasserstein distances between graph embeddings.

    Args:
        embeddings (list of csr_matrix): List of sparse node-feature matrices for each graph.

    Returns:
        np.ndarray: Symmetric matrix of Wasserstein distances (n_graphs x n_graphs).
    
    """
    n = len(embeddings)
    M = np.zeros((n, n))
    
    # Node subsampling - critical for memory if the graph has more than MAX_NODES,
    #  unless you want fried memory for lunchtime.
    emb_subsampled = []
    for emb in embeddings:
        if emb.shape[0] > MAX_NODES:
            idx = np.random.choice(emb.shape[0], MAX_NODES, replace=False)
            emb_subsampled.append(emb[idx].toarray())
        else:
            emb_subsampled.append(emb.toarray())
    
    # Create comparison pairs (upper triangle)
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def process_batch(batch, device_id):
        """Process batch on specified GPU"""
        device = torch.device(f'cuda:{device_id}')
        results = []
        for i, j in batch:
            X = torch.tensor(emb_subsampled[i], dtype=torch.float32, device=device)
            Y = torch.tensor(emb_subsampled[j], dtype=torch.float32, device=device)
            
            # Heuristic: Sliced for large, Sinkhorn for small.
            if X.shape[0] * Y.shape[0] > 1e6:
                dist = sliced_wasserstein(X, Y, SLICED_PROJECTIONS).item()
            else:
                dist = SamplesLoss("sinkhorn", p=2, blur=0.01)(X, Y).item()
            results.append((i, j, dist))
        return results
    
    # GPU batch processing
    batch_size = min(GPU_BATCH_SIZE, len(pairs) // max(1, len(GPU_DEVICES)))
    batches = [pairs[i:i+batch_size] for i in range(0, len(pairs), batch_size)]
    
    print(f"Computing {len(pairs)} distances on {len(GPU_DEVICES)} GPUs")
    
    all_results = []
    if USE_GPU and GPU_DEVICES:
        # Corrected GPU distribution logic I should write a test for this.
        # Distribute batches evenly across available GPUs
        batches_per_gpu = (len(batches) + len(GPU_DEVICES) - 1) // len(GPU_DEVICES)

        
        # Create GPU assignment plan , the batches are distributed evenly across the GPUs.
        # like a drill sargeant.
        # Each GPU gets a slice of the batches to process.
        gpu_assignments = []
        for gpu_idx in range(len(GPU_DEVICES)):
            start_idx = gpu_idx * batches_per_gpu
            end_idx = min((gpu_idx + 1) * batches_per_gpu, len(batches))
            if start_idx < len(batches):
                gpu_assignments.append((GPU_DEVICES[gpu_idx], batches[start_idx:end_idx]))
        # Each assignment is a tuple (gpu_id, list_of_batches)
        
        # Process in parallel , this is the main computation of the MWSPO kernel.
        with Parallel(n_jobs=len(gpu_assignments), backend="threading") as parallel:
            results = parallel(
                delayed(lambda args: [process_batch(batch, args[0]) for batch in args[1]])(assignment)
                for assignment in gpu_assignments
            )
            all_results = [res for gpu_results in results for batch_results in gpu_results for res in batch_results]
    else:

        # CPU fallback , "CAPITAN! , WE ARE OUT OF AMMO SIR!" , "NO! NO STEP BACK!, THROW YOUR CPU'S ON THEM!""
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
    
    np.fill_diagonal(M, 0)
    return (M + M.T) / 2  # Ensure symmetry

# ====================== GRAPH PROCESSING UTILS ======================
# ALL UNITS BE ADVISED! , SUSPECT IS A GRAPH, WE ARE GOING TO EXTRACT THE BFS TREE! 
# USE OF VIOLENCE IS AUTHORIZED! ARREST AT ALL COSTS! WANTED, DEAD OR ALIVE!
def gpu_bfs(G, source, depth): 
    """GPU-accelerated BFS tree extraction using cuGraph"""
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
        return []  # Fallback for isolated nodes , or potentially problematic graphs (rare case)

def path_to_feature_index(path, labels):
    """Convert path to fixed feature index using hashing
    
    Memory Optimization:
    - Uses SHA256 hash truncated to 64-bit integer
    - Modulo FEATURE_DIM ensures fixed-size feature space
    
    Args:
        path: String path (comma-separated node IDs)
        labels: Current node labels
        
    Returns:
        int: Feature index in [0, FEATURE_DIM-1]
    """
    # Convert node IDs to label sequence
    nodes = path.split(',')
    label_seq = tuple(labels[int(n)] for n in nodes)
    
    # Generate stable hash
    hash_str = hashlib.sha256(str(label_seq).encode()).hexdigest()
    hash_int = int(hash_str[:16], 16)  # 64-bit integer
    return hash_int % FEATURE_DIM

# ====================== EMBEDDING GENERATION ======================
def build_embeddings(graphs, maxh, depth):
    """Memory-optimized embedding construction
    
    Critical Optimizations:
    1. Hash-based feature indexing (fixed-size output)
    2. Sparse matrix storage (lil_matrix)
    3. Per-graph processing (no global storage)
    4. Hierarchical feature compression
    
    Args:
        graphs: List of GraphContainer objects
        maxh: Max BFS depth
        depth: Max path depth
        
    Returns:
        list: Sparse embedding matrices (csr_format)
    """
    n_graphs = len(graphs)
    all_labels = [g.node_tags for g in graphs]  # Initial labels
    
    # ========== Hierarchical Feature Extraction ==========
    for level in range(1, maxh):
        new_labels = []
        tree_dict = {}
        global_hashes = []
        
        with ResourceMonitor(f"BFS Level {level}"):
            for gidx, graph in enumerate(graphs):
                G = graph.g
                labels = all_labels[gidx]  # Current labels
                tree_hashes = []  # New labels for this graph
                
                for node in range(len(G)):
                    # Get BFS tree
                    if USE_GPU and len(G) > 100:
                        edges = gpu_bfs(G, node, level) # searching for the BFS tree using GPU
                    else:
                        edges = list(bfs_edges(G, labels, source=node, depth_limit=level))
                    
                    # Create tree signature
                    if edges:
                        # Signature: sorted edge label pairs
                        signature = tuple(sorted(
                            (labels[u], labels[v]) for u, v in edges
                        ))
                    else:
                        # Single-node tree
                        signature = (labels[node],)
                    
                    tree_hash = hash(signature)
                    tree_hashes.append(tree_hash)
                    global_hashes.append(tree_hash)
                
                new_labels.append(tree_hashes)
                del tree_hashes
                gc.collect()
            
            # Create global label mapping
            unique_hashes = set(global_hashes)
            tree_dict = {h: idx for idx, h in enumerate(unique_hashes)}
            
            # Update labels for next level
            all_labels = [
                [tree_dict[h] for h in graph_hashes] 
                for graph_hashes in new_labels
            ]
            del global_hashes, unique_hashes, new_labels
            gc.collect()
    
    # ========== Path-Based Embedding Generation ==========
    graph_embs = []
    
    with ResourceMonitor("Path Extraction"):
        for gidx, graph in enumerate(graphs):
            G = graph.g
            labels = all_labels[gidx]  # Final labels
            emb = lil_matrix((len(G), FEATURE_DIM), dtype=np.float32)
            
            for node in range(len(G)):
                # Node as single-node path
                h = hash((labels[node],)) % FEATURE_DIM
                emb[node, h] += 1
                
                # Get all paths from this node
                try:
                    if USE_GPU and len(G) > 50:
                        # GPU-accelerated SSSP
                        edges = list(G.edges())
                        df = cudf.DataFrame({
                            'src': [e[0] for e in edges],
                            'dst': [e[1] for e in edges]
                        })
                        cug = cugraph.Graph()
                        cug.from_cudf_edgelist(df, 'src', 'dst')
                        sssp = cugraph.sssp(cug, node, depth)
                        
                        for target in sssp['vertex'].to_array():
                            if target == node: 
                                continue
                            path = cugraph.utils.get_traversed_path_list(sssp, target)
                            if path:
                                path_str = ','.join(map(str, path))
                                h = path_to_feature_index(path_str, labels)
                                emb[node, h] += 1
                    else:
                        # CPU fallback
                        for target in nx.single_source_shortest_path_length(G, node, depth):
                            if target == node: 
                                continue
                            path = nx.shortest_path(G, node, target)
                            path_str = ','.join(map(str, path))
                            h = path_to_feature_index(path_str, labels)
                            emb[node, h] += 1
                except Exception as e:
                    print(f"The shortest path computation failed for node {node}: {str(e)}")
                    # this is a flag for bad graphs, or isolated nodes
                    # Skip problematic nodes
                    continue
            
            # Convert to compressed format and store
            graph_embs.append(csr_matrix(emb))
            del emb
            gc.collect()
    
    print(f"Generated embeddings: {len(graph_embs)} graphs")# this is the number of graphs in the dataset, or the number of ID cards we have created.
    print(f"Feature dimension: {FEATURE_DIM} (fixed)")
    return graph_embs

# ====================== EXPERIMENT PIPELINE ======================
# MAN THE CANONS , WE ARE GOING TO TAKE DOWN THAT DATASET!
def run_experiment(dataset, maxh, depth):
    """End-to-end experiment workflow
    
    1. Load data
    2. Generate embeddings
    3. Compute distance matrix
    4. Train SVM classifier
    5. Evaluate accuracy
    
    Args:
        dataset: Dataset name
        maxh: Max BFS depth
        depth: Max path depth
        
    Returns:
        tuple: (mean accuracy, std)
    """
    # Load and preprocess data
    with ResourceMonitor("Data Loading"):

        degree_tag = dataset in ['IMDB-BINARY', 'REDDIT-BINARY']# Use degree as node tags for specific datasets, not all datasets
        # for example for the fraking MUTAG dataset, the degree is not used as a tag, and uses the dataset information as tags.

        graphs, n_classes = load_data(dataset, degree_tag)# see the n_classes? the code does not use it, but it is useful for the future.
        labels = np.array([g.label for g in graphs])
    
    # Feature extraction pipeline
    with ResourceMonitor("Feature Extraction"):
        embeddings = build_embeddings(graphs, maxh, depth)# build ID cards for the graphs , very memory efficient, but still powerful.
    
    # Distance computation
    with ResourceMonitor("Wasserstein Distance"):# the main computation for the MWSPO kernel
        D = compute_wasserstein(embeddings)
    
    # Kernel matrix construction
    with ResourceMonitor("Kernel Construction"): # the kernel matrix is the main output of the MWSPO kernel
        gamma = 0.1 / np.median(D[D > 0])  # Adaptive scaling
        K = np.exp(-gamma * D)
    
    # Classification and evaluation
    with ResourceMonitor("SVM Training"):# the SVM is used to classify the graphs based on the kernel matrix
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

# ====================== MAIN EXECUTION ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Memory-Optimized MWSPO Graph Kernel",
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
    print(f"Feature Dimension: {FEATURE_DIM} (fixed)")
    print(f"Hardware: {len(GPU_DEVICES)} GPUs, {multiprocessing.cpu_count()} CPUs")
    print(f"Memory: {psutil.virtual_memory().total/(1024**3):.1f}GB RAM")
    print("="*70)
    
    if not GPU_DEVICES:# this is a fallback for CPU execution , but our clsuter will have GPUs , and this is just a safety check
        print("INFO: Running in CPU mode (no GPUs detected)")
    
    # Enforce safety limits
    run_experiment(
        dataset=args.dataset,
        maxh=min(args.maxh, MAXH_LIMIT),
        depth=min(args.depth, DEPTH_LIMIT)
    )
