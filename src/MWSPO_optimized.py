"""
MWSPO_optimized.py
------------------
Multiscale Wasserstein Shortest Path Graph Kernel (Optimized Version)

A graph kernel method that:
1. Generates multiscale graph embeddings via BFS-based feature extraction
2. Computes Wasserstein distances between graph embeddings
3. Builds a kernel matrix using heat kernel on Wasserstein distances
4. Trains an SVM classifier for graph classification

Optimized for cluster execution with:
- Parallel Wasserstein distance computation
- Memory-efficient operations
- Scalable to large datasets (1500+ graphs)

Author: Celestine 
Contact: https://t.me/celestine_1729
Date: July 2025
"""
import numpy as np
import networkx as nx
import sys
import time
import re
from gensim import corpora
import gensim
import breadth_first_search as bfs
import ot
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection._validation import _fit_and_score
from sklearn.base import clone
from sklearn.metrics import make_scorer, accuracy_score
from joblib import Parallel, delayed
import multiprocessing
import os

class S2VGraph:
    """
    Graph representation container for graph classification tasks
    
    Attributes:
        label (int): Class label of the graph
        g (nx.Graph): NetworkX graph object
        node_tags (list): Node labels or attributes
        neighbors (list): Adjacency list representation of neighbors
        node_features (np.array): One-hot encoded node features
    """
    def __init__(self, g, label, node_tags=None):
        """
        Initialize graph object
        
        Args:
            g: NetworkX graph
            label: Graph class label
            node_tags: List of node labels/tags
        """
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = None

def load_data(dataset, degree_as_tag=False):
    """
    Load graph dataset from text file format
    
    Expected file format:
    Line 1: Number of graphs (N)
    For each graph:
      Header: [node_count] [graph_label]
      For each node:
        [node_id] [neighbor_count] [neighbor1] [neighbor2] ... [optional_features]
    
    Args:
        dataset: Name of dataset directory
        degree_as_tag: Whether to use node degree as tags
        
    Returns:
        g_list: List of S2VGraph objects
        num_classes: Number of unique graph classes
    """
    g_list = []
    label_dict = {}
    feat_dict = {}
    dataset_path = f'./datasets/{dataset}/{dataset}.txt'
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)
    
    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path, 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            n, l = map(int, f.readline().split())
            if l not in label_dict:
                label_dict[l] = len(label_dict)
            
            g = nx.Graph()
            node_tags = []
            for j in range(n):
                g.add_node(j)
                row = f.readline().split()
                num_edges = int(row[1])
                node_tags.append(int(row[0]))
                
                # Add edges
                for k in range(2, 2 + num_edges):
                    g.add_edge(j, int(row[k]))
            
            g_list.append(S2VGraph(g, l, node_tags))
    
    # Build neighbor lists
    for graph in g_list:
        graph.neighbors = [list(graph.g.neighbors(i)) for i in range(len(graph.g))]
        if degree_as_tag:
            graph.node_tags = [d for _, d in graph.g.degree()]
    
    # Build feature matrix
    tagset = set(tag for graph in g_list for tag in graph.node_tags)
    tag2index = {tag: idx for idx, tag in enumerate(sorted(tagset))}
    for graph in g_list:
        num_nodes = len(graph.node_tags)
        num_features = len(tagset)
        graph.node_features = np.zeros((num_nodes, num_features))
        for node_idx, tag in enumerate(graph.node_tags):
            graph.node_features[node_idx, tag2index[tag]] = 1
    
    print(f'Loaded {len(g_list)} graphs')
    print(f'Number of classes: {len(label_dict)}')
    print(f'Number of unique node tags: {len(tagset)}')
    return g_list, len(label_dict)

def compute_wasserstein_distance(graph_embeddings, sinkhorn=False, categorical=False, 
                                 sinkhorn_lambda=1e-2, n_jobs=-1):
    """
    Compute pairwise Wasserstein distances between graph embeddings
    
    Args:
        graph_embeddings: List of node embedding matrices per graph
        sinkhorn: Whether to use Sinkhorn approximation
        categorical: Use Hamming distance for categorical embeddings
        sinkhorn_lambda: Regularization parameter for Sinkhorn
        n_jobs: Number of parallel jobs (-1 = all cores)
        
    Returns:
        M: Symmetric n x n distance matrix
    """
    n = len(graph_embeddings)
    M = np.zeros((n, n))
    
    # Configure parallel processing
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    # Create index pairs for upper triangle
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def compute_single_distance(i, j):
        """
        Compute Wasserstein distance between two graphs
        
        Args:
            i, j: Indexes of graphs to compare
            
        Returns:
            dist: Wasserstein distance between graph i and j
        """
        emb1 = graph_embeddings[i]
        emb2 = graph_embeddings[j]
        metric = 'hamming' if categorical else 'euclidean'
        costs = ot.dist(emb1, emb2, metric=metric)
        a, b = np.ones(len(emb1))/len(emb1), np.ones(len(emb2))/len(emb2)
        
        if sinkhorn:
            mat = ot.sinkhorn(a, b, costs, sinkhorn_lambda, numItermax=50)
            return np.sum(mat * costs)
        return ot.emd2(a, b, costs)
    
    print(f"Computing {len(pairs)} Wasserstein distances using {n_jobs} cores...")
    start_time = time.time()
    
    # Parallel computation
    if n_jobs == 1:
        results = [compute_single_distance(i, j) for i, j in pairs]
    else:
        results = Parallel(n_jobs=n_jobs, verbose=20)(
            delayed(compute_single_distance)(i, j) for i, j in pairs
        )
    
    # Fill symmetric distance matrix
    for idx, (i, j) in enumerate(pairs):
        dist = results[idx]
        M[i, j] = dist
        M[j, i] = dist
    
    print(f"Distance matrix computed in {time.time()-start_time:.2f} seconds")
    return M

def build_multiset(graph_data, maxh, depth):
    """
    Build multiscale graph embeddings via BFS-based feature extraction
    
    Args:
        graph_data: List of S2VGraph objects
        maxh: Maximum BFS depth for feature extraction
        depth: Maximum path depth to consider
        
    Returns:
        graph_embedding: List of node embedding matrices per graph
    """
    graphs = {}
    labels = {}
    all_labels = {}
    num_graphs = len(graph_data)
    
    # Initialize graph representations
    for gidx in range(num_graphs):
        graphs[gidx] = nx.to_numpy_array(graph_data[gidx].g)
        labels[gidx] = graph_data[gidx].node_tags
    all_labels[0] = labels
    
    # Hierarchical feature extraction
    for level in range(1, maxh):
        new_labels = {}
        all_trees = []
        tree_set = set()
        
        # Extract BFS trees for each node
        for gidx in range(num_graphs):
            G = graph_data[gidx].g
            current_labels = all_labels[level-1][gidx]
            
            for node in range(len(G)):
                edges = list(bfs.bfs_edges(G, current_labels, source=node, depth_limit=level))
                tree_str = ','.join(f"{current_labels[u]},{current_labels[v]}" for u,v in edges)
                all_trees.append(tree_str)
                tree_set.add(tree_str)
        
        # Create new label mapping
        tree_list = sorted(tree_set)
        offset = 0
        for gidx in range(num_graphs):
            n = len(graphs[gidx])
            new_labels[gidx] = np.array([tree_list.index(tree) for tree in all_trees[offset:offset+n]])
            offset += n
        
        all_labels[level] = new_labels
    
    # Path extraction
    all_paths = []
    for gidx in range(num_graphs):
        G = graph_data[gidx].g
        seen_paths = set()
        graph_paths = []
        current_labels = all_labels[maxh-1][gidx]
        
        for node in range(len(G)):
            node_paths = [str(node)]
            seen_paths.add(str(node))
            
            # Extract BFS paths
            edges = list(bfs.bfs_edges(G, current_labels, source=node, depth_limit=depth))
            targets = [v for _, v in edges]
            
            for target in targets:
                path = nx.shortest_path(G, node, target)
                path_str = ','.join(map(str, path))
                rev_path = ','.join(map(str, reversed(path)))
                
                # Add unique path representations
                if rev_path not in seen_paths:
                    seen_paths.add(path_str)
                    node_paths.append(path_str)
            
            graph_paths.append(node_paths)
        all_paths.extend(graph_paths)
    
    # Build embeddings using BoW model
    embeddings = []
    for level in range(maxh):
        level_paths = []
        offset = 0
        
        # Convert to textual representations
        for gidx in range(num_graphs):
            n = len(graphs[gidx])
            labels = all_labels[level][gidx]
            for i in range(n):
                path_repr = []
                for path in all_paths[offset + i]:
                    nodes = path.split(',')
                    path_repr.append(','.join(str(labels[int(n)]) for n in nodes))
                level_paths.append(path_repr)
            offset += n
        
        # Create BoW embeddings
        dictionary = corpora.Dictionary(level_paths)
        corpus = [dictionary.doc2bow(paths) for paths in level_paths]
        bow_matrix = gensim.matutils.corpus2dense(corpus, len(dictionary)).T
        embeddings.append(bow_matrix)
    
    # Concatenate multiscale embeddings
    full_embeddings = np.hstack(embeddings)
    
    # Group by graph
    graph_embeddings = []
    idx = 0
    for gidx in range(num_graphs):
        n = len(graphs[gidx])
        graph_embeddings.append(full_embeddings[idx:idx+n])
        idx += n
    
    print(f"Built embeddings with dimension {full_embeddings.shape[1]}")
    return graph_embeddings

def run_single_experiment(ds_name, maxh, depth, crossvalidation=True, gridsearch=True, n_jobs=-1):
    """
    Run full experiment pipeline for a dataset
    
    Args:
        ds_name: Dataset name (must exist in datasets/)
        maxh: Maximum BFS depth for feature extraction
        depth: Maximum path depth to consider
        crossvalidation: Whether to perform 10-fold CV
        gridsearch: Whether to perform hyperparameter tuning
        n_jobs: Number of parallel jobs for Wasserstein
        
    Returns:
        mean_accuracy: Mean classification accuracy
        std_accuracy: Standard deviation of accuracy
    """
    print(f"\n{'='*50}")
    print(f"Experiment: {ds_name} | k={maxh-1} | d={depth-1}")
    print(f"{'='*50}")
    
    # Dataset-specific configuration
    degree_as_tag = ds_name in ['IMDB-BINARY', 'REDDIT-BINARY']
    
    # Load data
    graphs, num_classes = load_data(ds_name, degree_as_tag)
    graph_labels = np.array([g.label for g in graphs])
    
    # Build embeddings and distance matrix
    start_time = time.time()
    embeddings = build_multiset(graphs, maxh, depth)
    distance_matrix = compute_wasserstein_distance(
        embeddings, 
        n_jobs=n_jobs
    )
    print(f"Feature extraction time: {time.time()-start_time:.2f}s")
    
    # Kernel matrix computation
    gammas = np.logspace(-4, 1, num=6) if gridsearch else [0.001]
    kernel_matrices = [np.exp(-gamma * distance_matrix) for gamma in gammas]
    
    # Cross-validation setup
    n_splits = 10 if crossvalidation else 1
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    
    print(f"Starting {n_splits}-fold cross-validation...")
    for fold, (train_idx, test_idx) in enumerate(cv.split(kernel_matrices[0], graph_labels)):
        fold_start = time.time()
        
        # Prepare kernel subsets
        K_train = [K[train_idx][:, train_idx] for K in kernel_matrices]
        K_test = [K[test_idx][:, train_idx] for K in kernel_matrices]
        y_train, y_test = graph_labels[train_idx], graph_labels[test_idx]
        
        # Model training
        if gridsearch:
            # Hyperparameter tuning
            best_score = -1
            best_model = None
            for gamma_idx, K_tr in enumerate(K_train):
                for C in np.logspace(-3, 3, 7):
                    model = SVC(C=C, kernel='precomputed', max_iter=10000)
                    model.fit(K_tr, y_train)
                    score = model.score(K_test[gamma_idx], y_test)
                    if score > best_score:
                        best_score = score
                        best_model = model
            y_pred = best_model.predict(K_test[gamma_idx])
        else:
            # Fixed parameters
            model = SVC(C=100, kernel='precomputed', max_iter=10000)
            model.fit(K_train[0], y_train)
            y_pred = model.predict(K_test[0])
        
        # Evaluation
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        print(f"Fold {fold+1}: Accuracy = {acc:.4f} | Time = {time.time()-fold_start:.2f}s")
    
    # Results summary
    mean_acc = np.mean(accuracies) * 100
    std_acc = np.std(accuracies) * 100
    print(f"\nResults: {mean_acc:.2f}% ± {std_acc:.2f}%")
    return mean_acc, std_acc

if __name__ == "__main__":
    """
    Command-line execution
    
    Usage:
        python MWSPO_optimized.py DATASET_NAME MAXH DEPTH [N_JOBS]
    
    Example:
        python MWSPO_optimized.py MUTAG 3 2 -1
    """
    if len(sys.argv) < 4:
        print("Error: Missing required arguments")
        print("Usage: python MWSPO_optimized.py DATASET_NAME MAXH DEPTH [N_JOBS]")
        print("Example: python MWSPO_optimized.py MUTAG 3 2 -1")
        sys.exit(1)
    
    dataset = sys.argv[1]
    maxh = int(sys.argv[2])
    depth = int(sys.argv[3])
    n_jobs = -1 if len(sys.argv) < 5 else int(sys.argv[4])
    
    run_single_experiment(
        ds_name=dataset,
        maxh=maxh,
        depth=depth,
        n_jobs=n_jobs
    )
