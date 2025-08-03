# Multiscale Wasserstein Shortest Path Graph Kernel (MWSP)

**A Scalable Graph Kernel Method for Graph Classification Using Wasserstein Distances and Multiscale Embeddings**

## Overview

The Multiscale Wasserstein Shortest Path Graph Kernel (MWSPO) is a state-of-the-art graph classification system that combines:

    Hierarchical structural feature extraction

    Optimal transport theory (Wasserstein distances)

    GPU-accelerated computation

    Support Vector Machines (SVM)

Designed for large-scale graph datasets (1,000+ graphs) with applications in:

    Bioinformatics (molecular property prediction)

    Social network analysis

    Fraud detection systems

    Recommendation engines
    

The system implements a novel graph kernel approach:

Key Innovations:

    Label-Aware BFS Traversal

        Performs breadth-first search while respecting node labels

        Generates consistent features for isomorphic graphs

        Captures local neighborhood structures

 Multiscale Feature Abstraction

        Hierarchical feature extraction at multiple depths

        Progressive abstraction of structural patterns

 Shortest Path Embedding

    Encodes all shortest paths from each node

    Represents paths as label sequences

    Uses hashing for fixed-dimension representation

Wasserstein Distance

    Measures distributional similarity between graphs

Hybrid computation strategy:

        Exact Sinkhorn for small graphs

        Sliced approximation for large graphs

Engineering Innovations:

    Memory-optimized graph processing pipeline

    Hybrid CPU-GPU workload balancing

    Automatic resource adaptation

    Cluster-scale execution design

  Memory Control Mechanisms:

    Node subsampling (MAX_NODES=180)

    Projection limiting (SLICED_PROJECTIONS=120)

    Batch processing (GPU_BATCH_SIZE=96)

    Depth limiting (MAXH_LIMIT=3, DEPTH_LIMIT=2)

  Memory Optimization Strategy:

    Hash-Based Compression

        Converts paths to 64-bit hashes (8 bytes vs 100+ bytes for strings)

        Fixed feature dimension (1,048,576) prevents vocabulary explosion

    Sparse Matrix Representation

        Uses LIL (List of Lists) format during construction

        Converts to CSR (Compressed Sparse Row) for storage

        Saves 100-1000× memory for sparse features

    Incremental Processing

        Processes graphs sequentially

        Clears intermediate results immediately

        Explicit garbage collection triggers

    GPU-CPU Hybrid Workload

        GPU: BFS/SSSP for large graphs (>100 nodes)

        CPU: Small graph processing

        Automatic fallback mechanism

 Resource Monitor Features:

    Tracks execution time per stage

    Measures RAM allocation/delta

    Records peak GPU memory usage

    Reports total memory consumption


  This system represents a significant advancement in graph representation learning, balancing expressiveness with computational feasibility for real-world large-scale graph analysis tasks. The memory-optimized design makes graph kernel methods accessible for datasets previously considered infeasible due to memory constraints.



## Installation & Usage


### Clone repository:

```
git clone https://github.com/celestine1729/MWSP.git
```

### run_cluster.sh
```

sbatch run_cluster.sh 
```


## Support

For questions or issues, please open a GitHub issue or contact Celestine1729@proton.me

