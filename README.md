# Multiscale Wasserstein Shortest Path Graph Kernel (MWSP)

**A Scalable Graph Kernel Method for Graph Classification Using Wasserstein Distances and Multiscale Embeddings**

## Overview

The Multiscale Wasserstein Shortest Path Kernel (MWSP) is an advanced graph kernel method that combines:

1. **Multiscale graph embeddings** via BFS-based feature extraction
2. **Wasserstein distance** to measure structural similarity between graphs
3. **Support Vector Machines** for graph classification tasks

This implementation is optimized for both laptop-scale experimentation (small datasets) and cluster-scale execution (large datasets with 1000+ graphs).


## Installation

### Requirements
- Python 3.7+
- Required packages:
  ```
  pip install numpy scipy networkx scikit-learn gensim POT joblib
  ```

  
  ## Setup

 ### Clone repository:

```
git clone https://github.com/yourusername/multiscale-wasserstein-graph-kernel.git
cd multiscale-wasserstein-graph-kernel
```
### Install dependencies:
```
pip install -r requirements.txt
```


### Usage
```
python MWSPO_optimized.py DATASET_NAME MAXH DEPTH [N_JOBS]
```
### Example

```
# Run on MUTAG dataset using 4 cores
python MWSPO_optimized.py MUTAG 3 2 4
```
### Expected Output

```
==================================================
Experiment: MUTAG | k=2 | d=1
==================================================
Loading dataset: ./datasets/MUTAG/MUTAG.txt
Loaded 188 graphs
Number of classes: 2
Number of unique node tags: 7
Computing 17578 Wasserstein distances using 4 cores...
[Parallel]: 100%|██████████| 17578/17578 [01:15<00:00]
Distance matrix computed in 75.32 seconds
Starting 10-fold cross-validation...
Fold 1: Accuracy = 0.8947 | Time = 0.12s
Fold 2: Accuracy = 0.8421 | Time = 0.11s
...
Fold 10: Accuracy = 0.8684 | Time = 0.10s

Results: 87.37% ± 3.21%
```

## Support

For questions or issues, please open a GitHub issue or contact AlirezaAhmadi@gmail.com.

