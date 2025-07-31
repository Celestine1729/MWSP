# Multiscale Wasserstein Shortest Path Graph Kernel (MWSP)

**A Scalable Graph Kernel Method for Graph Classification Using Wasserstein Distances and Multiscale Embeddings**

## Overview

The Multiscale Wasserstein Shortest Path Kernel (MWSP) is an advanced graph kernel method that combines:

1. **Multiscale graph embeddings** via BFS-based feature extraction
2. **Wasserstein distance** to measure structural similarity between graphs
3. **Support Vector Machines** for graph classification tasks

This implementation is optimized for both laptop-scale experimentation (small datasets) and cluster-scale execution (large datasets with 1000+ graphs).


## Installation & Usage

### Enviroment Setup
  ```
  python -m venv mwspo_env
  source mwspo_env/bin/activate

  pip install torch==2.0.1+cu117 torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
  pip install cugraph-cu11x cudf-cu11x cuml-cu11x --extra-index-url https://pypi.nvidia.com
  pip install geomloss scikit-learn networkx gensim joblib psutil argparse
  ```


 ### Clone repository:

```
git clone https://github.com/celestine1729/MWSP.git
```
### database file structure:
```
  datasets/
└── YOUR_DATASET/
    ├── dataset.txt
    └── (optional metadata files)
  
```

### run_cluster.sh
```
#!/bin/bash
#SBATCH --job-name=MWSPO
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=48
#SBATCH --mem=250G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:tesla_v100:2
#SBATCH --output=logs/%x_%j.log

# Cluster environment setup
module purge
module load cuda/11.7
module load python/3.10
module load gcc/9.3.0

# Python environment
python -m venv venv
source venv/bin/activate
pip install torch==2.0.1+cu117 torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install cugraph-cu11x cudf-cu11x cuml-cu11x --extra-index-url https://pypi.nvidia.com
pip install geomloss scikit-learn networkx gensim joblib psutil argparse

# Execute with optimal parameters for 1000+ graphs
python MWSPO_final.py YOUR_DATASET \
  --maxh 3 \
  --depth 2
  
```


### Example

```
# Single GPU run
python MWSPO.py MUTAG --maxh 3 --depth 2

# Cluster submission
sbatch run_cluster.sh   # enter the database name in the .sh file.
```
### Expected Output

```
======================================================================
MWSPO Graph Kernel Experiment: REDDIT-BINARY
Configuration: maxh=3, depth=2
Feature Dimension: 1048576 (fixed)
Hardware: 2 GPUs, 64 CPUs
Memory: 256.0GB RAM
======================================================================
Loading REDDIT-BINARY from datasets/REDDIT-BINARY/REDDIT-BINARY.txt
Loaded 1000 graphs | Classes: 2 | Node features: 12

[RESOURCE] Data Loading            |    12.3s | RAM: +   1.2GB (Total:  3.2GB) | GPU:   0.0GB
[RESOURCE] BFS Level 1             |   142.5s | RAM: +   2.5GB (Total:  5.7GB) | GPU:   1.2GB
[RESOURCE] BFS Level 2             |   158.2s | RAM: +   1.8GB (Total:  7.5GB) | GPU:   1.5GB
[RESOURCE] Path Extraction         |   225.7s | RAM: +   4.0GB (Total: 11.5GB) | GPU:   2.0GB
Generated embeddings: 1000 graphs
Feature dimension: 1048576 (fixed)
[RESOURCE] Feature Extraction      |   568.4s | RAM: +   9.5GB (Total: 13.5GB) | GPU:   2.0GB

Computing 499500 distances on 2 GPUs
[RESOURCE] Wasserstein Distance    |  1204.8s | RAM: +  15.0GB (Total: 28.5GB) | GPU:  10.0GB
[RESOURCE] Kernel Construction     |     0.1s | RAM: +   0.0GB (Total: 28.5GB) | GPU:   0.0GB

[RESOURCE] SVM Training            |    45.2s | RAM: +   0.5GB (Total: 29.0GB) | GPU:   0.0GB
Fold 1: Accuracy = 0.8500
Fold 2: Accuracy = 0.8200
Fold 3: Accuracy = 0.8700
Fold 4: Accuracy = 0.8400
Fold 5: Accuracy = 0.8300
Fold 6: Accuracy = 0.8600
Fold 7: Accuracy = 0.8500
Fold 8: Accuracy = 0.8100
Fold 9: Accuracy = 0.8400
Fold 10: Accuracy = 0.8300

Final Accuracy: 84.00% ± 1.52%
```

## Support

For questions or issues, please open a GitHub issue or contact Celestine1729@proton.me

