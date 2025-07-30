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
============================================================
MWSPO Graph Kernel Experiment: REDDIT-BINARY
Configuration: maxh=3, depth=2
Hardware: 2 GPUs, 96 CPUs
Memory: 256.0GB RAM
============================================================
Loading REDDIT-BINARY from datasets/REDDIT-BINARY/REDDIT-BINARY.txt
Loaded 2000 graphs | Classes: 2 | Node features: 7
[RESOURCE] Data Loading              |    12.5s | RAM:  15.3GB | GPU:   0.0GB
[RESOURCE] Feature Extraction        |   184.2s | RAM:  42.1GB | GPU:  18.7GB
[RESOURCE] Wasserstein Distance      |  2174.8s | RAM: 182.4GB | GPU:  28.7GB
[RESOURCE] Kernel Construction       |     0.1s | RAM:   0.0GB | GPU:   0.0GB
[RESOURCE] SVM Training              |   324.7s | RAM:   4.2GB | GPU:   0.0GB

Fold 1: Accuracy = 0.874
...
Final Accuracy: 86.42% ± 1.87%
```

## Support

For questions or issues, please open a GitHub issue or contact AlirezaAhmadi@gmail.com.

