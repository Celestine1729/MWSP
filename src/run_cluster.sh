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