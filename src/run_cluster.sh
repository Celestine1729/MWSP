#!/bin/bash
#SBATCH --job-name=MWSPO
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=250G              # Optimized for 256GB node
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:tesla_v100:2
#SBATCH --output=logs/%x_%j.log

echo "========== ENVIRONMENT SETUP =========="
echo "Loading modules..."
module purge
module load cuda/11.7
module load python/3.10
module load gcc/9.3.0

echo "========== VIRTUAL ENVIRONMENT =========="
if [ ! -d "venv" ]; then
    echo "Creating new Python virtual environment..."
    python -m venv venv
    source venv/bin/activate
    
    echo "Installing Python packages..."
    pip install --upgrade pip
    pip install torch==2.0.1+cu117 torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
    pip install cugraph-cu11x cudf-cu11x cuml-cu11x --extra-index-url https://pypi.nvidia.com
    pip install geomloss scikit-learn networkx gensim joblib psutil argparse scipy
else
    source venv/bin/activate
fi

echo "========== EXECUTION =========="
echo "Starting MWSPO graph kernel experiment..."
python MWSPO.py YOUR_DATASET_NAME_HERE \
  --maxh 3 \
  --depth 2

echo "========== CLEANUP =========="
echo "Forcing garbage collection..."
python -c "import gc; gc.collect()"
echo "Experiment finished."