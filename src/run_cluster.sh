#!/bin/bash
#SBATCH --nodes=4                  # Use 4 physical nodes
#SBATCH --ntasks-per-node=4        # 4 tasks per node = 1 per GPU
#SBATCH --cpus-per-task=5          # 5 cores per dataset
#SBATCH --mem=200G                 # 200GB ram per dataset
#SBATCH --gres=gpu:tesla_v100:1    # 1 GPU per dataset
#SBATCH --array=1-23%16            # 24 datasets, max 16 concurrent (4 nodes × 4 GPUs)
#SBATCH --time=48:00:00
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

mapfile -t DS_LIST < datasets.txt
CURRENT_DS=${DS_LIST[$SLURM_ARRAY_TASK_ID - 1]}

# Critical: Limit CPU threads
export OMP_NUM_THREADS=5
export OPENBLAS_NUM_THREADS=5

# Single dataset per task
srun --ntasks=1 --exclusive \
  python MWSPO.py $CURRENT_DS --maxh 3 --depth 2

echo "========== CLEANUP =========="
echo "Forcing garbage collection..."
python -c "import gc; gc.collect()"
echo "Experiment finished."