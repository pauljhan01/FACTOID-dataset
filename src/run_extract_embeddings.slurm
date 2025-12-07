#!/bin/bash
#SBATCH -J factoid_extract
#SBATCH -o extract_%j.out
#SBATCH -e extract_%j.err
#SBATCH -p gh-dev
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuean@utexas.edu

#############################################
# 1. Load required modules
#############################################
module purge
module load gcc
module load cuda
module load python3

#############################################
# 2. Activate your Python environment
#############################################
source $HOME/envs/factoid_gpu/bin/activate

#############################################
# 3. Install GPU PCA (cuML) if missing
#############################################
pip install cuml-cu12 --user --quiet || echo "cuML already installed"

#############################################
# 4. Move to project directory
#############################################
cd /work/09930/yue2324/vista/paul/FACTOID-dataset/src

#############################################
# 5. Run PCA extraction (optimized version)
#############################################
python extract_embeddings_1000.py \
    --sample_dir='../data/bert_embeddings/' \
    --checkpoint_dir='../results/checkpoints/' \
    --result_dir='../results/pca/' \
    --run_id='pca_extract_1000'

echo "=== PCA Extraction Completed ==="
