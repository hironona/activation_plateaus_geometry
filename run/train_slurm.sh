#!/bin/bash

#SBATCH --job-name=resnet-train
#SBATCH --array=0
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=05:00:00
#SBATCH --output=slurm_output/resnet-train-%j.out
#SBATCH --error=slurm_output/resnet-train-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${USER}@example.com

# Create output directory if it doesn't exist
mkdir -p slurm_output

# ============================================================================
# Optional: Create conda environment from uv.lock (uncomment if needed)
# ============================================================================
# If the system doesn't have uv, you can extract dependencies from uv.lock
# and create a conda environment instead.
#
# Uncomment the block below if you want to set up conda:
#
# CONDA_ENV_NAME="activation_plateau"
# if ! conda env list | grep -q "^${CONDA_ENV_NAME}"; then
#     echo "Creating conda environment from uv.lock..."
#     # Extract Python version and dependencies from uv.lock
#     # This is a simplified approach; adjust as needed for your uv.lock format
#     python -m pip install --upgrade pip setuptools wheel
#     # Install from requirements (you may need to convert uv.lock to requirements.txt)
#     pip install torch PyYAML tqdm numpy matplotlib scipy scikit-learn transformers
# fi
#
# # Activate environment
# source activate ${CONDA_ENV_NAME}
# ============================================================================

# Navigate to project root
cd "$(dirname "$0")/.." || exit 1

echo "Starting resnet-train job on $(date)"
echo "GPU: $(nvidia-smi --query-gpu=index,name --format=csv,noheader)"
echo "CPU: $(nproc) cores available"
echo "Memory: $(free -h | grep Mem)"
echo ""

# Run multi-seed training
python train/train.py --config train/config.yaml --multi_seed

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Job completed successfully on $(date)"
else
    echo "Job failed with exit code $EXIT_CODE on $(date)"
fi

exit $EXIT_CODE
