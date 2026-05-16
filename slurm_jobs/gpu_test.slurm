#!/bin/bash

#SBATCH --job-name=test_job

#SBATCH --partition=stud

#SBATCH --output=output/test.txt

#SBATCH --error=error/test.txt

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --time=0-00:01:00 	   # set limit 1 min for test job

# Virtual environment setup
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
SCRIPT_DIR="${ROOT_DIR}/slurm_jobs"
SRC_DIR="${ROOT_DIR}/src"

bash "${SCRIPT_DIR}/setup_env.sh"

# Detect GPUs and print the number of available GPUs
.venv/bin/python3 - <<'PYCODE'
import tensorflow as tf
print('Num GPUs Available before activation:', len(tf.config.experimental.list_physical_devices('GPU')))
PYCODE

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo 'nvidia-smi not found'
fi

export CUDNN_PATH="$HOME/.local/lib/python3.9/site-packages/nvidia/cudnn"
export LD_LIBRARY_PATH="${CUDNN_PATH}/lib:${LD_LIBRARY_PATH:-}"

# Detect GPUs and print the number of available GPUs
.venv/bin/python3 - <<'PYCODE'
import tensorflow as tf
print('Num GPUs Available after activation:', len(tf.config.experimental.list_physical_devices('GPU')))
print('If there are still no GPUs available, it may be due to missing CUDA or cuDNN libraries. Please ensure that they are properly installed and configured.')
PYCODE

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo 'nvidia-smi not found'
fi