#!/bin/bash

#SBATCH --job-name=full_cube_model

#SBATCH --partition=stud

#SBATCH --output=output/cube_model_lr_finetuning.txt

#SBATCH --error=error/cube_model_lr_finetuning.txt

#SBATCH --ntasks=1

#SBATCH --gres=gpu:1

#SBATCH --gpus=1                            # 1 GPU per job

#SBATCH --cpus-per-gpu=4

#SBATCH --nodelist=cadintulnic

#SBATCH --time=1-23:00:00 	   # set limit 1 d for test job


# Virtual environment setup
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
SCRIPT_DIR="${ROOT_DIR}/slurm_jobs"
SRC_DIR="${ROOT_DIR}/src"

bash "${SCRIPT_DIR}/setup_env.sh"

echo "Starting cube model training with learning rate and finetuning"
date

cd "${SRC_DIR}"

python3 -u cube_model_v3.py

echo "Cube model training completed"
date

