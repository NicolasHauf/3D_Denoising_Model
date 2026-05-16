#!/bin/bash

#SBATCH --job-name=post_process_cube_model

#SBATCH --partition=stud

#SBATCH --output=output/cube_model_postprocess.txt  # Unique output file for each job

#SBATCH --error=/error/cube_model_postprocess.txt    # Unique error file for each job

#SBATCH --ntasks=1

#SBATCH --gres=gpu:1

#SBATCH --gpus=1                            # 1 GPU per job

#SBATCH --cpus-per-gpu=4

#SBATCH --nodelist=cadintulnic

#SBATCH --time=0-10:00:00 	   # set limit 10 h for test job

# Virtual environment setup
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
SCRIPT_DIR="${ROOT_DIR}/slurm_jobs"
SRC_DIR="${ROOT_DIR}/src"

bash "${SCRIPT_DIR}/setup_env.sh"

cd "${SRC_DIR}"

echo "Starting post-processing of cube model results"
date
python3 -u postprocess.py

echo "Post-processing completed"
date

