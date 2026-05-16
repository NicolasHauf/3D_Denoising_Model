#!/bin/bash

#SBATCH --job-name=prep_job

#SBATCH --partition=stud

#SBATCH --output=output/preprocess.txt  # Unique output file for each job

#SBATCH --error=error/preprocess.txt    # Unique error file for each job

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --gres=gpu:1

#SBATCH --gpus=1

#SBATCH --nodelist=dedragoste

#SBATCH --time=1-00:00:00 	   # set limit 1 day for test job

# Virtual environment setup
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
SCRIPT_DIR="${ROOT_DIR}/slurm_jobs"
SRC_DIR="${ROOT_DIR}/src"

bash "${SCRIPT_DIR}/setup_env.sh"

echo "Starting pre-processing of image data"
date

cd "${SRC_DIR}"

python3 preprocessing.py

echo "Pre-processing completed"
date
