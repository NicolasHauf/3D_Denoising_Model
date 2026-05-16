#!/bin/bash

#SBATCH --job-name=copy_job

#SBATCH --partition=stud

#SBATCH --output=output/copy.txt

#SBATCH --error=error/copy.txt

#SBATCH --ntasks=1

#SBATCH --nodelist=znamennychant

#SBATCH --time=0-24:00:00 	   # set limit 24h for test job

# Define source and destination folders
SOURCE_DIR="/home/haufns/Documents/temp/org_frag_data_3d"
DEST_DIR="/scratch/users/haufns/denoising_model/training_data"

echo "Start copying data"
date
mkdir -p "$DEST_DIR"
cp -r "$SOURCE_DIR" "$DEST_DIR/"
echo "Copying completed"
date
