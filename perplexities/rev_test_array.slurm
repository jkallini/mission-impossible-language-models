#!/bin/bash
#SBATCH --account=nlp
#SBATCH --job-name=train_array
#SBATCH --output=rev_logs/job_%A_%a.out
#SBATCH --array=0-82  # adjust based on number of lines in args file
#SBATCH --partition=jag-standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --exclude=jagupard[19-20],jagupard[26-31],jagupard[37-39]

# Default to test_args.txt if not provided
ARGS_FILE=$1

# Activate your environment
source /nlp/scr/kallini/miniconda3/etc/profile.d/conda.sh
conda activate mission-impossible-2

# Read line from args.txt
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$ARGS_FILE")
read PERTURB PERTURB_TEST DATASET RUN_NAME SEED ARCH_OPTION <<< "$LINE"

# Run with or without architecture flag
python3 perplexities_test.py "$PERTURB" "$PERTURB_TEST" "$DATASET" "$RUN_NAME" "$SEED"
