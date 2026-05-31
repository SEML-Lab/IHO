#!/bin/bash
#SBATCH --job-name=judge_ablation
#SBATCH --output=slurm/slurm_logs/single_target_single_cycle_ablations/judge_ablation/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/single_target_single_cycle_ablations/judge_ablation/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --array=0-39   # 4 models × 10 rows = 40 jobs

########################################
# Grid
########################################

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "GraySwanAI/Llama-3-8B-Instruct-RR"
    "google/gemma-3-1b-it"
)

ROWS=(1 3 7 17 25 29 47 58 77 81)

########################################
# Index math
########################################

N_MODEL=${#MODELS[@]}
N_ROWS=${#ROWS[@]}

MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / N_ROWS))
ROW_INDEX=$((SLURM_ARRAY_TASK_ID % N_ROWS))

MODEL=${MODELS[$MODEL_INDEX]}
ROW=${ROWS[$ROW_INDEX]}

########################################
# Naming
########################################

RUN_NAME="judge_swapped"

echo "======================================="
echo "JOB ID:    $SLURM_JOB_ID"
echo "TASK ID:   $SLURM_ARRAY_TASK_ID"
echo "MODEL:     $MODEL"
echo "ROW:       $ROW"
echo "RUN NAME:  $RUN_NAME"
echo "======================================="

########################################
# Environment
########################################


########################################
# Launch
########################################

pixi run python -m slurm.slurm_wrapper \
    --operation train \
    --experiment-root evaluation/single_target_single_cycle_ablations/judge_ablation \
    --sub-experiment-name "$RUN_NAME" \
    --gpu-type h200 \
    --training-rows $ROW \
    --validation-rows $ROW \
    --dpo-rows $ROW \
    --mask-padding-tokens \
    --model "$MODEL" \
    --judge-training harmbench \
    --judge-validation strong_reject \
    --dpo-epochs 500 \
    --checkpoint-every 50 \
    --save-checkpoints true \
    --patience 3 \
    --num-training-samples 512 \
    --num-validation-samples 128 \
    --n-cycles 1
