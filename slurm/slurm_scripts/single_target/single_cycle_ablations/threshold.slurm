#!/bin/bash
#SBATCH --job-name=threshold_ablation
#SBATCH --output=slurm/slurm_logs/single_target/single_cycle_ablations/threshold_ablation/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/single_target/single_cycle_ablations/threshold_ablation/%x_%A_%a.err
#SBATCH --time=5:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --array=0-239   # adjust based on grid size

########################################
# Grid
########################################

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "GraySwanAI/Llama-3-8B-Instruct-RR"
    "google/gemma-3-1b-it"
)

THRESHOLDS=(0.1 0.2 0.3 0.4 0.5 0.6)

ROWS=(1 3 7 17 25 29 47 58 77 81)

########################################
# Index math
########################################

N_MODEL=4
N_THRESH=6
N_ROWS=10

TOTAL_PER_MODEL=$((N_THRESH * N_ROWS))  # 60

MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / TOTAL_PER_MODEL))
REM=$((SLURM_ARRAY_TASK_ID % TOTAL_PER_MODEL))

THRESH_INDEX=$((REM / N_ROWS))
ROW_INDEX=$((REM % N_ROWS))

MODEL=${MODELS[$MODEL_INDEX]}
THRESH=${THRESHOLDS[$THRESH_INDEX]}
ROW=${ROWS[$ROW_INDEX]}

########################################
# Naming
########################################

RUN_NAME="thr_${THRESH}"

echo "======================================="
echo "JOB ID:    $SLURM_JOB_ID"
echo "TASK ID:   $SLURM_ARRAY_TASK_ID"
echo "MODEL:     $MODEL"
echo "THRESH:    $THRESH"
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
    --experiment-root evaluation/single_target/single_cycle_ablations/threshold \
    --sub-experiment-name "$RUN_NAME" \
    --gpu-type a100 \
    --training-rows $ROW \
    --validation-rows $ROW \
    --dpo-rows $ROW \
    --model "$MODEL" \
    --dpo-epochs 500 \
    --checkpoint-every 50 \
    --save-checkpoints true \
    --patience 3 \
    --num-training-samples 512 \
    --num-validation-samples 128 \
    --num-dpo-samples 128 \
    --n-cycles 1 \
    --harmfulness-threshold $THRESH \
    --no-logprobs \
    --overwrite
