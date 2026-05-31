#!/bin/bash
#SBATCH --job-name=percent_chosen_ablation
#SBATCH --output=slurm/slurm_logs/single_target_single_cycle_ablations/percent_chosen_ablation/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/single_target_single_cycle_ablations/percent_chosen_ablation/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --array=0-47   # 4 models × 12 percents = 48 jobs

########################################
# Grid
########################################

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "GraySwanAI/Llama-3-8B-Instruct-RR"
    "google/gemma-3-1b-it"
)

PERCENTS=(0.001 0.005 0.01 0.02 0.05 0.075 0.1 0.125 0.15 0.2 0.25 0.33)

########################################
# Index math
########################################

N_PERCENT=${#PERCENTS[@]}   # 5
N_MODEL=${#MODELS[@]}       # 4

PERCENT_INDEX=$((SLURM_ARRAY_TASK_ID % N_PERCENT))
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / N_PERCENT))

PERCENT=${PERCENTS[$PERCENT_INDEX]}
MODEL=${MODELS[$MODEL_INDEX]}

########################################
# Naming
########################################

RUN_NAME="percent_${PERCENT}"

echo "======================================="
echo "TASK:    $SLURM_ARRAY_TASK_ID"
echo "MODEL:   $MODEL"
echo "PERCENT: $PERCENT"
echo "LR:      $LR"
echo "BETA:    $BETA"
echo "RUN:     $RUN_NAME"
echo "======================================="

########################################
# Env
########################################


########################################
# Launch
########################################

pixi run python -m slurm.slurm_wrapper \
    --operation train \
    --experiment-root evaluation/single_target_single_cycle_ablations/percent_chosen_ablation \
    --sub-experiment-name "$RUN_NAME" \
    --iterate-rows-separately \
    --gpu-type a100 \
    --training-rows THE_TRAINING_ONES_SMALL \
    --mask-padding-tokens \
    --model "$MODEL" \
    --greedy \
    --percent-chosen "$PERCENT" \
    --dpo-epochs 500 \
    --checkpoint-every 50 \
    --save-checkpoints true \
    --patience 3 \
    --num-training-samples 1024 \
    --num-validation-samples 128 \
    --n-cycles 1
