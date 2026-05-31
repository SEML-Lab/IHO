#!/bin/bash
#SBATCH --job-name=multi_target_single_cycle_ablations
#SBATCH --output=slurm/slurm_logs/multi_target_single_cycle_ablations/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/multi_target_single_cycle_ablations/%x_%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --array=0-0

########################################
# Models (only dimension we sweep)
########################################

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "GraySwanAI/Llama-3-8B-Instruct-RR"
    "google/gemma-3-1b-it"
)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

########################################
# Naming
########################################

RUN_NAME="sanity"

echo "======================================="
echo "TASK:   $SLURM_ARRAY_TASK_ID"
echo "MODEL:  $MODEL"
echo "RUN:    $RUN_NAME"
echo "======================================="

########################################
# Env
########################################


########################################
# Launch
########################################

pixi run python -m slurm.slurm_wrapper \
    --operation train \
    --experiment-root evaluation/multi_target_single_cycle_ablations \
    --sub-experiment-name "$RUN_NAME" \
    --gpu-type h200 \
    --training-rows THE_TRAINING_ONES_BIG \
    --validation-rows THE_VALIDATION_ONES \
    --attacker-input-size 32 \
    --mask-padding-tokens \
    --attacker-step-number 32 \
    --model "$MODEL" \
    --greedy \
    --learning-rate 0.0002 \
    --beta 0.15 \
    --dpo-epochs 250 \
    --checkpoint-every 40 \
    --save-checkpoints true \
    --patience 1 \
    --num-training-samples 20480 \
    --num-validation-samples 512 \
    --n-cycles 1
