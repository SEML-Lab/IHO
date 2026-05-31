#!/bin/bash
#SBATCH --job-name=warm_start
#SBATCH --output=slurm/slurm_logs/single_target_single_cycle_ablations/warm_start/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/single_target_single_cycle_ablations/warm_start/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --array=0-3

########################################
# Grid (models only)
########################################

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "GraySwanAI/Llama-3-8B-Instruct-RR"
    "google/gemma-3-1b-it"
)

########################################
# Index math
########################################

MODEL_INDEX=$SLURM_ARRAY_TASK_ID
MODEL=${MODELS[$MODEL_INDEX]}

########################################
# Naming
########################################

RUN_NAME="warm_start"

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
    --experiment-root evaluation/single_target_single_cycle_ablations \
    --sub-experiment-name "$RUN_NAME" \
    --iterate-rows-separately \
    --gpu-type h200 \
    --training-rows "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]" \
    --mask-padding-tokens \
    --model "$MODEL" \
    --greedy \
    --learning-rate 4e-5 \
    --dpo-epochs 500 \
    --checkpoint-every 50 \
    --save-checkpoints true \
    --patience 3 \
    --num-training-samples 1024 \
    --num-validation-samples 128 \
    --n-cycles 1
