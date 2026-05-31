#!/bin/bash
#SBATCH --job-name=optimal_single_cycle
#SBATCH --output=slurm/slurm_logs/single_target_multi_cycle_ablations/optimal_single_cycle/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/single_target_multi_cycle_ablations/optimal_single_cycle/%x_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --array=0-239   # 6 models × 60 rows = 240 jobs

########################################
# Models
########################################

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct-with-detector"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "GraySwanAI/Llama-3-8B-Instruct-RR"
    "GraySwanAI/Llama-3-8B-Instruct-RR-with-detector"
    "google/gemma-3-1b-it"
)

########################################
# Explicit training rows
########################################

ROWS=(0 2 5 6 8 9 10 14 15 16 17 19 20 21 22 25 27 28 31 34 35 36 37 39 40 42 43 45 46 49 50 54 55 56 57 59 61 62 64 65 66 68 71 74 76 77 78 79 81 82 83 84 88 89 91 92 93 97 98 99)

########################################
# Index decoding (model × row grid)
########################################

NUM_ROWS=${#ROWS[@]}

MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_ROWS))
ROW_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_ROWS))

MODEL=${MODELS[$MODEL_INDEX]}
ROW_VALUE=${ROWS[$ROW_INDEX]}

########################################
# Naming
########################################

RUN_NAME="optimal_single_cycle"

echo "======================================="
echo "TASK:   $SLURM_ARRAY_TASK_ID"
echo "MODEL:  $MODEL"
echo "ROW:    $ROW_VALUE"
echo "RUN:    $RUN_NAME"
echo "======================================="

########################################
# Environment
########################################


########################################
# Launch
########################################

pixi run python -m slurm.slurm_wrapper \
    --operation train \
    --experiment-root evaluation/single_target_multi_cycle_ablations \
    --sub-experiment-name "$RUN_NAME" \
    --gpu-type a100 \
    --training-rows "$ROW_VALUE" \
    --validation-rows "$ROW_VALUE" \
    --dpo-rows "$ROW_VALUE" \
    --attacker-input-size 32 \
    --attacker-step-number 32 \
    --mask-padding-tokens \
    --model "$MODEL" \
    --learning-rate 0.00005 \
    --beta 0.15 \
    --dpo-epochs 280 \
    --checkpoint-every 40 \
    --save-checkpoints true \
    --patience 2 \
    --num-training-samples 1024 \
    --num-validation-samples 128 \
    --percent-chosen 0.03125 \
    --harmfulness-threshold 0.35 \
    --n-cycles 6 \
    --judge-validation aegis_guard
