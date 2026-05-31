#!/bin/bash
#SBATCH --job-name=warmup_ablations
#SBATCH --output=slurm/slurm_logs/single_target_multi_cycle_ablations/warmup_ablations/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/single_target_multi_cycle_ablations/warmup_ablations/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --array=0-39   # 4 models × 10 rows = 40 jobs

########################################
# Models
########################################

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "GraySwanAI/Llama-3-8B-Instruct-RR"
    "google/gemma-3-1b-it"
)

########################################
# Explicit training rows
########################################

ROWS=(0 19 20 39 40 55 62 78 84 91)

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

RUN_NAME="warmup_ablations"

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
    --dpo-epochs 250 \
    --checkpoint-every 40 \
    --save-checkpoints true \
    --patience 1 \
    --num-training-samples 1024 \
    --num-validation-samples 128 \
    --percent-chosen 0.03125 \
    --harmfulness-threshold 0.35 \
    --n-cycles 4 \
    --warmup-epochs 10
