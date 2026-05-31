#!/bin/bash
#SBATCH --job-name=stratified_more_mask_tokens
#SBATCH --output=slurm/slurm_logs/multi_target_multi_cycle_ablations/stratified_more_mask_tokens/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/multi_target_multi_cycle_ablations/stratified_more_mask_tokens/%x_%A_%a.err
#SBATCH --time=23:59:59
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --array=0-5%6


MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "GraySwanAI/Llama-3-8B-Instruct-RR"
    "google/gemma-3-1b-it"
    "Qwen/Qwen2.5-7B-Instruct-with-detector"
    "GraySwanAI/Llama-3-8B-Instruct-RR-with-detector"
)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

########################################
# Naming
########################################

RUN_NAME="stratified_more_mask_tokens"

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
    --experiment-root evaluation/multi_target_multi_cycle_ablations \
    --sub-experiment-name "$RUN_NAME" \
    --gpu-type h200 \
    --training-rows THE_TRAINING_ONES_BIG_STRATIFIED \
    --validation-rows THE_VALIDATION_ONES_STRATIFIED \
    --dpo-rows THE_TRAINING_ONES_BIG_STRATIFIED \
    --attacker-input-size 64 \
    --attacker-step-number 64 \
    --mask-padding-tokens \
    --model "$MODEL" \
    --learning-rate 0.0002 \
    --beta 0.15 \
    --dpo-epochs 250 \
    --checkpoint-every 40 \
    --save-checkpoints true \
    --patience 1 \
    --num-training-samples 15360 \
    --num-validation-samples 2048 \
    --percent-chosen 0.0625 \
    --harmfulness-threshold 0.35 \
    --n-cycles 2
