#!/bin/bash
#SBATCH --job-name=stratified_lower_beta
#SBATCH --output=slurm/slurm_logs/multi_target_multi_cycle_ablations/stratified_lower_beta/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/multi_target_multi_cycle_ablations/stratified_lower_beta/%x_%A_%a.err
#SBATCH --time=23:59:59
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --array=0-5%6

set -euo pipefail

########################################
# Read n-cycles argument
########################################

if [ "$#" -lt 1 ]; then
    echo "Error: n-cycles argument missing"
    echo "Usage: sbatch stratified_lower_beta.slurm <n_cycles>"
    exit 1
fi

N_CYCLES="$1"

########################################
# Models
########################################

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

RUN_NAME="stratified_lower_beta"

echo "======================================="
echo "JOB:        $SLURM_JOB_ID"
echo "TASK:       $SLURM_ARRAY_TASK_ID"
echo "MODEL:      $MODEL"
echo "RUN:        $RUN_NAME"
echo "N_CYCLES:   $N_CYCLES"
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
    --model "$MODEL" \
    --learning-rate 0.0002 \
    --beta 0.05 \
    --dpo-epochs 250 \
    --checkpoint-every 40 \
    --save-checkpoints true \
    --patience 1 \
    --num-training-samples 15360 \
    --num-validation-samples 2048 \
    --percent-chosen 0.0625 \
    --harmfulness-threshold 0.35 \
    --n-cycles "$N_CYCLES"
