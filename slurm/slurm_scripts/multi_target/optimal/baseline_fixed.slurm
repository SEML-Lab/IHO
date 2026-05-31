#!/bin/bash
#SBATCH --job-name=baseline_fixed
#SBATCH --output=slurm/slurm_logs/multi_target_multi_cycle_ablations/baseline_fixed/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/multi_target_multi_cycle_ablations/baseline_fixed/%x_%A_%a.err
#SBATCH --time=11:59:59
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --array=0-0

set -euo pipefail

########################################
# Read n-cycles argument
########################################

if [ "$#" -lt 1 ]; then
    echo "Error: n-cycles argument missing"
    echo "Usage: sbatch baseline_fixed.slurm <n_cycles>"
    exit 1
fi

N_CYCLES="$1"

########################################
# Models
########################################

MODELS=(
    #"Qwen/Qwen2.5-7B-Instruct"
    #"Qwen/Qwen2.5-7B-Instruct-with-detector"
    #"Qwen/Qwen2.5-32B-Instruct"
    #"meta-llama/Meta-Llama-3-8B-Instruct"
    #"GraySwanAI/Llama-3-8B-Instruct-RR"
    #"GraySwanAI/Llama-3-8B-Instruct-RR-with-detector"
    #"LLM-LAT/robust-llama3-8b-instruct"
    #"google/gemma-3-1b-it"
    "CAT/local"
)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

########################################
# Naming
########################################

RUN_NAME="baseline_fixed"

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

GPU_TYPE="h200"

if [[ "$MODEL" == "Qwen/Qwen2.5-32B-Instruct" ]]; then
    GPU_TYPE="custom"
fi

pixi run python -m slurm.slurm_wrapper \
    --operation train \
    --experiment-root evaluation/multi_target_multi_cycle_ablations \
    --sub-experiment-name "$RUN_NAME" \
    --gpu-type "$GPU_TYPE" \
    --training-rows THE_TRAINING_ONES_BIG_STRATIFIED \
    --validation-rows THE_VALIDATION_ONES_STRATIFIED \
    --dpo-rows THE_TRAINING_ONES_BIG_STRATIFIED \
    --model "$MODEL" \
    --learning-rate 0.0002 \
    --beta 0.15 \
    --dpo-epochs 361 \
    --checkpoint-every 40 \
    --save-checkpoints true \
    --patience 3 \
    --num-training-samples 15360 \
    --num-validation-samples 2048 \
    --num-dpo-samples 1024 \
    --percent-chosen 0.0625 \
    --harmfulness-threshold 0.35 \
    --n-cycles "$N_CYCLES"
