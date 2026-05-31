#!/bin/bash
#SBATCH --job-name=sample_num
#SBATCH --output=slurm/slurm_logs/single_target_single_cycle_ablations/sample_num/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/single_target_single_cycle_ablations/sample_num/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --array=0-3

########################################
# Grid
########################################

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "GraySwanAI/Llama-3-8B-Instruct-RR"
    "google/gemma-3-1b-it"
)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "======================================="
echo "TASK:   $SLURM_ARRAY_TASK_ID"
echo "MODEL:  $MODEL"
echo "======================================="

########################################
# Env
########################################


########################################
# Launch
########################################

pixi run python -m slurm.slurm_wrapper \
    --operation sample \
    --experiment-root evaluation/single_target_single_cycle_ablations/sample_num \
    --sub-experiment-name samples_4096 \
    --iterate-rows-separately \
    --gpu-type h200 \
    --training-rows THE_TRAINING_ONES_SMALL \
    --mask-padding-tokens \
    --model "$MODEL" \
    --greedy \
    --num-training-samples 4096
