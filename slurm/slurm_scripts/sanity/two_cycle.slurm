#!/bin/bash
#SBATCH --job-name=sample_baseline_current_base_config
#SBATCH --output=slurm/slurm_logs/sampling/baselines/current_base_config/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/sampling/baselines/current_base_config/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --array=0-3

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "GraySwanAI/Llama-3-8B-Instruct-RR"
    "google/gemma-3-1b-it"
)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "================================================="
echo "SLURM JOB ID:      $SLURM_JOB_ID"
echo "ARRAY TASK ID:     $SLURM_ARRAY_TASK_ID"
echo "MODEL:             $MODEL"
echo "MODE:              NON-GREEDY SAMPLING"
echo "================================================="



pixi run python -m slurm.slurm_wrapper \
    --operation train \
    --experiment-root evaluation/sampling_baselines \
    --sub-experiment-name current_base_config \
    --gpu-type h200 \
    --training-rows ALL \
    --model "$MODEL" \
    --num-training-samples 128 \
    --n-cycles 1
