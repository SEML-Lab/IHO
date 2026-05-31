#!/bin/bash
#SBATCH --job-name=new_config
#SBATCH --output=slurm/slurm_logs/sampling/baselines/new_config/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/sampling/baselines/new_config/%x_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --array=0-5

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "GraySwanAI/Llama-3-8B-Instruct-RR"
    "google/gemma-3-1b-it"
    "Qwen/Qwen2.5-7B-Instruct-with-detector"
    "GraySwanAI/Llama-3-8B-Instruct-RR-with-detector"
)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "================================================="
echo "SLURM JOB ID:      $SLURM_JOB_ID"
echo "ARRAY TASK ID:     $SLURM_ARRAY_TASK_ID"
echo "MODEL:             $MODEL"
echo "================================================="



pixi run python -m slurm.slurm_wrapper \
    --operation sample \
    --experiment-root evaluation/sampling_baselines \
    --sub-experiment-name new_config \
    --gpu-type a100 \
    --training-rows ALL \
    --model "$MODEL" \
    --num-training-samples 5120
