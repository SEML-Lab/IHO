#!/bin/bash
#SBATCH --job-name=sample_baseline_current_base_config
#SBATCH --output=slurm/slurm_logs/sampling/baselines/current_base_config/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/sampling/baselines/current_base_config/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --array=0-3

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct-with-detector"
    "Qwen/Qwen2.5-32B-Instruct"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "GraySwanAI/Llama-3-8B-Instruct-RR"
    "GraySwanAI/Llama-3-8B-Instruct-RR-with-detector"
    "LLM-LAT/robust-llama3-8b-instruct"
    #"google/gemma-3-1b-it"
)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "================================================="
echo "SLURM JOB ID:      $SLURM_JOB_ID"
echo "ARRAY TASK ID:     $SLURM_ARRAY_TASK_ID"
echo "MODEL:             $MODEL"
echo "MODE:              NON-GREEDY SAMPLING"
echo "================================================="



pixi run python -m slurm.slurm_wrapper \
    --operation sample \
    --experiment-root evaluation/sanity \
    --sub-experiment-name sampling \
    --gpu-type a100 \
    --training-rows ALL \
    --model "$MODEL" \
    --num-training-samples 128
