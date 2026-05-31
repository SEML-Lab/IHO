#!/bin/bash
#SBATCH --job-name=sanity_sampling
#SBATCH --output=slurm/slurm_logs/sanity/sampling/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/sanity/sampling/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --array=0-1

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct-with-detector"
)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "================================================="
echo "SLURM JOB ID:      $SLURM_JOB_ID"
echo "ARRAY TASK ID:     $SLURM_ARRAY_TASK_ID"
echo "MODEL:             $MODEL"
echo "MODE:              SANITY"
echo "================================================="



pixi run python -m slurm.slurm_wrapper \
    --operation sample \
    --experiment-root evaluation/sanity \
    --sub-experiment-name sampling \
    --gpu-type a100 \
    --training-rows ALL \
    --model "$MODEL" \
    --num-training-samples 128
