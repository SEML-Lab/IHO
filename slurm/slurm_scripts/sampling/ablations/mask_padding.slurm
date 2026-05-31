#!/bin/bash
#SBATCH --job-name=mask_padding_ablation
#SBATCH --output=slurm/slurm_logs/sampling_ablations/mask_padding/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/sampling_ablations/mask_padding/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_a100
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
echo "SLURM JOB ID:       $SLURM_JOB_ID"
echo "SLURM ARRAY TASK:  $SLURM_ARRAY_TASK_ID"
echo "MODEL:             $MODEL"
echo "================================================="


pixi run python -m slurm.slurm_wrapper \
    --operation sample \
    --experiment-root evaluation/sampling_ablations/mask_padding \
    --sub-experiment-name with_masking \
    --gpu-type a100 \
    --overwrite \
    --training-rows ALL \
    --mask-padding-tokens \
    --model "$MODEL" \
    --greedy \
    --num-training-samples 512

pixi run python -m slurm.slurm_wrapper \
    --operation sample \
    --experiment-root evaluation/sampling_ablations/mask_padding \
    --sub-experiment-name without_masking \
    --gpu-type a100 \
    --overwrite \
    --training-rows ALL \
    --model "$MODEL" \
    --num-training-samples 512
