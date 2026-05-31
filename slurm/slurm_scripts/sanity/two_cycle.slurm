#!/bin/bash
#SBATCH --job-name=sanity_two_cycle
#SBATCH --output=slurm/slurm_logs/sanity/two_cycle/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/sanity/two_cycle/%x_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --array=0-0

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "================================================="
echo "SLURM JOB ID:      $SLURM_JOB_ID"
echo "ARRAY TASK ID:     $SLURM_ARRAY_TASK_ID"
echo "MODEL:             $MODEL"
echo "MODE:              SANITY"
echo "================================================="



pixi run python -m slurm.slurm_wrapper \
    --operation train \
    --experiment-root evaluation/sanity \
    --sub-experiment-name iho_pipeline \
    --gpu-type a100 \
    --training-rows ALL \
    --attacker-input-size 32 \
    --attacker-step-number 32 \
    --model "$MODEL" \
    --learning-rate 0.00005 \
    --beta 0.15 \
    --dpo-epochs 280 \
    --checkpoint-every 40 \
    --save-checkpoints true \
    --patience 2 \
    --num-training-samples 1024 \
    --num-validation-samples 128 \
    --percent-chosen 0.0625 \
    --harmfulness-threshold 0.35 \
    --n-cycles 2 \
    --no-logprobs