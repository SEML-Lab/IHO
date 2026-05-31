#!/bin/bash
#SBATCH --job-name=attack_size_ablation_paper
#SBATCH --output=slurm/slurm_logs/sampling/ablations/attack_size/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/sampling/ablations/attack_size/%x_%A_%a.err
#SBATCH --time=14:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=5G
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

ATTACK_SIZES=(8 16 32 64 96)

N_SIZES=${#ATTACK_SIZES[@]}   # 6

MODEL_IDX=$(( SLURM_ARRAY_TASK_ID % N_SIZES ))
SIZE_IDX=$(( SLURM_ARRAY_TASK_ID / N_SIZES ))

MODEL="${MODELS[$MODEL_IDX]}"
SIZE="${ATTACK_SIZES[$SIZE_IDX]}"

echo "================================================="
echo "SLURM JOB ID:      $SLURM_JOB_ID"
echo "SLURM ARRAY TASK:  $SLURM_ARRAY_TASK_ID"
echo "MODEL IDX:         $MODEL_IDX  →  $MODEL"
echo "SIZE IDX:          $SIZE_IDX   →  $SIZE"
echo "================================================="


echo "Running with attack_size = $SIZE (step_size = $SIZE)"

pixi run python -m slurm.slurm_wrapper \
    --operation sample \
    --experiment-root evaluation/sampling/ablations/attack_size \
    --sub-experiment-name "size_${SIZE}" \
    --gpu-type a100_small \
    --training-rows ALL \
    --attacker-input-size "$SIZE" \
    --attacker-step-number "$SIZE" \
    --model "$MODEL" \
    --num-training-samples 15000 \
    --no-logprobs \
    --overwrite
