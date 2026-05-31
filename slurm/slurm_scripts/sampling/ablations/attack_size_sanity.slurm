#!/bin/bash
#SBATCH --job-name=harmbench_attack_size
#SBATCH --output=slurm/slurm_logs/sampling_ablations/harmbench_attack_size/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/sampling_ablations/harmbench_attack_size/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --array=0-95

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct-with-detector"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "GraySwanAI/Llama-3-8B-Instruct-RR"
    "GraySwanAI/Llama-3-8B-Instruct-RR-with-detector"
    "google/gemma-3-1b-it"
)

ATTACK_SIZES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 24)

N_SIZES=${#ATTACK_SIZES[@]}   # 16

MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / N_SIZES ))
SIZE_IDX=$(( SLURM_ARRAY_TASK_ID % N_SIZES ))

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
    --experiment-root evaluation/sampling_ablations/harmbench_attack_size \
    --sub-experiment-name "size_${SIZE}" \
    --gpu-type a100_small \
    --training-rows ALL \
    --attacker-input-size "$SIZE" \
    --attacker-step-number "$SIZE" \
    --judge-training harmbench \
    --model "$MODEL" \
    --num-training-samples 5120
