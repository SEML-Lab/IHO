#!/bin/bash
#SBATCH --job-name=attacker_temp_ablation
#SBATCH --output=slurm/slurm_logs/sampling_ablations/attacker_temperature/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/sampling_ablations/attacker_temperature/%x_%A_%a.err
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

TEMPERATURES=(0.25 0.5 0.75 1.0 1.25)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "================================================="
echo "SLURM JOB ID:       $SLURM_JOB_ID"
echo "SLURM ARRAY TASK:  $SLURM_ARRAY_TASK_ID"
echo "MODEL:             $MODEL"
echo "================================================="


for TEMP in "${TEMPERATURES[@]}"; do
    echo "Running with temperature: $TEMP"
    
    pixi run python -m slurm.slurm_wrapper \
        --operation sample \
        --experiment-root evaluation/sampling_ablations/attacker_temperature \
        --sub-experiment-name "temp_${TEMP}" \
        --gpu-type a100_small \
        --overwrite \
        --training-rows ALL \
        --mask-padding-tokens \
        --attacker-temperature "$TEMP" \
        --model "$MODEL" \
        --num-training-samples 5120
done
