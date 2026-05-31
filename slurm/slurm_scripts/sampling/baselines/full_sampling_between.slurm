#!/bin/bash
#SBATCH --job-name=partial_sampling_between
#SBATCH --output=slurm/slurm_logs/sampling/baselines/partial_sampling_between/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/sampling/baselines/partial_sampling_between/%x_%A_%a.err
#SBATCH --time=3:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --array=0-99%10

MODELS=(
    #"Qwen/Qwen2.5-7B-Instruct"
    #"meta-llama/Meta-Llama-3-8B-Instruct"
    "GraySwanAI/Llama-3-8B-Instruct-RR"
    #"google/gemma-3-1b-it"
    #"Qwen/Qwen2.5-7B-Instruct-with-detector"
    #"GraySwanAI/Llama-3-8B-Instruct-RR-with-detector"
)

NUM_MODELS=${#MODELS[@]}

MODEL_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_MODELS))
ID_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_MODELS))

MODEL="${MODELS[$MODEL_INDEX]}"
SUB_EXPERIMENT_NAME="partial_sampling_between_id/${ID_INDEX}"

echo "================================================="
echo "SLURM JOB ID:         $SLURM_JOB_ID"
echo "ARRAY TASK ID:        $SLURM_ARRAY_TASK_ID"
echo "MODEL INDEX:          $MODEL_INDEX"
echo "ID INDEX:             $ID_INDEX"
echo "MODEL:                $MODEL"
echo "SUB EXPERIMENT NAME:  $SUB_EXPERIMENT_NAME"
echo "TRAINING ROWS:        ALL"
echo "================================================="


pixi run python -m slurm.slurm_wrapper \
    --operation sample \
    --experiment-root evaluation/sampling_baselines \
    --sub-experiment-name "$SUB_EXPERIMENT_NAME" \
    --gpu-type a100 \
    --training-rows ALL \
    --model "$MODEL" \
    --attacker-input-size 64 \
    --attacker-step-number 64 \
    --num-training-samples 2048 \
    --judge-validation-2 aegis_guard
