#!/bin/bash
#SBATCH --job-name=optimal_single_cycle
#SBATCH --output=slurm/slurm_logs/single_target/baselines/optimal_single_cycle/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/single_target/baselines/optimal_single_cycle/%x_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --array=0-799%12   # 6 models × 60 rows = 360 jobs

########################################
# Read n-cycles from first positional arg
########################################
N_CYCLES="${1}"

if [ -z "$N_CYCLES" ]; then
    echo "Error: n-cycles argument missing."
    echo "Usage: sbatch optimal_single_cycle.sbatch <n-cycles>"
    exit 1
fi

########################################
# Models
########################################
MODELS=(
    "CAT/local"
    "google/gemma-3-1b-it"
    "GraySwanAI/Llama-3-8B-Instruct-RR"
    "GraySwanAI/Llama-3-8B-Instruct-RR-with-detector"
    "LLM-LAT/robust-llama3-8b-instruct"    
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct-with-detector"
    #"Qwen/Qwen2.5-32B-Instruct"
)

########################################
# Explicit training rows
########################################
ROWS=($(seq 0 99))

########################################
# Index decoding (model × row grid)
########################################
NUM_ROWS=${#ROWS[@]}
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_ROWS))
ROW_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_ROWS))
MODEL=${MODELS[$MODEL_INDEX]}
ROW_VALUE=${ROWS[$ROW_INDEX]}

########################################
# Naming
########################################
RUN_NAME="optimal_single_target"

echo "======================================="
echo "TASK:      $SLURM_ARRAY_TASK_ID"
echo "MODEL:     $MODEL"
echo "ROW:       $ROW_VALUE"
echo "RUN:       $RUN_NAME"
echo "N_CYCLES:  $N_CYCLES"
echo "======================================="

########################################
# Environment
########################################

########################################
# Launch
########################################
pixi run python -m slurm.slurm_wrapper \
    --operation train \
    --experiment-root evaluation/single_target/baselines \
    --sub-experiment-name "$RUN_NAME" \
    --gpu-type a100 \
    --training-rows "$ROW_VALUE" \
    --validation-rows "$ROW_VALUE" \
    --dpo-rows "$ROW_VALUE" \
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
    --n-cycles "$N_CYCLES" \
    --no-logprobs
