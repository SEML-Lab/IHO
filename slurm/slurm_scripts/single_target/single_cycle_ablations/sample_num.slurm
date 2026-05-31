#!/bin/bash
#SBATCH --job-name=sample_num_ablation
#SBATCH --output=slurm/slurm_logs/single_target/single_cycle_ablations/sample_num_ablation/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/single_target/single_cycle_ablations/sample_num_ablation/%x_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --array=0-399%10   # 4 models × 10 rows × 10 sample counts = 400 jobs

########################################
# Models
########################################

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "GraySwanAI/Llama-3-8B-Instruct-RR"
    "google/gemma-3-1b-it"
)

########################################
# Explicit training rows
########################################

ROWS=(1 3 7 17 25 29 47 58 77 81)

########################################
# Sample counts
########################################

SAMPLES=(8 16 32 64 128 256 512 1024 2048 3072)

########################################
# Index decoding (model × row × sample grid)
########################################

NUM_MODELS=${#MODELS[@]}
NUM_ROWS=${#ROWS[@]}
NUM_SAMPLES=${#SAMPLES[@]}

JOBS_PER_MODEL=$((NUM_ROWS * NUM_SAMPLES))

MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / JOBS_PER_MODEL))
REMAINDER=$((SLURM_ARRAY_TASK_ID % JOBS_PER_MODEL))

ROW_INDEX=$((REMAINDER / NUM_SAMPLES))
SAMPLE_INDEX=$((REMAINDER % NUM_SAMPLES))

MODEL=${MODELS[$MODEL_INDEX]}
ROW_VALUE=${ROWS[$ROW_INDEX]}
NUM_SAMPLES_VALUE=${SAMPLES[$SAMPLE_INDEX]}

########################################
# Naming
########################################

RUN_NAME="samples_${NUM_SAMPLES_VALUE}"

echo "======================================="
echo "TASK:     $SLURM_ARRAY_TASK_ID"
echo "MODEL:    $MODEL"
echo "ROW:      $ROW_VALUE"
echo "SAMPLES:  $NUM_SAMPLES_VALUE"
echo "RUN:      $RUN_NAME"
echo "======================================="

########################################
# Environment
########################################


########################################
# Launch
########################################

pixi run python -m slurm.slurm_wrapper \
    --operation train \
    --experiment-root evaluation/single_target/single_cycle_ablations/sample_num \
    --sub-experiment-name "$RUN_NAME" \
    --gpu-type a100 \
    --training-rows "$ROW_VALUE" \
    --dpo-rows "$ROW_VALUE" \
    --model "$MODEL" \
    --dpo-epochs 500 \
    --checkpoint-every 50 \
    --save-checkpoints true \
    --patience 3 \
    --num-training-samples "$NUM_SAMPLES_VALUE" \
    --num-validation-samples 128 \
    --n-cycles 1 \
    --no-logprobs
