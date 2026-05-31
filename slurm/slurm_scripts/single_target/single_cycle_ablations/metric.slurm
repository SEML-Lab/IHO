#!/bin/bash
#SBATCH --job-name=metric_ablation
#SBATCH --output=slurm/slurm_logs/single_target_single_cycle_ablations/metric_ablation/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/single_target_single_cycle_ablations/metric_ablation/%x_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --array=0-179%8

########################################
# Grid
########################################

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct-with-detector"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "GraySwanAI/Llama-3-8B-Instruct-RR"
    "GraySwanAI/Llama-3-8B-Instruct-RR-with-detector"
    "google/gemma-3-1b-it"
)

DPO_METRICS=("max" "mean" "weighted")

ROWS=(1 3 7 17 25 29 47 58 77 81)

########################################
# Index math
########################################

N_MODEL=${#MODELS[@]}
N_METRIC=${#DPO_METRICS[@]}
N_ROWS=${#ROWS[@]}

TOTAL_PER_MODEL=$((N_METRIC * N_ROWS))

MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / TOTAL_PER_MODEL))
REM=$((SLURM_ARRAY_TASK_ID % TOTAL_PER_MODEL))

METRIC_INDEX=$((REM / N_ROWS))
ROW_INDEX=$((REM % N_ROWS))

MODEL=${MODELS[$MODEL_INDEX]}
METRIC=${DPO_METRICS[$METRIC_INDEX]}
ROW=${ROWS[$ROW_INDEX]}

########################################
# Naming
########################################

RUN_NAME="metric_${METRIC}"

echo "======================================="
echo "JOB ID:    $SLURM_JOB_ID"
echo "TASK ID:   $SLURM_ARRAY_TASK_ID"
echo "MODEL:     $MODEL"
echo "METRIC:    $METRIC"
echo "ROW:       $ROW"
echo "RUN NAME:  $RUN_NAME"
echo "======================================="

########################################
# Environment
########################################


########################################
# Launch
########################################

pixi run python -m slurm.slurm_wrapper \
    --operation train \
    --experiment-root evaluation/single_target_single_cycle_ablations/metric_ablation \
    --sub-experiment-name "$RUN_NAME" \
    --gpu-type a100 \
    --training-rows $ROW \
    --validation-rows $ROW \
    --dpo-rows $ROW \
    --model "$MODEL" \
    --greedy \
    --dpo-epochs 500 \
    --checkpoint-every 50 \
    --save-checkpoints true \
    --patience 2 \
    --num-training-samples 512 \
    --num-validation-samples 128 \
    --n-cycles 2 \
    --seed 0 \
    --dpo-metric $METRIC
