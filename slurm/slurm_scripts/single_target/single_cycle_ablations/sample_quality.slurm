#!/bin/bash
#SBATCH --job-name=sample_quality
#SBATCH --output=slurm/slurm_logs/single_target/single_cycle_ablations/sample_quality/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/single_target/single_cycle_ablations/sample_quality/%x_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --array=0-239%8   # 4 models × 10 rows × 6 quality thresholds = 240 jobs

set -euo pipefail

########################################
# Paths
########################################

PROJECT_ROOT="${IHO_PROJECT_ROOT:-$PWD}"
EXPERIMENT_REL="evaluation/single_target/single_cycle_ablations/sample_quality"
EXPERIMENT_ROOT="${PROJECT_ROOT}/${EXPERIMENT_REL}"

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
# Explicit training rows / jb_index values
########################################

ROWS=(0 19 20 39 40 55 62 78 84 91)

########################################
# Quality threshold folders
########################################

QUALITY_RUNS=(
    "quality_0.2_1.0"
    "quality_0.3_1.0"
    "quality_0.4_1.0"
    "quality_0.5_1.0"
    "quality_0.6_1.0"
    "quality_0.7_1.0"
)

########################################
# Fixed percent chosen
########################################

PERCENT_CHOSEN=0.5

########################################
# Index decoding: model × row × quality run
########################################

NUM_MODELS=${#MODELS[@]}
NUM_ROWS=${#ROWS[@]}
NUM_QUALITY_RUNS=${#QUALITY_RUNS[@]}

JOBS_PER_MODEL=$((NUM_ROWS * NUM_QUALITY_RUNS))
TOTAL_JOBS=$((NUM_MODELS * JOBS_PER_MODEL))

if [[ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_JOBS" ]]; then
    echo "Invalid SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID; TOTAL_JOBS=$TOTAL_JOBS"
    exit 1
fi

MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / JOBS_PER_MODEL))
REMAINDER=$((SLURM_ARRAY_TASK_ID % JOBS_PER_MODEL))

ROW_INDEX=$((REMAINDER / NUM_QUALITY_RUNS))
QUALITY_INDEX=$((REMAINDER % NUM_QUALITY_RUNS))

MODEL=${MODELS[$MODEL_INDEX]}
ROW_VALUE=${ROWS[$ROW_INDEX]}
RUN_NAME=${QUALITY_RUNS[$QUALITY_INDEX]}

# Folder names use "/" replaced by "_", e.g.
# google/gemma-3-1b-it -> google_gemma-3-1b-it
MODEL_DIR="${MODEL//\//_}"

SAMPLE_FILE="${EXPERIMENT_ROOT}/${RUN_NAME}/${MODEL_DIR}/row_${ROW_VALUE}/samples/cycle_0.parquet"

########################################
# Logging
########################################

echo "======================================="
echo "TASK:            $SLURM_ARRAY_TASK_ID"
echo "MODEL:           $MODEL"
echo "MODEL DIR:       $MODEL_DIR"
echo "ROW / JB_INDEX:  $ROW_VALUE"
echo "QUALITY RUN:     $RUN_NAME"
echo "PERCENT CHOSEN:  $PERCENT_CHOSEN"
echo "SAMPLE FILE:     $SAMPLE_FILE"
echo "======================================="

########################################
# Skip non-viable setups
########################################

if [[ ! -f "$SAMPLE_FILE" ]]; then
    echo "Skipping non-viable setup: sample file does not exist."
    echo "Missing: $SAMPLE_FILE"
    exit 0
fi

echo "Viable setup found. Launching training."

########################################
# Environment
########################################


cd "$PROJECT_ROOT"

########################################
# Launch
########################################

pixi run python -m slurm.slurm_wrapper \
    --operation train \
    --experiment-root "$EXPERIMENT_REL" \
    --sub-experiment-name "$RUN_NAME" \
    --gpu-type a100 \
    --training-rows "$ROW_VALUE" \
    --validation-rows "$ROW_VALUE" \
    --dpo-rows "$ROW_VALUE" \
    --model "$MODEL" \
    --percent-chosen "$PERCENT_CHOSEN" \
    --harmfulness-threshold 0.1 \
    --dpo-epochs 500 \
    --checkpoint-every 50 \
    --save-checkpoints true \
    --patience 3 \
    --num-training-samples 128 \
    --num-validation-samples 0 \
    --num-dpo-samples 128 \
    --n-cycles 1 \
    --no-logprobs
