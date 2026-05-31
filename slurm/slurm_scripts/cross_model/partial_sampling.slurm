#!/bin/bash
#SBATCH --job-name=sample_cross_model_cycle_partial
#SBATCH --output=slurm/slurm_logs/multi_target_cross_model_eval/baseline_partial/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/multi_target_cross_model_eval/baseline_partial/%x_%A_%a.err
#SBATCH --time=2:30:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --array=0-359%20

# -----------------------------
# Models
# -----------------------------
CHECKPOINT_MODELS=(
    "Qwen_Qwen2.5-7B-Instruct"
    "meta-llama_Meta-Llama-3-8B-Instruct"
    "GraySwanAI_Llama-3-8B-Instruct-RR"
    "google_gemma-3-1b-it"
    "Qwen_Qwen2.5-7B-Instruct-with-detector"
    "GraySwanAI_Llama-3-8B-Instruct-RR-with-detector"
)

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "GraySwanAI/Llama-3-8B-Instruct-RR"
    "google/gemma-3-1b-it"
    "Qwen/Qwen2.5-7B-Instruct-with-detector"
    "GraySwanAI/Llama-3-8B-Instruct-RR-with-detector"
)

# -----------------------------
# Dimensions
# -----------------------------
N_CHECKPOINTS=${#CHECKPOINT_MODELS[@]}
N_MODELS=${#MODELS[@]}
N_SEEDS=10

TOTAL_COMBINATIONS=$((N_CHECKPOINTS * N_MODELS))

# -----------------------------
# Index decoding
# -----------------------------
SEED_IDX=$((SLURM_ARRAY_TASK_ID / TOTAL_COMBINATIONS))
COMBO_IDX=$((SLURM_ARRAY_TASK_ID % TOTAL_COMBINATIONS))

CHECKPOINT_IDX=$((COMBO_IDX / N_MODELS))
MODEL_IDX=$((COMBO_IDX % N_MODELS))

# -----------------------------
# Assign values
# -----------------------------
CHECKPOINT_MODEL="${CHECKPOINT_MODELS[$CHECKPOINT_IDX]}"
MODEL="${MODELS[$MODEL_IDX]}"
SEED="$SEED_IDX"

SUB_EXPERIMENT_NAME="${CHECKPOINT_MODEL}/seed_${SEED}"

CHECKPOINT="${IHO_CHECKPOINT_ROOT:-evaluation/multi_target_multi_cycle_ablations/baseline}/${CHECKPOINT_MODEL}/THE_TRAINING_ONES_BIG/checkpoints/best_cycle_2"

# -----------------------------
# Debug info
# -----------------------------
echo "================================================="
echo "SLURM JOB ID:          $SLURM_JOB_ID"
echo "ARRAY TASK ID:         $SLURM_ARRAY_TASK_ID"
echo "SEED IDX:              $SEED_IDX"
echo "COMBO IDX:             $COMBO_IDX"
echo "CHECKPOINT IDX:        $CHECKPOINT_IDX"
echo "MODEL IDX:             $MODEL_IDX"
echo "SEED:                  $SEED"
echo "CHECKPOINT MODEL:      $CHECKPOINT_MODEL"
echo "TARGET MODEL:          $MODEL"
echo "SUB EXPERIMENT NAME:   $SUB_EXPERIMENT_NAME"
echo "CHECKPOINT PATH:       $CHECKPOINT"
echo "================================================="

# -----------------------------
# Environment
# -----------------------------

# -----------------------------
# Run
# -----------------------------
pixi run python -m slurm.slurm_wrapper \
    --operation sample \
    --experiment-root evaluation/multi_target_cross_model_eval_full \
    --sub-experiment-name "$SUB_EXPERIMENT_NAME" \
    --gpu-type a100 \
    --training-rows THE_TRAINING_ONES_BIG \
    --validation-rows THE_VALIDATION_ONES \
    --testing-rows THE_TEST_HELD_OUT_STRATIFIED \
    --model "$MODEL" \
    --lora-checkpoint-path "$CHECKPOINT" \
    --num-training-samples 5120 \
    --num-validation-samples 512 \
    --seed "$SEED"
