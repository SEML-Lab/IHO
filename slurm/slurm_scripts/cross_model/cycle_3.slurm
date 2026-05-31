#!/bin/bash
#SBATCH --job-name=sample_cross_model_cycle_3
#SBATCH --output=slurm/slurm_logs/cross_model/sampling/cycle_3/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/cross_model/sampling/cycle_3/%x_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15G
#SBATCH --ntasks=1
#SBATCH --array=0-79%8

# -----------------------------
# Models
# -----------------------------
CHECKPOINT_MODELS=(
    "CAT_local"
    "google_gemma-3-1b-it"
    "GraySwanAI_Llama-3-8B-Instruct-RR"
    "GraySwanAI_Llama-3-8B-Instruct-RR-with-detector"
    "LLM-LAT_robust-llama3-8b-instruct"    
    "meta-llama_Meta-Llama-3-8B-Instruct"
    "Qwen_Qwen2.5-7B-Instruct"
    "Qwen_Qwen2.5-7B-Instruct-with-detector"
    #"Qwen_Qwen2.5-32B-Instruct"
)

MODELS=(
    #"CAT/local"
    "google/gemma-3-1b-it"
    #"GraySwanAI/Llama-3-8B-Instruct-RR"
    #"GraySwanAI/Llama-3-8B-Instruct-RR-with-detector"
    #"LLM-LAT/robust-llama3-8b-instruct"    
    #"meta-llama/Meta-Llama-3-8B-Instruct"
    #"Qwen/Qwen2.5-7B-Instruct"
    #"Qwen/Qwen2.5-7B-Instruct-with-detector"
    #"Qwen/Qwen2.5-32B-Instruct"
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

SUB_EXPERIMENT_NAME="${CHECKPOINT_MODEL}_cycle_3/seed_${SEED}"

CHECKPOINT="${IHO_CHECKPOINT_ROOT:-evaluation/multi_target/multi_cycle_ablations/baseline_fixed}/${CHECKPOINT_MODEL}/THE_TRAINING_ONES_BIG_STRATIFIED/checkpoints/best_cycle_3"

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
    --experiment-root evaluation/cross_model/sampling/samples_partial \
    --sub-experiment-name "$SUB_EXPERIMENT_NAME" \
    --gpu-type a100 \
    --training-rows THE_TRAINING_ONES_BIG_STRATIFIED \
    --validation-rows THE_VALIDATION_ONES_STRATIFIED \
    --testing-rows THE_TEST_HELD_OUT_STRATIFIED \
    --model "$MODEL" \
    --lora-checkpoint-path "$CHECKPOINT" \
    --num-training-samples 7680 \
    --num-validation-samples 2560 \
    --num-test-samples 2560 \
    --seed "$SEED" \
    --no-logprobs
