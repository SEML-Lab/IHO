#!/bin/bash
#SBATCH --job-name=single_target_lora_modules
#SBATCH --output=slurm/slurm_logs/single_target_single_cycle_ablations/lora_modules/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/single_target_single_cycle_ablations/lora_modules/%x_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --array=0-79

########################################
# Grid
########################################

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "GraySwanAI/Llama-3-8B-Instruct-RR"
    "google/gemma-3-1b-it"
)

STRUCTURES=("baseline" "four")

ROWS=(1 3 7 17 25 29 47 58 77 81)

########################################
# Index math
########################################

N_ROWS=10
N_STRUCT=2
N_MODEL=4

TOTAL_PER_MODEL=$((N_STRUCT * N_ROWS))  # 20

MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / TOTAL_PER_MODEL))
REM=$((SLURM_ARRAY_TASK_ID % TOTAL_PER_MODEL))

STRUCT_INDEX=$((REM / N_ROWS))
ROW_INDEX=$((REM % N_ROWS))

MODEL=${MODELS[$MODEL_INDEX]}
STRUCT=${STRUCTURES[$STRUCT_INDEX]}
ROW=${ROWS[$ROW_INDEX]}

########################################
# LoRA configuration
########################################

RANK=8
ALPHA=16
DROPOUT=0.05
RS=false

if [ "$STRUCT" = "baseline" ]; then
    TARGETS="q_proj,v_proj"
else
    TARGETS="q_proj,k_proj,v_proj,o_proj"
fi

########################################
# Naming
########################################

RUN_NAME="${STRUCT}"

echo "======================================="
echo "JOB ID:    $SLURM_JOB_ID"
echo "TASK ID:   $SLURM_ARRAY_TASK_ID"
echo "MODEL:     $MODEL"
echo "STRUCTURE: $STRUCT"
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
    --experiment-root evaluation/single_target_single_cycle_ablations/lora_modules \
    --sub-experiment-name "$RUN_NAME" \
    --gpu-type a100 \
    --training-rows $ROW \
    --mask-padding-tokens \
    --model "$MODEL" \
    --greedy \
    --dpo-epochs 200 \
    --checkpoint-every 201 \
    --save-checkpoints true \
    --patience 1 \
    --num-training-samples 512 \
    --num-validation-samples 128 \
    --n-cycles 1 \
    --lora-r $RANK \
    --lora-alpha $ALPHA \
    --lora-dropout $DROPOUT \
    --lora-target-modules $TARGETS \
    --lora-use-rslora $RS
