#!/bin/bash
#SBATCH --job-name=beta_lr_ablation
#SBATCH --output=slurm/slurm_logs/single_target_single_cycle_ablations/beta_lr_ablation/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/single_target_single_cycle_ablations/beta_lr_ablation/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --ntasks=1
#SBATCH --array=0-15

########################################
# Grid (BETA fixed to 0.3)
########################################

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "GraySwanAI/Llama-3-8B-Instruct-RR"
    "google/gemma-3-1b-it"
)

LRS=(1e-4 5e-4 1e-5 5e-5)

BETA=0.3

########################################
# Index math
########################################

N_LR=${#LRS[@]}        # 4
N_MODEL=${#MODELS[@]}  # 4

LR_INDEX=$((SLURM_ARRAY_TASK_ID % N_LR))
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / N_LR))

LR=${LRS[$LR_INDEX]}
MODEL=${MODELS[$MODEL_INDEX]}

########################################
# Naming
########################################

RUN_NAME="beta_${BETA}_lr_${LR}"

echo "======================================="
echo "TASK:   $SLURM_ARRAY_TASK_ID"
echo "MODEL:  $MODEL"
echo "BETA:   $BETA"
echo "LR:     $LR"
echo "RUN:    $RUN_NAME"
echo "======================================="

########################################
# Env
########################################


########################################
# Launch
########################################

pixi run python -m slurm.slurm_wrapper \
    --operation train \
    --experiment-root evaluation/single_target_single_cycle_ablations/beta_lr_ablation \
    --sub-experiment-name "$RUN_NAME" \
    --iterate-rows-separately \
    --gpu-type a100 \
    --training-rows THE_TRAINING_ONES_SMALL \
    --mask-padding-tokens \
    --model "$MODEL" \
    --greedy \
    --learning-rate "$LR" \
    --beta "$BETA" \
    --dpo-epochs 500 \
    --checkpoint-every 50 \
    --save-checkpoints true \
    --patience 3 \
    --num-training-samples 1024 \
    --num-validation-samples 128 \
    --n-cycles 1
