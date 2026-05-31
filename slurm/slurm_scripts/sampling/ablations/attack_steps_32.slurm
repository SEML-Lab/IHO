#!/bin/bash
#SBATCH --job-name=attack_step_32_ablation_paper
#SBATCH --output=slurm/slurm_logs/sampling/ablations/attack_steps/32/%x_%A_%a.out
#SBATCH --error=slurm/slurm_logs/sampling/ablations/attack_steps/32/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=5G
#SBATCH --ntasks=1
#SBATCH --array=0-23   # 4 models × 4 steps = 16 jobs

MODELS=(
    #"CAT/local"
    "google/gemma-3-1b-it"
    "GraySwanAI/Llama-3-8B-Instruct-RR"
    "GraySwanAI/Llama-3-8B-Instruct-RR-with-detector"
    #"LLM-LAT/robust-llama3-8b-instruct"    
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct-with-detector"
    #"Qwen_Qwen2.5-32B-Instruct"
)

ATTACK_STEPS=(8 16 24 32)

FIXED_ATTACK_SIZE=32

NUM_STEPS=${#ATTACK_STEPS[@]}

# Map SLURM index → model + step
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / NUM_STEPS))
STEP_IDX=$((SLURM_ARRAY_TASK_ID % NUM_STEPS))

MODEL="${MODELS[$MODEL_IDX]}"
STEP="${ATTACK_STEPS[$STEP_IDX]}"

echo "================================================="
echo "SLURM JOB ID:      $SLURM_JOB_ID"
echo "SLURM ARRAY TASK:  $SLURM_ARRAY_TASK_ID"
echo "MODEL IDX:         $MODEL_IDX"
echo "STEP IDX:          $STEP_IDX"
echo "MODEL:             $MODEL"
echo "ATTACK STEPS:      $STEP"
echo "================================================="


pixi run python -m slurm.slurm_wrapper \
    --operation sample \
    --experiment-root evaluation/sampling/sampling/ablations/attack_steps/attack_size_32 \
    --sub-experiment-name "steps_${STEP}" \
    --gpu-type a100 \
    --training-rows ALL \
    --attacker-input-size "$FIXED_ATTACK_SIZE" \
    --attacker-step-number "$STEP" \
    --model "$MODEL" \
    --num-training-samples 15000 \
    --no-logprobs
