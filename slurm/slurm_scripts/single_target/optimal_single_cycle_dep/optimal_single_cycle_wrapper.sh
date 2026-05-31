#!/bin/bash

SBATCH_FILE="IHO/slurm/slurm_scripts/single_target/optimal_single_cycle_dep/optimal_single_cycle_template.slurm"

prev_job_id=""

for cycle in 1 2 3 4 5 6; do
    if [ -z "$prev_job_id" ]; then
        submit_output=$(sbatch "$SBATCH_FILE" "$cycle")
    else
        submit_output=$(sbatch --dependency=afterany:${prev_job_id} "$SBATCH_FILE" "$cycle")
    fi

    job_id=$(echo "$submit_output" | awk '{print $4}')

    if [ -z "$job_id" ]; then
        echo "Failed to submit cycle $cycle"
        echo "sbatch output: $submit_output"
        exit 1
    fi

    echo "Submitted cycle $cycle with job ID $job_id"
    prev_job_id="$job_id"
done