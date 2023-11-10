#!/bin/bash
#SBATCH -p jic-medium
#SBATCH --mem 32000
#SBATCH --job-name="random_search_array"
#SBATCH -o slurm.random_search_array_%a.out
#SBATCH -e slurm.random_search_array_%a.err
#SBATCH --array=1-96

# Here I use the jic-medium queue, because I know that my scripts normally take between 2-48 hours.
# Also the --array value should match that in random_search_array_sample

# Randomly sample from the hyperparameter grid, producing CSV files
srun singularity exec ../tensorflow_model_train.img python3 random_search_array_sample.py --arraylen ${SLURM_ARRAY_TASK_COUNT} --perfile 36

# Make folders for each task to store the output
mkdir array_task${SLURM_ARRAY_TASK_ID}

# Run random search on each CSV file, saving output to individual folders
# This uses my singularity container. You don't have to worry about that for the time being, we will go over that next week.

singularity exec ../tensorflow_model_train.img python3 random_search_array.py --iteration ${SLURM_ARRAY_TASK_ID}

# Move output, error, and data files to their respective folders
if [ ${SLURM_ARRAY_TASK_ID} -eq 1 ]; then
    for ((task_id = 1; task_id <= SLURM_ARRAY_TASK_COUNT; task_id++)); do
        mv "random_samples${task_id}.csv" "./array_task${task_id}/"
        mv "slurm.random_search_array_${task_id}.out" "./array_task${task_id}/"
        mv "slurm.random_search_array_${task_id}.err" "./array_task${task_id}/"
    done
fi