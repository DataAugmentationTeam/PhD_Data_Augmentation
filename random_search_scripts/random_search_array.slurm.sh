#!/bin/bash
#SBATCH -p jic-medium
#SBATCH --mem 32000
#SBATCH --job-name="random_search_array"
#SBATCH -o slurm.random_search_array_%a.out
#SBATCH -e slurm.random_search_array_%a.err
#SBATCH --array=1-3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jowillia@nbi.ac.uk

model_number="99"


# STEP 1: DATASET CREATION AND TRAINING HYPERPARAMETER SAMPLING

# Flag to make sure that training only begins once the dataset and csvs have been built
flag_start="flag_start.flag"

# Check if this is the first task in the array
if [ "${SLURM_ARRAY_TASK_ID}" -eq 1 ]; then
    # Randomly sample from the hyperparameter grid, producing CSV files
    srun singularity exec ../tensorflow_model_train.img python3 random_search_array_sample.py --arraylen ${SLURM_ARRAY_TASK_COUNT} --perfile 3 --model_number "${model_number}"

    # Create a flag file to signal completion
    touch "${flag_start}"
else
    # Wait for the setup_complete_flag file to be created by the first task
    while [ ! -f "${flag_start}" ]; do
        sleep 10 # Wait for 10 seconds before checking again
    done
fi


# STEP 2: MODEL TRAINING

# Raise a flag for each array once the model training is complete
flagdir="flags"
mkdir -p "${flag_dir}"
task_flag="${flag_dir}/flag_${SLURM_ARRAY_TASK_ID}.flag"

# Make folders for each task to store the output
mkdir array_task${SLURM_ARRAY_TASK_ID}

# Run random search on each CSV file, saving output to individual folders
singularity exec ../tensorflow_model_train.img python3 random_search_array.py --iteration ${SLURM_ARRAY_TASK_ID} --model_number "${model_number}"

# Move output, error, and data files to their respective folders
if [ ${SLURM_ARRAY_TASK_ID} -eq 1 ]; then
    for ((task_id = 1; task_id <= SLURM_ARRAY_TASK_COUNT; task_id++)); do
        mv "random_samples${task_id}.csv" "./array_task${task_id}/"
        mv "slurm.random_search_array_${task_id}.out" "./array_task${task_id}/"
        mv "slurm.random_search_array_${task_id}.err" "./array_task${task_id}/"
    done
fi

touch "${task_flag}" # Raise a flag for this array once complete


# STEP 3: COMBINING RESULTS

# If all other tasks are complete (the flags have been raised)
if [ "${SLURM_ARRAY_TASK_ID}" -eq 1 ]; then
    # Wait for all other tasks to complete
    expected_flags=$((SLURM_ARRAY_TASK_COUNT-1))
    while [ $(ls ${flag_dir} | wc -l) -lt ${expected_flags} ]; do
        sleep 60 # Wait for 10 seconds before checking again
    done

    # Combine outputs when all tasks are complete
    srun singularity exec ../tensorflow_model_train.img python3 random_search_array_combine.py
fi