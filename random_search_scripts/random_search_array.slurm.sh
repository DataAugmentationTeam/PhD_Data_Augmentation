#!/bin/bash
#SBATCH -p jic-medium
#SBATCH --mem 32000
#SBATCH --job-name="random_search_array"
#SBATCH -o slurm.random_search_array_%a.out
#SBATCH -e slurm.random_search_array_%a.err
#SBATCH --array=1-3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jowillia@nbi.ac.uk


: <<'END_COMMENT'

USING THIS SCRIPT:

- This is the bash script which runs the machine learning pipeline.
- There are several things you can change here.
    a) The number of arrays can be changed in line 7, changing 1-3 to 1-X, where X is your chosen value.
    b) The number of models to be trained in each array, changing the perfile value below.
    c) The model number (this is an ID where you can keep track of which you trained), changing model_number below
    d) The email address to notify when the script is done (you can replace mine with yours)

    To be added by Josh:
    e) The dataset: currently this has to be changed on line 69 of random_search_array_sample.py

- If you want to run all the hyperparameter combinations from random_search_array_sample.py line 24,
  you can find the total number of combinations by multiplying the number of options in each class.
  e.g. 4*2*2*1*1*3*1*1*2*3*4*3 = 3456 models. Since 3456 = 96 * 36, we could therefore run this script
  with array=1-96, perfile=36, then our script would run 96 times in parallel, with each training 36
  models.

- I would recommend just increasing the model number by 1 each time you run a new analysis.

END_COMMENT

# YOUR CHANGES HERE
model_number="1"
perfile=3





model_number=$(printf "%04d" $model_number)

# STEP 1: DATASET CREATION AND TRAINING HYPERPARAMETER SAMPLING

# Flag to make sure that training only begins once the dataset and csvs have been built
flag_start="flag_start.flag"

# Check if this is the first task in the array
if [ "${SLURM_ARRAY_TASK_ID}" -eq 1 ]; then
    # Randomly sample from the hyperparameter grid, producing CSV files
    srun singularity exec ../tensorflow_model_train.img python3 random_search_array_sample.py --arraylen ${SLURM_ARRAY_TASK_COUNT} --perfile ${perfile} --model_number "${model_number}"

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