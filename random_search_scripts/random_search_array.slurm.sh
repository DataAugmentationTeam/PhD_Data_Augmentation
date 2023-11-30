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
- There are two places you can make changes to this script:
1) The #SBATCH header:
    a) Change line 9 from my email to yours. When the script errors or finishes running it will email you.
    b) Change line 7 from 1-3 to 1-X, where X is the number of arrays (scripts run in parallel). e.g. 1-96
2) The "YOUR CHANGES HERE" section:
    a) model_number: This is an ID value you should change each time you run a new analysis.
    b) perfile: This is the number of models which will be trained for each script, and should be 3 or more.
    c) data_dir: This is the location of the CDA images that the model will use for training and testing.
    d) img: This is the location of the disk image which will run the scripts.

Tip for choosing array and perfile numbers:
- The number of models that will be trained in total is array*perfile. So if you run 3 arrays and 4 models in each, you will have 12 models.
- You can do all combinations of the hyperparameter grid in random_search_array_sample.py line 24 by using --array=1-96, perfile=36. 

If you don't yet have an image, please work through the file building_singularity_container.txt

END_COMMENT

# YOUR CHANGES HERE ----
model_number="1"
perfile=3
data_dir="../../data/images_combined/"
img="../tensorflow_model_train.img"
img_size=224
train=60
val=20
# ----



model_number=$(printf "%04d" $model_number)

# STEP 1: DATASET CREATION AND TRAINING HYPERPARAMETER SAMPLING

# Flag to make sure that training only begins once the dataset and csvs have been built
flag_start="flag_start.flag"

# Check if this is the first task in the array
if [ "${SLURM_ARRAY_TASK_ID}" -eq 1 ]; then
    # Randomly sample from the hyperparameter grid, producing CSV files
    srun singularity exec ${img} python3 random_search_array_sample.py --arraylen ${SLURM_ARRAY_TASK_COUNT} --perfile ${perfile} --model_number "${model_number}" --data_dir "${data_dir}" --image_size ${img_size} --train ${train} --val ${val}

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
flag_dir="flags"
mkdir -p "${flag_dir}"
task_flag="${flag_dir}/flag_${SLURM_ARRAY_TASK_ID}.flag"

# Make folders for each task to store the output
mkdir array_task${SLURM_ARRAY_TASK_ID}

# Run random search on each CSV file, saving output to individual folders
singularity exec ${img} python3 random_search_array.py --iteration ${SLURM_ARRAY_TASK_ID} --model_number "${model_number}" --image_size ${img_size}

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
    srun singularity exec ${img} python3 random_search_array_combine.py
fi