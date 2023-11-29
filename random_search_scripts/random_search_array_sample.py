# 12/09/23

# Import required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
import random
from typing import Any, List, Tuple
import os
import cv2 as cv
from sklearn.utils import shuffle
import argparse

parser = argparse.ArgumentParser(add_help=True, formatter_class=argparse.RawDescriptionHelpFormatter, description = "random_search_slurm_array")
parser.add_argument("-a", "--arraylen", help="number of csv files", type=int, default=None)
parser.add_argument("-p", "--perfile", help="number of sets per csv", type=int, default=None)
parser.add_argument("-n", "--model_number", help="the number of the model", type=str, default=None)
args = parser.parse_args()
arraylen, perfile, model_number = args.arraylen, args.perfile, "00" + args.model_number

# 96 36 to get all

# Parameter grid to sample from
param_grid = {
    'num_filters': [2, 4, 8, 16],
    'filter_size': [3, 5],
    'learning_rate': [0.001, 0.0001],
    'epochs': [50],
    'k': [5],
    'num_layers': [1, 2, 3],
    'pooling_size': [2],
    'activation_function': ['relu'],
    'batch_size': [64, 128],
    'reg': [None, "L1", "L2"],
    'opt': ["Adam", "SGD", "Momentum", "RMSProp"],
    'dropout': [0, 0.2, 0.5]
}


# Randomly select from ParameterGrid
grid = list(ParameterGrid(param_grid))
random.shuffle(grid)

dfs = []
lengths = []

# Create 10 CSV files of 100 sampled parameter sets
for csv in range(arraylen):
    random_samples = []
    for sample in range(perfile): # Change this to 100 once testing over
        random_sample = grid.pop()
        print(len(grid))
        random_samples.append(random_sample)
    output_df = pd.DataFrame(random_samples)
    lengths.append(output_df.shape[0])
    dfs.append(output_df)

lengths = pd.DataFrame(lengths)
lengths.to_csv("lengths.csv", index=False)

for df_index, df in enumerate(dfs):
    if not df.empty:
        with open(f"random_samples{df_index + 1}.csv", 'w', newline='') as file:
            df.to_csv(file, index=False)


# Create dataset
def load_images_and_labels() -> Tuple[List[Any], List[int]]:
    data_dir = "../../data/images_combined/" # Relative path to the folders containing the CDAs
    class_labels = os.listdir(data_dir)
    class_labels = [label for label in class_labels if label != '.DS_Store'] # Remove .DS_Store

    images = []
    labels = []
    test_ids = []

    for label_idx, class_label in enumerate(class_labels):
        class_dir = os.path.join(data_dir, class_label)     
        image_files = os.listdir(class_dir) # List of cropped images in the current class dir

        for image_file in image_files:

            image_path = os.path.join(class_dir, image_file)
            image = cv.imread(image_path)
            image = cv.resize(image, (64, 64)) # Preprocessing the cropped images to be the same size.
            
            images.append(image)
            labels.append(class_label)
            test_ids.append(label_idx)
    return images, labels

def train_val_test(images: List[Any], labels: List[Any], filepath: str, model_number: str,
                   train_ratio: float = 0.6, val_ratio: float = 0.2) -> None:
    train_out = os.path.join(filepath, ("train_data" + model_number + ".npy"))
    val_out = os.path.join(filepath, ("val_data" + model_number + ".npy"))
    test_out = os.path.join(filepath, ("test_data" + model_number + ".npy"))

    if os.path.isfile(test_out):
        print("The data exists already")
        return

    total_samples = len(images)

    images, labels = shuffle(images, labels)

    train_samples = int(total_samples * train_ratio)
    val_samples = int(total_samples * val_ratio)

    train_images = np.array(images[:train_samples])
    train_labels = np.array(labels[:train_samples])

    val_images = np.array(images[train_samples:train_samples + val_samples])
    val_labels = np.array(labels[train_samples:train_samples + val_samples])

    test_images = np.array(images[train_samples + val_samples:])
    test_labels = np.array(labels[train_samples + val_samples:])

    def save_dataset(images, labels, file_name):
        data = {'images': images, 'labels': labels}
        np.save(file_name, data, allow_pickle=True)

    save_dataset(train_images, train_labels, train_out)
    save_dataset(val_images, val_labels, val_out)
    save_dataset(test_images, test_labels, test_out)

# Load in data, normalize, and split into train:val:test
print("Creating dataset")
images, labels = load_images_and_labels()
train_val_test(images, labels, ".", model_number, train_ratio=0.8, val_ratio=0.0) # Consider changing this ratio when considering the undersampling
