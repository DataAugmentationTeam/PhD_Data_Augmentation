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
parser.add_argument("-d", "--data_dir", help="the directory containing the images", type=str, default=None)
parser.add_argument("-s", "--image_size", help="the size to rescale the images to", type=int, default=64)
parser.add_argument("-t", "--train", help="percentage of dataset used for training", type=int, default=60)
parser.add_argument("-v", "--val", help="percentage of dataset used for validation", type=int, default=20)
args = parser.parse_args()
arraylen, perfile, model_number, data_dir, image_size = args.arraylen, args.perfile, args.model_number, args.data_dir, args.image_size
train_split, val_split = args.train, args.val

# 96 4 to get all

# Parameter grid to sample from
param_grid = {
    'layer_size': [4, 8, 16, 32],
    'learning_rate': [0.01, 0.001, 0.0001, 0.00001],
    'epochs': [50],
    'k': [5],
    'num_layers': [1, 2, 3, 4],
    'activation_function': ['relu'],
    'batch_size': [64],
    'opt': ["Adam", "Momentum", "RMSProp"],
    'doubling': [True, False]
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
        #print(len(grid))
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
            image = cv.resize(image, (image_size, image_size)) # Preprocessing the cropped images to be the same size.
            
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
    
    print("Creating dataset")

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

# Load in data, and split into train:val:test
images, labels = load_images_and_labels()
train_val_test(images, labels, ".", model_number, train_ratio=train_split, val_ratio=val_split) # Consider changing this ratio when considering the undersampling
