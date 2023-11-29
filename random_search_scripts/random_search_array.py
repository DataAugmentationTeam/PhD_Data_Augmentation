# 12/09/23

# Import required libraries
import datetime
start_time = datetime.datetime.now()

import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np
from tensorflow.keras import layers
from typing import Any, List, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import datetime
from sklearn.utils import resample
import pandas as pd
import os
from sklearn.utils import shuffle
import argparse

parser = argparse.ArgumentParser(add_help=True, formatter_class=argparse.RawDescriptionHelpFormatter, description = "random_search_slurm_array")
parser.add_argument("-i", "--iteration", help="slurm array iteration value", type=int, default=None)
parser.add_argument("-n", "--model_number", help="the number of the model", type=str, default=None)
args = parser.parse_args()
iteration, model_number = args.iteration, "00" + args.model_number

# Helper functions
def norm_pixels(image: np.ndarray, max_val: float = 255.0) -> np.ndarray:
    return image / max_val

def undersample_train(train_images, train_labels):
    # Combine train_images and train_labels into a single array
    train_data = np.column_stack((train_images.reshape(len(train_images), -1), train_labels))

    # Separate the samples by class
    class_samples = {}
    for label in np.unique(train_labels):
        class_samples[label] = train_data[train_data[:, -1] == label]

    # Determine the minimum number of samples in any class
    min_samples = min(len(samples) for samples in class_samples.values())

    # Undersample the majority classes
    undersampled_samples = []
    minority_class = min(class_samples, key=lambda x: len(class_samples[x]))
    for label, samples in class_samples.items():
        if label != minority_class:
            undersampled_samples.append(resample(samples, replace=False, n_samples=min_samples, random_state=42))

    # Combine the undersampled samples
    undersampled_data = np.concatenate([class_samples[minority_class]] + undersampled_samples)

    # Shuffle the data
    np.random.shuffle(undersampled_data)

    # Split the data into images and labels
    height, width, channels = 64, 64, 3
    undersampled_images = undersampled_data[:, :-1].reshape(len(undersampled_data), height, width, channels)
    undersampled_labels = undersampled_data[:, -1]
    
    return undersampled_images, undersampled_labels

def plot_confusion_matrix_sum(class_labels: list, confusion_matrices: list, model_name: str) -> np.ndarray:
    # Calculate the sum of confusion matrices
    num_labels = len(class_labels)
    confusion_matrix_sum = np.zeros((num_labels, num_labels), dtype=np.int32)

    for obj in confusion_matrices:
        predicted_values = obj['predicted_values']
        ground_truth_values = obj['ground_truth_values']
        confusion_matrix_sum += confusion_matrix(y_true=ground_truth_values, y_pred=predicted_values)
    
    # Plot the confusion matrix
    plt.figure()
    plt.imshow(confusion_matrix_sum, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title(f'Confusion Matrix: Rank {model_name} model')
    plt.colorbar()
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    filename = model_name + "_confusion_matrix.png"
    plt.savefig(filename)

    return confusion_matrix_sum

def plot_acc_scatter(model_name, training_cycles, val_final_acc):
    # Scatter plot of val_final_acc values
    plt.figure()
    plt.scatter(range(1, training_cycles+1), val_final_acc)
    plt.xlabel('Train Cycle')
    plt.ylabel('Validation Accuracy')
    plt.title(f'Val acc per fold: Rank {model_name} model')
    plt.ylim(0, 1)  # Set y-axis limits to 0 and 1
    avg_val_acc = np.mean(val_final_acc)
    plt.axhline(avg_val_acc, color='r', linestyle='--', label=f'Average Accuracy: {avg_val_acc:.2f}')
    plt.legend()
    filename = model_name + "_scatter_plot.png"
    plt.savefig(filename)

def plot_train_val_acc(model_name: str, train_accuracies: np.ndarray, epochs: int, val_accuracies: np.ndarray, k: int) -> None:
    filename = model_name + "_accuracies.png"

    plt.figure()

    # Plot the individual training accuracies per epoch for validation and training for each fold
    for i in range(k):
        plt.plot(range(1, epochs+1), train_accuracies[i], color="red", alpha=0.2)
        plt.plot(range(1, epochs+1), val_accuracies[i], color="green", alpha=0.2)

    # Plot the mean accuracies per epoch for validation and training.
    plt.plot(range(1, epochs+1), np.mean(train_accuracies, axis=0), label='Train Accuracy', color="red", linewidth=2)
    plt.plot(range(1, epochs+1), np.mean(val_accuracies, axis=0), label='Validation Accuracy', color="green", linewidth=2)

    # Labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Train/val acc per epoch: Rank {model_name} model')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(filename)

def shuffle_train(train_images: List[Any], train_labels: List[Any]) -> None:
    train_images, train_labels = shuffle(train_images, train_labels)

def load_dataset(file_name: str) -> Tuple[Any, Any]:
    data = np.load(file_name, allow_pickle=True)
    images = data.item().get('images')
    labels = data.item().get('labels')
    labels = [float(value) for value in labels]
    return images, labels

train_images, train_labels = load_dataset("train_data" + model_number + ".npy") # This train dataset will be split into train and val during KFold cross-validation
test_images, test_labels = load_dataset("test_data" + model_number + ".npy")

class_labels = [0, 1, 2, 3, 4, 5, 6]

train_images, test_images = norm_pixels(train_images), norm_pixels(test_images) # Normalize the pixel values
train_images, train_labels = undersample_train(train_images, train_labels) # Undersample so that the class sizes are the same

# Model building and training functions
def kfold_validation(train_images, train_labels, k, class_labels, learning_rate, epochs, num_filters, filter_size, num_layers, pooling_size, activation_function, batch_size, reg, opt, dropout):
    val_accuracies_fold = []
    train_accuracies_fold = []
    val_accuracies_epoch = []
    train_accuracies_epoch = []
    confusion_matrices = []

    kf = KFold(n_splits=k, shuffle=True)

    for train_index, test_index in kf.split(train_images):
        model = create_model(num_filters, filter_size, num_layers, pooling_size, activation_function, batch_size, reg, opt, dropout)

        # Split data into train and test sets for the current fold
        x_train, x_test = train_images[train_index], train_images[test_index]
        y_train, y_test = train_labels[train_index], train_labels[test_index]
        y_train_encoded = tf.keras.utils.to_categorical(y_train, len(class_labels))
        y_test_encoded = tf.keras.utils.to_categorical(y_test, len(class_labels))

        # Compile and train your model

        if opt == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif opt == "SGD":
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif opt == "Momentum":
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif opt == "RMSProp":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train_encoded, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test_encoded), verbose=0)

        # Store the accuracies per fold for training and validation in a list of integers
        val_loss_fold, val_accuracy_fold = model.evaluate(x_test, y_test_encoded, verbose=0)
        val_accuracies_fold.append(val_accuracy_fold)
        train_loss_fold, train_accuracy_fold = model.evaluate(x_train, y_train_encoded, verbose=0)
        train_accuracies_fold.append(train_accuracy_fold)

        # Store the accuracies for each epoch of each fold for training and validation in a list of 1D arrays
        train_accuracies_epoch.append(history.history['accuracy'])
        val_accuracies_epoch.append(history.history['val_accuracy'])

        # Calculate predictions on the validation set
        y_pred = model.predict(x_test, verbose=0)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_test_encoded, axis=1)

        # Generate the confusion matrix for the current fold
        confusion_matrices.append({'predicted_values': y_pred_labels, 'ground_truth_values': y_true_labels})

    return val_accuracies_fold, train_accuracies_fold, val_accuracies_epoch, train_accuracies_epoch, confusion_matrices, model

def create_model(num_filters, filter_size, num_layers, pooling_size, activation_function, batch_size, reg, opt, dropout):
    # Regularization
    if reg == "L1":
        regularization = tf.keras.regularizers.l1(0.01)
    elif reg == "L2":
        regularization = tf.keras.regularizers.l2(0.01)
    else:
        regularization = None
    
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(num_filters, filter_size, activation=activation_function, input_shape=(64, 64, 3), kernel_regularizer=regularization))
    model.add(layers.MaxPooling2D(pool_size=(pooling_size,pooling_size)))
    for layer in range(num_layers-1):
        model.add(layers.Conv2D(num_filters, filter_size, activation=activation_function, kernel_regularizer=regularization))
        model.add(layers.MaxPooling2D(pool_size=(pooling_size,pooling_size)))
    model.add(layers.Flatten())
    
    # Dropout
    if dropout > 0:
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(len(class_labels), activation='softmax', kernel_regularizer=regularization))
    return model

def build_train_model(num_filters, filter_size, learning_rate, epochs, k, num_layers, pooling_size, activation_function, batch_size, reg, opt, dropout):
    # This function is to convert the model building and training into something that can take input from ParameterGrid

    vaf, taf, vae, tae, cm, model = kfold_validation(train_images, train_labels, k, class_labels, learning_rate, epochs, num_filters, filter_size, num_layers, pooling_size, activation_function, batch_size, reg, opt, dropout)

    return vaf, taf, vae, tae, cm, model

# Perform the random search
total_VAF = []
total_TAF = []
total_VAE = []
total_TAE = []
total_CM = []
total_Models = []
total_Params = []

input_data = pd.read_csv(f"random_samples{iteration}.csv")

# For some reason k is loading as a float -> convert to int.
input_data['k'] = input_data['k'].astype(int)

# num_filters, filter_size, learning_rate, epochs, k, num_layers, pooling_size, activation_function, batch_size, reg, opt, dropout
random_samples = []
for index, row in input_data.iterrows():
    dictionary = {"num_filters": row.num_filters,
                  "filter_size": row.filter_size,
                  "learning_rate": row.learning_rate,
                  "epochs": row.epochs,
                  "k": row.k,
                  "num_layers": row.num_layers,
                  "pooling_size": row.pooling_size,
                  "activation_function": row.activation_function,
                  "batch_size": row.batch_size,
                  "reg": row.reg,
                  "opt": row.opt,
                  "dropout": row.dropout}
    random_samples.append(dictionary)

acc_threshold = 0.8 # Target accuracy threshold
for params in random_samples:
    print("Training with params:", params)

    vaf, taf, vae, tae, cm, model = build_train_model(**params)
    total_VAF.append(vaf)
    total_TAF.append(taf)
    total_VAE.append(vae)
    total_TAE.append(tae)
    total_CM.append(cm)
    total_Models.append(model)
    total_Params.append(params)

    if np.mean(vaf) >= acc_threshold:
        break

# Output results to CSV
avg_VAFs = np.mean(total_VAF, axis=1)
avg_TAFs = np.mean(total_TAF, axis=1)
df = pd.DataFrame(random_samples)
df['val_acc'] = avg_VAFs
divergence = [t - v for t, v in zip(avg_TAFs, avg_VAFs)]
df['divergence'] = divergence

df = df.sort_values(by="val_acc", ascending=False)
results_path = os.path.join(f"array_task{iteration}", f"random_search_array{iteration}.csv" )
df.to_csv(results_path, index=False)

# Display plots of Top 3 models
def model_plots(rank, name):

    if name == None:
        name = str(rank)

    avg_VAFs = np.mean(total_VAF, axis=1)
    ranked_VAFs = np.sort(avg_VAFs)[::-1]
    value = ranked_VAFs[rank-1]

    position = np.where(avg_VAFs == value)[0][0]
    

    vaf = total_VAF[position]
    taf = total_TAF[position]
    vae = total_VAE[position]
    tae = total_TAE[position]
    cm = total_CM[position]
    model = total_Models[position]
    params = total_Params[position]

    print(f"Model parameters: {params}")
    print(f"Model validation accuracy: {avg_VAFs[position]}")

    # Training vs validation accuracy over time
    plot_train_val_acc(name, tae, params['epochs'], vae, params['k'])

    # Scatter of validation accuracies per fold
    plot_acc_scatter(name, params['k'], vaf)

    # Confusion matrix
    confusion_matrix_sum = plot_confusion_matrix_sum(class_labels, cm, name)

    # Per-class validation accuracies
    class_accuracy = np.diag(confusion_matrix_sum) / confusion_matrix_sum.sum(axis=1)
    print(f"Class accuracies:")
    for class_acc in range(0, len(class_accuracy)):
        print(f"Class {class_acc}: {class_accuracy[class_acc]}")

model_plots(1, f"./array_task{iteration}/1_random_search_array")
model_plots(2, f"./array_task{iteration}/2_random_search_array")
model_plots(3, f"./array_task{iteration}/3_random_search_array")

data = {
    'total_VAF': total_VAF,
    'total_TAF': total_TAF,
    'total_VAE': total_VAE,
    'total_TAE': total_TAE,
    'total_CM': total_CM,
    'total_Params': total_Params
}

# Save the arrays to a single .npz file
np.savez(f'./array_task{iteration}/random_search_array_training_data.npz', **data)

# Save each of the models in the Models folder

model_save_dir = f"./array_task{iteration}/models/"

for i, model in enumerate(total_Models):
    model_path = os.path.join(model_save_dir, f'model_{i}.h5')
    model.save(model_path)

end_time = datetime.datetime.now()
execution_time = end_time - start_time
print(f"Script execution time: {execution_time}")