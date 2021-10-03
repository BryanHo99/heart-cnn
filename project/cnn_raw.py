"""
Time series CNN classification implemented from:
https://keras.io/examples/timeseries/timeseries_classification_from_scratch/
"""

import glob                         # Package to search for files in a given directory
from tqdm import tqdm               # Package to show progress bar of a loop
import scipy.io as sio              # Package to load matlab files (*.mat) of ECG-PCG data
import pandas as pd                 # Package to read labels from csv/excel file
import os                           # Package used to process system paths
import numpy as np                  # Package for numerical analysis
import matplotlib.pyplot as plt     # Package for graphical visualization

from tensorflow import keras                            # Package for CNN classifier
from sklearn.model_selection import train_test_split    # Package to split training and test data

# ECG - PCG sample rate (samples per second or Hz).
sample_rate = 1000

# 0 for PCG, 1 for ECG.
# Change this to switch between classifying different signal types.
signal_type = 1

# Discard signals that have lengths shorter that this value.
# Change this to modify the minimum threshold.
discard_value = 20000

# Number of convolutional layers to use.
num_conv_layers = 3


def read_mat():
    """
    Function that reads the MATLAB files given and generates the numpy arrays
    for ECG data and the labels.
    :return: A tuple of (mat_data, data_labels) where
                mat_data - 2D numpy array of ECG/PCG data
                labels - 1D numpy array of labels
    """
    # Search for all MAT files in data path "./training-a/".
    # Each subject has 2 rows. The first row is PCG (heart sound) and the second row is ECG.
    list_files = glob.glob("./training-a/*.mat")

    # Load the demographic information from csv file.
    # Demo contains "data name", "label -> -1:normal, 1:abnormal",
    # "signal quality index (sqi) -> 0:noisy, 1:clean".
    demo = pd.read_csv('REFERENCE_withSQI.csv')

    # Find subject with the shortest data length greater than 30000
    # and round it down to the nearest 1000.
    time_length = float("inf")
    shortest_length = 0
    for file in list_files:
        current_length = len(sio.loadmat(file)['val'][signal_type, :])
        if current_length < discard_value:
            continue

        if time_length > current_length:
            shortest_length = current_length
            time_length = current_length - (current_length % 1000)

    # Initialize empty 2D numpy array to store ECG data.
    # Initialize empty 1D numpy array to store labels.
    mat_data = np.empty((0, time_length), int)
    data_labels = np.empty(0, int)

    num_subjects = len(list_files)
    num_used = 0
    num_discarded = 0
    num_normal = 0
    num_abnormal = 0

    for file in tqdm(list_files):
        # Load data from mat file and extract the ECG data only.
        # If the data length is less than 30000, skip it.
        raw_data = sio.loadmat(file)['val'][signal_type, :time_length]
        if len(raw_data) < discard_value:
            num_discarded += 1
            continue

        # Append data into mat_data.
        mat_data = np.append(mat_data, np.array([raw_data]), axis=0)
        num_used += 1

        # Find the MAT file name of the loaded data.
        # Then, remove the file extension from the filename.
        path, filename = os.path.split(file)
        filename = filename.split(".")[0][:-1]

        # Extract the label of the particular "filename" data.
        # Then, append label to data_labels.
        label = demo.label[demo['name'] == filename].values
        data_labels = np.append(data_labels, np.array(label), axis=0)
        if label == -1:
            num_normal += 1
        else:
            num_abnormal += 1

    print("Total number of subjects:", num_subjects)
    print("Total number of used subjects:", num_used)
    print("Total number of discarded subjects:", num_discarded)
    print("Total number of normal signals:", num_normal)
    print("Total number of abnormal signals:", num_abnormal)
    print("Shortest length:", shortest_length)
    print("Rounded down shortest length:", time_length)
    print()

    return mat_data, data_labels


def make_model(input_shape, num_of_classes):
    """
    Function that makes a Fully Convolutional Neural Network proposed in this paper:
    https://arxiv.org/abs/1611.06455.

    The implementation is based on the following link:
    https://github.com/hfawaz/dl-4-tsc/

    :param input_shape      : A shape tuple (integers)
    :param num_of_classes   : The number of classes
    :return                 : A Keras model
    """
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    current_conv = conv1

    for i in range(num_conv_layers - 1):
        next_conv = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(current_conv)
        next_conv = keras.layers.BatchNormalization()(next_conv)
        next_conv = keras.layers.ReLU()(next_conv)
        current_conv = next_conv

    gap = keras.layers.GlobalAveragePooling1D()(current_conv)

    output_layer = keras.layers.Dense(num_of_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


if __name__ == "__main__":
    # STEP 1: Read the Data
    # Read the MATLAB data and then split into training and testing instances.
    print("Creating Training and Testing Sets...")
    data, labels = read_mat()
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=42)

    # STEP 2: Visualize the Data
    print("Visualizing Data...")
    classes = np.unique(np.concatenate((y_train, y_test), axis=0))

    # Plot the graph.
    plt.figure()
    for c in classes:
        c_x_train = x_train[y_train == c]
        plt.plot(c_x_train[0], label="class " + str(c))
    plt.legend(loc="best")
    plt.show()
    plt.close()

    # STEP 3: Standardize the Data
    # Reshape into a 3D array since CNN needs 3D array as input.
    # Reshaping idea: numpy.reshape((row, column, depth))
    # We are reshaping it to be (No. of subjects, Time steps, Features)
    print("Standardizing Data...")
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] // sample_rate, sample_rate))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] // sample_rate, sample_rate))

    # Count number of classes (aka labels).
    # In this case, there are 2 classes: -1 and 1.
    num_classes = len(np.unique(y_train))

    # Shuffle training set.
    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    # Standardize labels to positive integers. -1 is changed to 0.
    # Expected labels will be 0 and 1.
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    # STEP 4: Build a Model
    print("Building Model...")
    model = make_model(x_train.shape[1:], num_classes)
    model.summary()

    # STEP 5: Train the Model
    print("Training Model...")
    epochs = 200
    batch_size = 32

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "best_model.h5", save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=1,
    )

    # STEP 6: Evaluate the Model on Test Data
    print("Evaluating Model...")
    model = keras.models.load_model("best_model.h5")

    # Obtain test accuracy and test loss.
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print()
    print("Test Accuracy:", test_acc)
    print("Test Loss:", test_loss)

    # Plot model's training and validation loss.
    metric = "sparse_categorical_accuracy"
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("Model Sparse Categorical Accuracy")
    plt.ylabel("Sparse Categorical Accuracy", fontsize="large")
    plt.xlabel("Epoch", fontsize="large")
    plt.legend(["Training Accuracy", "Validation Accuracy"], loc="best")
    plt.show()
    plt.close()
