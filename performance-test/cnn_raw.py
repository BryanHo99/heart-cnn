"""
Time series CNN classification implemented from:
https://keras.io/examples/timeseries/timeseries_classification_from_scratch/

Example to execute the file in the command line:
python cnn.py [folder name of MATLAB data] [SQL CSV file] [pcg/ecg]
e.g. python cnn.py training-data REFERENCE_withSQI.csv pcg
"""
import os
import glob
import pandas as pd
import scipy.io as sio
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# Package to load model.
from tensorflow import keras
from keras.optimizers import Adam
import sys                          # Package to read command line arguments.
import numpy as np                  # Package for numerical analysis.
import matplotlib.pyplot as plt
from datetime import datetime
# Local imports for preprocessing, visualization and building model.
from configs.cnn_config import cnn_config

def read_mat(signal_type_index):
    """
    Function that reads the MATLAB files given and generates the numpy arrays
    for ECG data and the labels.
    :return: A tuple of (mat_data, data_labels) where
                mat_data - 2D numpy array of ECG/PCG data
                labels - 1D numpy array of labels
    """
    # Discard signals that have lengths shorter that this value.
    # Change this to modify the minimum threshold.
    discard_value = 20000
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
        current_length = len(sio.loadmat(file)['val'][signal_type_index, :])
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
        raw_data = sio.loadmat(file)['val'][signal_type_index, :time_length]
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

    # print("Total number of subjects:", num_subjects)
    # print("Total number of used subjects:", num_used)
    # print("Total number of discarded subjects:", num_discarded)
    # print("Total number of normal signals:", num_normal)
    # print("Total number of abnormal signals:", num_abnormal)
    # print("Shortest length:", shortest_length)
    # print("Rounded down shortest length:", time_length)
    # print()

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

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])

if __name__ == "__main__":
    try:
        assert len(sys.argv) == 4, "Invalid number of command line arguments."

        _, folder_name, sqi_csv, signal_type = sys.argv

        assert signal_type.lower() == "pcg" or signal_type.lower() == "ecg", "Unknown signal type"

        if signal_type.lower() == "pcg":
            signal_type_index = 0
        elif signal_type.lower() == "ecg":
            signal_type_index = 1
        else:
            raise ValueError("Signal type can only be PCG or ECG.")

        # Number of convolutional layers to use.
        num_conv_layers = 3

        # Read the Data
        # Read the MATLAB data and then split into training and testing instances.
        # print("Creating Training and Testing Sets...")
        
        # data, labels = read_mat(folder_name, sqi_csv, signal_type)
        # x_train, x_test, y_train, y_test = split_train_test(data, labels)
        # STEP 1: Read the Data
        # Read the MATLAB data and then split into training and testing instances.
        # print("Creating Training and Testing Sets...")
        data, labels = read_mat(signal_type_index)
        x_train, x_test, y_train, y_test = train_test_split(data, labels, stratify=labels, test_size=0.20, random_state=42)

        # Visualize the Data
        # print("Visualizing Data...")
        # if sys.argv[3].lower() == "pcg":
        #     visualize_mfcc(data, labels)
        # else:
        #     visualize_wavelet(data, labels)

        # Standardize the Data
        # print("Standardizing Data...")
        # ECG - PCG sample rate (samples per second or Hz).
        sample_rate = 1000
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] // sample_rate, sample_rate))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] // sample_rate, sample_rate))

        # Count number of classes (aka labels).
        # In this case, there are 2 classes: -1 and 1.
        num_classes = len(np.unique(y_train))

        # Standardize labels to positive integers. -1 is changed to 0.
        # Expected labels will be 0 and 1.
        y_train[y_train == -1] = 0
        y_test[y_test == -1] = 0

        # Train the Model
        training_iterations = 3     # Number of times to repeat training
        learning_rates = [0.001] # lr=0.001 same as default for tf.keras.optimizers.Adam
        batch_sizes = [32]
        # print("Training Model...")
        start_now = datetime.now()
        start_time = start_now.strftime("%H:%M:%S")
        print('CNN', signal_type, 'training start time:', start_time)
        # Initialize results file
        results_filename = 'cnn_raw_best-'+signal_type+'-results.txt'
        f = open(results_filename, 'w')
        f.write('CNN-'+signal_type+'-training\n')
        f.write('start time: '+start_time+'\n')
        f.close()
        for lr in learning_rates:
            for bs in batch_sizes:
                # Set learning rate and batch size in cnn_config
                cnn_config['learning_rate'] = lr
                cnn_config['batch_size'] = bs
                # print('training cnn with lr, bs', cnn_config['learning_rate'], cnn_config['batch_size'])
                for i in range(training_iterations):
                    # Set model and metrics plots filenames for current trianing iteration
                    filename = 'cnn_raw_best'+'-'+\
                                        signal_type+'-'\
                                        'lr'+str(cnn_config['learning_rate'])+'-'+\
                                        'bs'+str(cnn_config['batch_size'])+'-'+\
                                        'i'+str(i+1)
                    # print('curr iteration filename:', filename)
                    saved_model_name = filename + cnn_config['saved_model_format']
                    metric_plot_name = filename + cnn_config['saved_plot_format']
                    cnn_config['saved_model_name'] = saved_model_name
                    cnn_config['saved_plot_name'] = metric_plot_name

                    # Build a Model
                    model = make_model(x_train.shape[1:], num_classes)

                    # Train CNN with current iteration parameters, save as cnn_config['saved_model_name']
                    epochs = 200

                    callbacks = [
                        keras.callbacks.ModelCheckpoint(
                            cnn_config['saved_model_name'],
                            save_best_only=True,
                            save_weights_only=True,
                            monitor="val_loss"
                        ),
                        keras.callbacks.ReduceLROnPlateau(
                            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
                        ),
                        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=0),
                    ]
                    
                    adam = Adam(cnn_config['learning_rate'])
                    model.compile(
                        optimizer=adam,
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"],
                    )

                    history = model.fit(
                        x_train,
                        y_train,
                        batch_size=cnn_config['batch_size'],
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_split=0.2,
                        verbose=0,
                    )

                    # Plot accuracy and loss, save as cnn_config['saved_plot_name']
                    plt.figure(figsize=(13, 7))
                    plt.subplot(1, 2, 1)
                    plot_graphs(history, 'accuracy')
                    plt.ylim(0, 1)
                    plt.subplot(1, 2, 2)
                    plot_graphs(history, 'loss')
                    plt.ylim(0, None)
                    # plt.show()
                    plt.savefig(cnn_config['saved_plot_name'])
                    plt.clf()
                    plt.close()

                    # Load the saved model weights
                    del model
                    model = make_model(x_train.shape[1:], num_classes)
                    # compile the model
                    adam = Adam(cnn_config['learning_rate'])
                    model.compile(
                        optimizer=adam,
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"],
                    )
                    model.load_weights(cnn_config['saved_model_name'])

                    # Evaluate the Model on Test Data
                    # print("Evaluating Model...")
                    # Obtain test accuracy and test loss.
                    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
                    print(filename, "test accuracy, loss:", np.round(test_acc, 4), np.round(test_loss, 4))
                    f = open(results_filename, 'a')
                    results_output = filename+' -- '+'test accuracy, loss: '+\
                                        str(np.round(test_acc, 4))+', '+\
                                        str(np.round(test_loss, 4))+'\n'
                    f.write(results_output)
                    f.close()
                    del model # make sure model isn't reused in next iteration

        end_now = datetime.now()
        end_time = end_now.strftime("%H:%M:%S")
        print('CNN', signal_type, 'training end time:', end_time)

        elapsed_time = end_now - start_now
        print('CNN', signal_type, 'training elapsed time:', elapsed_time)
        f = open(results_filename, 'a')
        f.write('end time: '+end_time+'\n')
        f.write('elapsed time: '+str(elapsed_time)+'\n')
        f.close()

    except (AssertionError, ValueError) as e:
        print("ERROR:", e)
        print("Exiting program.")
        exit()
