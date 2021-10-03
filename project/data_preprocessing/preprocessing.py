import glob                         # Package to search for files in a given directory.
import os                           # Package used to process system paths.

from tqdm import tqdm               # Package to show progress bar of a loop.
import scipy.io as sio              # Package to load matlab files (*.mat) of ECG-PCG data.
import pandas as pd                 # Package to read labels from CSV/Excel file.
import numpy as np                  # Package for numpy array initialization.

# MFCC feature extraction for PCG.
from feature_extraction.feature_extraction import mfcc, wavelet_transform

# Package to split training and test data.
from sklearn.model_selection import train_test_split


def read_mat(folder_name, sqi_csv, signal_type):
    """
    Function that reads the MATLAB files given and generates the numpy arrays
    for ECG/PCG data and the labels.

    The MATLAB files must be in the root of the folder's directory.
    The SQI CSV file must have the following headers as its structure:
    - Data name
    - Label -> -1: Normal, 1: Abnormal
    - Signal Quality Index (SQI) -> 0: Noisy, 1: Clean

    :return: A tuple of (mat_data, data_labels) where
                mat_data - 3D numpy array of ECG/PCG data
                labels - 1D numpy array of labels
    """
    check_folder = f"./{folder_name}"
    check_csv = f"./{sqi_csv}"
    assert os.path.isdir(check_folder), "Dataset folder specified is not a folder"
    assert os.path.isfile(check_csv), "SQI CSV file specified is not a csv file"
    assert signal_type == "ecg" or signal_type == "pcg", "Specified signal type does not exist. PCG/ECG only."

    # 0 for PCG, 1 for ECG.
    # Change this to switch between classifying different signal types.
    signal_type_index = set_signal_type_index(signal_type)

    # Search for all MAT files in folder_name.
    # Each subject has 2 rows. The first row is PCG (heart sound) and the second row is ECG.
    directory = f"./{folder_name}/*.mat"
    list_files = glob.glob(directory)

    # Load the demographic information from CSV file.
    demo = pd.read_csv(sqi_csv)

    assert len(list_files) >= 2 and len(demo) >= 2, \
        "Must have at least 2 subjects -- 1 normal, 1 abnormal."

    # Filter off the signals that do not have signals greater than the discard threshold.
    filtered_files, signals = remove_signals(list_files, signal_type_index)

    # Get shortest length and time length.
    shortest_length, time_length = find_shortest_length(signals)

    # Preprocess MATLAB ECG/PCG data and binary classification labels.
    mat_data = preprocess_signals(signals, time_length, signal_type_index)
    data_labels = preprocess_labels(filtered_files, demo)
    print_results(list_files, mat_data, data_labels, shortest_length, time_length)

    validate_data(mat_data, data_labels)

    return mat_data, data_labels


def preprocess_signals(signals, time_length, signal_type_index):
    """
    Function that preprocesses the ECG/PCG signals.
    It also performs feature extraction based on the signal type.
    It returns a 3D numpy array of the extracted signals.

    :param signals: The list of ECG/PCG numpy arrays
    :param time_length: The rounded down value of the shortest length (to the nearest 1000)
    :param signal_type_index: The signal type (0 for PCG, 1 for ECG)
    :return: A 3D numpy array of the feature extracted signals.
    """
    assert type(signals) == list, "preprocess signals array must be a numpy array"
    assert len(signals) != 0, "signals array should not be empty"
    assert signal_type_index == 0 or signal_type_index == 1, "invalid signal type"

    # ECG - PCG sample rate (samples per second or Hz).
    sample_rate = 1000

    # MFCC required parameters.
    n_mfcc = 40
    hop_length = 256

    # Wavelet Transform required parameters.
    widths = 10

    # Initialize empty 3D numpy array for PCG/ECG respectively.
    if signal_type_index == 0:
        mat_data = np.empty((0, n_mfcc, (time_length // hop_length) + 1), float)
    else:
        mat_data = np.empty((0, widths, time_length), float)

    for signal in tqdm(signals):
        raw_data = signal[:time_length].astype(float)

        # Perform feature extraction on raw data.
        # Then append data into mat_data.
        if signal_type_index == 0:
            # MFCC for PCG signal.
            mat_data = np.append(mat_data, [mfcc(raw_data, sample_rate, n_mfcc, hop_length)], axis=0)
        else:
            # Wavelet Transform for ECG signal.
            mat_data = np.append(mat_data, [wavelet_transform(raw_data, widths)], axis=0)

    return mat_data


def preprocess_labels(list_files, demo):
    """
    Function that preprocess the data labels.
    It returns a 1D numpy array consisting of binary classification labels
    -1 for normal, 1 for abnormal.

    :param list_files: The list of MATLAB files
    :param demo: The SQI CSV file for the labels
    :return: 1D numpy array of the data labels.
    """

    assert len(list_files) != 0, "Empty list of MATLAB files"
    assert type(demo) == pd.DataFrame, "Attribute demo should be a DataFrame type"
    assert len(demo) != 0, "Attribute demo is empty"

    # Initialize empty 1D numpy array to store labels.
    data_labels = np.empty(0, int)

    for file in list_files:
        # Find the MAT file name of the loaded data.
        # Then, remove the file extension from the filename.
        path, filename = os.path.split(file)
        filename = filename.split(".")[0][:-1]

        # Extract the label of the particular "filename" data.
        # Then, append label to data_labels.
        label = demo.label[demo['name'] == filename].values
        data_labels = np.append(data_labels, np.array(label), axis=0)

    return data_labels


def remove_signals(list_files, signal_type_index):
    """
    Function that removes any signals that are lesser
    than the discard value.

    :param list_files: The list of MATLAB files
    :param signal_type_index: The signal type index
    :return: (filtered_files, signals)
             where filtered_files are the MATLAB files that have signals > discard value
                                  which is needed to filter the labels
                   signals are the actual ECG/PCG numpy arrays needed to be preprocessed.
    """
    assert len(list_files) != 0, "Empty list of MATLAB files"
    assert signal_type_index == 0 or signal_type_index == 1, "Invalid signal type"

    # Discard signals that have lengths shorter that this value.
    # Change this to modify the minimum threshold.
    discard_value = 20000
    filtered_files = []
    signals = []

    for file in list_files:
        signal = sio.loadmat(file)['val'][signal_type_index, :]
        current_length = len(signal)
        if current_length < discard_value:
            continue

        filtered_files.append(file)
        signals.append(signal)

    return filtered_files, signals


def find_shortest_length(signals):
    """
    Function that finds the shortest length greater than the discard threshold.
    Returns the shortest length and the rounded value of the shortest length.

    :param signals: The list of ECG/PCG signals.
    :return: (shortest_length, time_length)
             where shortest_length is the shortest length amongst the subjects.
                   time_length is the rounded down shortest length to the nearest 1000.
    """
    assert len(signals) != 0, "List of signals can't be empty"

    # Find subject with the shortest data length greater than 30000
    # and round it down to the nearest 1000.
    time_length = float("inf")
    shortest_length = 0
    for signal in signals:
        current_length = len(signal)

        if time_length > current_length:
            shortest_length = current_length
            time_length = current_length - (current_length % 1000)

    return shortest_length, time_length


def set_signal_type_index(signal_type):
    """
    Function that sets the index of the signal type.
    0 for PCG.
    1 for ECG.

    :param signal_type: The signal type string
    :return: The signal type index
    """
    assert signal_type.lower() == "pcg" or signal_type.lower() == "ecg", "Unknown signal type"

    if signal_type.lower() == "pcg":
        signal_type_index = 0
    elif signal_type.lower() == "ecg":
        signal_type_index = 1
    else:
        raise ValueError("Signal type can only be PCG or ECG.")

    return signal_type_index


def print_results(list_files, mat_data, data_labels, shortest_length, time_length):
    """
    Function to print out preprocessing results of the ECG/PCG data.
    Notable data includes total number of subjects used and discarded
    and total number of normal and abnormal signals.

    :param list_files: All the MATLAB files
    :param mat_data: The MATLAB files that are used
    :param data_labels: The binary classification labels (normal/abnormal)
    :param shortest_length: The shortest length amongst the subjects
    :param time_length: The rounded down value of the shortest length (to the nearest 1000)
    :return: None
    """
    num_subjects = len(list_files)
    num_used = len(mat_data)
    num_discarded = len(list_files) - len(mat_data)
    num_normal = len(data_labels[data_labels == -1])
    num_abnormal = len(data_labels[data_labels == 1])

    print("Total number of subjects:", num_subjects)
    print("Total number of used subjects:", num_used)
    print("Total number of discarded subjects:", num_discarded)
    print("Total number of normal signals:", num_normal)
    print("Total number of abnormal signals:", num_abnormal)
    print("Shortest length:", shortest_length)
    print("Rounded down shortest length:", time_length)
    print()


def split_train_test(data, labels):
    """
    Function that splits the data into training and test data for the model.
    Training data: 80%
    Test data: 20%

    :param data: The numpy array data
    :param labels: The normal/abnormal labels
    :return: A tuple of (x_train, x_test, y_train, y_test)
                where   x_train - The ECG/PCG training data
                        x_test - The ECG/PCG test data
                        y_train - The label training data
                        y_test - The label test data
    """
    assert type(data) == np.ndarray, "Data is not a numpy array"
    assert type(labels) == np.ndarray, "Labels is not a numpy array"
    assert data.size != 0, "Data array is empty"
    assert labels.size != 0, "Labels array is empty"

    return train_test_split(data, labels, test_size=0.20, random_state=42)


def validate_data(mat_data, data_labels):
    assert len(mat_data) >= 2 and len(data_labels) >= 2, "Data to be classified must have at least 2 subjects."

    assert len(data_labels[data_labels == -1]) >= 1, "There must be at least 1 normal subject."
    assert len(data_labels[data_labels == 1]) >= 1, "There must be at least 1 abnormal subject."
