import glob                         # Package to search for files in a given directory.
import os                           # Package used to process system paths
from tqdm import tqdm               # Package to show progress bar of a loop.
import scipy.io as sio              # Package to load matlab files (*.mat) of ECG-PCG data.
import pandas as pd                 # Package to read labels from CSV/Excel file.
import numpy as np                  # Package for numpy array initialization.
from sklearn.preprocessing import StandardScaler

def load_files(folder_name, sqi_csv, signal_type):
    """
    Load MATLAB files and corresponding labels.
    The MATLAB files must be in the root of the folder's directory.
    The SQI CSV file must have the following headers as its structure:
        - Data name
        - Label -> -1: Normal, 1: Abnormal
        - Signal Quality Index (SQI) -> 0: Noisy, 1: Clean
    :return: (raw_signals, labels)
                mat_data - 2D numpy array of ECG/PCG data
                labels - 1D numpy array of labels
    """
    assert signal_type.lower() == 'ecg' or signal_type.lower() == 'pcg', 'invalid signal_type in load_files(...)'
    assert folder_name and len(folder_name) > 0, 'invalid folder_name or sqi_csv in load_files(...)'
        
    # 0 for PCG, 1 for ECG.
    if signal_type.lower() == 'ecg':
        signal_type_index = 1
    elif signal_type.lower() == 'pcg':
        signal_type_index = 0

    # Load MAT files in folder_name.
    directory = f"./{folder_name}/*.mat"
    list_files = glob.glob(directory)
    # Load the demographic information from csv file.
    demo = pd.read_csv(sqi_csv)
    assert len(list_files) >= 2 and len(demo) >= 2,\
        "Must have at least 2 subjects -- 1 normal, 1 abnormal."

    raw_signals = []    # Array stores raw ECG or PCG signals
    labels = []         # Array stores labels for corresponding signals
    for file in tqdm(list_files):
        data = sio.loadmat(file)['val']
        # data = data[signal_type_index, :]
        data = data[signal_type_index, :].astype(float)
        raw_signals.append(data)

        _, filename = os.path.split(file)                 # Find the MAT file name of the loaded data
        filename = filename.split(".")[0][:-1]               # Remove the file extension from the filename
        label = demo.label[demo['name'] == filename].values  # Extract the label of the particular "filename" data
        label[label == -1] = 0                               # change normal label from -1 to 0
        label[label == 1] = 1
        labels.append(label)                                 # Append label to array "labels"

    raw_signals = np.array(raw_signals, dtype=object)
    labels = np.array(labels).squeeze()
    return raw_signals, labels

def truncate_data(raw_signals, labels, config={}):
    """
    Discard signals with length below threshold, truncate all signals to length of
    shortest remaining signal
    :return: (data, labels) tuple containing truncated raw signals with corresponding labels
    """
    assert 'min_signal_length' in config and 'round_down' in config, 'missing keys in config'
    filtered_data = []
    filtered_labels = []
    shortest = len(raw_signals[0])
    for i in range(len(raw_signals)):
        # Discard signals shorter than threshold
        if len(raw_signals[i]) < config['min_signal_length']:
            continue
        filtered_data.append(raw_signals[i])
        filtered_labels.append(labels[i])
        # Find shortest remaining signal in dataset
        if len(raw_signals[i]) < shortest:
            shortest = len(raw_signals[i])
    # Truncate remaining signals to shortest signal, optionally round down to nearest thousand
    if config['round_down']:
        shortest = shortest // 1000 * 1000
    for i in range(len(filtered_data)):
        filtered_data[i] = filtered_data[i][:shortest]

    filtered_data = np.array(filtered_data)
    filtered_labels = np.array(filtered_labels)
    return filtered_data, filtered_labels

def standardize(signals):
    """
    Standadizes the input dataset
    :return: Standardized data
    """
    scaler = StandardScaler().fit(signals)
    scaled = scaler.transform(signals)
    return scaled