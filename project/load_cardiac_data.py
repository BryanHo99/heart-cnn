# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:38:36 2020

@author: FUAD
"""
# package to search for files in a given directory
import glob
# package to show progress bar of a loop
from tqdm import tqdm
# package to load matlab files (*.mat) of ECG-PCG data
import scipy.io as sio
# package to read labels from csv/excel file
import pandas as pd
# package used to process system paths
import os
# package for numerical analysis
import numpy as np
# package for visualization (plots)
import matplotlib.pyplot as plt
# local helper functions for ECG/PCG filtering
from filters import remove_baseline_wander

# %%
# ECG - PCG sample rate (samples per second or Hz)
sample_rate = 1000
# search for all MAT files in data path "./training-a/"
listFiles = glob.glob("./training-a/*.mat")
# load the demographic information from csv file.
# demo contains "data name", "label ->-1:normal, 1:abnormal", 
# signal quality index (sqi) "0: noisy, 1:clean"
demo = pd.read_csv('REFERENCE_withSQI.csv')

# create empty list array to store the ECG-PCG data
ecg_pcg_data = []
# create empty list array to store classification labels
labels = []

for file in tqdm(listFiles):
    data = sio.loadmat(file)['val']  # load data from mat file
    ecg_pcg_data.append(data)  # append data to array "ecg_pcg_data"
    path, filename = os.path.split(file)  # find the MAT file name of the loaded data
    filename = filename.split(".")[0][:-1]  # remove the file extension from the filename
    label = demo.label[demo['name'] == filename].values  # extract the label of the particular "filename" data
    labels.append(label)  # append label to array "labels"

labels = np.array(labels).squeeze()  # convert list array to numpy array

# %%
normal = 6  # select normal subject number
abnormal = 1  # select abnormal subject number

# ================================ Plot Normal Example
# extract heart sound (PCG) data, channel 0 of ecg_pcg_data.
PCG_data = ecg_pcg_data[normal][0][500:].squeeze()
# extract ECG data, channel 1 of ecg_pcg_data. ignore the first 500 (noisy) samples 
ECG_data = ecg_pcg_data[normal][1][500:].squeeze()
# remove the baseline drift using notch filter
ECG_data = remove_baseline_wander(ECG_data, sample_rate, cutoff=0.01)

# remove the mean of ECG data
ECG_data = ECG_data - np.mean(ECG_data)

fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # create subplot figure of 2 rows and 1 columns
axs[0].plot(PCG_data)  # plot PCG data for healthy subject No 6
axs[0].set_title('Normal - PCG')  # add title

axs[1].plot(ECG_data)  # plot ECG data for healthy subject No 6
axs[1].set_title('Normal - ECG')  # add title

# ================================ Plot Abnormal Example
PCG_data = ecg_pcg_data[abnormal][0][500:].squeeze()
ECG_data = ecg_pcg_data[abnormal][1][500:].squeeze()
ECG_data = remove_baseline_wander(ECG_data, sample_rate, cutoff=0.01)

ECG_data = ECG_data - np.mean(ECG_data)

fig, axs = plt.subplots(2, 1, figsize=(10, 8))
axs[0].plot(PCG_data)
axs[0].set_title('Abnormal - PCG')

axs[1].plot(ECG_data)
axs[1].set_title('Abnormal - ECG')
plt.show()
