# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 10:49:45 2021
for more details refer to:
    https://librosa.org/doc/main/generated/librosa.feature.mfcc.html
@author: FUAD

"""
import librosa
import librosa.display
import scipy.io as sio
import matplotlib.pyplot as plt

# ECG - PCG sample rate (samples per second or Hz)
Fs = 1000
# load 2-channel PCG/ECG data
data = sio.loadmat('a0001m.mat')['val']

PCG = data[0, :].astype(float)
ECG = data[1, :].astype(float)

# extract MFCC from PCG
'''
inputs:
    PCG: one-dimensional time-series 
    sr: sampling rate of PCG
    n_mfcc: number of mfcc coefficients
    hop_length: sliding window length
Output:
    a (n_mfcc,L) matrix of mfcc coeffs 
    where L is the total sliding windows
    (i.e., len(PCG)/hop_length)
'''
mfccs = librosa.feature.mfcc(PCG, sr=Fs, n_mfcc=20, hop_length=128)

fig, ax = plt.subplots()
img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
fig.colorbar(img, ax=ax)
ax.set(title='MFCC')
plt.show()

print(len(PCG))
