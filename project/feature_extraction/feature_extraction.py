import librosa                      # Package for MFCC feature extraction.
from scipy import signal            # Package for Wavelet Transform feature extraction.
import numpy as np                  # Package for numerical analysis.


def mfcc(pcg_data, sample_rate, n_mfcc, hop_length):
    """
    Function that uses Librosa for MFCC feature extraction on PCG signals.
    Referenced from: https://librosa.org/doc/main/generated/librosa.feature.mfcc.html

    :param pcg_data: The MATLAB PCG data
    :param sample_rate: The sample rate of the PCG data
    :param n_mfcc: The number of MFCCs to return
    :param hop_length: Sliding window length
    :return: A 2D numpy matrix of MFCC coefficients of dimensions: (n_mfcc, L)
             where L is the total number of sliding windows (i.e., len(pcg_data)/hop_length)
    """
    assert type(pcg_data) == np.ndarray, "array must be a numpy array"
    assert pcg_data.size != 0, "numpy array must not be empty"

    return librosa.feature.mfcc(pcg_data, sr=sample_rate, n_mfcc=n_mfcc, hop_length=hop_length)


def wavelet_transform(ecg_data, widths):
    """
    Function that uses SciPy for Wavelet Transform feature extraction on ECG signals.
    Referenced from: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cwt.html

    Wavelet function: Ricker Wavelet / Mexican Hat Wavelet
    Referenced from: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ricker.html#scipy.signal.ricker

    :param ecg_data: The MATLAB ECG data
    :param widths: Widths to use for the transform
    :return: A 2D numpy matrix of Wavelet coefficients of dimensions: (M, N)
             where M = len(np.arange(1, widths + 1))
                   N = len(ecg_data)
    """
    assert type(ecg_data) == np.ndarray, "array must be a numpy array"
    assert ecg_data.size != 0, "numpy array must not be empty"

    return signal.cwt(ecg_data, signal.ricker, np.arange(1, widths + 1))
