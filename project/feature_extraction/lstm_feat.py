from librosa.feature import mfcc, delta
from scipy import signal
import numpy as np

# from librosa.display import specshow

def extract_mfcc(signals, config={}):
    """
    Extract Mel-frequency cepstral coefficients (MFCC), deltas and delta-deltas from PCG data,
    Savitsky-Golay filter smoothing is performed on MFCC deltas and delta-deltas as 
    the difference operator is sensitive to noise.
    Referenced from: https://librosa.org/doc/main/generated/librosa.feature.mfcc.html
        and https://librosa.org/doc/main/generated/librosa.feature.delta.html
    :return: 3D numpy array containing the extracted MFCC 
    """
    if 'sample_rate' not in config:
        raise Exception('key: sample_rate, not found in config')
    if 'n_mfcc' not in config:
        raise Exception('key: n_mfcc, not found in config')
    if 'hop_length' not in config:
        raise Exception('key: hop_length, not found in config')
        
    mfccs = []
    for signal in signals:
        mfcc_order0 = mfcc(signal, sr=config['sample_rate'] ,n_mfcc=config['n_mfcc'], hop_length=config['hop_length'])
        mfcc_order1 = delta(mfcc_order0)
        mfcc_order2 = delta(mfcc_order0, order=2)
        mfcc_concatenated = np.vstack((mfcc_order0, mfcc_order1))
        mfcc_concatenated = np.vstack((mfcc_concatenated, mfcc_order2))
        mfccs.append(mfcc_concatenated)
    mfccs = np.array(mfccs)
    return mfccs


def extract_wavelet(signals, config={}):
    """
    SciPy for Wavelet Transform feature extraction on ECG signals.
    Referenced from: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cwt.html
    Wavelet function: Ricker Wavelet / Mexican Hat Wavelet.
    Referenced from: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ricker.html#scipy.signal.ricker

    :return: 3D numpy array containing the extracted wavelets 
    """
    if not config:
        raise Exception('empty config in extract_mfcc(...)')
    wavelets = []
    for data in signals:
        wavelet = signal.cwt(data, signal.ricker, np.arange(1, config['widths'] + 1))
        wavelets.append(wavelet)
    wavelets = np.array(wavelets)
    return wavelets

# def visualize_mfcc(data):
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
#     img1 = specshow(mfccs[0], ax=ax[0], x_axis='time')
#     ax[0].set(title='MFCC')
#     ax[0].label_outer()
#     img2 = specshow(mfccs_delta[0], ax=ax[1], x_axis='time')
#     ax[1].set(title=r'MFCC-$\Delta$')
#     ax[1].label_outer()
#     img3 = specshow(mfccs_delta2[0], ax=ax[2], x_axis='time')
#     ax[2].set(title=r'MFCC-$\Delta^2$')
#     fig.colorbar(img1, ax=[ax[0]])
#     fig.colorbar(img2, ax=[ax[1]])
#     fig.colorbar(img3, ax=[ax[2]])
#     plt.show()
