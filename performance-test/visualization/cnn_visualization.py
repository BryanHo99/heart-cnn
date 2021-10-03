import matplotlib.pyplot as plt     # Package for graphical visualization.
import librosa                      # Package for MFCC feature extraction.
import librosa.display              # Package for visualizing MFCC.
import numpy as np                  # Package for selecting a subject to be displayed.


def visualize_mfcc(mfcc_data, labels):
    """
    Function that displays a spectrogram to visualize an
    MFCC series of PCG normal and abnormal signals.

    :param mfcc_data: The MFCC numpy array
    :param labels: The binary classification labels (normal/abnormal)
    :return: None
    """
    assert type(mfcc_data) == np.ndarray, "mfcc_data array must be a numpy array"
    assert type(labels) == np.ndarray, "labels array must be a numpy array"
    assert mfcc_data.size != 0, "mfcc_data array must not be an empty array"
    assert labels.size != 0, "labels array must not be an empty array"

    normal_subject, abnormal_subject = select_random_subjects(mfcc_data, labels)

    fig, (ax1, ax2) = plt.subplots(2)
    fig.tight_layout(pad=3.5)

    # Display MFCC spectrogram of a random normal subject.
    img1 = librosa.display.specshow(normal_subject, x_axis='time', ax=ax1)
    fig.colorbar(img1, ax=ax1)
    ax1.set(title='MFCC Normal')

    # Display MFCC spectrogram of a random abnormal subject.
    img2 = librosa.display.specshow(abnormal_subject, x_axis='time', ax=ax2)
    fig.colorbar(img2, ax=ax2)
    ax2.set(title='MFCC Abnormal')

    plt.show()
    plt.close()


def visualize_wavelet(wavelet_data, labels):
    """
    Function that displays a spectrogram to visualize a
    Wavelet Transform series of ECG normal and abnormal signals.

    :param wavelet_data: The Wavelet Transform numpy array
    :param labels: The binary classification labels (normal/abnormal)
    :return: None
    """
    assert type(wavelet_data) == np.ndarray, "wavelet_data array must be a numpy array"
    assert type(labels) == np.ndarray, "labels array must be a numpy array"
    assert wavelet_data.size != 0, "wavelet_data array must not be an empty array"
    assert labels.size != 0, "labels array must not be an empty array"

    normal_subject, abnormal_subject = select_random_subjects(wavelet_data, labels)

    fig, (ax1, ax2) = plt.subplots(2)
    fig.tight_layout(pad=3.5)

    # Display Wavelet Transform image of a random normal subject.
    ax1.imshow(normal_subject, extent=[-1, 1, 1, 100], cmap='PRGn', aspect='auto',
               vmax=abs(normal_subject).max(), vmin=-abs(normal_subject).max())
    ax1.set(title='Wavelet Normal')

    # Display Wavelet Transform image of a random abnormal subject.
    ax2.imshow(abnormal_subject, extent=[-1, 1, 1, 100], cmap='PRGn', aspect='auto',
               vmax=abs(abnormal_subject).max(), vmin=-abs(abnormal_subject).max())
    ax2.set(title='Wavelet Abnormal')

    plt.show()
    plt.close()


def select_random_subjects(data, labels):
    """
    Function that randomly selects a normal subject and abnormal subject to have their
    feature extracted ECG/PCG visualized.

    :param data: The ECG/PCG data of all subjects
    :param labels: The binary classification labels (normal/abnormal)
    :return: (normal_subject, abnormal_subject) where both are 2D numpy arrays
    """
    assert type(data) == np.ndarray, "data array must be a numpy array"
    assert type(labels) == np.ndarray, "labels array must be a numpy array"
    assert data.size != 0, "data array should not be empty"
    assert labels.size != 0, "labels array should not be empty"

    normal_subject = data[labels == -1]
    normal_subject = normal_subject[np.random.choice(len(normal_subject))]

    abnormal_subject = data[labels == 1]
    abnormal_subject = abnormal_subject[np.random.choice(len(abnormal_subject))]

    return normal_subject, abnormal_subject


def visualize_cnn_accuracy_loss(history):
    """
    Function that displays a graph of the training and validation
    accuracy/loss after evaluating the CNN model.

    :param history: The History object from Keras after training the model.
    :return: None
    """
    fig, (ax1, ax2) = plt.subplots(2)
    fig.tight_layout(pad=3.5)

    # Plot model's training and validation loss.
    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.set_title('Training Loss vs Validation Loss')
    ax1.set(xlabel='Epoch', ylabel='Loss')
    ax1.legend(['Training Loss', 'Validation Loss'], loc='best')

    # Plot model's training and validation accuracy.
    ax2.plot(history.history['sparse_categorical_accuracy'])
    ax2.plot(history.history['val_sparse_categorical_accuracy'])
    ax2.set_title('Training Accuracy vs Validation Accuracy')
    ax2.set(xlabel='Epoch', ylabel='Accuracy')
    ax2.legend(['Training Accuracy', 'Validation Accuracy'], loc='best')

    plt.show()
    plt.close()
