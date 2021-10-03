# Heart CNN
Final year project focusing on deep learning classification of ECG and PCG signals for cardiac abnormality detection.

## Resources
- [Final Report](Final&#32;Report.pdf)
- [User and Technical Guide](User&#32;and&#32;Technical&#32;Guide.pdf)

## Summary

#### Preprocessing
1. Parse MATLAB dataset into NumPy arrays.
2. Discard ECG/PCG signals lower than the given threshold.
3. Find shortest length from remaining signals.
4. Truncate remaining signals to shortest signal, optionally rounding down to nearest thousand.
5. Standardize features with Standard Scaler.
6. Extract features of ECG/PCG signals with CWT and MFCC.
7. Stratify dataset during train-test split.

#### Feature Extraction
- Mel-frequency Cepstrum Coefficients (MFCC)
- Continuous Wavelet Transform (CWT)

#### Classification
- Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)
