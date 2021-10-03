# Header
from configs.lstm_config import lstm_config
from data_preprocessing.lstm_pre import load_files, truncate_data, standardize
from feature_extraction.lstm_feat import extract_mfcc, extract_wavelet
from sklearn.model_selection import train_test_split
from implement_model.lstm_model import build_lstm, train_lstm
import sys                          # Package to read command line arguments.
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
# import numpy as np                  # Package for numerical analysis.
def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])

if __name__ == "__main__":
    # assert len(sys.argv) == 4, "Invalid number of command line arguments."
    _, folder_name, sqi_csv, signal_type = sys.argv
    # Load dataset
    # raw_data, labels = load_files('training-a', 'REFERENCE_withSQI.csv', 'pcg', lstm_config)
    raw_data, labels = load_files(folder_name, sqi_csv, signal_type)
    # Truncate signals to the shortest signal longer than min_signal_length1
    data, labels = truncate_data(raw_data, labels, lstm_config)
    normal_count = 0
    abnormal_count = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            abnormal_count += 1
        else:
            normal_count += 1
    # print("normal, abnormal")
    # print(normal_count, abnormal_count)
    # Standardize data
    data = standardize(data)
    # print('raw shape', data.shape)
    # Feature extraction
    if signal_type.lower() == 'ecg':
        data = extract_wavelet(data, lstm_config)
    elif signal_type.lower() == 'pcg':
        data = extract_mfcc(data, lstm_config)
    else:
        raise Exception('invalid signal type in main-lstm')
    # print('feat extracted shape', data.shape)
    # Split dataset - 60:20:20 ratio of training, testing and validation respectively
    x_train, x_test, y_train, y_test \
        = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)      # split 20% testing set
    x_train, x_val, y_train, y_val \
        = train_test_split(x_train, y_train, test_size=0.25, stratify=y_train, random_state=42) # split 20% validation set
    # Build LSTM
    model = build_lstm(data.shape, lstm_config)
    print(model.summary())
    # Train LSTM
    history = train_lstm(model, (x_train,y_train), (x_val,y_val), lstm_config)

    # Plot accuracy and loss
    plt.figure(figsize=(13, 7))
    plt.subplot(1, 2, 1)
    plot_graphs(history, 'accuracy')
    plt.ylim(0, 1)
    plt.subplot(1, 2, 2)
    plot_graphs(history, 'loss')
    plt.ylim(0, None)
    plt.show()
    plt.clf()
    plt.close()

    import numpy as np
    best_epoch_i = np.argmax(history.history['val_accuracy'])
    print('best epoch was number', best_epoch_i+1)
    print('best epoch had train_ acc, loss, spec, sens',
        np.round(history.history['accuracy'][best_epoch_i], 4),
        np.round(history.history['loss'][best_epoch_i], 4),
        np.round(history.history['specificity'][best_epoch_i], 4),
        np.round(history.history['sensitivity'][best_epoch_i], 4)
    )
    print('best epoch had val_ acc, loss, spec, sens',
        np.round(history.history['val_accuracy'][best_epoch_i], 4),
        np.round(history.history['val_loss'][best_epoch_i], 4),
        np.round(history.history['val_specificity'][best_epoch_i], 4),
        np.round(history.history['val_sensitivity'][best_epoch_i], 4)
    )
    print('completed epochs', len(history.history['val_accuracy']))

    # Evaluate
    print('Loading model named:', lstm_config['best_model_name'])
    del model
    model = build_lstm(data.shape, lstm_config)
    model.compile(loss=lstm_config['loss'],
                    optimizer=Adam(lstm_config['learning_rate']),
                    metrics=lstm_config['metrics'])
    model.load_weights(lstm_config['best_model_name'])
    loss, acc, spec, sens, f1, recall, prec = model.evaluate(x_test, y_test, verbose=0)
    print('Saved model performance on test set:')
    print('    Accuracy:', acc)
    print('    Loss:', loss)
    print('    Sensitivity:', sens)
    print('    Specificity:', spec)

    # Predict
    train_preds = model.predict(x_train, verbose=0)
    test_preds = model.predict(x_test, verbose=0)
    # print(train_preds[:10])
    # print(test_preds[:10])
    # score > 0.5 : 1--abnormal
    # score <= 0.5 : 0--normal
    threshold_train_preds = np.where(train_preds > 0.5, 1, 0)
    threshold_test_preds = np.where(test_preds > 0.5, 1, 0)
    # print(threshold_train_preds[:10])
    # print(threshold_test_preds[:10])
    # compute accuracy score
    train_accuracy = accuracy_score(y_train, threshold_train_preds)
    test_accuracy = accuracy_score(y_test, threshold_test_preds)
    print('prediction accuracy on training set:', np.round(train_accuracy, 4))
    print('prediction accuracy on test set:', np.round(test_accuracy, 4), '(same as model.evaluate() accuracy score)')
