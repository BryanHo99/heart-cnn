# Header
from configs.lstm_config import lstm_config
from data_preprocessing.lstm_pre import load_files, truncate_data, standardize
from feature_extraction.lstm_feat import extract_mfcc, extract_wavelet
from sklearn.model_selection import train_test_split
from implement_model.lstm_model import build_lstm, train_lstm
import sys
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

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

    # normal_count = 0
    # abnormal_count = 0
    # for i in range(len(labels)):
    #     if labels[i] == 1:
    #         abnormal_count += 1
    #     else:
    #         normal_count += 1
    # print("--normal, abnormal--")
    # print(normal_count, abnormal_count)
    # print('normal:', np.round(normal_count/len(data), 4))
    # print('abnormal:', np.round(abnormal_count/len(data), 4))

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
    # print(model.summary())

    # training_iterations = 2     # used for demo
    training_iterations = 3     # Number of times to repeat training
    learning_rates = [0.0001, 0.00005]  # used for demo
    batch_sizes = [16, 20]      # used for demo
    # learning_rates = [0.0001,
    #                 0.00005,
    #                 0.00001,
    #                 0.000005,
    #                 0.000001]
    # batch_sizes = [14,
    #                 16,
    #                 18,
    #                 20,
    #                 22]
    start_now = datetime.now()
    start_time = start_now.strftime("%H:%M:%S")
    print('LSTM', signal_type, 'training start time:', start_time)
    # Initialize results file
    results_filename = 'lstm_best-'+signal_type+'-results.txt'
    f = open(results_filename, 'w')
    f.write('LSTM-'+signal_type+'-training\n')
    f.write('start time: '+start_time+'\n')
    f.close()

    for lr in learning_rates:
            for bs in batch_sizes:
                # Set learning rate and batch size in lstm_config
                lstm_config['learning_rate'] = lr
                lstm_config['batch_size'] = bs
                # print('training lstm with lr, bs', lstm_config['learning_rate'], lstm_config['batch_size'])
                for i in range(training_iterations):
                    # Set model and metrics plots filenames for current trianing iteration
                    filename = 'lstm_best'+'-'+\
                                    signal_type+'-'\
                                    'lr'+str(lstm_config['learning_rate'])+'-'+\
                                    'bs'+str(lstm_config['batch_size'])+'-'+\
                                    'i'+str(i+1)
                    # print('curr iteration filename:', filename)
                    saved_model_name = filename + lstm_config['saved_model_format']
                    metric_plot_name = filename + lstm_config['saved_plot_format']
                    lstm_config['saved_model_name'] = saved_model_name
                    lstm_config['saved_plot_name'] = metric_plot_name

                    # Build LSTM
                    model = build_lstm(data.shape, lstm_config)

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
                    # plt.show()
                    plt.savefig(lstm_config['saved_plot_name'])
                    plt.clf()
                    plt.close()

                    # Evaluate
                    # print('Loading model:', lstm_config['saved_model_name'])
                    del model
                    model = build_lstm(data.shape, lstm_config)
                    model.load_weights(lstm_config['saved_model_name'])
                    loss, acc, spec, sens, f1, recall, prec = model.evaluate(x_test, y_test, verbose=0)
                    print(filename, "test accuracy, loss:", np.round(acc, 4), np.round(loss, 4))
                    # print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
                    f = open(results_filename, 'a')
                    results_output = filename+' -- '+'test accuracy, loss: '+\
                                        str(np.round(acc, 4))+', '+\
                                        str(np.round(loss, 4))+'\n'
                    f.write(results_output)
                    f.close()
                    del model # make sure model isn't reused in next iteration

    end_now = datetime.now()
    end_time = end_now.strftime("%H:%M:%S")
    print('LSTM', signal_type, 'training end time:', end_time)

    elapsed_time = end_now - start_now
    print('LSTM', signal_type, 'training elapsed time:', elapsed_time)
    f = open(results_filename, 'a')
    f.write('end time: '+end_time+'\n')
    f.write('elapsed time: '+str(elapsed_time)+'\n')
    f.close()

    # Predict - on entire test dataset
    # train_preds = model.predict(x_train, verbose=1)
    # test_preds = model.predict(x_test, verbose=1)
    # print(train_preds[:10])
    # print(test_preds[:10])
    # score > 0.5 : 1--abnormal
    # score <= 0.5 : 0--normal
    # threshold_train_preds = np.where(train_preds > 0.5, 1, 0)
    # threshold_test_preds = np.where(test_preds > 0.5, 1, 0)
    # print(threshold_train_preds[:10])
    # print(threshold_test_preds[:10])
    # compute accuracy score
    # train_accuracy = accuracy_score(y_train, threshold_train_preds)
    # test_accuracy = accuracy_score(y_test, threshold_test_preds)
    # print('prediction accuracy on training set:', np.round(train_accuracy, 4))
    # print('prediction accuracy on test set:', np.round(test_accuracy, 4))
