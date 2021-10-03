"""
Time series CNN classification implemented from:
https://keras.io/examples/timeseries/timeseries_classification_from_scratch/

Example to execute the file in the command line:
python cnn.py [folder name of MATLAB data] [SQL CSV file] [pcg/ecg]
e.g. python cnn.py training-data REFERENCE_withSQI.csv pcg
"""
# Package to load model.
from tensorflow import keras
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import sys                          # Package to read command line arguments.
import numpy as np                  # Package for numerical analysis.
import matplotlib.pyplot as plt
from datetime import datetime
# Local imports for preprocessing, visualization and building model.
from data_preprocessing.preprocessing import read_mat, split_train_test
from implement_model.cnn_model import make_cnn_model, train_cnn_model
from configs.cnn_config import cnn_config


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])

if __name__ == "__main__":
    try:
        assert len(sys.argv) == 4, "Invalid number of command line arguments."

        _, folder_name, sqi_csv, signal_type = sys.argv

        # Number of convolutional layers to use.
        num_conv_layers = 3

        data, labels = read_mat(folder_name, sqi_csv, signal_type)
        x_train, x_test, y_train, y_test = split_train_test(data, labels)

        # Count number of classes (aka labels).
        # In this case, there are 2 classes: -1 and 1.
        num_classes = len(np.unique(y_train))

        # Standardize labels to positive integers. -1 is changed to 0.
        # Expected labels will be 0 and 1.
        y_train[y_train == -1] = 0
        y_test[y_test == -1] = 0


        # Train the Model
        # training_iterations = 2     # Number of times to repeat training
        training_iterations = 3     # Number of times to repeat training
        # learning_rates = [0.001,  # used for testing
        #                 0.0005]
        # batch_sizes = [28,        # used for testing
        #                 30]
        learning_rates = [0.001,
                        0.0005,
                        0.0001,
                        0.00005,
                        0.00001]
        batch_sizes = [28,
                        30,
                        32,
                        34,
                        36]
        # print("Training Model...")
        start_now = datetime.now()
        start_time = start_now.strftime("%H:%M:%S")
        print('CNN', signal_type, 'training start time:', start_time)
        # Initialize results file
        results_filename = 'cnn_best-'+signal_type+'-results.txt'
        f = open(results_filename, 'w')
        f.write('CNN-'+signal_type+'-training\n')
        f.write('start time: '+start_time+'\n')
        f.close()
        for lr in learning_rates:
            for bs in batch_sizes:
                # Set learning rate and batch size in cnn_config
                cnn_config['learning_rate'] = lr
                cnn_config['batch_size'] = bs
                # print('training cnn with lr, bs', cnn_config['learning_rate'], cnn_config['batch_size'])
                for i in range(training_iterations):
                    # Set model and metrics plots filenames for current trianing iteration
                    filename = 'cnn_best'+'-'+\
                                        signal_type+'-'\
                                        'lr'+str(cnn_config['learning_rate'])+'-'+\
                                        'bs'+str(cnn_config['batch_size'])+'-'+\
                                        'i'+str(i+1)
                    # print('curr iteration filename:', filename)
                    saved_model_name = filename + cnn_config['saved_model_format']
                    metric_plot_name = filename + cnn_config['saved_plot_format']
                    cnn_config['saved_model_name'] = saved_model_name
                    cnn_config['saved_plot_name'] = metric_plot_name

                    # Build a Model
                    # print("Building Model...")
                    model = make_cnn_model(x_train.shape[1:], num_conv_layers, num_classes)

                    # Train CNN with current iteration parameters, save as cnn_config['saved_model_name']
                    history = train_cnn_model(model, x_train, y_train, cnn_config)
                    # Plot accuracy and loss, save as cnn_config['saved_plot_name']
                    plt.figure(figsize=(13, 7))
                    plt.subplot(1, 2, 1)
                    plot_graphs(history, 'accuracy')
                    plt.ylim(0, 1)
                    plt.subplot(1, 2, 2)
                    plot_graphs(history, 'loss')
                    plt.ylim(0, None)
                    # plt.show()
                    plt.savefig(cnn_config['saved_plot_name'])
                    plt.clf()
                    plt.close()

                    # Load the saved model weights
                    # print('Loading model:', cnn_config['saved_model_name'])
                    del model
                    model = make_cnn_model(x_train.shape[1:], num_conv_layers, num_classes)
                    # compile the model
                    adam = Adam(cnn_config['learning_rate'])
                    model.compile(
                        optimizer=adam,
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"],
                    )
                    model.load_weights(cnn_config['saved_model_name'])

                    # Evaluate the Model on Test Data
                    # print("Evaluating Model...")
                    # Obtain test accuracy and test loss.
                    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
                    print(filename, "test accuracy, loss:", np.round(test_acc, 4), np.round(test_loss, 4))
                    f = open(results_filename, 'a')
                    results_output = filename+' -- '+'test accuracy, loss: '+\
                                        str(np.round(test_acc, 4))+', '+\
                                        str(np.round(test_loss, 4))+'\n'
                    f.write(results_output)
                    f.close()
                    del model # make sure model isn't reused in next iteration

        end_now = datetime.now()
        end_time = end_now.strftime("%H:%M:%S")
        print('CNN', signal_type, 'training end time:', end_time)

        elapsed_time = end_now - start_now
        print('CNN', signal_type, 'training elapsed time:', elapsed_time)
        f = open(results_filename, 'a')
        f.write('end time: '+end_time+'\n')
        f.write('elapsed time: '+str(elapsed_time)+'\n')
        f.close()

    except (AssertionError, ValueError) as e:
        print("ERROR:", e)
        print("Exiting program.")
        exit()
