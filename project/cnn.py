"""
Time series CNN classification implemented from:
https://keras.io/examples/timeseries/timeseries_classification_from_scratch/

Example to execute the file in the command line:
python cnn.py [folder name of MATLAB data] [SQL CSV file] [pcg/ecg]
e.g. python cnn.py training-data REFERENCE_withSQI.csv pcg
"""
import sys                          # Package to read command line arguments.
import numpy as np                  # Package for numerical analysis.

# Package to load model.
from tensorflow import keras

# Local imports for preprocessing, visualization and building model.
from data_preprocessing.preprocessing import read_mat, split_train_test
from visualization.cnn_visualization import visualize_mfcc, visualize_cnn_accuracy_loss, visualize_wavelet
from implement_model.cnn_model import make_cnn_model, train_cnn_model


if __name__ == "__main__":
    try:
        assert len(sys.argv) == 4, "Invalid number of command line arguments."

        # Number of convolutional layers to use.
        num_conv_layers = 3

        # STEP 1: Read the Data
        # Read the MATLAB data and then split into training and testing instances.
        print("Creating Training and Testing Sets...")
        data, labels = read_mat(sys.argv[1], sys.argv[2], sys.argv[3])
        x_train, x_test, y_train, y_test = split_train_test(data, labels)

        # STEP 2: Visualize the Data
        print("Visualizing Data...")
        if sys.argv[3].lower() == "pcg":
            visualize_mfcc(data, labels)
        else:
            visualize_wavelet(data, labels)

        # STEP 3: Standardize the Data
        print("Standardizing Data...")

        # Count number of classes (aka labels).
        # In this case, there are 2 classes: -1 and 1.
        num_classes = len(np.unique(y_train))

        # Standardize labels to positive integers. -1 is changed to 0.
        # Expected labels will be 0 and 1.
        y_train[y_train == -1] = 0
        y_test[y_test == -1] = 0

        # STEP 4: Build a Model
        print("Building Model...")
        model = make_cnn_model(x_train.shape[1:], num_conv_layers, num_classes)
        model.summary()

        # STEP 5: Train the Model
        print("Training Model...")
        history = train_cnn_model(model, x_train, y_train)

        # STEP 6: Evaluate the Model on Test Data
        print("Evaluating Model...")
        model = keras.models.load_model("best_model.h5")

        # Obtain test accuracy and test loss.
        test_loss, test_acc = model.evaluate(x_test, y_test)
        print()
        print("Test Accuracy:", test_acc)
        print("Test Loss:", test_loss)

        visualize_cnn_accuracy_loss(history)

    except (AssertionError, ValueError) as e:
        print("ERROR:", e)
        print("Exiting program.")
        exit()
