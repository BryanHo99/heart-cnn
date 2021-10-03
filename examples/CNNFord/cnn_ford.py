"""
Time series CNN classification implemented from:
https://keras.io/examples/timeseries/timeseries_classification_from_scratch/

Dataset Description:
The dataset used is FordA taken from the UCR archive. The dataset contains
3601 training data and 1320 testing data. Each time series represents the
measurement of engine noise.

We are attempting to develop a CNN classification model to detect the
difference between normal/abnormal engine noise.

IMPORTANT NOTE:
Make sure to run the command "pip install pydot"
"""


from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def read_ucr(filename):
    """
    Function that preprocesses the data into separate x and y arrays.
    e.g.
    data = [
        [1, 2, 3, 4],
        [0, 3, 6, 9, 10],
        [1, 1, 1, 2, 3]
    ]

    y = [1, 0, 1]
    x = [[2, 3, 4], [3, 6, 9, 10], [1, 1, 2, 3]]

    :param filename : The CSV/TSV file
    :return         : A tuple of (x, y) where
                        x - 2D array
                        y - 1D array where all elements are casted to integer type
    """
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]          # For every array, take the 0th element (forms 1D array)
    x = data[:, 1:]         # For every array, take the 1st to the last element (forms 2D array)
    return x, y.astype(int)


def make_model(input_shape):
    """
    Function that makes a Fully Convolutional Neural Network proposed in this paper:
    https://arxiv.org/abs/1611.06455.

    The implementation is based on the following link:
    https://github.com/hfawaz/dl-4-tsc/

    :param input_shape  : A shape tuple (integers)
    :return             : A Keras model
    """
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


# STEP 1: Read TSV data
print("CREATING TRAINING AND TESTING SETS")
root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

# Create training and testing instances of the data.
x_train, y_train = read_ucr(root_url + "FordA_TRAIN.tsv")
x_test, y_test = read_ucr(root_url + "FordA_TEST.tsv")


# STEP 2: Visualize the data
print("VISUALIZING DATA")
classes = np.unique(np.concatenate((y_train, y_test), axis=0))

# Plot the graph.
plt.figure()
for c in classes:
    c_x_train = x_train[y_train == c]
    plt.plot(c_x_train[0], label="class " + str(c))
plt.legend(loc="best")
plt.show()
plt.close()


# STEP 3: Standardize the data
# Reshape into a 3D array since CNN needs 3D array as input.
# Reshaping idea: numpy.reshape((row, column, depth))
print("STANDARDIZING DATA")
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# Count number of classes (aka labels).
# In this case, there are 2 classes: -1 and 1.
num_classes = len(np.unique(y_train))

# Shuffle training set.
idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

# Standardize labels to positive integers. -1 is changed to 0.
# Expected labels will be 0 and 1.
y_train[y_train == -1] = 0
y_test[y_test == -1] = 0


# STEP 4: Build a model
print("BUILDING MODEL")
model = make_model(input_shape=x_train.shape[1:])


# STEP 5: Train the model
print("TRAINING MODEL")
epochs = 200
batch_size = 32

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model.h5", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.2,
    verbose=1,
)


# STEP 6: Evaluate the model on test data
print("EVALUATING MODEL")
model = keras.models.load_model("best_model.h5")

test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test accuracy", test_acc)
print("Test loss", test_loss)


# STEP 7: Plot model's training and validation loss
metric = "sparse_categorical_accuracy"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.show()
plt.close()
