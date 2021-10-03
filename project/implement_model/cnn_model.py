from tensorflow import keras                     # Package for CNN classifier.
import numpy as np

def conv(current_layer):
    """
    Function that adds a convolutional layer from the previous layer.
    It performs Conv1D, then Batch Normalization, and add a ReLU activation function.

    :param current_layer: The current layer to connect a new convolutional layer to
    :return: The next convolutional layer
    """
    next_layer = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(current_layer)
    next_layer = keras.layers.BatchNormalization()(next_layer)
    next_layer = keras.layers.ReLU()(next_layer)
    return next_layer


def make_cnn_model(input_shape, num_conv_layers, num_classes):
    """
    Function that makes a Fully Convolutional Neural Network proposed in this paper:
    https://arxiv.org/abs/1611.06455.

    The implementation is based on the following link:
    https://github.com/hfawaz/dl-4-tsc/

    :param input_shape: A shape tuple (integers)
    :param num_conv_layers:
    :param num_classes: The number of classes
    :return: A Keras CNN model
    """
    assert type(input_shape) is tuple, "Input_shape is not a tuple"
    assert num_classes >= 1, "num_classes must not be empty"

    input_layer = keras.layers.Input(input_shape)
    current_layer = input_layer

    for i in range(num_conv_layers):
        current_layer = conv(current_layer)

    gap = keras.layers.GlobalAveragePooling1D()(current_layer)

    output_layer = keras.layers.Dense(num_classes, activation="sigmoid")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def train_cnn_model(model, x_train, y_train):
    """
    Function that trains the CNN model.
    Returns a History object for reviewing training and validation losses
    as well as metrics at successive epochs.

    :param model: The Keras CNN model
    :param x_train: The ECG/PCG training data
    :param y_train: The label training data
    :return: The History object.
    """
    assert type(x_train) == np.ndarray, "Invalid x_train parameter"
    assert type(y_train) == np.ndarray, "Invalid y_train parameter"
    assert x_train.size != 0, "Empty x_train numpy array"
    assert y_train.size != 0, "Empty y_train numpy array"

    epochs = 200
    batch_size = 32

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "best_model.h5", save_best_only=True, monitor="val_loss", verbose=1
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

    # Train CNN model.
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=1,
    )

    return history
