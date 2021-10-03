# import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from sklearn.utils import class_weight
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

def build_lstm(data_shape, config={}):
    if 'optimizer' not in config:
        raise Exception('key: optimizer, not found in config')
    if 'learning_rate' not in config:
        raise Exception('key: learning_rate, not found in config')
    if 'loss' not in config:
        raise Exception('key: loss, not found in config')
    if 'metrics' not in config:
        raise Exception('key: metrics, not found in config')
        
    if len(data_shape) != 3:
        raise Exception('expected 3D model input data shape, got', data_shape)

    # Build single layer LSTM network
    model = Sequential()
    # model for PCG
    model.add(LSTM(128, input_shape=(data_shape[1], data_shape[2]), return_sequences=True)) # if want add more layer, put in first layer return_sequences=True
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(256))
    model.add(Dropout(0.6))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid')) # softmax for classification, for this need encode labels in binary (one hot encoding)
    optimizer_fn = config['optimizer']
    optimizer = optimizer_fn(config['learning_rate'])
    model.compile(loss=config['loss'],
                    optimizer=optimizer,
                    metrics=config['metrics'])
    return model

def train_lstm(model, training_set, validation_set, config={}):
    if not config:
        raise Exception('empty config in train_lstm(...)')

    x_train, y_train = training_set
    x_val, y_val = validation_set

    chk = [
        ModelCheckpoint(
            config['saved_model_name'],     # *.tf generates multiple files, *.h5 generates one
            monitor='val_accuracy',
            save_weights_only=True,
            save_best_only=True,
            verbose=0
            ),
        EarlyStopping(
            monitor='val_loss',
            mode='auto',
            patience=10
            )
        ]

    class_weights = class_weight.compute_class_weight('balanced',classes=np.unique(y_train),y=y_train)
    class_weights_dict = dict(enumerate(class_weights))
    
    history = model.fit(x_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'],
                    validation_data=(x_val,y_val),
                    shuffle=config['shuffle'],
                    callbacks=chk,
                    class_weight=class_weights_dict,
                    verbose=0)
    return history