from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint
from performance_metrics.metrics import recall, precision, sensitivity, specificity, f1

# Data parameters
lstm_config = {
    # Dataset/preprocessing related
    'sample_rate': 1000,
    'min_signal_length': 20000,
    'round_down': True,
    # Feature extraction related
    'n_mfcc': 12,
    'hop_length': 128,
    'widths': 10,
    # Model related
    'optimizer': Adam,
    'learning_rate': 0.00001,    # Gets overwritten during performance test
    # 'learning_rate': 0.000005,
    'saved_model_name': 'lstm_best',    # Gets overwritten during performance test
    'saved_model_format': '.h5',
    'mc_monitor': 'val_accuracy',
    'save_weights_only': True,
    'save_best_only': True,
    'mc_verbose': 1,
    'es_monitor': 'val_loss',
    'es_mode': 'auto',
    'loss': 'binary_crossentropy',
    'metrics': ['accuracy', specificity, sensitivity, f1, recall, precision],
    'epochs': 200,
    'batch_size': 16,   # Gets overwritten during performance test
    'shuffle': True,
    'saved_plot_name': 'lstm_best', # Gets overwritten during performance test
    'saved_plot_format': '.png'
}