import unittest

from data_preprocessing.lstm_pre import load_files, truncate_data, standardize
from feature_extraction.lstm_feat import extract_mfcc, extract_wavelet
from implement_model.lstm_model import build_lstm
from configs.lstm_config import lstm_config

from keras.models import Sequential
import numpy as np

class TestLstm(unittest.TestCase):
    global pcg_data, ecg_data, pcg_labels, ecg_labels, shortest_signal
    pcg_data, pcg_labels = load_files('training-a','REFERENCE_withSQI.csv','pcg')
    ecg_data, ecg_labels = load_files('training-a','REFERENCE_withSQI.csv','ecg')
    shortest_signal = truncate_data(pcg_data, pcg_labels, lstm_config)[0].shape[1]
    TOTAL_SUBJECTS = 405

    def test_load_files_pcg_ecg_labels(self):
        # Assert labels are the same when loading ecg or pcg data
        self.assertEqual(len(pcg_labels), len(ecg_labels))
        for i in range(len(pcg_labels)):
            self.assertEqual(pcg_labels[i], ecg_labels[i])
    
    def test_load_files_load_all_subjects(self):
        # Assert all subjects from MATLAB files and their corresponding labels are being loaded
        self.assertEqual(pcg_data.shape[0], self.TOTAL_SUBJECTS)
        self.assertEqual(ecg_data.shape[0], self.TOTAL_SUBJECTS)
        self.assertEqual(pcg_labels.shape[0], self.TOTAL_SUBJECTS)
        self.assertEqual(ecg_labels.shape[0], self.TOTAL_SUBJECTS)

    def test_load_files_ecg_pcg_notequal(self):
        # Test when loading 'pcg' or 'ecg' data, returned data is actually different
        self.assertNotEqual(pcg_data, ecg_data, 'ECG and PCG data are the same')
    
    def test_truncate_data_config_keys_present(self):
        # Assert necessary keys exist in config
        self.assertIn('min_signal_length', lstm_config)
        self.assertIn('round_down', lstm_config)
    
    def test_truncate_data_discard(self):
        mock_signal = [
            np.ones(2000),
            np.ones(500),
            np.ones(1111),
            np.ones(3500)
        ]
        mock_labels = [
            1,
            0,
            1,
            0
        ]
        mock_config = {
            'min_signal_length': 1000,
            'round_down': False
        }
        data, labels = truncate_data(mock_signal, mock_labels, mock_config)
        # Assert signals below min_signal_length were discarded
        for signal in data:
            self.assertTrue(len(signal) >= mock_config['min_signal_length'])
        # Assert equal number of signals and labels after truncating
        self.assertEqual(data.shape[0], labels.shape[0])
    
    def test_truncate_data_min_signal_length(self):
        data, _ = truncate_data(pcg_data, pcg_labels, lstm_config)
        for signal in data:
            self.assertTrue(len(signal) >= lstm_config['min_signal_length'])
    
    def test_truncate_data_output_shape(self):
        data, _ = truncate_data(pcg_data, pcg_labels, lstm_config)
        # Assert signals were truncated, resulting in 2D numpy array
        self.assertTrue(len(data.shape) == 2)
        self.assertEqual(data.shape[1], len(data[0]))

    def test_standardize_output_shape(self):
        # Uses external function to standardize, assume values are correct
        # Assert that shape of input data equals shape of output data
        mock_data = np.random.randn(3,10000) # Mock 2D numpy array of floats with shape (3,10000)
        output = standardize(mock_data)
        self.assertEqual(mock_data.shape, output.shape)

    def test_extract_mfcc_config_keys_present(self):
        # Assert necessary keys exist in config
        self.assertIn('sample_rate', lstm_config)
        self.assertIn('n_mfcc', lstm_config)
        self.assertIn('hop_length', lstm_config)
    
    def test_extract_mfcc_output_shape(self):
        data, _ = truncate_data(pcg_data, pcg_labels, lstm_config)
        mfcc = extract_mfcc(data, lstm_config)
        # Assert that shape of output is as expected--(subjects, n_mfcc*3, signal_length//hop_length + 1)
        self.assertEqual(mfcc.shape, (data.shape[0], lstm_config['n_mfcc']*3, (shortest_signal//lstm_config['hop_length'])+1))
    
    def test_extract_wavelet_config_keys_present(self):
        # Assert necessary key exists in config
        self.assertIn('widths', lstm_config)
    
    def test_extract_wavelet_output_shape(self):
        data, _ = truncate_data(ecg_data, ecg_labels, lstm_config)
        wavelets = extract_wavelet(data, lstm_config)
        # Assert that shape of output is as expected--(subjects,config['widths'],signal_length)
        self.assertEqual(wavelets.shape, (data.shape[0], lstm_config['widths'], data.shape[-1]))
    
    def test_build_lstm_config_keys_present(self):
        # Assert necessary keys exist in config
        self.assertIn('optimizer', lstm_config)
        self.assertIn('learning_rate', lstm_config)
        self.assertIn('loss', lstm_config)
        self.assertIn('metrics', lstm_config)
        self.assertIn('min_signal_length', lstm_config)
    
    def test_build_lstm_return_type(self):
        mock_input_shape = (400, 20, 10000)
        model = build_lstm(mock_input_shape, lstm_config)
        # Assert a Sequential model is returned
        self.assertIsInstance(model, type(Sequential()))

    def test_train_lstm_config_keys_present(self):
        # Assert necessary keys exist in config
        # model.fit callback function array config keys
        self.assertIn('best_model_name', lstm_config)
        self.assertIn('mc_monitor', lstm_config)
        self.assertIn('save_weights_only', lstm_config)
        self.assertIn('save_best_only', lstm_config)
        self.assertIn('mc_verbose', lstm_config)
        self.assertIn('es_monitor', lstm_config)
        self.assertIn('es_mode', lstm_config)
        # model.fit parameter config keys
        self.assertIn('epochs', lstm_config)
        self.assertIn('batch_size', lstm_config)
        self.assertIn('shuffle', lstm_config)



if __name__ == '__main__':
    unittest.main()