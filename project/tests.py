import unittest
from unittest.mock import Mock, patch
import numpy as np
from io import StringIO
import pandas as pd

from data_preprocessing import preprocessing
from implement_model import cnn_model
from visualization import cnn_visualization
from feature_extraction import feature_extraction

class all_test(unittest.TestCase):
    global data, labels
    global data2, labels2
    data, labels = preprocessing.read_mat('training-a', 'REFERENCE_withSQI.csv', 'pcg')
    data2, labels2 = preprocessing.read_mat('training-a', 'REFERENCE_withSQI.csv', 'ecg')

    # FOLDER: DATA_PREPROCESSING
    def test_data_preprocessing_read_mat_non_folder(self):
        self.assertRaises(AssertionError, preprocessing.read_mat, 'testing.py', 'REFERENCE_withSQI.csv', 'pcg')

    def test_data_preprocessing_read_mat_non_csv(self):
        self.assertRaises(AssertionError, preprocessing.read_mat, 'training-a', 'REFERENCE.txt', 'pcg')

    def test_data_preprocessing_read_mat_non_csv(self):
        self.assertRaises(AssertionError, preprocessing.read_mat, 'training-a', 'REFERENCE_withSQI.csv', 'bitcoin')

    @patch('sys.stdout', new_callable=StringIO)
    def test_data_preprocessing_read_mat_output(self, mock_stdout):
        output = "Total number of subjects: "+str(405)+"\n"
        output += "Total number of used subjects: "+str(404)+"\n"
        output += "Total number of discarded subjects: "+str(1)+"\n"
        output += "Total number of normal signals: "+str(116)+"\n"
        output += "Total number of abnormal signals: "+str(288)+"\n"
        output += "Shortest length: "+str(25955)+"\n"
        output += "Rounded down shortest length: "+str(25000)+"\n"
        output += ""+"\n"
        preprocessing.read_mat('training-a', 'REFERENCE_withSQI.csv', 'pcg')

        self.assertEqual(mock_stdout.getvalue(), output)

    def test_data_preprocessing_preprocess_signals_empty_signals(self):
        empty_array = []
        self.assertRaises(AssertionError, preprocessing.preprocess_signals, empty_array, 10, 0)

    def test_data_preprocessing_preprocess_signals_invalid_signals(self):
        test_array = np.empty(0)
        self.assertRaises(AssertionError, preprocessing.preprocess_signals, test_array, 10, 0)

    def test_data_preprocessing_preprocess_signals_invalid_type(self):
        signal_type = 2
        self.assertRaises(AssertionError, preprocessing.preprocess_signals, data, 10, signal_type)

    def test_data_preprocessing_preprocess_labels_empty_list_files(self):
        empty_list = []
        self.assertRaises(AssertionError, preprocessing.preprocess_labels, empty_list, 'REFERENCES_withSQI.csv')

    def test_data_preprocessing_preprocess_labels_empty_demo(self):
        empty_df = pd.DataFrame.empty
        self.assertRaises(AssertionError, preprocessing.preprocess_labels, data, empty_df)

    def test_data_preprocessing_preprocess_labels_invalid_demo(self):
        empty_list = []
        self.assertRaises(AssertionError, preprocessing.preprocess_labels, data, empty_list)

    def test_data_preprocessing_remove_signals_empty_list_files(self):
        empty_list = []
        self.assertRaises(AssertionError, preprocessing.preprocess_labels, empty_list, 1)

    def test_data_preprocessing_remove_signals_invalid_signal_type(self):
        self.assertRaises(AssertionError, preprocessing.preprocess_labels, data, 2)

    def test_data_preprocessing_find_shortest_length_empty_signals(self):
        empty_list = []
        self.assertRaises(AssertionError, preprocessing.find_shortest_length, empty_list)

    def test_data_preprocessing_set_signal_type_index_invalid_type(self):
        signal_type = "dogecoin"
        self.assertRaises(AssertionError, preprocessing.set_signal_type_index, signal_type)

    def test_data_preprocessing_split_train_test_empty_data(self):
        empty_data = np.empty(0)
        self.assertRaises(AssertionError, preprocessing.split_train_test, empty_data, labels)

    def test_data_preprocessing_split_train_test_empty_labels(self):
        empty_data = np.empty(0)
        self.assertRaises(AssertionError, preprocessing.split_train_test, data, empty_data)

    def test_data_preprocessing_split_train_test_invalid_data_type(self):
        test_list = [1, 2, 3]
        self.assertRaises(AssertionError, preprocessing.split_train_test, test_list, labels)

    def test_data_preprocessing_split_train_test_invalid_labels_type(self):
        test_list = [1, 2, 3]
        self.assertRaises(AssertionError, preprocessing.split_train_test, data, test_list)

    def test_data_preprocessing_validate_data_mat_data_empty(self):
        test_list = []
        self.assertRaises(AssertionError, preprocessing.validate_data, test_list, labels)

    def test_data_preprocessing_validate_data_data_labels_empty(self):
        test_list = []
        self.assertRaises(AssertionError, preprocessing.validate_data, data, test_list)

    def test_data_preprocessing_validate_data_normal_requirement(self):
        test_labels = np.array([-1, -1])
        self.assertRaises(AssertionError, preprocessing.validate_data, data, test_labels)

    def test_data_preprocessing_validate_data_abnormal_requirement(self):
        test_labels = np.array([1, 1])
        self.assertRaises(AssertionError, preprocessing.validate_data, data, test_labels)

    # FOLDER: FEATURE EXTRACTION
    # This tests if an AssertionError is raised for an empty numpy array
    def test_feature_extraction_mfcc_empty_array(self):
        empty_numpy = np.empty(0)
        sample_rate = 1000
        n_mfcc = 40
        hop_length = 256
        self.assertRaises(AssertionError, feature_extraction.mfcc, empty_numpy, sample_rate, n_mfcc, hop_length)

    # This tests if an AssertionError is raised for invalid array type
    def test_feature_extraction_mfcc_invalid_array_type(self):
        test_data = [1, 2, 3]
        sample_rate = 1000
        n_mfcc = 40
        hop_length = 256
        self.assertRaises(AssertionError, feature_extraction.mfcc, test_data, sample_rate, n_mfcc, hop_length)

    # This tests if an AssertionError is raised for an empty numpy array
    def test_feature_extraction_wt_empty_array(self):
        empty_numpy = np.empty(0)
        widths = 10
        self.assertRaises(AssertionError, feature_extraction.wavelet_transform, empty_numpy, widths)

    # This tests if an AssertionError is raised for invalid array type
    def test_feature_extraction_wt_invalid_array_type(self):
        test_data = [1, 2, 3]
        widths = 10
        self.assertRaises(AssertionError, feature_extraction.wavelet_transform, test_data, widths)

    # FOLDER: IMPLEMENT_MODEL
    def test_implement_model_make_cnn_model_invalid_input_shape(self):
        shape = 40
        self.assertRaises(AssertionError, cnn_model.make_cnn_model, shape, 3, 2)

    def test_implement_model_make_cnn_model_empty_num_classes(self):
        num_class = 0
        self.assertRaises(AssertionError, cnn_model.make_cnn_model, (40, 98), 3, num_class)

    def test_implement_model_train_cnn_model_invalid_type_x_train(self):
        test_array = []
        dummy_model = None
        y_train = np.array((1, -1))
        self.assertRaises(AssertionError, cnn_model.train_cnn_model, dummy_model, test_array, y_train)

    def test_implement_model_train_cnn_model_invalid_type_y_train(self):
        test_array = []
        dummy_model = None
        x_train = np.array((1, -1))
        self.assertRaises(AssertionError, cnn_model.train_cnn_model, dummy_model, x_train, test_array)

    def test_implement_model_train_cnn_model_empty_x_train(self):
        test_array = np.empty(0)
        dummy_model = None
        y_train = np.array((1, -1))
        self.assertRaises(AssertionError, cnn_model.train_cnn_model, dummy_model, test_array, y_train)

    def test_implement_model_train_cnn_model_empty_y_train(self):
        test_array = np.empty(0)
        dummy_model = None
        x_train = np.array((1, -1))
        self.assertRaises(AssertionError, cnn_model.train_cnn_model, dummy_model, x_train, test_array)

    # FOLDER: VISUALIZATION
    # This tests if plt.show is called
    @patch("matplotlib.pyplot.show")
    def test_visualize_mfcc_plt(self, mock_show):
        cnn_visualization.visualize_mfcc(data, labels)
        self.assertEqual(mock_show.called, True)

    def test_visualize_mfcc_invalid_data_type_array(self):
        test_data = [1, 2, 3]
        self.assertRaises(AssertionError, cnn_visualization.visualize_mfcc, test_data, labels)

    def test_visualize_mfcc_invalid_label_type_array(self):
        test_data = [1, 2, 3]
        self.assertRaises(AssertionError, cnn_visualization.visualize_mfcc, data, test_data)

    def test_visualize_mfcc_empty_data(self):
        empty_data = np.empty(0)
        self.assertRaises(AssertionError, cnn_visualization.visualize_mfcc, empty_data, labels)

    def test_visualize_mfcc_empty_label(self):
        empty_data = np.empty(0)
        self.assertRaises(AssertionError, cnn_visualization.visualize_mfcc, data, empty_data)

    # This tests if plt.show is called
    @patch("matplotlib.pyplot.show")
    def test_visualize_wavelet_plt(self, mock_show):
        cnn_visualization.visualize_wavelet(data2, labels2)
        self.assertEqual(mock_show.called, True)

    def test_visualize_wavelet_invalid_data_type_array(self):
        test_data = [1, 2, 3]
        self.assertRaises(AssertionError, cnn_visualization.visualize_wavelet, test_data, labels2)

    def test_visualize_wavelet_invalid_label_type_array(self):
        test_data = [1, 2, 3]
        self.assertRaises(AssertionError, cnn_visualization.visualize_wavelet, data2, test_data)

    def test_visualize_wavelet_empty_data(self):
        empty_data = np.empty(0)
        self.assertRaises(AssertionError, cnn_visualization.visualize_wavelet, empty_data, labels2)

    def test_visualize_wavelet_empty_label(self):
        empty_data = np.empty(0)
        self.assertRaises(AssertionError, cnn_visualization.visualize_wavelet, data2, empty_data)

    def test_visualize_srs_invalid_data_type_array(self):
        test_data = [1, 2, 3]
        self.assertRaises(AssertionError, cnn_visualization.select_random_subjects, test_data, labels2)

    def test_visualize_srs_invalid_label_type_array(self):
        test_data = [1, 2, 3]
        self.assertRaises(AssertionError, cnn_visualization.select_random_subjects, data2, test_data)

    # This tests if an AssertionError is raised for an empty data numpy array
    def test_visualizaton_srs_empty_data(self):
        empty_numpy = np.empty(0)
        self.assertRaises(AssertionError, cnn_visualization.select_random_subjects, empty_numpy, labels)

    # This tests if an AssertionError is raised for an empty labels numpy array
    def test_visualizaton_srs_empty_labels(self):
        empty_numpy = np.empty(0)
        self.assertRaises(AssertionError, cnn_visualization.select_random_subjects, data, empty_numpy)

    # This tests if plt.show is called
    @patch("sys.stdout")
    @patch("matplotlib.pyplot.show")
    def test_visualization_cnn_accuracy_loss_plt(self, mock_show, mock_stdout):
        x_train, x_test, y_train, y_test = preprocessing.split_train_test(data, labels)
        num_classes = len(np.unique(y_train))
        y_train[y_train == -1] = 0
        y_test[y_test == -1] = 0
        num_conv_layers = 3
        model = cnn_model.make_cnn_model(x_train.shape[1:], num_conv_layers, num_classes)
        history = cnn_model.train_cnn_model(model, x_train, y_train)
        cnn_visualization.visualize_cnn_accuracy_loss(history)
        self.assertEqual(mock_show.called, True)


def main():
    # Create the test suite from the cases above.
    suite = unittest.TestLoader().loadTestsFromTestCase(all_test)
    # This will run the test suite.
    unittest.TextTestRunner(verbosity=2).run(suite)
