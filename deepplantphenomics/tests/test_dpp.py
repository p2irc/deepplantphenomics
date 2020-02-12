import shutil

import pytest
# import unittest.mock as mock
import numpy as np
import os.path
import random
import tensorflow.compat.v1 as tf
import deepplantphenomics as dpp
from deepplantphenomics import loaders, layers
from deepplantphenomics.tests.mock_dpp_model import MockDPPModel


# public setters and adders, mostly testing for type and value errors
@pytest.fixture(scope="module")
def test_data_dir():
    return os.path.join(os.path.dirname(__file__), 'test_data')


@pytest.fixture()
def model():
    model = MockDPPModel()
    model.set_image_dimensions(1, 1, 1)
    return model


def test_set_number_of_threads(model):
    with pytest.raises(TypeError):
        model.set_number_of_threads(5.0)
    with pytest.raises(ValueError):
        model.set_number_of_threads(-1)


def test_set_number_of_gpus(model):
    assert model._num_gpus == 1
    assert model._batch_size == 1
    assert model._subbatch_size == 1
    model._max_gpus = 2

    # Test sanity checks for type and value range
    with pytest.raises(ValueError):
        model.set_number_of_gpus(0)
    with pytest.raises(TypeError):
        model.set_number_of_gpus('2')

    # Test subbatch sets and errors when enough GPUs are available
    with pytest.raises(RuntimeError):
        model.set_number_of_gpus(2)  # Can't split 1 item across 2 GPUs
    model._batch_size = 2
    model.set_number_of_gpus(2)
    assert model._num_gpus == 2
    assert model._subbatch_size == 1

    # Test subbatch sets and errors when enough GPUs aren't available
    model._max_gpus = 1
    model.set_number_of_gpus(2)
    assert model._num_gpus == 1
    assert model._subbatch_size == 2

    model._max_gpus = 0
    model.set_number_of_gpus(2)
    assert model._num_gpus == 1
    assert model._subbatch_size == 2


def test_set_batch_size(model):
    assert model._batch_size == 1
    assert model._subbatch_size == 1

    # Test sanity checks for type and value range
    with pytest.raises(TypeError):
        model.set_batch_size(5.0)
    with pytest.raises(ValueError):
        model.set_batch_size(-1)

    # Test normal batch size sets (i.e. with 1 GPU)
    model.set_batch_size(2)
    assert model._batch_size == 2
    assert model._subbatch_size == 2

    # Test batch size sets with multiple GPUs and errors
    model._num_gpus = 2
    with pytest.raises(RuntimeError):
        model.set_batch_size(3)  # Can't split 3 items across 2 GPUs
    model.set_batch_size(4)
    assert model._batch_size == 4
    assert model._subbatch_size == 2


def test_set_test_split(model):
    assert model._test_split == 0.10
    assert model._validation_split == 0.10

    with pytest.raises(TypeError):
        model.set_test_split('0.2')
    with pytest.raises(ValueError):
        model.set_test_split(-0.1)
    with pytest.raises(ValueError):
        model.set_test_split(1.1)

    # No testing
    model.set_test_split(0)
    assert not model._testing
    assert model._test_split == 0
    assert model._validation_split == 0.10

    # Regular testing and validation splits
    model.set_test_split(0.25)
    assert model._testing
    assert model._test_split == 0.25
    assert model._validation_split == 0.10

    # Exactly half of the data is for training; warning shouldn't trigger
    model.set_test_split(0.40)
    assert model._testing
    assert model._test_split == 0.40
    assert model._validation_split == 0.10

    # Less than half of the data is for testing; warning should trigger
    with pytest.warns(Warning):
        model.set_test_split(0.50)
        assert model._testing
        assert model._test_split == 0.50
        assert model._validation_split == 0.10


def test_set_validation_split(model):
    assert model._test_split == 0.10
    assert model._validation_split == 0.10

    # No testing
    model.set_validation_split(0)
    assert not model._validation
    assert model._validation_split == 0
    assert model._test_split == 0.10

    # Regular testing and validation splits
    model.set_validation_split(0.25)
    assert model._validation
    assert model._validation_split == 0.25
    assert model._test_split == 0.10

    # Exactly half of the data is for training; warning shouldn't trigger
    model.set_validation_split(0.40)
    assert model._validation
    assert model._validation_split == 0.40
    assert model._test_split == 0.10

    # Less than half of the data is for testing; warning should trigger
    with pytest.warns(Warning):
        model.set_validation_split(0.50)
        assert model._validation
        assert model._validation_split == 0.50
        assert model._test_split == 0.10


def test_force_split_shuffle(model):
    assert not model._force_split_partition

    with pytest.raises(TypeError):
        model.force_split_shuffle('True')

    model.force_split_shuffle(True)
    assert model._force_split_partition


def test_set_random_seed(model):
    with pytest.raises(TypeError):
        model.set_random_seed('7')

    # The only real way to check that we're setting the seed properly is to do some random stuff twice and check for
    # the same values
    def get_random_sequences():
        with model._graph.as_default():
            model.set_random_seed(7)
            py_seq = [random.random() for _ in range(10)]
            np_seq = np.random.random_sample((10,))
            tf_seq = model._session.run(tf.random.uniform([10]))
        return py_seq, np_seq, tf_seq

    # We need reproducibility across script runs; making a new graph with new ops is what essentially happens
    py_seq_1, np_seq_1, tf_seq_1 = get_random_sequences()
    model._reset_graph()
    model._reset_session()
    py_seq_2, np_seq_2, tf_seq_2 = get_random_sequences()
    assert np.all(py_seq_1 == py_seq_2)
    assert np.all(np_seq_1 == np_seq_2)
    assert np.all(tf_seq_1 == tf_seq_2)


def test_set_num_regression_outputs():
    model = dpp.RegressionModel()

    with pytest.raises(TypeError):
        model.set_num_regression_outputs(5.0)
    with pytest.raises(ValueError):
        model.set_num_regression_outputs(-1)


def test_set_density_map_sigma():
    model = dpp.HeatmapObjectCountingModel()
    assert model._density_sigma == 5

    with pytest.raises(TypeError):
        model.set_density_map_sigma('4')
    model.set_density_map_sigma(2.0)
    assert model._density_sigma == 2.0


def test_set_maximum_training_epochs(model):
    with pytest.raises(TypeError):
        model.set_maximum_training_epochs(5.0)
    with pytest.raises(ValueError):
        model.set_maximum_training_epochs(-1)


def test_set_learning_rate(model):
    # Give the type checking a workout
    with pytest.raises(TypeError):
        model.set_learning_rate("5")
    with pytest.raises(ValueError):
        model.set_learning_rate(-0.001)

    # Ensure a good value sets the rate properly
    model.set_learning_rate(0.01)
    assert model._learning_rate == 0.01

    # Ensure that the internal learning rate setter doesn't touch it (due to no decay settings)
    model._set_learning_rate()
    assert model._learning_rate == 0.01


def test_set_crop_or_pad_images(model):
    with pytest.raises(TypeError):
        model.set_crop_or_pad_images("True")


def test_set_resize_images(model):
    with pytest.raises(TypeError):
        model.set_resize_images("True")


def test_set_augmentation_flip_horizontal():
    model1 = dpp.RegressionModel()
    model2 = dpp.SemanticSegmentationModel()

    with pytest.raises(TypeError):
        model1.set_augmentation_flip_horizontal("True")
    with pytest.raises(RuntimeError):
        model2.set_augmentation_flip_horizontal(True)
    model1.set_augmentation_flip_horizontal(True)


def test_set_augmentation_flip_vertical():
    model1 = dpp.RegressionModel()
    model2 = dpp.SemanticSegmentationModel()

    with pytest.raises(TypeError):
        model1.set_augmentation_flip_vertical("True")
    with pytest.raises(RuntimeError):
        model2.set_augmentation_flip_vertical(True)
    model1.set_augmentation_flip_vertical(True)


def test_set_augmentation_crop():
    model1 = dpp.RegressionModel()
    model2 = dpp.SemanticSegmentationModel()

    with pytest.raises(TypeError):
        model1.set_augmentation_crop("True", 0.5)
    with pytest.raises(TypeError):
        model1.set_augmentation_crop(True, "5")
    with pytest.raises(ValueError):
        model1.set_augmentation_crop(False, -1.0)
    with pytest.raises(RuntimeError):
        model2.set_augmentation_crop(True)
    model1.set_augmentation_crop(True)


def test_set_augmentation_brightness_and_contrast():
    model1 = dpp.RegressionModel()
    model2 = MockDPPModel()
    model2._supported_augmentations = []

    with pytest.raises(TypeError):
        model1.set_augmentation_crop("True")
    with pytest.raises(RuntimeError):
        model2.set_augmentation_brightness_and_contrast(True)
    model1.set_augmentation_brightness_and_contrast(True)


def test_set_augmentation_rotation():
    model1 = dpp.RegressionModel()
    model2 = dpp.SemanticSegmentationModel()

    # Check the type-checking
    with pytest.raises(TypeError):
        model1.set_augmentation_rotation("True")
    with pytest.raises(TypeError):
        model1.set_augmentation_rotation(True, crop_borders="False")
    with pytest.raises(RuntimeError):
        model2.set_augmentation_rotation(True)

    # Check that rotation augmentation can be turned on the simple way
    model1.set_augmentation_rotation(True)
    assert model1._augmentation_rotate is True
    assert model1._rotate_crop_borders is False

    # Check that it can be turned on with a border cropping setting
    model1.set_augmentation_rotation(False, crop_borders=True)
    assert model1._augmentation_rotate is False
    assert model1._rotate_crop_borders is True


def test_set_regularization_coefficient(model):
    with pytest.raises(TypeError):
        model.set_regularization_coefficient("5")
    with pytest.raises(ValueError):
        model.set_regularization_coefficient(-0.001)


def test_set_learning_rate_decay(model):
    # Give the type checking a workout
    with pytest.raises(TypeError):
        model.set_learning_rate_decay("5", 1)
    with pytest.raises(ValueError):
        model.set_learning_rate_decay(-0.001, 1)
    with pytest.raises(TypeError):
        model.set_learning_rate_decay(0.5, 5.0)
    with pytest.raises(ValueError):
        model.set_learning_rate_decay(0.5, -1)

    # Ensure that a good set of inputs sets model parameters properly
    model.set_learning_rate_decay(0.01, 100)
    assert model._lr_decay_factor == 0.01
    assert model._epochs_per_decay == 100
    assert model._lr_decay_epochs is None

    # Ensure that the internal learning rate setter handles decay properly
    model._total_training_samples = 100
    model._test_split = 0.20
    model._learning_rate = 0.1
    model._global_epoch = 0
    model._set_learning_rate()
    assert model._lr_decay_epochs == 8000
    assert isinstance(model._learning_rate, tf.Tensor)
    with tf.Session() as sess:
        assert sess.run(model._learning_rate) == pytest.approx(0.1)


def test_set_optimizer(model):
    with pytest.raises(TypeError):
        model.set_optimizer(5)
    with pytest.raises(ValueError):
        model.set_optimizer('Nico')
    model.set_optimizer('adam')
    assert model._optimizer == 'adam'
    model.set_optimizer('Adam')
    assert model._optimizer == 'adam'
    model.set_optimizer('ADAM')
    assert model._optimizer == 'adam'
    model.set_optimizer('SGD')
    assert model._optimizer == 'sgd'
    model.set_optimizer('sgd')
    assert model._optimizer == 'sgd'
    model.set_optimizer('sGd')
    assert model._optimizer == 'sgd'


def test_set_weight_initializer(model):
    with pytest.raises(TypeError):
        model.set_weight_initializer(5)
    with pytest.raises(ValueError):
        model.set_weight_initializer('Nico')
    model.set_weight_initializer('normal')
    assert model._weight_initializer == 'normal'
    model.set_weight_initializer('Normal')
    assert model._weight_initializer == 'normal'
    model.set_weight_initializer('NORMAL')
    assert model._weight_initializer == 'normal'


def test_set_image_dimensions(model):
    with pytest.raises(TypeError):
        model.set_image_dimensions(1.0, 1, 1)
    with pytest.raises(ValueError):
        model.set_image_dimensions(-1, 1, 1)
    with pytest.raises(TypeError):
        model.set_image_dimensions(1, 1.0, 1)
    with pytest.raises(ValueError):
        model.set_image_dimensions(1, -1, 1)
    with pytest.raises(TypeError):
        model.set_image_dimensions(1, 1, 1.0)
    with pytest.raises(ValueError):
        model.set_image_dimensions(1, 1, -1)


def test_set_original_image_dimensions(model):
    with pytest.raises(TypeError):
        model.set_original_image_dimensions(1.0, 1)
    with pytest.raises(ValueError):
        model.set_original_image_dimensions(-1, 1)
    with pytest.raises(TypeError):
        model.set_original_image_dimensions(1, 1.0)
    with pytest.raises(ValueError):
        model.set_original_image_dimensions(1, -1)


def test_set_patch_size(model):
    with pytest.raises(TypeError):
        model.set_patch_size(1.0, 1)
    with pytest.raises(ValueError):
        model.set_patch_size(-1, 1)
    with pytest.raises(TypeError):
        model.set_patch_size(1, 1.0)
    with pytest.raises(ValueError):
        model.set_patch_size(1, -1)


@pytest.mark.parametrize("model,bad_loss,good_loss",
                         [(dpp.ClassificationModel(), 'l2', 'softmax cross entropy'),
                          (dpp.RegressionModel(), 'softmax cross entropy', 'l2'),
                          (dpp.SemanticSegmentationModel(), 'l2', 'sigmoid cross entropy'),
                          (dpp.ObjectDetectionModel(), 'l2', 'yolo'),
                          (dpp.CountCeptionModel(), 'l2', 'l1'),
                          (dpp.HeatmapObjectCountingModel(), 'sigmoid cross entropy', 'l2')])
def test_set_loss_function(model, bad_loss, good_loss):
    with pytest.raises(TypeError):
        model.set_loss_function(0)
    with pytest.raises(ValueError):
        model.set_loss_function(bad_loss)
    model.set_loss_function(good_loss)


def test_set_num_segmentation_classes():
    model = dpp.SemanticSegmentationModel()
    assert model._num_seg_class == 2
    assert model._loss_fn == 'sigmoid cross entropy'

    with pytest.raises(TypeError):
        model.set_num_segmentation_classes('2')
    with pytest.raises(ValueError):
        model.set_num_segmentation_classes(1)

    model.set_num_segmentation_classes(5)
    assert model._num_seg_class == 5
    assert model._loss_fn == 'softmax cross entropy'

    model.set_num_segmentation_classes(2)
    assert model._num_seg_class == 2
    assert model._loss_fn == 'sigmoid cross entropy'


def test_set_yolo_parameters():
    model = dpp.ObjectDetectionModel()
    with pytest.raises(RuntimeError):
        model.set_yolo_parameters()
    model.set_image_dimensions(448, 448, 3)
    model.set_yolo_parameters()

    with pytest.raises(TypeError):
        model.set_yolo_parameters(True, ['plant', 'knat'], [(100, 30), (200, 10), (50, 145)])
    with pytest.raises(TypeError):
        model.set_yolo_parameters(13, ['plant', 'knat'], [(100, 30), (200, 10), (50, 145)])
    with pytest.raises(TypeError):
        model.set_yolo_parameters([13], ['plant', 'knat'], [(100, 30), (200, 10), (50, 145)])
    with pytest.raises(TypeError):
        model.set_yolo_parameters([13, 13], 'plant', [(100, 30), (200, 10), (50, 145)])
    with pytest.raises(TypeError):
        model.set_yolo_parameters([13, 13], ['plant', 2], [(100, 30), (200, 10), (50, 145)])
    with pytest.raises(TypeError):
        model.set_yolo_parameters([13, 13], ['plant', 'knat'], 100)
    with pytest.raises(TypeError):
        model.set_yolo_parameters([13, 13], ['plant', 'knat'], [(100, 30), (200, 10), 50])
    with pytest.raises(TypeError):
        model.set_yolo_parameters([13, 13], ['plant', 'knat'], [(100, 30), (200, 10), (145,)])
    with pytest.raises(TypeError):
        model.set_yolo_parameters([13, 13], ['plant', 'knat'], [(100, 30), (200, 10), (145, 'a')])
    model.set_yolo_parameters([13, 13], ['plant', 'knat'], [(100, 30), (200, 10), (50, 145)])


# adding layers may require some more indepth testing
def test_add_input_layer(model):
    model.set_batch_size(1)
    model.set_image_dimensions(1, 1, 1)
    model.add_input_layer()
    assert isinstance(model._last_layer(), dpp.layers.inputLayer)
    with pytest.raises(RuntimeError):
        model.add_input_layer()


# need to come back to this one
# need to add exceptions to real function, and set up the layer for the test better
# def test_add_moderation_layer(model):
#     mf = np.array([[0, 1, 2]])
#     model.add_moderation_features(mf)
#     model.add_moderation_layer()
#     assert isintance(model._DPPModel__last_layer(), layers.moderationLayer)
#     model.add_moderation_layer()
#     assert isinstance(model._DPPModel__last_layer(), layers.moderationLayer)


def test_add_convolutional_layer(model):
    with pytest.raises(RuntimeError):
        model.add_convolutional_layer([1, 2.0, 3, 4], 1, 'relu')
    model.add_input_layer()
    with pytest.raises(TypeError):
        model.add_convolutional_layer([1, 2.0, 3, 4], 1, 'relu')
    with pytest.raises(TypeError):
        model.add_convolutional_layer([1, 2], 1, 'relu')
    with pytest.raises(TypeError):
        model.add_convolutional_layer([1, 2, 3, 4], 1.0, 'relu')
    with pytest.raises(ValueError):
        model.add_convolutional_layer([1, 2, 3, 4], -1, 'relu')
    with pytest.raises(TypeError):
        model.add_convolutional_layer([1, 2, 3, 4], 1, 555)
    with pytest.raises(ValueError):
        model.add_convolutional_layer([1, 2, 3, 4], 1, 'Nico')
    model.add_convolutional_layer(np.array([1, 1, 1, 1]), 1, 'relu')
    assert isinstance(model._last_layer(), dpp.layers.convLayer)


def test_add_paral_conv_block(model):
    with pytest.raises(RuntimeError):
        model.add_paral_conv_block([1, 1, 1, 1], [1, 1, 1, 1])
    model.add_input_layer()
    with pytest.raises(TypeError):
        model.add_paral_conv_block([1, 1, 1, 1], [1, 2.0, 1, 1])
    with pytest.raises(TypeError):
        model.add_paral_conv_block([1, 1, 1, 1], [1, 1, 1])
    with pytest.raises(TypeError):
        model.add_paral_conv_block([1, 1, 1, 1], 1)
    model.add_paral_conv_block([1, 1, 1, 1], [1, 1, 1, 1])
    assert isinstance(model._last_layer(), dpp.layers.paralConvBlock)


def test_add_skip_connection(model):
    model.set_image_dimensions(50, 50, 16)
    model.set_batch_size(1)

    with pytest.raises(RuntimeError):
        model.add_skip_connection(downsampled=False)
    model.add_input_layer()

    model.add_skip_connection(downsampled=False)
    assert isinstance(model._last_layer(), dpp.layers.skipConnection)
    assert model._last_layer().output_size == [1, 50, 50, 16]

    model.add_skip_connection(downsampled=True)
    assert isinstance(model._last_layer(), dpp.layers.skipConnection)
    assert model._last_layer().output_size == [1, 25, 25, 16]


def test_add_pooling_layer(model):
    with pytest.raises(RuntimeError):
        model.add_pooling_layer(1, 1, 'avg')
    model.add_input_layer()
    with pytest.raises(TypeError):
        model.add_pooling_layer(1.5, 1)
    with pytest.raises(ValueError):
        model.add_pooling_layer(-1, 1)
    with pytest.raises(TypeError):
        model.add_pooling_layer(1, 1.5)
    with pytest.raises(ValueError):
        model.add_pooling_layer(1, -1)
    with pytest.raises(TypeError):
        model.add_pooling_layer(1, 1, 5)
    with pytest.raises(ValueError):
        model.add_pooling_layer(1, 1, 'Nico')
    model.add_pooling_layer(1, 1, 'avg')
    assert isinstance(model._last_layer(), dpp.layers.poolingLayer)


@pytest.mark.parametrize("kernel_size,stride,output_size", [(2, 2, 3), (3, 3, 2), (2, 1, 5)])
def test_pooling_layer_output_size(model, kernel_size, stride, output_size):
    model.set_image_dimensions(5, 5, 1)
    model.add_input_layer()
    model.add_pooling_layer(kernel_size, stride)
    assert model._last_layer().output_size == [1, output_size, output_size, 1]


def test_add_normalization_layer(model):
    with pytest.raises(RuntimeError):
        model.add_normalization_layer()
    model.add_input_layer()
    model.add_normalization_layer()
    assert isinstance(model._last_layer(), dpp.layers.normLayer)


def test_add_dropout_layer(model):
    with pytest.raises(RuntimeError):
        model.add_dropout_layer(0.4)
    model.add_input_layer()
    with pytest.raises(TypeError):
        model.add_dropout_layer("0.5")
    with pytest.raises(ValueError):
        model.add_dropout_layer(1.5)
    model.add_dropout_layer(0.4)
    assert isinstance(model._last_layer(), dpp.layers.dropoutLayer)


def test_add_batch_norm_layer(model):
    with pytest.raises(RuntimeError):
        model.add_batch_norm_layer()
    model.add_input_layer()
    model.add_batch_norm_layer()
    assert isinstance(model._last_layer(), dpp.layers.batchNormLayer)


def test_add_fully_connected_layer(model):
    with pytest.raises(RuntimeError):
        model.add_fully_connected_layer(1, 'tanh', 0.3)
    model.add_input_layer()
    with pytest.raises(TypeError):
        model.add_fully_connected_layer(2.3, 'relu', 1.8)
    with pytest.raises(ValueError):
        model.add_fully_connected_layer(-3, 'relu', 1.8)
    with pytest.raises(TypeError):
        model.add_fully_connected_layer(2, 5, 1.8)
    with pytest.raises(ValueError):
        model.add_fully_connected_layer(3, 'Nico', 1.8)
    with pytest.raises(TypeError):
        model.add_fully_connected_layer(2, 'relu', "1.8")
    with pytest.raises(ValueError):
        model.add_fully_connected_layer(3, 'relu', -1.5)
    model.add_fully_connected_layer(1, 'tanh', 0.3)
    assert isinstance(model._last_layer(), dpp.layers.fullyConnectedLayer)


def test_add_output_layer():
    model1 = dpp.ClassificationModel()
    model2 = dpp.SemanticSegmentationModel()
    model3 = dpp.CountCeptionModel()
    model1.set_image_dimensions(5, 5, 3)
    model2.set_image_dimensions(5, 5, 3)

    with pytest.raises(RuntimeError):
        model1.add_output_layer(2.5, 3)
    model1.add_input_layer()
    model2.add_input_layer()
    model3.add_input_layer()
    with pytest.raises(TypeError):
        model1.add_output_layer("2")
    with pytest.raises(ValueError):
        model1.add_output_layer(-0.4)
    with pytest.raises(TypeError):
        model1.add_output_layer(2.0, 3.4)
    with pytest.raises(ValueError):
        model1.add_output_layer(2.0, -4)
    with pytest.raises(RuntimeError):
        model2.add_output_layer(output_size=3)  # Semantic segmentation needed for this runtime error to occur

    model1.add_output_layer(2.5, 3)
    assert isinstance(model1._last_layer(), dpp.layers.fullyConnectedLayer)
    with pytest.warns(Warning):
        model2.add_output_layer(regularization_coefficient=2.0)
    assert isinstance(model2._last_layer(), dpp.layers.convLayer)
    model3.add_output_layer()
    assert isinstance(model3._last_layer(), dpp.layers.inputLayer)


# more loading data tests!!!!
def test_load_dataset_from_directory_with_csv_labels(model, test_data_dir):
    im_path = os.path.join(test_data_dir, 'test_dir_csv_labels', '')
    label_path = os.path.join(test_data_dir, 'test_csv_labels.txt')
    with pytest.raises(TypeError):
        model.load_dataset_from_directory_with_csv_labels(5, label_path)
    with pytest.raises(TypeError):
        model.load_dataset_from_directory_with_csv_labels(im_path, 5)
    with pytest.raises(ValueError):
        model.load_dataset_from_directory_with_csv_labels(os.path.join(test_data_dir, 'test_dir_csv_images', ''),
                                                          label_path)
    model.load_dataset_from_directory_with_csv_labels(im_path, label_path)


def test_load_ippn_leaf_count_dataset_from_directory(test_data_dir):
    # The following tests take the format laid out in the documentation of an example
    # for training a leaf counter, and leave out key parts to see if the program
    # throws an appropriate exception, or executes as intended due to using a default setting
    data_path = os.path.join(test_data_dir, 'test_Ara2013_Canon', '')

    # forgetting to set image dimensions
    model = dpp.RegressionModel(debug=False, save_checkpoints=False, report_rate=20)
    # channels = 3
    model.set_batch_size(4)
    # model.set_image_dimensions(128, 128, channels)
    model.set_resize_images(True)
    model.set_num_regression_outputs(1)
    model.set_test_split(0.1)
    model.set_weight_initializer('xavier')
    model.set_maximum_training_epochs(1)
    model.set_learning_rate(0.0001)
    with pytest.raises(RuntimeError):
        model.load_ippn_leaf_count_dataset_from_directory(data_path)

    # forgetting to set num epochs
    model = dpp.RegressionModel(debug=False, save_checkpoints=False, report_rate=20)
    channels = 3
    model.set_batch_size(4)
    model.set_image_dimensions(128, 128, channels)
    model.set_resize_images(True)
    model.set_num_regression_outputs(1)
    model.set_test_split(0.1)
    model.set_weight_initializer('xavier')
    # model.set_maximum_training_epochs(1)
    model.set_learning_rate(0.0001)
    with pytest.raises(RuntimeError):
        model.load_ippn_leaf_count_dataset_from_directory(data_path)

    # the following shouldn't raise any issues since there should be defaults for
    # batch_size, train_test_split, and learning_rate
    model = dpp.RegressionModel(debug=False, save_checkpoints=False, report_rate=20)
    channels = 3
    # model.set_batch_size(4)
    model.set_image_dimensions(128, 128, channels)
    model.set_resize_images(True)
    model.set_num_regression_outputs(1)
    # model.set_test_split(0.2)
    model.set_weight_initializer('xavier')
    model.set_maximum_training_epochs(1)
    # model.set_learning_rate(0.0001)
    model.load_ippn_leaf_count_dataset_from_directory(data_path)


def test_heatmap_csv_data_load(test_data_dir):
    im_dir = os.path.join(test_data_dir, 'test_Ara2013_heatmap')
    expected_heatmap_dir = os.path.join(os.path.curdir, 'generated_heatmaps')
    if os.path.exists(expected_heatmap_dir):
        shutil.rmtree(expected_heatmap_dir)

    model = dpp.HeatmapObjectCountingModel()
    model.set_image_dimensions(128, 128, 3)
    assert model._raw_image_files is None
    assert model._raw_labels is None

    base_names = ['ara2013_plant007_rgb', 'ara2013_plant008_rgb', 'ara2013_plant001_rgb',
                  'ara2013_plant002_rgb', 'ara2013_plant003_rgb', 'ara2013_plant004_rgb',
                  'ara2013_plant005_rgb', 'ara2013_plant006_rgb']
    expected_images = [os.path.join(im_dir, '{}.png'.format(x)) for x in base_names]
    expected_labels = [os.path.join(expected_heatmap_dir, '{}.npy'.format(x)) for x in base_names]

    # Load data from CSV; generate heatmaps
    model.load_heatmap_dataset_with_csv_from_directory(im_dir, 'point_labels.csv', ext='png')
    assert model._raw_image_files == expected_images
    assert model._raw_labels == expected_labels

    # Load data from CSV and pre-existing heatmaps
    model._raw_image_files = None
    model._raw_labels = None
    model.load_heatmap_dataset_with_csv_from_directory(im_dir, 'point_labels.csv', ext='png')
    assert model._raw_image_files == expected_images
    assert model._raw_labels == expected_labels

    shutil.rmtree(expected_heatmap_dir)


def test_heatmap_json_data_load(test_data_dir):
    im_dir = os.path.join(test_data_dir, 'test_Ara2013_heatmap')
    expected_heatmap_dir = os.path.join(os.path.curdir, 'generated_heatmaps')
    if os.path.exists(expected_heatmap_dir):
        shutil.rmtree(expected_heatmap_dir)

    model = dpp.HeatmapObjectCountingModel()
    model.set_image_dimensions(128, 128, 3)
    assert model._raw_image_files is None
    assert model._raw_labels is None

    base_names = ['ara2013_plant001_rgb', 'ara2013_plant002_rgb', 'ara2013_plant003_rgb',
                  'ara2013_plant004_rgb', 'ara2013_plant005_rgb', 'ara2013_plant006_rgb',
                  'ara2013_plant007_rgb', 'ara2013_plant008_rgb']
    expected_images = [os.path.join(im_dir, '{}.png'.format(x)) for x in base_names]
    expected_labels = [os.path.join(expected_heatmap_dir, '{}.npy'.format(x)) for x in base_names]

    # Load data from JSON; generate heatmaps
    model.load_heatmap_dataset_with_json_files_from_directory(im_dir)
    assert model._raw_image_files == expected_images
    assert model._raw_labels == expected_labels

    # Load data from JSON and pre-existing heatmaps
    model._raw_image_files = None
    model._raw_labels = None
    model.load_heatmap_dataset_with_json_files_from_directory(im_dir)
    assert model._raw_image_files == expected_images
    assert model._raw_labels == expected_labels

    shutil.rmtree(expected_heatmap_dir)


# seems to be some issue with tensorflow not using the same graph when run inside pytest framework
# def test_begin_training():
#     model = dpp.DPPModel(debug=False, save_checkpoints=False, report_rate=20)
#     channels = 3
#     model.set_batch_size(4)
#     model.set_image_dimensions(128, 128, channels)
#     model.set_resize_images(True)
#     model.set_problem_type('regression')
#     model.set_num_regression_outputs(1)
#     model.set_train_test_split(0.8)
#     model.set_weight_initializer('xavier')
#     model.set_maximum_training_epochs(1)
#     model.set_learning_rate(0.0001)
#     model.load_ippn_leaf_count_dataset_from_directory('test_data/Ara2013-Canon')
#     model.add_input_layer()
#     model.add_convolutional_layer(filter_dimension=[5, 5, channels, 32], stride_length=1, activation_function='tanh')
#     model.add_pooling_layer(kernel_size=3, stride_length=2)
#
#     model.add_convolutional_layer(filter_dimension=[5, 5, 32, 64], stride_length=1, activation_function='tanh')
#     model.add_pooling_layer(kernel_size=3, stride_length=2)
#
#     model.add_convolutional_layer(filter_dimension=[3, 3, 64, 64], stride_length=1, activation_function='tanh')
#     model.add_pooling_layer(kernel_size=3, stride_length=2)
#
#     model.add_convolutional_layer(filter_dimension=[3, 3, 64, 64], stride_length=1, activation_function='tanh')
#     model.add_pooling_layer(kernel_size=3, stride_length=2)
#     model.add_output_layer()
#     model.begin_training()

def test_forward_pass_residual():
    model = dpp.SemanticSegmentationModel()
    model.set_image_dimensions(50, 50, 1)
    model.set_batch_size(1)

    # Set up a small deterministic network with residuals
    model.add_input_layer()
    model.add_skip_connection(downsampled=False)
    model.add_skip_connection(downsampled=False)

    # Create an input image and its expected output
    test_im = np.full([50, 50, 1], 0.5, dtype=np.float32)
    expected_im = np.full([50, 50, 1], 1.0, dtype=np.float32)

    # Add the layers and get the forward pass
    model._add_layers_to_graph()
    out_im = model.forward_pass(test_im)

    assert out_im.size == expected_im.size
    assert np.all(out_im == expected_im)


def test_graph_problem_loss_semantic():
    model = dpp.SemanticSegmentationModel()
    assert model._loss_fn == 'sigmoid cross entropy'
    assert model._num_seg_class == 2

    in_batch_binary = np.array([[[[1.0], [0.9]],
                                 [[0.1], [0.0]]],
                                [[[1.0], [0.0]],
                                 [[0.8], [0.2]]]], np.float32)
    in_label_binary = np.array([[[[1.0], [0.0]],
                                 [[0.0], [1.0]]],
                                [[[1.0], [0.0]],
                                 [[0.0], [1.0]]]], np.float32)
    out_loss_binary = np.array([0.7480, 0.6939], np.float32)
    # Correct outputs are one-hot encoded but as inputs to softmax; -50 should turn into a small probability and 0
    # should turn into a probability close to 1 (i.e. softmax(0, -50, -50) ~= [1, 0, 0])
    in_batch_multi = np.array([[[[0.0, -50.0, -50.0], [-50.0, 0.0, -50.0]],
                                [[-50.0, -50.0, 0.0], [-50.0, 0.0, -50.0]]],
                               [[[-50.0, 0.0, -50.0], [-50.0, 0.0, -50.0]],
                                [[-2.0, 0.0, -2.0], [-50.0, -50.0, 0.0]]]], np.float32)
    in_label_multi = np.array([[[[0], [1]],
                                [[2], [1]]],
                               [[[1], [1]],
                                [[0], [2]]]], np.int32)
    out_loss_multi = np.array([0.0000, 0.5599], np.float32)

    with pytest.raises(RuntimeError):
        model._loss_fn = 'sigmoid cross entropy'
        model._num_seg_class = 3
        model._graph_problem_loss(in_batch_multi, in_label_multi)
    with pytest.raises(RuntimeError):
        model._loss_fn = 'softmax cross entropy'
        model._num_seg_class = 2
        model._graph_problem_loss(in_batch_binary, in_label_binary)

    model._loss_fn = 'sigmoid cross entropy'
    model._num_seg_class = 2
    with tf.Session() as sess:
        out_binary_tensor = model._graph_problem_loss(in_batch_binary, in_label_binary)
        out_binary = sess.run(out_binary_tensor)
        assert np.all(out_binary.shape == (2,))
        assert np.allclose(out_binary, out_loss_binary, atol=0.0001)

    model._loss_fn = 'softmax cross entropy'
    model._num_seg_class = 3
    with tf.Session() as sess:
        out_multi_tensor = model._graph_problem_loss(in_batch_multi, in_label_multi)
        out_multi = sess.run(out_multi_tensor)
        assert np.all(out_multi.shape == (2,))
        assert np.allclose(out_multi, out_loss_multi, atol=0.0001)


def test_det_random_mask(model, test_data_dir):
    data_path = os.path.join(test_data_dir, 'test_Ara2013_Canon', '')

    model.set_validation_split(0.25)
    model.set_test_split(0.25)
    model.set_maximum_training_epochs(1)
    model.load_ippn_leaf_count_dataset_from_directory(data_path)

    def get_random_mask():
        model.set_random_seed(7)
        labels = [' '.join(map(str, label)) for label in model._raw_labels]
        return loaders._get_split_mask(model._test_split, model._validation_split,
                                       len(labels), force_mask_creation=True)

    mask_1 = get_random_mask()
    model._reset_graph()
    model._reset_session()
    mask_2 = get_random_mask()
    assert np.all(mask_1 == mask_2)


def test_det_random_set_split(test_data_dir):
    model = dpp.RegressionModel()
    data_path = os.path.join(test_data_dir, 'test_Ara2013_Canon', '')

    model.set_validation_split(0.25)
    model.set_test_split(0.25)
    model.set_maximum_training_epochs(1)
    model.set_image_dimensions(128, 128, 3)
    model.load_ippn_leaf_count_dataset_from_directory(data_path)

    def get_random_splits():
        with model._graph.as_default():
            model.set_random_seed(7)
            trn_im, trn_lab, _, tst_im, tst_lab, _, val_im, val_lab, _ \
                = loaders.split_raw_data(model._raw_image_files, model._raw_labels,
                                         model._test_split, model._validation_split,
                                         split_labels=True, force_mask_creation=True)
            return model._session.run([trn_im, trn_lab, tst_im, tst_lab, val_im, val_lab])

    splits_1 = get_random_splits()
    model._reset_graph()
    model._reset_session()
    splits_2 = get_random_splits()
    assert np.all([np.all(x == y) for x, y in zip(splits_1, splits_2)])


def test_det_random_augmentations(test_data_dir):
    model = dpp.RegressionModel()
    data_path = os.path.join(test_data_dir, 'test_Ara2013_Canon', '')

    model.set_validation_split(0)
    model.set_test_split(0)
    model.set_maximum_training_epochs(1)
    model.set_image_dimensions(128, 128, 3)
    model.set_resize_images(True)
    model.set_augmentation_brightness_and_contrast(True)
    model.set_augmentation_flip_horizontal(True)
    model.set_augmentation_flip_vertical(True)
    model.load_ippn_leaf_count_dataset_from_directory(data_path)

    def get_random_augmentations():
        with model._graph.as_default():
            model.set_random_seed(7)
            labels = [' '.join(map(str, label)) for label in model._raw_labels]
            model._parse_dataset(model._raw_image_files, labels, None, None, None, None, None, None, None)
            data_iter = model._train_dataset.make_one_shot_iterator().get_next()

            data = []
            for _ in range(len(model._raw_image_files)):
                xy = model._session.run(data_iter)
                data.append(xy)
            return data

    data_1 = get_random_augmentations()
    model._reset_graph()
    model._reset_session()
    data_2 = get_random_augmentations()
    assert np.all([np.all(x[0] == y[0]) and x[1] == y[1] for x, y in zip(data_1, data_2)])


def test_det_shuffle_dataset(model, test_data_dir):
    data_path = os.path.join(test_data_dir, 'test_Ara2013_Canon', '')

    model.set_maximum_training_epochs(1)
    model.set_batch_size(1)
    model.load_ippn_leaf_count_dataset_from_directory(data_path)

    def get_shuffled_dataset():
        with model._graph.as_default():
            model.set_random_seed(7)
            ds = tf.data.Dataset.from_tensor_slices(model._raw_image_files)
            ds = model._batch_and_iterate(ds, shuffle=True)
            data_iter = ds.get_next()

            data = []
            for _ in range(len(model._raw_image_files)):
                xy = model._session.run(data_iter)
                data.append(xy[0])
            return data

    data_1 = get_shuffled_dataset()
    model._reset_graph()
    model._reset_session()
    data_2 = get_shuffled_dataset()
    assert np.all(data_1 == data_2)


def test_det_dropout(model, test_data_dir):
    data_path = os.path.join(test_data_dir, 'test_Ara2013_Canon', '')

    model.set_maximum_training_epochs(1)
    model.set_batch_size(1)
    model.load_ippn_leaf_count_dataset_from_directory(data_path)

    def get_dropout_result():
        with model._graph.as_default():
            model.set_random_seed(7)
            drop_in = [float(x[0]) for x in model._raw_labels]
            drop_layer = layers.dropoutLayer([8], 0.5)
            drop_result = drop_layer.forward_pass(drop_in, deterministic=False)
            return model._session.run(drop_result)

    drop_1 = get_dropout_result()
    model._reset_graph()
    model._reset_session()
    drop_2 = get_dropout_result()
    assert np.all(drop_1 == drop_2)
