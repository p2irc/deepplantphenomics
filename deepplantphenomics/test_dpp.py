import pytest
import numpy as np
from . import deepplantpheno as dpp
from . import definitions
from . import layers

# public setters and adders, mostly testing for type and value errors
@pytest.fixture
def model(scope="module"):
    model = dpp.DPPModel()
    model.set_image_dimensions(1, 1, 1)
    return model

def test_set_number_of_threads(model):
    with pytest.raises(TypeError):
        model.set_number_of_threads(5.0)
    with pytest.raises(ValueError):
        model.set_number_of_threads(-1)

def test_set_processed_images_dir(model):
    with pytest.raises(TypeError):
        model.set_processed_images_dir(5)

def test_set_batch_size(model):
    with pytest.raises(TypeError):
        model.set_batch_size(5.0)
    with pytest.raises(ValueError):
        model.set_batch_size(-1)

def test_set_num_regression_outputs(model):
    with pytest.raises(RuntimeError):
        model.set_num_regression_outputs(1)
    model.set_problem_type('regression')
    with pytest.raises(TypeError):
        model.set_num_regression_outputs(5.0)
    with pytest.raises(ValueError):
        model.set_num_regression_outputs(-1)

def test_set_maximum_training_epochs(model):
    with pytest.raises(TypeError):
        model.set_maximum_training_epochs(5.0)
    with pytest.raises(ValueError):
        model.set_maximum_training_epochs(-1)

def test_set_learning_rate(model):
    with pytest.raises(TypeError):
        model.set_learning_rate("5")
    with pytest.raises(ValueError):
        model.set_learning_rate(-0.001)

def test_set_crop_or_pad_images(model):
    with pytest.raises(TypeError):
        model.set_crop_or_pad_images("True")

def test_set_resize_images(model):
    with pytest.raises(TypeError):
        model.set_resize_images("True")

def test_set_augmentation_flip_horizontal(model):
    with pytest.raises(TypeError):
        model.set_augmentation_flip_horizontal("True")

def test_set_augmentation_flip_vertical(model):
    with pytest.raises(TypeError):
        model.set_augmentation_flip_vertical("True")

def test_set_augmentation_crop(model):
    with pytest.raises(TypeError):
        model.set_augmentation_crop("True", 0.5)
    with pytest.raises(TypeError):
        model.set_augmentation_crop(True, "5")
    with pytest.raises(ValueError):
        model.set_augmentation_crop(False, -1.0)

def test_set_augmentation_brightness_and_contrast(model):
    with pytest.raises(TypeError):
        model.set_augmentation_crop("True")

def test_set_regularization_coefficient(model):
    with pytest.raises(TypeError):
        model.set_regularization_coefficient("5")
    with pytest.raises(ValueError):
        model.set_regularization_coefficient(-0.001)

def test_set_learning_rate_decay(model):
    with pytest.raises(TypeError):
        model.set_learning_rate_decay("5", 1)
    with pytest.raises(ValueError):
        model.set_learning_rate_decay(-0.001, 1)
    with pytest.raises(TypeError):
        model.set_learning_rate_decay(0.5, 5.0)
    with pytest.raises(ValueError):
        model.set_learning_rate_decay(0.5, -1)
    with pytest.raises(RuntimeError):
        model.set_learning_rate_decay(0.5, 1)

def test_set_optimizer(model):
    with pytest.raises(TypeError):
        model.set_optimizer(5)
    with pytest.raises(ValueError):
        model.set_optimizer('Nico')
    model.set_optimizer('adam')
    assert model._DPPModel__optimizer == 'adam'
    model.set_optimizer('Adam')
    assert model._DPPModel__optimizer == 'adam'
    model.set_optimizer('ADAM')
    assert model._DPPModel__optimizer == 'adam'
    model.set_optimizer('SGD')
    assert model._DPPModel__optimizer == 'sgd'
    model.set_optimizer('sgd')
    assert model._DPPModel__optimizer == 'sgd'
    model.set_optimizer('sGd')
    assert model._DPPModel__optimizer == 'sgd'

def test_set_weight_initializer(model):
    with pytest.raises(TypeError):
        model.set_weight_initializer(5)
    with pytest.raises(ValueError):
        model.set_weight_initializer('Nico')
    model.set_weight_initializer('normal')
    assert model._DPPModel__weight_initializer == 'normal'
    model.set_weight_initializer('Normal')
    assert model._DPPModel__weight_initializer == 'normal'
    model.set_weight_initializer('NORMAL')
    assert model._DPPModel__weight_initializer == 'normal'

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

def test_add_preprocessor(model):
    with pytest.raises(TypeError):
        model.add_preprocessor(5)
    with pytest.raises(ValueError):
        model.add_preprocessor('Nico')

def test_set_problem_type(model):
    with pytest.raises(TypeError):
        model.set_problem_type(5)
    with pytest.raises(ValueError):
        model.set_problem_type('Nico')
    model.set_problem_type('classification')
    assert model._DPPModel__problem_type == definitions.ProblemType.CLASSIFICATION
    model.set_problem_type('regression')
    assert model._DPPModel__problem_type == definitions.ProblemType.REGRESSION
    model.set_problem_type('semantic_segmentation')
    assert model._DPPModel__problem_type == definitions.ProblemType.SEMANTICSEGMETNATION

# adding layers may require some more indepth testing
def test_add_input_layer(model):
    model.set_batch_size(1)
    model.set_image_dimensions(1, 1, 1)
    model.add_input_layer()
    assert isinstance(model._DPPModel__last_layer(), layers.inputLayer)
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
    with pytest.raises(TypeError):
        model.add_convolutional_layer([1, 2, 3, 4], 1, 'relu', "5")
    with pytest.raises(ValueError):
        model.add_convolutional_layer([1, 2, 3, 4], 1, 'relu', -1.0)
    model.add_convolutional_layer(np.array([1,1,1,1]), 1, 'relu')
    assert isinstance(model._DPPModel__last_layer(), layers.convLayer)

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
    assert isinstance(model._DPPModel__last_layer(), layers.poolingLayer)

def test_add_normalization_layer(model):
    with pytest.raises(RuntimeError):
        model.add_normalization_layer()
    model.add_input_layer()
    model.add_normalization_layer()
    assert isinstance(model._DPPModel__last_layer(), layers.normLayer)

def test_add_dropout_layer(model):
    with pytest.raises(RuntimeError):
        model.add_dropout_layer(0.4)
    model.add_input_layer()
    with pytest.raises(TypeError):
        model.add_dropout_layer("0.5")
    with pytest.raises(ValueError):
        model.add_dropout_layer(1.5)
    model.add_dropout_layer(0.4)
    assert isinstance(model._DPPModel__last_layer(), layers.dropoutLayer)

def test_add_batch_norm_layer(model):
    with pytest.raises(RuntimeError):
        model.add_batch_norm_layer()
    model.add_input_layer()
    model.add_batch_norm_layer()
    assert isinstance(model._DPPModel__last_layer(), layers.batchNormLayer)

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
    assert isinstance(model._DPPModel__last_layer(), layers.fullyConnectedLayer)

def test_add_output_layer(model):
    with pytest.raises(RuntimeError):
        model.add_output_layer(2.5, 3)
    model.add_input_layer()
    with pytest.raises(TypeError):
        model.add_output_layer("2")
    with pytest.raises(ValueError):
        model.add_output_layer(-0.4)
    with pytest.raises(TypeError):
        model.add_output_layer(2.0, 3.4)
    with pytest.raises(ValueError):
        model.add_output_layer(2.0, -4)
    model.set_problem_type('semantic_segmentation') # needed for following runetime error to occur
    with pytest.raises(RuntimeError):
        model.add_output_layer(2.0, 3)
    model.set_problem_type('classification')
    model.add_output_layer(2.5, 3)
    assert isinstance(model._DPPModel__last_layer(), layers.fullyConnectedLayer)

# having issue with not being able to create a new model, they all seem to inherit the fixture model
# used in previous test functions and thus can't properly add a new outputlayer for this test
# @pytest.fixture
# def model2():
#     model2 = dpp.DPPModel()
#     return model2
# def test_add_output_layer_2(model2): # semantic_segmentation problem type
#     model2.set_batch_size(1)
#     model2.set_image_dimensions(1, 1, 1)
#     model2.add_input_layer()
#     model2.set_problem_type('semantic_segmentation')
#     model2.add_output_layer(2.5)
#     assert isinstance(model2._DPPModel__last_layer(), layers.convLayer)

# more loading data tests!!!!
def test_load_dataset_from_directory_with_csv_labels(model):
    with pytest.raises(TypeError):
        model.load_dataset_from_directory_with_csv_labels(5, 'img_001')

def test_load_ippn_leaf_count_dataset_from_directory():
    # The following tests take the format laid out in the documentation of an example
    # for training a leaf counter, and leave out key parts to see if the program
    # throws an appropriate exception, or executes as intended due to using a default setting

    # forgetting to set image dimensions
    model = dpp.DPPModel(debug=False, save_checkpoints=False, report_rate=20)
    channels = 3
    model.set_batch_size(4)
    # model.set_image_dimensions(128, 128, channels)
    model.set_resize_images(True)
    model.set_problem_type('regression')
    model.set_num_regression_outputs(1)
    model.set_test_split(0.1)
    model.set_weight_initializer('xavier')
    model.set_maximum_training_epochs(1)
    model.set_learning_rate(0.0001)
    with pytest.raises(RuntimeError):
        model.load_ippn_leaf_count_dataset_from_directory('test_data/test_Ara2013_Canon')

    # forgetting to set num epochs
    model = dpp.DPPModel(debug=False, save_checkpoints=False, report_rate=20)
    channels = 3
    model.set_batch_size(4)
    model.set_image_dimensions(128, 128, channels)
    model.set_resize_images(True)
    model.set_problem_type('regression')
    model.set_num_regression_outputs(1)
    model.set_test_split(0.1)
    model.set_weight_initializer('xavier')
    # model.set_maximum_training_epochs(1)
    model.set_learning_rate(0.0001)
    with pytest.raises(RuntimeError):
        model.load_ippn_leaf_count_dataset_from_directory('test_data/test_Ara2013_Canon')

    # the following shouldn't raise any issues since there should be defaults for
    # batch_size, train_test_split, and learning_rate
    model = dpp.DPPModel(debug=False, save_checkpoints=False, report_rate=20)
    channels = 3
    # model.set_batch_size(4)
    model.set_image_dimensions(128, 128, channels)
    model.set_resize_images(True)
    model.set_problem_type('regression')
    model.set_num_regression_outputs(1)
    # model.set_train_test_split(0.8)
    model.set_weight_initializer('xavier')
    model.set_maximum_training_epochs(1)
    # model.set_learning_rate(0.0001)
    model.load_ippn_leaf_count_dataset_from_directory('test_data/test_Ara2013_Canon')




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