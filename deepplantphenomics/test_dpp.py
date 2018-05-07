import pytest
import datetime
import numpy as np
import deepplantphenomics as dpp
from . import definitions
from . import layers

def test_init():
    model = dpp.DPPModel(tensorboard_dir="test_dir")
    # test tensorboard directory is being formatted correctly
    assert model._DPPModel__tb_dir == "test_dir/"+datetime.datetime.now().strftime("%d%B%Y%I:%M%p")

# public setters and adders, mostly testing for type and value errors
@pytest.fixture
def model(scope="module"):
    model = dpp.DPPModel()
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
        model.set_batch_size(-3)
    model.set_batch_size(-1)

def test_set_num_regression_outputs(model):
    with pytest.raises(TypeError):
        model.set_num_regression_outputs(5.0)
    with pytest.raises(ValueError):
        model.set_num_regression_outputs(-1)

def test_set_train_test_split(model):
    with pytest.raises(TypeError):
        model.set_train_test_split(5)
    with pytest.raises(ValueError):
        model.set_train_test_split(1.5)

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
    model.set_train_test_split(0.5)  # need this so next test catches a different RuntimeException
    with pytest.raises(RuntimeError):
        model.set_learning_rate_decay(0.5, 1)

def test_set_optimizer(model):
    with pytest.raises(TypeError):
        model.set_optimizer(5)
    with pytest.raises(ValueError):
        model.set_optimizer('Nico')
    model.set_optimizer('adam')
    assert model._DPPModel__optimizer == 'Adam'
    model.set_optimizer('Adam')
    assert model._DPPModel__optimizer == 'Adam'
    model.set_optimizer('ADAM')
    assert model._DPPModel__optimizer == 'Adam'

def test_set_weight_initializer(model):
    with pytest.raises(TypeError):
        model.set_weight_initializer(5)
    with pytest.raises(ValueError):
        model.set_weight_initializer('Nico')
    model.set_weight_initializer('normal')
    assert model._DPPModel__weight_initializer == 'Normal'
    model.set_weight_initializer('Normal')
    assert model._DPPModel__weight_initializer == 'Normal'
    model.set_weight_initializer('NORMAL')
    assert model._DPPModel__weight_initializer == 'Normal'

def test_set_image_dimensions(model):
    with pytest.raises(TypeError):
        model.set_image_dimensions(1.0, 1, 1)
    with pytest.raises(ValueError):
        model.set_image_dimensions(-2, 1, 1)
    with pytest.raises(TypeError):
        model.set_image_dimensions(1, 1.0, 1)
    with pytest.raises(ValueError):
        model.set_image_dimensions(1, -3, 1)
    with pytest.raises(TypeError):
        model.set_image_dimensions(1, 1, 1.0)
    with pytest.raises(ValueError):
        model.set_image_dimensions(1, 1, -4)
    model.set_image_dimensions(-1, -1, -1)


def test_set_original_image_dimensions(model):
    with pytest.raises(TypeError):
        model.set_original_image_dimensions(1.0, 1)
    with pytest.raises(ValueError):
        model.set_original_image_dimensions(-4, 1)
    with pytest.raises(TypeError):
        model.set_original_image_dimensions(1, 1.0)
    with pytest.raises(ValueError):
        model.set_original_image_dimensions(1, -7)
    model.set_original_image_dimensions(-1, -1)

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
    model.add_input_layer()
    assert isinstance(model._DPPModel__last_layer(), layers.inputLayer)

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
    model.set_batch_size(1)
    model.set_image_dimensions(1, 1, 1)
    model.add_input_layer()
    model.add_convolutional_layer(np.array([1,1,1,1]), 1, 'relu')
    assert isinstance(model._DPPModel__last_layer(), layers.convLayer)

def test_add_pooling_layer(model):
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
    model.set_batch_size(1)
    model.set_image_dimensions(1, 1, 1)
    model.add_input_layer()
    model.add_pooling_layer(1, 1, 'avg')
    assert isinstance(model._DPPModel__last_layer(), layers.poolingLayer)

def test_add_normalization_layer(model):
    model.set_batch_size(1)
    model.set_image_dimensions(1, 1, 1)
    model.add_input_layer()
    model.add_normalization_layer()
    assert isinstance(model._DPPModel__last_layer(), layers.normLayer)

def test_add_dropout_layer(model):
    with pytest.raises(TypeError):
        model.add_dropout_layer("0.5")
    with pytest.raises(ValueError):
        model.add_dropout_layer(1.5)
    model.set_batch_size(1)
    model.set_image_dimensions(1, 1, 1)
    model.add_input_layer()
    model.add_dropout_layer(0.4)
    assert isinstance(model._DPPModel__last_layer(), layers.dropoutLayer)

def test_add_batch_norm_layer(model):
    model.set_batch_size(1)
    model.set_image_dimensions(1, 1, 1)
    model.add_input_layer()
    model.add_batch_norm_layer()
    assert isinstance(model._DPPModel__last_layer(), layers.batchNormLayer)

def test_add_fully_connected_layer(model):
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
    model.set_batch_size(1)
    model.set_image_dimensions(1, 1, 1)
    model.add_input_layer()
    model.add_fully_connected_layer(1, 'tanh', 0.3)
    assert isinstance(model._DPPModel__last_layer(), layers.fullyConnectedLayer)

def test_add_output_layer(model):
    with pytest.raises(TypeError):
        model.add_output_layer("2")
    with pytest.raises(ValueError):
        model.add_output_layer(-0.4)
    with pytest.raises(TypeError):
        model.add_output_layer(2.0, 3.4)
    with pytest.raises(ValueError):
        model.add_output_layer(2.0, -4)
    model.set_batch_size(1)
    model.set_image_dimensions(1, 1, 1)
    model.add_input_layer()
    model.set_problem_type('classification')
    model.add_output_layer(2.5, 10)
    assert isinstance(model._DPPModel__last_layer(), layers.fullyConnectedLayer)
    new_model = dpp.DPPModel()
    new_model.set_batch_size(1)
    new_model.set_image_dimensions(1, 1, 1)
    new_model.add_input_layer()
    new_model.set_problem_type('semantic_segmentation')
    new_model.add_output_layer(2.5)
    assert isinstance(new_model._DPPModel__last_layer(), layers.convLayer)