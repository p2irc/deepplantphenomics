#
# Demonstrates training a general-purpose image classifier on the popular CIFAR10 image classification dataset.
# Assumes that you downloaded CIFAR10 image files via nvidia DIGITS.
#

import deepplantphenomics as dpp

model = dpp.DPPModel(debug=True, load_from_saved=False)

# 3 channels for colour, 1 channel for greyscale
channels = 3

# Setup and hyperparameters
model.set_batch_size(128)
model.set_number_of_threads(4)
model.set_image_dimensions(32, 32, channels)

model.set_regularization_coefficient(0.004)
model.set_learning_rate(0.001)
model.set_weight_initializer('normal')
model.set_maximum_training_epochs(700)

# Augmentation options
model.set_augmentation_flip_horizontal(True)
model.set_augmentation_crop(True)
model.set_augmentation_brightness_and_contrast(True)

# Load dataset
model.load_cifar10_dataset_from_directory('./data/cifar10')

# Simple CIFAR-10 model
model.add_input_layer()

model.add_convolutional_layer(filter_dimension=[5, 5, channels, 32], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
model.add_pooling_layer(kernel_size=3, stride_length=2)

model.add_convolutional_layer(filter_dimension=[5, 5, 32, 32], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
model.add_pooling_layer(kernel_size=3, stride_length=2)

model.add_convolutional_layer(filter_dimension=[5, 5, 32, 64], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
model.add_pooling_layer(kernel_size=3, stride_length=2)

model.add_fully_connected_layer(output_size=256, activation_function='relu')

model.add_output_layer(regularization_coefficient=0.0)

# Train!
model.begin_training()
