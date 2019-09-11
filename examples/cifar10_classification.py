#
# Demonstrates training a general-purpose image classifier on the popular CIFAR10 image classification dataset.
# Assumes that you downloaded CIFAR10 image files via nvidia DIGITS.
#

import deepplantphenomics as dpp

model = dpp.ClassificationModel(debug=True, load_from_saved=False)

# 3 channels for colour, 1 channel for greyscale
channels = 3

# Setup and hyper-parameters
model.set_image_dimensions(32, 32, channels)

model.set_regularization_coefficient(0.004)
model.set_batch_size(32)
model.set_learning_rate(0.0001)
model.set_maximum_training_epochs(25)

# Augmentation options
model.set_augmentation_flip_horizontal(True)
model.set_augmentation_crop(True)
model.set_augmentation_brightness_and_contrast(True)

# Load dataset
model.load_cifar10_dataset_from_directory('./data/cifar10')

# Use a VGG-16 network
model.use_predefined_model('vgg-16')

# Train!
model.begin_training()
