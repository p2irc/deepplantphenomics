#
# An example of training a model for the Days After Germination (DAG) task from the IPPN dataset.
#

import deepplantphenomics as dpp

model = dpp.RegressionModel(debug=True, save_checkpoints=False, report_rate=20)

# 3 channels for colour, 1 channel for greyscale
channels = 3

# Setup and hyper-parameters
model.set_batch_size(16)
model.set_number_of_threads(8)
model.set_image_dimensions(128, 128, channels)
model.set_resize_images(True)

model.set_num_regression_outputs(1)
model.set_test_split(0.2)
model.set_validation_split(0.0)
model.set_regularization_coefficient(0.001)
model.set_learning_rate(0.0001)
model.set_weight_initializer('normal')
model.set_maximum_training_epochs(1000)

# Augmentation options
model.set_augmentation_brightness_and_contrast(True)
model.set_augmentation_flip_horizontal(True)
model.set_augmentation_flip_vertical(True)
model.set_augmentation_crop(True, crop_ratio=0.8)

# Load dataset
model.load_ippn_dataset_from_directory('./data/Ara2013-Canon', 'DAG')

# Define a model architecture
model.add_input_layer()

model.add_convolutional_layer(filter_dimension=[3, 3, channels, 16], stride_length=1, activation_function='relu')
model.add_pooling_layer(kernel_size=3, stride_length=2)

model.add_convolutional_layer(filter_dimension=[3, 3, 16, 64], stride_length=1, activation_function='relu')
model.add_pooling_layer(kernel_size=3, stride_length=2)

model.add_fully_connected_layer(output_size=2048, activation_function='relu')

model.add_output_layer()

# Begin training the regression model
model.begin_training()
