#
# Used to train the DAG-regressor model
#

import deepplantphenomics as dpp
import numpy as np

model = dpp.DPPModel(debug=True, save_checkpoints=False, tensorboard_dir='/home/jordan/tensorlogs', report_rate=20)

# 3 channels for colour, 1 channel for greyscale
channels = 3

# Setup and hyperparameters
model.set_batch_size(16)
model.set_number_of_threads(8)
model.set_image_dimensions(128, 128, channels)
model.set_resize_images(True)

model.set_problem_type('regression')
model.set_num_regression_outputs(1)
model.set_train_test_split(0.8)
model.set_regularization_coefficient(0.04)
model.set_learning_rate(0.00001)
model.set_weight_initializer('normal')
model.set_maximum_training_epochs(1000)

# Augmentation options
model.set_augmentation_brightness_and_contrast(True)
model.set_augmentation_flip_horizontal(True)
model.set_augmentation_flip_vertical(True)
model.set_augmentation_crop(True)

# Use the species network to get one-hot class labels for all of the training images
# print('Performing strain classification...')
#
# net = dpp.networks.arabidopsisStrainClassifier()
# raw_species = net.forward_pass(model.all_training_filenames)
# net.shut_down()
#
# # Convert network responses to one-hot matrix
# idx = np.argmax(raw_species, axis=1)
# onehot = np.zeros((idx.size, idx.max()+1))
# onehot[np.arange(idx.size), idx] = 1
#
# print('Done')

# LOL jk here it is hardcoded

onehot = np.array([[ 0,  1,  0,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 0,  0,  0,  0,  1],
 [ 0,  0,  1,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 0,  0,  1,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 1,  0,  0,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 0,  0,  1,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 0,  0,  1,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 0,  0,  1,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 0,  0,  1,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 0,  0,  1,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 1,  0,  0,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 0,  0,  0,  1,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 1,  0,  0,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  0,  1,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 1,  0,  0,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 0,  0,  1,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 1,  0,  0,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 0,  0,  0,  1,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 1,  0,  0,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 1,  0,  0,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  0,  0,  1,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 1,  0,  0,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 1,  0,  0,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 0,  0,  0,  1,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 1,  0,  0,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 0,  1,  0,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 1,  0,  0,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 0,  1,  0,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 1,  0,  0,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 0,  0,  0,  1,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 0,  0,  1,  0,  0],
 [ 1,  0,  0,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 0,  1,  0,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 1,  0,  0,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 1,  0,  0,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  0,  1,  0],
 [ 1,  0,  0,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 1,  0,  0,  0,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  0,  0,  1,  0],
 [ 0,  0,  1,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 0,  1,  0,  0,  0],
 [ 1,  0,  0,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 1,  0,  0,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 1,  0,  0,  0,  0],
 [ 0,  0,  0,  1,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 1,  0,  0,  0,  0],
 [ 0,  0,  0,  0,  1],
 [ 1,  0,  0,  0,  0],
 [ 0,  1,  0,  0,  0],
 [ 0,  0,  0,  1,  0],
 [ 0,  0,  1,  0,  0]])

model.add_moderation_features(onehot)

# Load dataset
model.load_ippn_dataset_from_directory('./data/Ara2013-Canon', 'DAG')

# Define a model architecture
model.add_input_layer()

model.add_convolutional_layer(filter_dimension=[5, 5, channels, 16], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
model.add_pooling_layer(kernel_size=3, stride_length=2)

model.add_convolutional_layer(filter_dimension=[5, 5, 16, 64], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
model.add_pooling_layer(kernel_size=3, stride_length=2)

model.add_convolutional_layer(filter_dimension=[5, 5, 64, 64], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
model.add_pooling_layer(kernel_size=3, stride_length=2)

model.add_convolutional_layer(filter_dimension=[5, 5, 64, 64], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
model.add_pooling_layer(kernel_size=3, stride_length=2)

model.add_moderation_layer()

model.add_fully_connected_layer(output_size=128, activation_function='relu')
model.add_fully_connected_layer(output_size=128, activation_function='relu')

model.add_output_layer(regularization_coefficient=0.0)

# Begin training the regression model
model.begin_training()