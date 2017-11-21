#
# An example of semantic segmentation
#

import deepplantphenomics as dpp

model = dpp.DPPModel(debug=True, save_checkpoints=False, report_rate=1)

# 3 channels for colour, 1 channel for greyscale
channels = 3

# Setup and hyperparameters
model.set_batch_size(4)
#model.set_number_of_threads(8)
model.set_image_dimensions(128, 128, channels)
model.set_resize_images(True)

model.set_problem_type('semantic_segmentation')
model.set_train_test_split(0.8)
#model.set_regularization_coefficient(0.001)
model.set_learning_rate(0.0001)
model.set_weight_initializer('normal')
model.set_maximum_training_epochs(1000)

# Augmentation options
model.set_augmentation_brightness_and_contrast(True)
#model.set_augmentation_flip_horizontal(True)
#model.set_augmentation_flip_vertical(True)
#model.set_augmentation_crop(True, crop_ratio=0.8)

# Load dataset
model.load_dataset_from_directory_with_segmentation_masks('./data/Ara2013-Canon', './data/Ara2013-Canon/segmented')

# Define a model architecture
model.add_input_layer()

model.add_convolutional_layer(filter_dimension=[3, 3, channels, 4], stride_length=1, activation_function='tanh')
model.add_convolutional_layer(filter_dimension=[3, 3, 4, 8], stride_length=1, activation_function='tanh')

model.add_output_layer()

# Begin training the segmentation model
model.begin_training()