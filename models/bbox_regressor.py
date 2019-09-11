#
# Used to train the bbox-regressor-lemnatec model
#

import deepplantphenomics as dpp

model = dpp.RegressionModel(debug=True, load_from_saved=False, save_checkpoints=False, report_rate=20)

# 3 channels for colour, 1 channel for greyscale
channels = 3

# Setup and hyper-parameters
model.set_batch_size(4)
model.set_number_of_threads(8)
model.set_original_image_dimensions(2056, 2454)
model.set_image_dimensions(257, 307, channels)
model.set_resize_images(True)

model.set_num_regression_outputs(4)
model.set_test_split(0.2)
model.set_validation_split(0.0)
model.set_regularization_coefficient(0.01)
model.set_learning_rate(0.0001)
model.set_weight_initializer('normal')
model.set_maximum_training_epochs(1000)

# Load bounding box labels from Pascal VOC format
model.load_pascal_voc_labels_from_directory('./annotations')

# Load all VIS images from a Lemnatec image repository
model.load_lemnatec_images_from_directory('./data')

# Define a model architecture
model.add_input_layer()

model.add_convolutional_layer(filter_dimension=[5, 5, channels, 16], stride_length=1, activation_function='relu')
model.add_pooling_layer(kernel_size=3, stride_length=2)

model.add_convolutional_layer(filter_dimension=[5, 5, 16, 64], stride_length=1, activation_function='relu')
model.add_pooling_layer(kernel_size=3, stride_length=2)

model.add_convolutional_layer(filter_dimension=[5, 5, 64, 64], stride_length=1, activation_function='relu')
model.add_pooling_layer(kernel_size=3, stride_length=2)

model.add_convolutional_layer(filter_dimension=[5, 5, 64, 64], stride_length=1, activation_function='relu')
model.add_pooling_layer(kernel_size=3, stride_length=2)

model.add_fully_connected_layer(output_size=384, activation_function='relu')

model.add_output_layer(regularization_coefficient=0.0)

# Begin training the regression model
model.begin_training()
