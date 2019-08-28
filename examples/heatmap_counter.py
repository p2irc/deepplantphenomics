import deepplantphenomics as dpp

model = dpp.HeatmapObjectCountingModel(debug=True, load_from_saved=False)

# 3 channels for colour, 1 channel for greyscale
channels = 3

# Setup and hyper-parameters
model.set_image_dimensions(128, 128, channels)
model.set_batch_size(32)
model.set_number_of_threads(4)

model.set_learning_rate(0.0001)
model.set_maximum_training_epochs(25)
model.set_test_split(0.75)
model.set_validation_split(0.0)

# Load dataset
model.set_density_map_sigma(4.0)
model.load_heatmap_dataset_with_csv_from_directory('./data', 'point_labels.csv')

# Define a model architecture
model.add_input_layer()

model.add_convolutional_layer(filter_dimension=[3, 3, 3, 16], stride_length=1, activation_function='relu')
model.add_convolutional_layer(filter_dimension=[3, 3, 16, 32], stride_length=1, activation_function='relu')
model.add_convolutional_layer(filter_dimension=[5, 5, 32, 32], stride_length=1, activation_function='relu')

model.add_output_layer()

# Train!
model.begin_training()
