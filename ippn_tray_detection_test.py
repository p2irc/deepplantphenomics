import deepplantphenomics as dpp

model = dpp.DPPModel(debug=True, load_from_saved=False, initialize=False)

# 3 channels for colour, 1 channel for greyscale
channels = 3

# Setup and hyperparameters
model.set_batch_size(4)
model.set_number_of_threads(8)
model.set_original_image_dimensions(2324, 3108)
model.set_image_dimensions(257, 307, channels)
model.set_resize_images(True)

model.set_problem_type('regression')
model.set_num_regression_outputs(4)
model.set_train_test_split(0.8)
model.set_regularization_coefficient(0.01)
model.set_learning_rate(0.0001)
model.set_weight_initializer('normal')
model.set_maximum_training_epochs(1000)

model.load_ippn_tray_dataset_from_directory('./data/Tray/Ara2012')