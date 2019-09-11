#
# Used to train the Count-ception model
#

import deepplantphenomics as dpp

model = dpp.CountCeptionModel(debug=True, load_from_saved=False, save_checkpoints=False, report_rate=20)

patch_size = 32

# Setup and hyperparameters
model.set_loss_function('l1')
model.set_batch_size(2)
model.set_number_of_threads(4)
model.set_image_dimensions(300, 300, 3)

model.set_test_split(0.3)
model.set_validation_split(0.2)
model.set_learning_rate(0.0001)
model.set_weight_initializer('xavier')
model.set_maximum_training_epochs(10)

# Load images and ground truth from a pickle file
model.load_countception_dataset_from_pkl_file('MBM-dataset.pkl')

# Define a model architecture
model.use_predefined_model("countception")

# Begin training the regression model
model.begin_training()