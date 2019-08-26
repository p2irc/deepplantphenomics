#
# Used to train the count_ception_model
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
model.set_maximum_training_epochs(1000)

# Load images and ground truth from a pickle file
model.load_countception_dataset_from_pkl_file('countception_MBM_dataset.pkl')

# Define a model architecture
model.add_input_layer()
model.add_convolutional_layer(filter_dimension=[3, 3, 3, 64],
                              stride_length=1,
                              activation_function='lrelu',
                              padding=patch_size,
                              batch_norm=True,
                              epsilon=1e-5,
                              decay=0.9)
model.add_paral_conv_block(filter_dimension_1=[1, 1, 0, 16],
                           filter_dimension_2=[3, 3, 0, 16])
model.add_paral_conv_block(filter_dimension_1=[1, 1, 0, 16],
                           filter_dimension_2=[3, 3, 0, 32])
model.add_convolutional_layer(filter_dimension=[14, 14, 0, 16],
                              stride_length=1,
                              activation_function='lrelu',
                              padding=0,
                              batch_norm=True,
                              epsilon=1e-5,
                              decay=0.9)
model.add_paral_conv_block(filter_dimension_1=[1, 1, 0, 112],
                           filter_dimension_2=[3, 3, 0, 48])
model.add_paral_conv_block(filter_dimension_1=[1, 1, 0, 64],
                           filter_dimension_2=[3, 3, 0, 32])
model.add_paral_conv_block(filter_dimension_1=[1, 1, 0, 40],
                           filter_dimension_2=[3, 3, 0, 40])
model.add_paral_conv_block(filter_dimension_1=[1, 1, 0, 32],
                           filter_dimension_2=[3, 3, 0, 96])
model.add_convolutional_layer(filter_dimension=[18, 18, 0, 32],
                              stride_length=1,
                              activation_function='lrelu',
                              padding=0,
                              batch_norm=True,
                              epsilon=1e-5,
                              decay=0.9)
model.add_convolutional_layer(filter_dimension=[1, 1, 0, 64],
                              stride_length=1,
                              activation_function='lrelu',
                              padding=0,
                              batch_norm=True,
                              epsilon=1e-5,
                              decay=0.9)
model.add_convolutional_layer(filter_dimension=[1, 1, 0, 64],
                              stride_length=1,
                              activation_function='lrelu',
                              padding=0,
                              batch_norm=True,
                              epsilon=1e-5,
                              decay=0.9)
model.add_convolutional_layer(filter_dimension=[1, 1, 0, 1],
                              stride_length=1,
                              activation_function='lrelu',
                              padding=0,
                              batch_norm=True,
                              epsilon=1e-5,
                              decay=0.9)

# Begin training the regression model
model.begin_training()