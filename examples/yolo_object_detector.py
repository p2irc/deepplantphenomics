#
# Demonstrates the process of training a YOLO-based object detector in DPP.
#

import deepplantphenomics as dpp

model = dpp.DPPModel(debug=True, load_from_saved='./saved_state')

# 3 channels for colour, 1 channel for greyscale
channels = 3

# Setup and hyperparameters
model.set_batch_size(1)
model.set_number_of_threads(4)
model.set_image_dimensions(448,448, channels)
model.set_resize_images(False)

model.set_problem_type('object_detection')
model.set_test_split(0.1)
model.set_validation_split(0)
model.set_learning_rate(0.000001)
model.set_weight_initializer('xavier')
model.set_maximum_training_epochs(100)

model.load_yolo_dataset_from_directory('./yolo_data', 'labels.json', 'images')

# Define the YOLOv2 model architecture
model.use_predefined_model('yolov2')

# Begin training the YOLOv2 model
model.begin_training()