#
# Demonstrates the process of training a YOLO-based object detector in DPP.
#

import deepplantphenomics as dpp

model = dpp.ObjectDetectionModel(debug=True, save_checkpoints=False, tensorboard_dir='tensor_logs', report_rate=20)

# 3 channels for colour, 1 channel for greyscale
channels = 3

# Setup and hyper-parameters
model.set_batch_size(1)
model.set_number_of_threads(4)
model.set_image_dimensions(448, 448, channels)
model.set_resize_images(False)
model.set_patch_size(448, 448)

# model.set_yolo_parameters() is not called here because we are using all of the default values
model.set_test_split(0.1)
model.set_validation_split(0)
model.set_learning_rate(0.000001)
model.set_weight_initializer('xavier')
model.set_maximum_training_epochs(100)

model.load_yolo_dataset_from_directory('./yolo_data', label_file='labels.json', image_dir='images')

# Define the YOLOv2 model architecture
model.use_predefined_model('yolov2')

# Begin training the YOLOv2 model
model.begin_training()
