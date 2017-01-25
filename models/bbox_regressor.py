#
# Used to train the boundingBoxRegressor regression model
#

import deepplantphenomics as dpp

model = dpp.DPPModel(debug=True, load_from_saved=False, tensorboard_dir='/home/jordan/tensorlogs', report_rate=20)

# 3 channels for colour, 1 channel for greyscale
channels = 3

# Setup and hyperparameters
model.setBatchSize(4)
model.setNumberOfThreads(8)
model.setOriginalImageDimensions(2056, 2454)
model.setImageDimensions(257, 307, channels)
model.setResizeImages(True)

model.setProblemType('regression')
model.setTrainTestSplit(0.8)
model.setRegularizationCoefficient(0.01)
model.setLearningRate(0.0001)
model.setWeightInitializer('normal')
model.setMaximumTrainingEpochs(1000)

# Load bounding box labels from Pascal VOC format
model.loadPascalVOCLabelsFromDirectory('./data/danforth-annotations')

# Load all VIS images from a Lemnatec image repository
model.loadLemnatecImagesFromDirectory('./data/danforth-sample')

# Define a model architecture
model.addInputLayer()

model.addConvolutionalLayer(filter_dimension=[5, 5, channels, 64], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
model.addPoolingLayer(kernel_size=3, stride_length=2)

model.addConvolutionalLayer(filter_dimension=[5, 5, 64, 128], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
model.addPoolingLayer(kernel_size=3, stride_length=2)

model.addConvolutionalLayer(filter_dimension=[5, 5, 128, 128], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
model.addPoolingLayer(kernel_size=3, stride_length=2)

model.addFullyConnectedLayer(output_size=384, activation_function='relu')

model.addOutputLayer(regularization_coefficient=0.0)

# Begin training the regression model
model.beginTraining()