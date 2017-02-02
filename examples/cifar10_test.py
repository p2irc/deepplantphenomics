import deepplantphenomics as dpp

model = dpp.DPPModel(debug=True, load_from_saved=False, tensorboard_dir='/home/jordan/tensorlogs')
#model = dpp.DPPModel(debug=True, load_from_saved='./tfhSaved.meta')

# 3 channels for colour, 1 channel for greyscale
channels = 3

# Setup and hyperparameters
model.setBatchSize(128)
model.setNumberOfThreads(4)
model.setImageDimensions(32, 32, channels)

model.setRegularizationCoefficient(0.004)
model.setLearningRate(0.001)
model.setWeightInitializer('normal')
model.setMaximumTrainingEpochs(700)

# Augmentation options
model.setAugmentationFlip(True)
model.setAugmentationCrop(True)
model.setAugmentationBrightnessAndContrast(True)

# Load dataset
model.loadCIFAR10DatasetFromDirectory('./data/cifar10')

# Simple CIFAR-10 model
model.addInputLayer()

model.addConvolutionalLayer(filter_dimension=[5, 5, channels, 32], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
model.addPoolingLayer(kernel_size=3, stride_length=2)

model.addConvolutionalLayer(filter_dimension=[5, 5, 32, 32], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
model.addPoolingLayer(kernel_size=3, stride_length=2)

model.addConvolutionalLayer(filter_dimension=[5, 5, 32, 64], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
model.addPoolingLayer(kernel_size=3, stride_length=2)

model.addFullyConnectedLayer(output_size=256, activation_function='relu')

model.addOutputLayer(regularization_coefficient=0.0)

# Train!
model.beginTraining()
