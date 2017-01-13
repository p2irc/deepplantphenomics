from tfHelper import tfHelper

tfh = tfHelper(debug=True, load_from_saved=False, tensorboard_dir='/home/jordan/tensorlogs')

# 3 channels for colour, 1 channel for greyscale
channels = 3

# Setup and hyperparameters
tfh.setBatchSize(128)
tfh.setNumberOfThreads(4)
tfh.setImageDimensions(32, 32, channels)

tfh.setRegularizationCoefficient(0.004)
tfh.setLearningRate(0.001)
tfh.setWeightInitializer('normal')
tfh.setMaximumTrainingEpochs(700)

# Augmentation options
tfh.setAugmentationFlip(True)
tfh.setAugmentationCrop(True)
tfh.setAugmentationBrightnessAndContrast(True)

# Load dataset
tfh.loadCIFAR10DatasetFromDirectory('./data/cifar10')

# Simple CIFAR-10 model
tfh.addInputLayer()

tfh.addConvolutionalLayer(filter_dimension=[5, 5, channels, 32], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
tfh.addPoolingLayer(kernel_size=3, stride_length=2)

tfh.addConvolutionalLayer(filter_dimension=[5, 5, 32, 32], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
tfh.addPoolingLayer(kernel_size=3, stride_length=2)

tfh.addConvolutionalLayer(filter_dimension=[5, 5, 32, 64], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
tfh.addPoolingLayer(kernel_size=3, stride_length=2)

tfh.addFullyConnectedLayer(output_size=256, activation_function='relu', shakeweight_p=0.5)
tfh.addFullyConnectedLayer(output_size=256, activation_function='relu', shakeweight_p=0.5)

tfh.addOutputLayer(regularization_coefficient=0.0)

# Train!
tfh.beginTraining()
