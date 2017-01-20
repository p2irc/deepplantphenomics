from deepplantpheno import DPPModel

model = DPPModel(debug=True, load_from_saved=False)

# 3 channels for colour, 1 channel for greyscale
channels = 3

# Setup and hyperparameters
model.setBatchSize(8)
model.setNumberOfThreads(4)
model.setOriginalImageDimensions(2056, 2454)
model.setImageDimensions(514, 614, channels)
model.setResizeImages(True)

model.setProblemType('regression')
model.setTrainTestSplit(0.7)
model.setRegularizationCoefficient(0.004)
model.setLearningRate(0.001)
model.setWeightInitializer('normal')
model.setMaximumTrainingEpochs(700)

# Set image pre-processing steps
#model.addPreprocessor('auto-segmentation')

# Load bounding box labels from Pascal VOC format
model.loadPascalVOCLabelsFromDirectory('./data/danforth-annotations')

# Load all VIS images from a Lemnatec image repository
model.loadLemnatecImagesFromDirectory('./data/danforth-sample')

# Define a model architecture
model.addInputLayer()

model.addConvolutionalLayer(filter_dimension=[5, 5, channels, 16], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
model.addPoolingLayer(kernel_size=3, stride_length=2)

model.addConvolutionalLayer(filter_dimension=[5, 5, 16, 16], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
model.addPoolingLayer(kernel_size=3, stride_length=2)

model.addConvolutionalLayer(filter_dimension=[5, 5, 16, 20], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
model.addPoolingLayer(kernel_size=3, stride_length=2)

model.addFullyConnectedLayer(output_size=64, activation_function='relu')

model.addOutputLayer(regularization_coefficient=0.0)

# Begin training the regression model
model.beginTraining()