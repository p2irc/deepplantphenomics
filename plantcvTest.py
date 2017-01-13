from deepplantpheno import DPPModel

model = DPPModel(debug=True, load_from_saved=False)

# 3 channels for colour, 1 channel for greyscale
channels = 3

# Setup and hyperparameters
model.setBatchSize(128)
model.setNumberOfThreads(12)
model.setImageDimensions(32, 32, channels)

model.setTrainTestSplit(0.7)
model.setRegularizationCoefficient(0.004)
model.setLearningRate(0.001)
model.setWeightInitializer('normal')
model.setMaximumTrainingEpochs(700)

model.addPreprocessor('Auto-segment')

model.loadLemnatecDatasetFromDirectory('./data/danforth-sample')