from tfHelper import tfHelper

tfh = tfHelper(debug=True, load_from_saved=False)

# 3 channels for colour, 1 channel for greyscale
channels = 3

# Setup and hyperparameters
tfh.setBatchSize(128)
tfh.setNumberOfThreads(12)
tfh.setImageDimensions(32, 32, channels)

tfh.setTrainTestSplit(0.7)
tfh.setRegularizationCoefficient(0.004)
tfh.setLearningRate(0.001)
tfh.setWeightInitializer('normal')
tfh.setMaximumTrainingEpochs(700)

tfh.addPreprocessor('Auto-segment')

tfh.loadLemnatecDatasetFromDirectory('./data/danforth-sample')