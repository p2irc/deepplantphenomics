import deepplantphenomics as dpp

model = dpp.DPPModel(debug=True, load_from_saved=False, initialize=False)

# 3 channels for colour, 1 channel for greyscale
channels = 3

# Setup and hyperparameters
model.setNumberOfThreads(12)
model.setImageDimensions(2056, 2454, channels)

# Add auto-segment preprocessor
model.addPreprocessor('auto-segmentation')

# Load all VIS images from a Lemnatec image repository
model.loadLemnatecImagesFromDirectory('./data/danforth-sample-2')

