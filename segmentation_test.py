from deepplantpheno import DPPModel

model = DPPModel(debug=True, load_from_saved=False)

# 3 channels for colour, 1 channel for greyscale
channels = 3

# Setup and hyperparameters
model.setBatchSize(4)
model.setNumberOfThreads(8)
model.setImageDimensions(2056, 2454, channels)

# Add auto-segment preprocessor
model.addPreprocessor('auto-segmentation')

# Load bounding box labels from Pascal VOC format
model.loadPascalVOCLabelsFromDirectory('./data/danforth-annotations')

# Load all VIS images from a Lemnatec image repository
model.loadLemnatecImagesFromDirectory('./data/danforth-sample')

