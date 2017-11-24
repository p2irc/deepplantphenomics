#
# This example shows a model which uses the auto-segmentation pre-processor.
# The model is never defined, so running this file only demonstrates auto-segmentation.
#

import deepplantphenomics as dpp

model = dpp.DPPModel(debug=True, load_from_saved=False, initialize=False)

# 3 channels for colour, 1 channel for greyscale
channels = 3

# Setup and hyperparameters
model.set_number_of_threads(12)
model.set_image_dimensions(2056, 2454, channels)

# Add auto-segment preprocessor
model.add_preprocessor('auto-segmentation')

# Load all VIS images from a Lemnatec image repository
model.load_lemnatec_images_from_directory('./data')

