import deepplantphenomics as dpp
import os

dir = '/home/nico/Plant_Phenotyping_Datasets/Plant/Ara2013-Canon'
#dir = './data/Ara2013-Canon'

images = [os.path.join(dir, name) for name in os.listdir(dir) if
          os.path.isfile(os.path.join(dir, name)) & name.endswith('_rgb.png')]
images = sorted(images)

dpp.tools.predict_rosette_leaf_count(images)
# Returns `array[17.]`

dpp.tools.predict_rosette_leaf_count(images)
# Raises aforementioned error.