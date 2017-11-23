#
# Demonstrates the use of tools.segment_vegetation on images of rosettes.
#

import deepplantphenomics as dpp
import numpy
# Required to write the results as images
from PIL import Image
import os

dir = './data/Ara2013-Canon'

output_dir = './segmented-images'

images = [os.path.join(dir, name) for name in os.listdir(dir) if
          os.path.isfile(os.path.join(dir, name)) & name.endswith('_rgb.png')]

print('Performing segmentation...')

y = dpp.tools.segment_vegetation(images)

for i, img in enumerate(y):
    filename = os.path.join(output_dir, os.path.basename(images[i]))
    result = Image.fromarray((img * 255).astype(numpy.uint8))
    result.save(filename)

print('Done')