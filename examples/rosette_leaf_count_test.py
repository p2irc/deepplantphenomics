#
# Demonstrates the use of tools.predict_rosette_leaf_count on images of rosettes.
# Loads filenames from the IPPN dataset by default.
#

import deepplantphenomics as dpp
import os

dir = './data/Ara2013-Canon'

images = [os.path.join(dir, name) for name in os.listdir(dir) if
          os.path.isfile(os.path.join(dir, name)) & name.endswith('_rgb.png')]

# Sort so the outputs match the order in the labels file
images = sorted(images)

print('Performing leaf estimation...')

y = dpp.tools.predict_rosette_leaf_count(images)

for k,v in zip(images, y):
    print('%s: %d' % (os.path.basename(k), v))

print('Done')