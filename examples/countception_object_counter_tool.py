#
# Demonstrates the use of tools.count_flowers on images of canola flowers.
# Loads image files from a folder.
#

import deepplantphenomics as dpp
import os

dir = './data/MBM_images'

images = [os.path.join(dir, name) for name in os.listdir(dir)]

print('Performing flower counting...')

y = dpp.tools.object_count_countception(images)

for k,v in zip(images, y):
    print('%s: %d' % (os.path.basename(k), v))

print('Done')