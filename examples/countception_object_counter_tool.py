#
# Demonstrates the use of tools.count_flowers on images of canola flowers.
# Loads image files from a folder.
#

from .. import deepplantphenomics as dpp
import os

dir = './data/canola_flowers'

images = [os.path.join(dir, name) for name in os.listdir(dir)]

print('Performing flower counting...')

y = dpp.tools.count_flowers(images)

for k,v in zip(images, y):
    print('%s: %d' % (os.path.basename(k), v))

print('Done')