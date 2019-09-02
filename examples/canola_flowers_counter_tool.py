#
# Demonstrates the use of tools.count_flowers on images of canola flowers.
# Loads image files from a folder.
#

import deepplantphenomics as dpp
import os

m_path, _ = os.path.split(__file__)
re_path = '../deepplantphenomics/tests/test_data/test_countception_MBM_images'

dir = os.path.join(m_path, re_path)

images = [os.path.join(dir, name) for name in os.listdir(dir)]

print('Performing flower counting...')

y = dpp.tools.count_canola_flowers(images)

for k, v in zip(images, y):
    print('%s: %d' % (os.path.basename(k), v))

print('Done')
