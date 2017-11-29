#
# Demonstrates the use of tools.segment_vegetation on images of rosettes.
#

import deepplantphenomics as dpp
import numpy as np
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
    # Get original image dimensions
    org_filename = images[i]
    org_img = Image.open(org_filename)
    org_width, org_height = org_img.size
    org_array = np.array(org_img)

    # Resize mask
    mask_img = Image.fromarray((img * 255).astype(np.uint8))
    mask_array = np.array(mask_img.resize((org_width, org_height))) / 255

    # Apply mask
    img_seg = np.array([org_array[:,:,0] * mask_array, org_array[:,:,1] * mask_array, org_array[:,:,2] * mask_array]).transpose()

    # Write output file
    filename = os.path.join(output_dir, os.path.basename(images[i]))
    result = Image.fromarray(img_seg.astype(np.uint8))
    result.save(filename)

print('Done')