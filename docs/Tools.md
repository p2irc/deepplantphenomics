"Tools" are stand-alone functions which provide useful functionality. They use pre-trained models and can be used out of the box without training or re-training.

## Vegetation Segmentation Network

The vegetation segmentation network can perform automatic segmentation of foreground pixels from background pixels. It outputs arrays which can be output to a file using a Python library like [Pillow](https://python-pillow.org/).

```
import deepplantphenomics as dpp
import numpy as np
from PIL import Image
import os

output_dir = './segmented-images'

my_files = ['one.png', 'two.png', 'three.png']

y = dpp.tools.segment_vegetation(images)

for i, img in enumerate(y):
    # Get original image dimensions
    org_filename = my_files[i]
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
```

## Rosette Leaf Counter

The rosette leaf counter provides an estimate of the number of leaves on a rosette plant using a pre-trained convolutional neural network.

```
import deepplantphenomics as dpp

my_files = ['one.png', 'two.png', 'three.png']

leaf_counts = dpp.tools.predict_rosette_leaf_count(my_files)
```

## Canola Flower Counter

The canola flower counter provides an estimate of the number of flowers in an image.

```
import deepplantphenomics as dpp

image_files = ['one.png', 'two.png', 'three.png']

y = dpp.tools.count_canola_flowers(image_files)

for k, v in zip(image_files, y):
    print('%s: %d' % (k, v))
```
