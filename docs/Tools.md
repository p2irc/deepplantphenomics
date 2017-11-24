"Tools" are stand-alone functions which provide useful functionality. They use pre-trained models and can be used out of the box without training or re-training.

## Vegetation Segmentation Network

The vegetaion segmentation network can perform automatic segmentation of foreground pixels from background pixels. It outputs arrays which can be output to a file using a Python library like [Pillow](https://python-pillow.org/).

```
import deepplantphenomics as dpp
import numpy
from PIL import Image
import os

output_dir = './segmented-images'

my_files = ['one.png', 'two.png', 'three.png']
y = dpp.tools.segment_vegetation(images)

for i, img in enumerate(y):
    filename = os.path.join(output_dir, os.path.basename(images[i]))
    result = Image.fromarray((img * 255).astype(numpy.uint8))
    result.save(filename)

print('Done')
```

## Rosette Leaf Counter

The rosette leaf counter provides an estimate of the number of leaves on a rosette plant using a pre-trained convolutional neural network.

```
import deepplantphenomics as dpp

my_files = ['one.png', 'two.png', 'three.png']

leaf_counts = dpp.tools.predict_rosette_leaf_count(my_files)
```

## Arabidopsis Strain (Mutant) Classifier

The strain classsifier is used to classify the species/strain/mutant using top-down images of arabidopsis thaliana rosettes.

```
import deepplantphenomics as dpp

my_files = ['one.png', 'two.png', 'three.png']

species = dpp.tools.classify_arabidopsis_strain(my_files)
```