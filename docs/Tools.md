"Tools" are stand-alone functions which provide useful functionality.

## Rosette Leaf Counter

The rosette leaf counter provides an estimate of the number of leaves on a rosette plant using a pre-trained convolutional neural network.

```
import deepplantphenomics as dpp

my_files = ['one.png', 'two.png', 'three.png']

leaf_counts = dpp.tools.predict_rosette_leaf_count(my_files)
```