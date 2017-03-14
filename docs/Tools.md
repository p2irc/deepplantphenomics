"Tools" are stand-alone functions which provide useful functionality. They use pre-trained models and can be used out of the box without training or re-training.

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