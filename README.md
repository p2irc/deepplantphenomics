# Deep Plant Phenomics

Deep Plant Phenomics (DPP) is a platform for plant phenotyping using deep learning. Think of it as [Keras](https://keras.io/) for plant scientists.

DPP integrates [Tensorflow](https://www.tensorflow.org/) for learning. This means that it is able to run on both CPUs and GPUs, and scale easily across devices.

Read the [doumentation](http://deep-plant-phenomics.readthedocs.io/en/latest/) for tutorials, or see the included examples. You can also read the [paper](http://journal.frontiersin.org/article/10.3389/fpls.2017.01190/full).

DPP is maintained at the [Plant Phenotyping and Imaging Research Center (P2IRC)](http://p2irc.usask.ca/) at the [University of Saskatchewan](https://www.usask.ca/). 🌾🇨🇦

## What's Deep Learning?

Principally, DPP provides deep learning functionality for plant phenotyping and related applications. Deep learning is a category of techniques which encompasses many different types of neural networks. Deep learning techniques lead the state of the art in many image-based tasks, including image classification, object detection and localization, image segmentation, and others.

## What Can I Do With This?

This package provides two things:

### 1. Useful tools made possible using pre-trained neural networks

For example, calling `tools.predict_rosette_leaf_count(my_files)` will use a pre-trained convolutional neural network to estimate the number of leaves on each rosette plant.

### 2. An easy way to train your own models

For example, using a few lines of code you can easily use your data to train a convolutional neural network to rate plants for biotic stress. See the [tutorial](http://deep-plant-phenomics.readthedocs.io/en/latest/Tutorial-Training-The-Leaf-Counter/) for how the leaf counting model was built.

## Features

- Several [trained networks](http://deep-plant-phenomics.readthedocs.io/en/latest/Tools/) for common plant phenotyping tasks.
- Easy ways to load data.
    - Loaders for some popular plant phenotyping datasets.
    - Plenty of [different loaders](http://deep-plant-phenomics.readthedocs.io/en/latest/Loaders/) for your own data, however it exists.
- Support for [semantic segmentation](http://deep-plant-phenomics.readthedocs.io/en/latest/Semantic-Segmentation/).
- Support for [object detection](http://deep-plant-phenomics.readthedocs.io/en/latest/Tutorial-Training-An-Object-Detector).
- Support for object counting via [density estimation](http://deep-plant-phenomics.readthedocs.io/en/latest/Tutorial-Object-Counting-with-Heatmaps), including [Countception networks](http://deep-plant-phenomics.readthedocs.io/en/latest/Tutorial-Object-Counting-with-Countception/).
- Support for classification and [regression](http://deep-plant-phenomics.readthedocs.io/en/latest/Tutorial-Training-The-Leaf-Counter) tasks.
- Tensorboard integration for visualization.
- Easy-to-use API for building new models.
    - [Pre-defined neural network architectures](http://deep-plant-phenomics.readthedocs.io/en/latest/Predefined-Model-Architectures) so you don't have to make your own.
    - Several data augmentation options.
    - Many ready-to-use [neural network layers](http://deep-plant-phenomics.readthedocs.io/en/latest/Neural-Network-Layers/).
- Easy to [deploy](http://deep-plant-phenomics.readthedocs.io/en/latest/Tutorial-Deployment/) your own models as a Python function!

## Example Usage

Train a simple regression model:

```python
import deepplantphenomics as dpp

model = dpp.RegressionModel(debug=True)

# 3 channels for colour, 1 channel for greyscale
channels = 3

# Setup and hyperparameters
model.set_batch_size(64)
model.set_image_dimensions(256, 256, channels)
model.set_maximum_training_epochs(25)
model.set_test_split(0.2)
model.set_validation_split(0.0)

# Load dataset of images and ground-truth labels
model.load_multiple_labels_from_csv('./data/my_labels.csv')
model.load_images_with_ids_from_directory('./data')

# Use a predefined model
model.use_predefined_model('vgg-16')

# Train!
model.begin_training()
```

## Installation

1. `git clone https://github.com/p2irc/deepplantphenomics.git`
2. `pip install ./deepplantphenomics`

**Note**: The package now requires Python 3.6 or greater. Python 2.7 is no longer supported.

## Contributing

Contributions are always welcome. If you would like to make a contribution, please fork from the develop branch.

## Help

If you are interested in research collaborations or want more information regarding this package, please email `jordan.ubbens@usask.ca`.

If you have a feature request or bug report, please open a new issue.
