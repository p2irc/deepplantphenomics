### The [DPP paper](http://journal.frontiersin.org/article/10.3389/fpls.2017.01190/full) is now available online!

# Deep Plant Phenomics

Deep Plant Phenomics (DPP) is a platform for plant phenotyping using deep learning. Think of it as [Keras](https://keras.io/) for plant scientists.

DPP integrates [Tensorflow](https://www.tensorflow.org/) for learning and [PlantCV](http://plantcv.danforthcenter.org/) for image processing. This means that it is able to run on both CPUs and GPUs, and scale easily across devices.

Read the [doumentation](http://deep-plant-phenomics.readthedocs.io/en/latest/) for tutorials, or see the included examples.

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

- Several [pre-made networks](http://deep-plant-phenomics.readthedocs.io/en/latest/Tools/) for common plant phenotyping tasks.
- Automatic batching and input pipeline.
    - Loaders for some popular plant phenotyping datasets.
    - Plenty of [different loaders](http://deep-plant-phenomics.readthedocs.io/en/latest/Loaders/) for your own data, however it exists.
    - Predict classes, values, bounding boxes, or segmentations.
- Support for [semantic segmentation](http://deep-plant-phenomics.readthedocs.io/en/latest/Semantic-Segmentation/) with fully convolutional networks.
- Tensorboard integration.
- Easy-to-use API for building new models.
    - Several data augmentation options.
    - Many ready-to-use [neural network layers](http://deep-plant-phenomics.readthedocs.io/en/latest/Neural-Network-Layers/).
- Easy to [deploy](http://deep-plant-phenomics.readthedocs.io/en/latest/Tutorial-Deployment/) your own models as a Python function!

## Example Usage

Train a simple model to classify species:

```python
import deepplantphenomics as dpp

model = dpp.DPPModel(debug=True)

# 3 channels for colour, 1 channel for greyscale
channels = 3

# Setup and hyperparameters
model.set_batch_size(128)
model.set_image_dimensions(256, 256, channels)
model.set_learning_rate(0.001)
model.set_maximum_training_epochs(700)
model.set_train_test_split(0.75)

# Load dataset
model.load_dataset_from_directory_with_auto_labels('./data')

# Specify pre-processing steps
model.add_preprocessing_step('auto-segmentation')

# Simple convolutional neural network model
model.add_input_layer()

model.add_convolutional_layer(filter_dimension=[5, 5, channels, 32], stride_length=1, activation_function='relu')
model.add_pooling_layer(kernel_size=3, stride_length=2)

model.add_convolutional_layer(filter_dimension=[5, 5, 32, 32], stride_length=1, activation_function='relu')
model.add_pooling_layer(kernel_size=3, stride_length=2)

model.add_convolutional_layer(filter_dimension=[5, 5, 32, 64], stride_length=1, activation_function='relu')
model.add_pooling_layer(kernel_size=3, stride_length=2)

model.add_fully_connected_layer(output_size=256, activation_function='relu')

model.add_output_layer()

# Train!
model.begin_training()
```

## Installation

0. The package should work on Python 2.7 or 3.x, but if using 2.7 you need the [enum34](https://pypi.python.org/pypi/enum34) package installed.
1. Install the following dependencies, following the directions provided according to your platform and requirements:
    - [Tensorflow](https://www.tensorflow.org/) (1.0 or later)
    - [PlantCV](http://plantcv.danforthcenter.org/) (Only required for the `auto-segmentation` preprocessor)
3. `git clone https://github.com/p2irc/deepplantphenomics.git` 
4. `python setup.py install`

## Downloading Pre-trained Networks

The package uses [Git Large File Storage](https://git-lfs.github.com/) (git-lfs) to handle the saved network states included in this repository, as they can sometimes be very large.

If you had git-lfs installed when you installed the packages, then you automatically downloaded the saved networks. If you want to download the states after installing the package, then install git-lfs and run ``git lfs fetch`` and then ``git lfs pull``.

## Contributing

Contributions are always welcome. If you would like to make a contribution, please fork from the develop branch.

## Help

If you are interested in research collaborations or want more information regarding this package, please email `jordan.ubbens@usask.ca`.

If you have a feature request or bug report, please open a new issue.
