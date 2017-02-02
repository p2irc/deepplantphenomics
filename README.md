# Deep Plant Phenomics

Deep Plant Phenomics (DPP) is an integrated platform for plant phenotyping using deep learning. Think of it as [Keras](https://keras.io/) for plant scientists.

DPP integrates [Tensorflow](https://www.tensorflow.org/) for learning and [PlantCV](http://plantcv.danforthcenter.org/) for image processing. This means that it is able to run on both CPUs and GPUs, and scale easily across devices.

DPP is maintained at the [Plant Phenotyping and Imaging Research Center (PIRC)](http://p2irc.usask.ca/) at the [University of Saskatchewan](https://www.usask.ca/). 🌾🇨🇦

# What's Deep Learning?

Principally, DPP provides deep learning functionality for plant phenotyping and related applications. Deep learning is a category of techniques which encompasses many different types of neural networks. Deep learning techniques lead the state of the art in many image-based tasks, including image classification, object detection and localization, image segmentation, and others.

# What Can I Do With This?

This package provides two things:

### 1. Useful tools made possible using pre-trained neural networks

For example, calling `addPreprocessingStep('auto-segmentation')` will use a pre-trained bounding box regression network to segment the plant from the background.

### 2. An easy way to train your own models

For example, using a few lines of code you can easily use your data to train a convolutional neural network to rate plants for the presence of disease.

# Example Usage

Train a simple model to classify species:

```python
import deepplantphenomics as dpp

model = dpp.DPPModel(debug=True)

# 3 channels for colour, 1 channel for greyscale
channels = 3

# Setup and hyperparameters
model.setBatchSize(128)
model.setImageDimensions(256, 256, channels)
model.setLearningRate(0.001)
model.setMaximumTrainingEpochs(700)
model.setTrainTestSplit(0.75)

# Load dataset
model.loadDatasetFromDirectoryWithAutoLabels('./data')

# Specify pre-processing steps
model.addPreprocessingStep('auto-segment')

# Simple convolutional neural network model
model.addInputLayer()

model.addConvolutionalLayer(filter_dimension=[5, 5, channels, 32], stride_length=1, activation_function='relu')
model.addPoolingLayer(kernel_size=3, stride_length=2)

model.addConvolutionalLayer(filter_dimension=[5, 5, 32, 32], stride_length=1, activation_function='relu')
model.addPoolingLayer(kernel_size=3, stride_length=2)

model.addConvolutionalLayer(filter_dimension=[5, 5, 32, 64], stride_length=1, activation_function='relu')
model.addPoolingLayer(kernel_size=3, stride_length=2)

model.addFullyConnectedLayer(output_size=256, activation_function='relu')

model.addOutputLayer()

# Train!
model.beginTraining()
```

# Installation

1. Install the following dependencies, following the directions provided according to your platform and requirements:
    - [Tensorflow](https://www.tensorflow.org/) (R0.12 or later)
    - [PlantCV](http://plantcv.danforthcenter.org/)
    - [joblib](https://pythonhosted.org/joblib/installing.html) 
3. `git clone https://github.com/jubbens/deepplantphenomics.git` 
4. Test installation using `python -c 'import deepplantphenomics'`

# Contributing

Contributions are always welcome. If you would like to make a contribution, please fork from the develop branch.

# Help

If you are interested in research collaborations or want more information regarding this package, please email `jordan.ubbens@usask.ca`.

If you have a feature request or bug report, please open a new issue.