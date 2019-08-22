DPP provides the model of count-ception object counter (https://arxiv.org/abs/1703.08710) for tasks like counting dense canola flowers in images. In count-ception model, a fully convolutional network with a small receptive field is used to run over the input image and counts the number of objects in its receptive field. A redundant count map is generated and then processed to get the total count of objects in the input image.


The overall structure and process of setting up and training a model is similar to other DPP models (see the [Leaf Counter training tutorial](Tutorial-Training-The-Leaf-Counter.md) for a detailed description of this). This tutorial largely covers the differences in model setup and data/label loading specific to training an countception object counter in DPP.

## Full Example

Below is a working example of training a count-ception object counter in DPP. 

```python
#
# Demonstrates the process of training a countception object counter in DPP.
#

import deepplantphenomics as dpp

model = dpp.CountCeptionModel(debug=True, save_checkpoints=False, report_rate=20)

# Setup and hyperparameters
model.set_loss_function('l1')
model.set_batch_size(2)
model.set_number_of_threads(4)
model.set_image_dimensions(300, 300, 3)

model.set_test_split(0.2)
model.set_validation_split(0.1)
model.set_learning_rate(0.0001)
model.set_weight_initializer('xavier')
model.set_maximum_training_epochs(1000)

# Load images and ground truth from a pickle file
model.load_dataset_from_pkl_file('MBM-dataset.pkl')

# Define the countception model architecture
model.use_predefined_model('countception')

# Begin training the countception model
model.begin_training()
```

## CountCeption Network Layers

The countception network consists of six convolutional layers and six parallel convolutional blocks. 
The counterception network can be created from layers using:

```python
patch_size = 32
model.add_input_layer()
model.add_convolutional_layer([3, 3, 3, 64], 1, 'lrelu', patch_size, True, 1e-5, 0.9)
model.add_paral_conv_block([1, 1, 0, 16], [3, 3, 0, 16])
model.add_paral_conv_block([1, 1, 0, 16], [3, 3, 0, 32])
model.add_convolutional_layer([14, 14, 0, 16], 1, 'lrelu', 0, True, 1e-5,0.9)
model.add_paral_conv_block([1, 1, 0, 112], [3, 3, 0, 48])
model.add_paral_conv_block([1, 1, 0, 64], [3, 3, 0, 32])
model.add_paral_conv_block([1, 1, 0, 40], [3, 3, 0, 40])
model.add_paral_conv_block([1, 1, 0, 32], [3, 3, 0, 96])
model.add_convolutional_layer([18, 18, 0, 32], 1, 'lrelu', 0, True, 1e-5, 0.9)
model.add_convolutional_layer([1, 1, 0, 64], 1, 'lrelu', 0, True, 1e-5, 0.9)
model.add_convolutional_layer([1, 1, 0, 64], 1, 'lrelu', 0, True, 1e-5, 0.9)
model.add_convolutional_layer([1, 1, 0, 1], 1, 'lrelu', 0, True, 1e-5,0.9)
```
Different from convolutional layers used in other networks, in countception each convolutional layer is followed immediately by a batch normalization layer. The last three parameters of add_convolutional_layer() method are used for setting up the batch normalization layer used in this way.

```python
add_convolutional_layer(self, filter_dimension, stride_length, activation_function,
                        padding=None, batch_norm=False, epsilon=1e-5, decay=0.9)
```
In current implementation of the countception network, a receptive field of 32 is used. For this setting, the first convolutional layer uses 3x3 filter and padding = 32, the second convolutional layer uses 14x14 filter and padding = 0 and the third convolutional layer uses 18x18 filter and padding = 0. To use other receptive fields, the above mentioned parameters should be modified and set properly. 

In each convolutional layer, the 'xavier' weight initializer and the 'LeakyReLu' activation function are used.

## Predefined CountCeption Network

Rather than having to create the layers yourself, the countception network is available as a predefined model in DPP. After configuring the model settings and loading in the dataset, the model layers can be setup using:

```python
model.use_predefined_model('countception')
```

## Data/Label Loading

For the countception network, image and ground truth data is loaded from a pickle file using:

```python
model.load_dataset_from_pkl_file()
```


