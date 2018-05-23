## Compression

DeepPlantPhenomics offers compression similar to [DeepCompression](https://arxiv.org/abs/1510.00149).

The compression pipeline consists of 2 steps: pruning and quantization, and is to be combined with `gzip` 
or a similar compression tool in order to achieve the highest level of compression possible. Pruning is 
iterative, so the more compression runs that are made the longer the compression takes to run.

**NOTE** Due to a bug in tensorflow, quantization is not yet possible.

The following will build a fully compressed model.

```
import deepplantphenomics as dpp

model = dpp.DPPModel(debug=True, save_checkpoints=True, tensorboard_dir='./tensorlogs', report_rate=1000)
# 3 channels for colour, 1 channel for greyscale
channels = 3

# Setup and hyperparameters
model.set_batch_size(4)
model.set_number_of_threads(8)
model.set_image_dimensions(128, 128, channels)
model.set_resize_images(True)
model.set_problem_type('classification')
model.set_train_test_split(0.8)
model.set_learning_rate(0.0001)
model.set_weight_initializer('xavier')
model.set_maximum_training_epochs(200)

# Augmentation options
# model.set_augmentation_brightness_and_contrast(True)
# model.set_augmentation_flip_horizontal(True)
# model.set_augmentation_flip_vertical(True)
# model.set_augmentation_crop(True)


# Load all data for IPPN leaf counting dataset
model.load_ippn_dataset_from_directory('./data/Plant_Phenotyping_Datasets/Plant/Ara2013-Canon/')

# Define a model architecture
model.add_input_layer()

model.add_convolutional_layer(filter_dimension=[5, 5, channels, 32], stride_length=1, activation_function='tanh')
model.add_pooling_layer(kernel_size=3, stride_length=2)

model.add_convolutional_layer(filter_dimension=[5, 5, 32, 64], stride_length=1, activation_function='tanh')
model.add_pooling_layer(kernel_size=3, stride_length=2)

model.add_convolutional_layer(filter_dimension=[3, 3, 64, 64], stride_length=1, activation_function='tanh')
model.add_pooling_layer(kernel_size=3, stride_length=2)

model.add_fully_connected_layer(output_size=512, activation_function='relu')
model.add_dropout_layer(0.6)
model.add_fully_connected_layer(output_size=256, activation_function='relu')
model.add_output_layer()
# Begin training the regression model
model.begin_training(shut_down=False)

model.compress(5, quantize=True, debug=True)

```

This will create a protobuf file, `quantized.pb`, that then must be converted into a tensorflow lite quantized model

```
bazel-bin/tensorflow/contrib/lite/toco/toco --input_file=../quantized.pb --output_file=model.tflite --input_format TENSORFLOW_GRAPHDEF --output_format=TFLITE --inferfence_type=QUANTIZED_UINT8 --input_shape="1, 128, 128, 3" --input_array=conv1 --output_array=output_weights --std_value=127.5 --mean=127.5
```
More information about tensorflow lite and compression can be found [here](https://www.tensorflow.org/performance/quantization<Paste>a).

If the quantization step is skipped (`quantize=False`) the model will still be pruned, and will need to be gzipped in order 
to observe any actual compression.
