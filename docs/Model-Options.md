## Multithreading Options

```
set_number_of_threads()
```

Set number of threads for input queue runners and preprocessing tasks. Using more threads won't accelerate training or inference, but if you're using a GPU then you should make sure that you're using enough threads that no single thread is running at 100% load if possible.

Note that all pre-trained networks operate with only one thread to avoid random orderings due to threading.

## Learning Hyperparameters

```
set_batch_size()
```

Sets the number of examples in each mini-batch. Recall that smaller batches mean more gradient updates per epoch.

```
set_maximum_training_epochs()
```

Sets the number of epochs to train to before stopping. An epoch is one full cycle through the entire training set.

```
set_learning_rate()
```

Set the initial learning rate. If you're not sure what learning rate is appropriate, err on the side of a smaller learning rate.

```
set_optimizer()
```

Set the optimization algorithm to use. Default is `'Adam'`. Other options are `'SGD'` (Stochastic Gradient Descent), `'Adadelta'`, and `'Adagrad'`.

```
set_learning_rate_decay(decay_factor, epochs_per_decay)
```

Manually anneal the learning rate every `epochs_per_decay` epochs. This isn't necessary for gradient-adaptive optimizers like `'Adam'`.

```
set_regularization_coefficient()
```

Set the coefficient for L2 weight decay (regularization).

```
set_weight_initializer()
```

Set the weight initialization scheme for convolutional and fully connected layers. Default is `'normal'`, other option is `'xavier'`. Note that you may experience gradient problems with relu activations and xavier initialization.

```
set_problem_type()
```

Set the type of problem for the model. Default is `'classification'`, other options are `'regression'` or `'semantic_segmentation'` (really just pixel-wise regression with a fully convolutional network, but useful for segmentation applications. See [semantic segmentation](/Semantic-Segmentation/)).

```
set_train_test_split()
```

Set the ratio of training samples to testing samples.

## Input Options

```
set_image_dimensions(image_height, image_width, image_depth)
```

Specify the image dimensions for images in the dataset (depth is the number of channels). This can be the dimensions you want to resize to if you're using `set_resize_images()` or `set_crop_or_pad_images()`.

```
set_original_image_dimensions(image_height, image_width)
```

Specify the original size of the image, before resizing. This is only needed in special cases, for instance, if you are resizing input images but using image coordinate labels which reference the original size.

```
add_preprocessor()
```

Add pre-processors. For more information see the documentation for pre-processors.

```
set_crop_or_pad_images(True)
```

Resize images by cropping or padding, as opposed to plain resizing.

```
set_resize_images(True)
```

Up-sample or down-sample images to specified size.

```
set_processed_images_dir()
```

Set the location to save processed images when pre-processing is used.

## Data Augmentation Options

```
set_augmentation_flip_horizontal(True)
```

Randomly flip training images horizontally.

```
set_augmentation_flip_vertical(True)
```

Randomly flip training images vertically.

```
set_augmentation_crop(True, crop_ratio)
```

Randomly crop images during training, and crop images to center during testing. The size of the crop is specified by `crop_ratio`, and defaults to `0.75`, or 75% of the original image.

```
set_augmentation_brightness_and_contrast(True)
```

Randomly adjust contrast and/or brightness on training images.


```
load_training_augmentation_dataset_from_directory_with_csv_labels(dirname, labels_file, column_number, id_column_number)
```

Load a second set of images with corresponding labels in a csv file to augment the training set with. This is a good option if your chosen augmentation is not listed above - you can create the augmented examples yourself and load them with this function. `column_number` should have the label and `id_column_number` should have the filename.