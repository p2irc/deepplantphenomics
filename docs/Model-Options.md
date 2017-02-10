# Model Options

This page details the different options which are available in DPP for training.

## Multithreading Options

```
set_number_of_threads()
```

Set number of threads for input queue runners and preprocessing tasks. Using more threads won't accelerate training or inference, but if you're using a GPU then you should make sure that you're using enough threads that no single thread is running at 100% load if possible.

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

Set the optimization algorithm to use. Default is `'Adam'`. Other options are `'SGD'` (Stochastic Gradient Descent), `'Adadelta'`, and `'Adagrad'`, 

```
set_learning_rate_decay(decay_factor, epochs_per_decay)
```

Manually anneal the learning rate every `epochs_per_decay` epochs. This isn't necessary for gradient-adaptive optimizers like `'Adam'`.

```
set_weight_initializer()
```

Set the weight initialization scheme for convolutional and fully connected layers. Default is `'xavier'`, other option is `'normal'`.

```
set_problem_type()
```

Set the type of problem for the model. Default is `'classification'`, other option is `'regression'`.

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
set_augmentation_crop(True)
```

Randomly crop images during training, and crop images to center during testing.

```
set_augmentation_brightness_and_contrast(True)
```

Randomly adjust contrast and/or brightness on training images.