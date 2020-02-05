## Multithreading and Multi-GPU Options

```
set_number_of_threads(1)
```

Set the number of threads for preprocessing tasks. Using more threads won't accelerate training or inference, but if you're using a GPU then you should make sure that you're using enough threads that no single thread is running at 100% load if possible.

Note that all pre-trained networks operate with only one thread to avoid random orderings due to threading.

```
set_number_of_gpus(1)
```

Sets the number of GPUs to use for model training. This should be set to at least 1. Setting it higher than the number of GPUs available is equivalent to setting it to exactly that number (i.e. setting it to 4 with 2 GPUs will set it to 2).

Using 2+ GPUs can make model training faster, provided that the model is complex enough for the GPU operations to be slower than the overhead from transferring data to/from the GPUs. Otherwise, the speedup from a multi-GPU setup will be overshadowed by the time required to move data between the GPUs and system memory.

Setting this after setting the batch size will also check whether batches can be evenly split across the desired number of GPUs; an error is raised if they can't be evenly split.

## Learning Hyperparameters
#### All Models

```
set_batch_size()
```

Sets the number of examples in each mini-batch. Defaults to 1. Recall that smaller batches mean more gradient updates per epoch.

Setting this after setting the number of GPUs for multi-GPU training will also check whether the batch size can be evenly split across the current number of GPUs; an error is raised if they can't be evenly split.

```
set_maximum_training_epochs()
```

Sets the number of epochs to train to before stopping. An epoch is one full cycle through the entire training set.

```
set_learning_rate()
```

Set the initial learning rate. Defaults to 0.001. If you're not sure what learning rate is appropriate, err on the side of a smaller learning rate.

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

Set the weight initialization scheme for convolutional and fully connected layers. Default is `'xavier'`, other option is `'normal'`. Note that you may experience gradient problems with relu activations and xavier initialization.

```
set_test_split()
```

Set the ratio of the total number of samples to use as a testing set after training. Defaults to 0.10 (i.e. 10% of the samples).

```
set_validation_split()
```

Set the ratio of the total number of samples to use as a validation set during training. Defaults to 0.10 (i.e. 10% of the samples).

```
force_split_shuffle()
```

Sets whether to force shuffling of a loaded dataset into train, test, and validation partitions. These partitions are shuffled and saved the first time a dataset is used for training. By default, this is turned off and subsequent training runs load and reuse this partitioning, to avoid leaking data from the initially selected training and validation sets into the test set, and vice versa.

```
set_gen_data_overwrite()
```

Sets the treatment of generated data like image patches and heatmaps. If true, existing generated data is overwritten. If false, existing generated data will be checked for and loaded when possible.

```
set_random_seed()
```

Sets an integer seed for random operations during training (shuffling, augmentations, etc.). This is used to make training results reproducible across multiple attempts at training a model as a support in testing and debugging them.

```
set_loss_function()
```

Sets the loss function to be used by the model during training and testing. The supported loss functions vary with the specific problem type/`Model`:

- `ClassificationModel`: `softmax cross entropy` only
- `RegressionModel`: `l2`, `l1`, and `smooth l1`
- `SemanticSegmentationModel`: `sigmoid cross entropy` and `softmax cross entropy`
- `ObjectDetectionModel`: `yolo` only
- `CountCeptionModel`: `l1` only
- `HeatmapObjectCountingModel`: `l2`, `l1`, and `smooth l1`

#### Regression Models Only

```
set_num_regression_outputs()
```

Sets the number of response variables for the regression model.

#### Semantic Segmentation Models Only

```
set_num_segmentation_classes()
```

Sets the number of per-pixel classes to segment images into. This defaults to 2 (for binary segmentation) and should be at least 2. The loss function will also be set automatically based on whether the segmentations will be binary or multi-class.


#### Object Detection Models Only

```
set_yolo_parameters(grid_size=[7,7],
                    class_list=['plant'], 
                    anchors=[[159, 157], [103, 133], [91, 89], [64, 65], [142, 101]])
```

Sets several parameters needed for the Yolo-based object detector.

- Yolo splits images into a grid and makes bounding box predictions in each grid square.`grid_size` defines the number of grid squares along the image width and height. Defaults to [7,7].
- `class_list` is a list of names for possible object classes in images. This defaults to a single 'plant' class. (DPP currently only supports one class at a time)
- `anchors` defines the widths and heights of anchors/prior boxes which the bounding box predictions use as a basis for detecting objects of various sizes and aspect ratios. The five anchors listed above are the default values and should be fine for most detectors.

```
set_yolo_thresholds(thresh_sig=0.6, 
                    thresh_overlap=0.3, 
                    thresh_correct=0.5)
```

Sets the Intersection-over-Union (IoU) thresholds internally used by the YOLO model to detect objects and calculate average precision. `thresh_sig` controls the minimum IoU for taking a detection as significant, `thresh_overlap` controls the minimum IoU for overlapping detections (at which point only the more confidant one is taken), and `thresh_correct` controls the minimum IoU for saying a detection is correct during validation and testing.

#### Heatmap Object Counting Models Only

```
set_density_map_sigma(sigma)
```

Sets the standard deviation used for gaussians when generating ground truth heatmaps from point locations of objects. See [the heatmap dataset loader](Loaders.md) for more info.

## Input Options

```
set_image_dimensions(image_height, image_width, image_depth)
```

Specify the image dimensions for images in the dataset, taking depth as the number of channels. These will be the dimensions you want to resize to if you're using `set_resize_images()` or `set_crop_or_pad_images()`.

```
set_original_image_dimensions(image_height, image_width)
```

Specify the original size of the image, before resizing. This is only needed in special cases; for instance, you would use this if you are resizing input images but using image coordinate labels which reference the original size.

```
set_crop_or_pad_images(True)
```

Resize images by either cropping or padding them, as opposed to plain resizing.

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
set_augmentation_crop(True, crop_ratio)
```

Randomly crop images during training, and crop images to their center during testing. The size of the crop is specified by `crop_ratio`, and defaults to `0.75` (i.e. 75% of the original image).

```
set_augmentation_brightness_and_contrast(True)
```

Randomly adjust the contrast and/or brightness on training images.

```
set_augmentation_rotation(True, crop_borders=False)
```

Randomly rotate training images by any angle within 0-360 degrees (for classification and regression tasks). Parts of the image may get rotated outside of the image after rotation. By default, this will also leave black borders around the image. Setting `crop_borders` to `True` will perform a centre crop to remove the black borders generated by rotation.

**A warning on using centre cropping with rotation augmentation:** In order to maintain similar feature scales between images, the cropping uses the tightest possible crop required for any given image to remove black borders (i.e. the crop required for 45 degree rotation). This will crop out at least 50% of the image (more for higher aspect ratios). If using this, ensure that the main content of the images is in the centre.


```
load_training_augmentation_dataset_from_directory_with_csv_labels(dirname, labels_file, column_number, id_column_number)
```

Load a second set of images with corresponding labels in a csv file to augment the training set with. This is a good option if your chosen augmentation is not listed above - you can create the augmented examples yourself and load them with this function. `column_number` should have the label and `id_column_number` should have the filename.

```
set_patch_size(height=128, width=128)
```

When loading datasets for semantic segmentation, heatmap counting, or object detection, this enables automatic patching of the input image dataset in order. This facilitates training models on large images that won't fit into memory during training. In this case, the patch size and image size should match.

When running inference on trained models, this splits the inference images into patches, runs a forward pass on the patches, and either stitches the full image back together (for semantic segmentation and heatmap counting) or returns predictions for the patches instead (for object detection).

See [this page](Automatic-Image-Patching.md) for more info about this automatic patching.
