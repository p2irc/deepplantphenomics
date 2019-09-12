## Input Layer

The first layer which needs to be added to the model is an input layer.

```
model.add_input_layer()
```

This input layer represents the input values to the network.

## Fully Connected Layer

Fully connected layers can be added with a specified output size, corresponding to the number of units in the layer.

```
model.add_fully_connected_layer(output_size=64, activation_function='relu')
```

The optional `activation_function` parameter specifies the nonlinear activation function (or *transfer function*) to apply (`'relu'` or `'tanh'`).

## Convolutional Layer

Convolutional layers are specialized network layers which are composed of filters applied in strided convolutions. The output of a convolutional layer is a volume, unlike a normal fully connected layer.

```
model.add_convolutional_layer(filter_dimension=[5, 5, 3, 32], stride_length=1, activation_function='relu')
```

The filter dimension is in the order `[x_size, y_size, depth, num_filters]` where `x_size` and `y_size` are spatial dimensions, `depth` is the full depth of the input volume, and `num_filters` is the desired number of filters in this layer. The output of this layer will be spatially smaller if a `stride_length` > 1 is used, and will always have a depth of `num_filters`.

Replication padding is used at the boundaries if no explicit padding is given, otherwise the given explicit padding will be used.

## Parallel Convolutional Block

The parallel convolutional block consists of two parallel convolutional layers of different filter dimensions. The input of this block goes to both convolutional layers and the outputs of the two convolutional layers are stacked together to form the output of this block.

```
model.add_paral_conv_block(filter_dimension_1, filter_dimension_2)
```

Both filter_dimension_1 and filter_dimension_2 are in the order `[x_size, y_size, depth, num_filters]`.

## Pooling Layer

The pooling layer spatially downsamples an input volume using max pooling. These are typically used following convolutional layers to decrease spatial resolution.

```
model.add_pooling_layer(kernel_size=3, stride_length=2, pooling_type='max')
```

The `kernel_size` parameter specifies the spatial diameter of the downsampling operation. For example, if max pooling is used with `kernel_size=3`, then the value at a particular position is the maximum of the 3x3 neighbourhood centered at that position.

The optional `pooling_type` parameter specifies the type of pooling operation, which defaults to `'max'` for max pooling but can also be set to `'avg'` for average pooling.

## Upsampling Layer

This layer is also often referred to as a deconvolutional layer or a convolutional transpose layer because of how the upsampling is performed.

The upsampling layer increases an input volume through the use of upsampling. The primary use of these layers is to increase the spatial resolution back to the original dimensions of the image after the use of pooling layers. For example, when performing semantic segmentation the final layer needs to evaluate the cost function pixel-wise and thus requires the spatial resolution to be the same as the original input image. In such an example, if you want to make use of max pooling you will need upsampling before the final layer in order to ensure the dimensions are matching.

```
model.add_upsampling_layer(filter_size=3, num_filters=32, upscale_factor=2)
```

The `filter_size` parameter defines the height and width of the filter performing the upsampling. The `num_filters` parameter represents how many filters this layer will have; this will define the depth of the output dimensions. The `upscale_factor` represents how much the height and width of the image will be scaled.

## DropOut Layer

The DropOut layer implements the DropOut operation (Srivastava et al. 2014), typically following fully connected layers.

```
model.add_dropout_layer(0.5)
```

The only parameter to this layer is `p`, indicating the "keep probability". Unit activations are randomly set to zero during training with probability (*1-p*). This operation serves to downsample the network during training to help prevent overfitting.

## Local Response Normalization Layer

This layer applies the [local response normalization](https://www.tensorflow.org/api_docs/python/nn/normalization#local_response_normalization) (Krizhevsky et al. 2012) operation to inputs. It has no parameters.

```
model.add_local_response_normalization_layer()
```

## Batch Normalization Layer

This layer applies batch normalization (Ioffe & Szegedy, 2015) to the activations of the previous layer. This is an alternative to the pre-activation batch normalization offered by the convolutional layer.

Note that using batch normalization may be detrimental to some regression problems. You should always try your network without batch norm before adding it in and re-tuning your hyperparameters.

```
model.add_batch_norm_layer()
```

## Residual/Skip Connection

This layer creates a residual connection that skips over blocks of layers in the network (He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition*. In Proceedings of the IEEE conference on computer vision and pattern recognition).

The first skip connection layer will have no effect. The second skip connection adds the output from the previous layer with the output at the location of the first skip connection. The third skip connection does the same with the output at the location of the second skip connection, and so on.

```
model.add_skip_connection(downsampled=False)
```

The `downsampled` flag is used when the passed values from a previous skip connection form a larger volume than the volume it needs to be added to. When `downsampled` is `True`, a 1x1 linear filter with a stride of 2 is used to spatially downsample the network by half, while doubling the depth dimension.

## Output Layer

The output layer is the final layer in the network.

```
model.add_output_layer()
```

The number of units in this layer corresponds to the number of outputs - for example, the number of regression values, or the number of classes in the classification task. If the task being performed is semantic segmentation, then the output size is the same width and height as the input.

The `output_size` parameter is optional and only used in rare cases where you want to override the calculated output size - for example, when the number of classes is not known because the dataset has not been loaded yet.
