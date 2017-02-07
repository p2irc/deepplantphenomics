# Neural Network Layers

DPP offers several different layers which can be stacked together to build models.

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

The optional `activation_function` parameter specifies the nonlinear activation (or *transfer*) function to apply.

## Convolutional Layer

Convolutional layers are specialized network layers which are composed of filters applied in strided convolutions. The output of a convolutional layer is a volume, unlike a normal fully connected layer.

```
model.add_convolutional_layer(filter_dimension=[5, 5, 3, 32], stride_length=1, activation_function='relu')
```

The filter dimension is in the order `[x_size, y_size, depth, num_filters]` where `x_size` and `y_size` are spatial dimensions, `depth` is the full depth of the input volume, and `num_filters` is the desired number of filters in this layer. The output of this layer will be spatially smaller if a `stride_length` > 1 is used, and will always have a depth of `num_filters`.

Replication padding is used at the boundaries. 

## Pooling Layer

The pooling layer spatially downsamples an input volume using max pooling. These are typically used following convolutional layers to decrease spatial resolution.

```
model.add_pooling_layer(kernel_size=3, stride_length=2)
```

The `kernel_size` parameter specifies the spatial diameter of the downsampling operation. For example, if max pooling is used with `kernel_size=3`, then the value at a particular position is the maximum of the 3x3 neighbourhood centered at that position.

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

## Output Layer

The output layer is the final layer in the network.

```
model.add_output_layer()
```

The number of units in this layer corresponds to the number of outputs - for example, the number of regression values, or the number of classes in the classification task.