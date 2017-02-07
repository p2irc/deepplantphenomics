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

## Convolutional Layer

## Pooling Layer

## DropOut Layer

## Local Response Normalization Layer

## Output Layer

The output layer is the final layer in the network.

```
model.add_output_layer()
```

The number of units in this layer corresponds to the number of outputs - for example, the number of regression values, or the number of classes in the classification task.