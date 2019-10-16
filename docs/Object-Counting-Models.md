DPP provides two different models for counting objects in images: an implementation of the Countception model and a more generic object counter that predicts heatmaps for object locations.

### Countception Object Counting

The Countception model uses a fully convolutional network with a small receptive field to run over images and count the number of objects. This results in a redundant count map, which is then processed to get the total number of objects in the image.

DPP provides the Countception model through a model class and a predefined network. The `CountCeptionModel` provides the necessary loaders, graph constructors, and forward pass functions. The network required, meanwhile is provided as the `countception` option to `set_predefined_model()`. More details are in [the specific tutorial](Tutorial-Object-Counting-with-Countception.md).

### Heatmap Object Counting

The heatmap object counter, meanwhile, uses an internal structure similar to that of semantic segmentation to predict a heatmap of object locations. The ground truth heatmaps used as inputs are a grayscale image of 2D gaussian distributions at every object location in the corresponding image. Sums over the heatmap labels and predictions should then give the number of objects in the image.

This more general object counter is provided via the `HeatmapObjectCountingModel` class. More details on its settings and loaders can be found in [its own tutorial](Tutorial-Object-Counting-with-Heatmaps.md).