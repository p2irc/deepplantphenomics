DPP provides two different techniques that can be used for object counting. One of those models is an object counter that predicts heatmaps of object locations (also known as *density estimation*).

The structure and process of training a heatmap-based object counter is similar to other models (see the [Leaf Counter training tutorial](Tutorial-Training-The-Leaf-Counter.md) for more details). This mostly covers the settings and data loading differences for heatmap object counters.

### Full Example

The below code is a full working example of how to train an object counting using heatmaps.

```python
import deepplantphenomics as dpp

model = dpp.HeatmapObjectCountingModel(debug=True, load_from_saved=False)

# 3 channels for colour, 1 channel for greyscale
channels = 3

# Setup and hyper-parameters
model.set_image_dimensions(128, 128, channels)
model.set_batch_size(32)

model.set_learning_rate(0.0001)
model.set_maximum_training_epochs(25)
model.set_test_split(0.75)
model.set_validation_split(0.0)

# Load dataset
model.set_density_map_sigma(4.0)
model.load_heatmap_dataset_with_csv_from_directory('./data', 'point_labels.csv')

# Define a model architecture
model.add_input_layer()

model.add_convolutional_layer(filter_dimension=[3, 3, 3, 16], stride_length=1, activation_function='relu')
model.add_convolutional_layer(filter_dimension=[3, 3, 16, 32], stride_length=1, activation_function='relu')
model.add_convolutional_layer(filter_dimension=[5, 5, 32, 32], stride_length=1, activation_function='relu')

model.add_output_layer()

# Train!
model.begin_training()
```

### Data/Label Loading

There are three ways to load in a dataset for training a heatmap-based object counter. The first way involves places the training images and a CSV file in a directory and calling:

```python
model.load_heatmap_dataset_with_csv_from_directory(dirname, label_file)
```

The CSV file should contain a mapping of image names to the coordinates of multiple objects in x,y,x,y,... order. This will take the object locations and generate the ground truth heatmap, placing a 2D gaussian distribution at every labeled location. The standard deviation (and thus size) of the gaussians can be set with `set_density_map_sigma(sigma)`. An example generated heatmap is shown below.

![Example Generated Heatmap](heatmap_labels.png)

Alternatively, the point labels can be placed into JSON files (1 per image) in the same directory as the images. These can then be loaded using:

```python
model.load_heatmap_with_json_files_from_directory(dirname)
```

The JSON label files, however, have a different format to the CSV labels:

```json
{"x": {"p1": x1, "p2": x2, ...}, 
"y": {"p1": y1, "p2": y2, ...}}
```

The other way to load datasets in is to use one of the semantic segmentation loaders:

```python
model.load_dataset_from_directory_with_segmentation_masks(dirname, seg_dirname)
```

This can be used to load in images with pre-made heatmaps, provided that the image and heatmap images are separated into different directories.
