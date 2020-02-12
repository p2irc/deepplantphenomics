## Image Loaders

These loaders can be used to load images, typically after you have already loaded labels for those images.

#### Load Images from a List of Filenames

Load images from a Python list of filename strings.

```
load_images_from_list(image_files)
```

#### Load Images from Lemnatec Plant Scanner

Loads the side-view VIS images from a collection of images provided by a Lemnatec plant scanner. Assumes each plant (timepoint) is in a subdirectory.

```
load_lemnatec_images_from_directory(dirname)
```

#### Load Images for Pre-specified IDs from Directory

If you have specified a list of files (for example, using the ID column in `load_multiple_labels_from_csv()`), then you can use this function to load those images from a directory. 

```
load_images_with_ids_from_directory(dirname)
```

## Label Loaders

These functions load labels for the dataset in multiple formats.

#### Load Multiple Labels from a CSV File

Load one or more labels per instance from a CSV file, for instance, values for regression. Parameter `id_column` (optional, zero-indexed, default 0) is the column number specifying the image file name.

Following this step, you can proceed to load the images specified by filename in `id_column` with `load_images_with_ids_from_directory(dirname)`

```
load_multiple_labels_from_csv(filepath, id_column)
```

#### Load Pascal VOC Bounding Box Coordinates from Directory

Loads single per-image bounding boxes from XML files in Pascal VOC format. The corresponding image filename should be specified in the XML file.

With the `ObjectDetectionModel`, this will also convert the labels into a format compatible with the output of the YOLO model.

```
load_pascal_voc_labels_from_directory(dirname)
```

#### Load Bounding Box Coordinates from a JSON File

Loads multiple per-image bounding boxes from a JSON file with the following format:

```
{
"<filename>.png": {
  "width": w,
  "height": h,
  "plants": [
    {"all_points_x": [x1, x2], "all_points_y": [y1, y2]}, 
    ...
    ]
  },
...
}
```

With the `ObjectDetectionModel`, this will also convert the labels into a format compatible with the output of the YOLO model.

```
load_json_labels_from_file(filename)
```

## Dataset Loaders

These functions load both images and labels simultaneously.

#### Load Dataset From Directory With CSV Labels

Provide `.png` images in a directory, and provide a CSV file where one of the columns (`column_number`, optional, zero-indexed) represents the label for each image. The labels should be sorted alphabetically by image name.

```
load_dataset_from_directory_with_csv_labels(dirname, labels_file, column_number)
```

#### Load Dataset from Directory with Auto-Labels

Put your images in a directory, organized with images of each class in a separate sub-directory. The names of the subdirectories are ignored. **Requires using the `ClassificationModel`**.

```
load_dataset_from_directory_with_auto_labels(dirname)
```

#### Load Dataset with Segmentation Ground-Truth

Loads the `.png` images from a directory, along with the ground-truth segmentation masks from another directory. File names should match exactly between the images and the corresponding ground truth images. **Requires using either the `SemanticSegmentationModel` or `HeatmapObjectCountingModel`**.

```
load_dataset_from_directory_with_segmentation_masks(dirname, truth_dirname)
```

#### Load Dataset from IPPN for Classification or Regression

Loads the RGB images and classification labels from the [International Plant Phenotyping Network](http://www.plant-phenotyping.org/) "PRL" dataset. Depending on the task you want labels for, you can pass the values `strain`, `treatment`, or `DAG` (Hours After Germination). **Requires using either the `ClassificationModel` or `RegressionModel`**.

```
load_ippn_dataset_from_directory(dirname, column)
```

#### Load Dataset from IPPN for Tray Segmentation

Loads the RGB tray images and plant bounding box labels from the [International Plant Phenotyping Network](http://www.plant-phenotyping.org/) datasets.

With the `ObjectDetectionModel`, this will also convert the labels into a format compatible with the output of the YOLO model.

```
load_ippn_tray_dataset_from_directory(dirname):
```

#### Load Dataset from IPPN for Leaf Counting

Loads the RGB images and leaf count labels from the [International Plant Phenotyping Network](http://www.plant-phenotyping.org/) datasets.

```
load_ippn_leaf_count_dataset_from_directory(dirname)
```

#### Load Dataset from INRA

Loads an arabidopsis dataset downloaded from INRA.

```
load_inra_dataset_from_directory(dirname)
```

#### Load CIFAR-10 Dataset From Directory

DPP can be used as a general-purpose deep learning platform, and you can use it with general image classification datasets such as CIFAR-10. This function assumes that you downloaded the dataset from [nVidia DIGITS](https://developer.nvidia.com/digits).

```
load_cifar10_dataset_from_directory(dirname)
```

#### Load YOLO Dataset From Directory

Loads a dataset for object detection with a JSON file of labels and a sub-directory of images. Unlike other label and dataset loaders for object detection, this can automatically patch the dataset; see [the object detection tutorial](Tutorial-Training-An-Object-Detector.md) for more info on automatic patching. **Requires the `ObjectDetectionModel`**.

```
load_yolo_dataset_from_directory(dirname, label_file, image_dir)
```

#### Load Dataset Saved in a Pickle File

Loads an object counting dataset from a pickle file. **Requires using the `CountCeptionModel`**.

The pickled dataset is stored in the following format:

```
[(image_data, count_map_data), ...]
```

Each image is represented by a tuple of two matrices: one with the image data and another that contains the count map data.

For more information about how the count map is generated, please refer to the original paper (https://arxiv.org/abs/1703.08710).

```
load_dataset_from_pkl_file(pkl_file_name)
```

#### Load Heatmap-based Counting Dataset From Directory

Two variants of this loader exist. **Both require using the `HeatmapObjectCountingModel`**.

##### With CSV Labels

Loads a dataset for object counting using heatmaps from directory with images and a CSV file of object locations for each image. For each image, the labels should be the x and y point coordinates of each object location such that x and y alternate with each other (i.e. `filename, x1, y1, x2, y2, ...`).

The object location labels will be used to automatically generate the ground truth heatmap for the corresponding image. The generated heatmap consists of a 2D gaussian placed at every location in the image; the size of the gaussians is controlled by setting the standard deviation using `set_density_map_sigma(sigma)`.

```
load_heatmap_dataset_with_csv_from_directory(dirname, label_file)
```

##### With JSON Labels

Loads a similar dataset whose labels are a JSON file per image of object locations. JSON files should have the same name as the corresponding image file and be in the same directory. The JSON labels should be in the following format (though key names like p1, p2, ... are arbitrary):

```json
{"x": {"p1": x1, "p2": x2, ...}, 
"y": {"p1": y1, "p2": y2, ...}}
```

```
load_heatmap_dataset_with_json_files_from_directory(dirname)
```
