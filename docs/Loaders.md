# Loaders

This page details the different functions which are provided to load images, labels, or both.

## Image Loaders

### Load Images from a List of Filenames

Load images from a Python list of filename strings.

```
load_images_from_list(image_files)
```

### Load Images from Lemnatec Plant Scanner

Loads the side-view VIS images from a collection of images provided by a Lemnatec plant scanner. Assumes each plant (timepoint) is in a subdirectory.

```
load_lemnatec_images_from_directory(dirname)
```

## Label Loaders

### Load Multiple Labels from a CSV File

Load multiple labels from a CSV file, for instance values for regression. Parameter `id_column` (optional, zero-indexed, default 0) is the column number specifying the image file name.

```
load_multiple_labels_from_csv(filepath, id_column)
```

### Load Pascal VOC Bounding Box Coordinates from Directory

Loads single per-image bounding boxes from XML files in Pascal VOC format. The corresponding image filename should be specified in the XML file.

```
load_pascal_voc_labels_from_directory(dirname)
```

## Dataset (Image+Labels) Loaders

### Load Dataset From Directory With CSV Labels

Provide `.png` images in a directory, and provide a CSV file where one of the columns (`column_number`, optional, zero-indexed) represents the label for each image. The labels should be sorted alphabetically by image name.

```
load_dataset_from_directory_with_csv_labels(dirname, labels_file, column_number)
```

### Load Dataset from Directory with Auto-Labels

Put your images in a directory, organized with images of each class in a separate sub-directory. For classification tasks only. The names of the subdirectories are ignored.

```
load_dataset_from_directory_with_auto_labels(dirname)
```

### Load Dataset from IPPN for Classification

Loads the RGB images and classification labels from the [International Plant Phenotyping Network dataset](http://www.plant-phenotyping.org/). Depending on the task you want labels for, you can pass the values `strain`, `treatment`, or `DAG` (Days After Germination).

```
load_ippn_classification_dataset_from_directory(dirname, column)
```

### Load Dataset from IPPN for Tray Segmentation

Loads the RGB tray images and plant bounding box labels from the [International Plant Phenotyping Network dataset](http://www.plant-phenotyping.org/).

```
load_ippn_tray_dataset_from_directory(dirname):
```

### Load Dataset from IPPN for Leaf Counting

Loads the RGB images and leaf count labels from the [International Plant Phenotyping Network dataset](http://www.plant-phenotyping.org/).

```
load_ippn_leaf_count_dataset_from_directory(dirname)
```

### Load Dataset from INRA

Loads a dataset downloaded from INRA.

```
load_inra_dataset_from_directory(dirname)
```

### Load CIFAR-10 Dataset From Directory

DPP can be used as a general-purpose deep learning platform, and you can use it with general image classification datasets such as CIFAR-10. This function assumes that you downloaded the dataset from [nVidia DIGITS](https://developer.nvidia.com/digits)

```
load_cifar10_dataset_from_directory(dirname)
```