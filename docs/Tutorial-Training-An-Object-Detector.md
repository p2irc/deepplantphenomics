Among the kinds of tasks DPP can train models for, it can be used to train a single-class object detector. The object detection in DPP is based on the YOLO object detector ([YOLOv2 specifically](https://arxiv.org/pdf/1612.08242.pdf)). YOLO's methodology splits images into grids and makes multiple bounding box predictions in each grid square alongside a prediction for whether there is an object and what kind of object it is (in multi-class detectors).

The overall structure and process of setting up and training a model is similar to other DPP models (see the [Leaf Counter training tutorial](/Tutorial-Training-The-Leaf-Counter/) for a detailed description of this). This tutorial largely covers the differences in model setup and data/label loading specific to training YOLO object detectors in DPP.

## YOLO v2 Network Layers

Rather than having to create the layers yourself, the YOLO v2 network is available as a predifined model in DPP. After configuring the model settings and loading in the dataset, the model layers can be setup using:

```python
model.use_predefined_model('yolov2')
```

## YOLO Object Detector Settings

When training a YOLO object detector, the settings and hyperparameters will typically look like this: 

```python
# 3 channels for colour, 1 channel for greyscale
channels = 3

# Setup and hyperparameters
model.set_batch_size(1)
model.set_number_of_threads(4)
model.set_image_dimensions(448, 448, channels)
model.set_resize_images(True)

# YOLO-specific setup
model.set_problem_type('object_detection')
prior_boxes = [[159, 157], [103, 133], [91, 89], [64, 65], [142, 101]]
model.set_yolo_parameters(grid_size=[7,7], labels=['plant'],
                          anchors=prior_boxes)
```

`set_yolo_parameters()` is used to set most of the settings specific to YOLO object detection. The grid size that it uses (typically with the same, odd,  width & height) and a list of the classes/labels have to be provided. YOLO also requires a list of anchors/priors for each bounding box; these are the widths and heights of reference boxes that allow the multiple bounding box predictions to detect objects of various aspect ratios.

The YOLO specific parameters default to a grid size of 7x7, a single 'plant' label, and anchors of [[159, 157], [103, 133], [91, 89], [64, 65], [142, 101]]. These settings are typical for most cases, and calling `set_yolo_parameters` is optional if all of these are left at their default values.

## Data/Label Loading

The main change for loading images and labels for YOLO object detection is that the labels need to be converted to a format with a similar box and prediction encoding to YOLO's output. This conversion happens automatically so long as the problem type is set to `object_detection` and certain functions are used to load in the images and labels.

Loaders with automatic conversion to YOLO labels include:

- `load_ippn_tray_dataset_from_directory(dirname)`: Load IPPN tray images and labels and convert labels to YOLO format.
- `load_pascal_voc_labels_from_directory(dirname)`: Load Pascal VOC labels from a directory of XML files and convert them to YOLO format, then load in images with `load_images_with_id_from_directory(dirname)`.
- `load_yolo_dataset_from_directory(dirname, label_file, image_dir)` and `load_json_labels_from_file(filename)`: Load labels from a custom JSON format file, then load in images with `load_images_from_list(files)`. The expectecd JSON format for the labels is as follows:

```json
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

## Automatic Patching of Large Images

Object detection also has special support for automatically splitting large input images into smaller patches of the right size when they're loaded with `load_yolo_dataset_from_directory`. This requires extra settings in the model for turning off image resizing and setting the size of the image patches.

```python
model.set_resize_images(False)
model.set_patch_size(448, 448)
```

With those settings, the labels should then be in a JSON file compatible with `load_json_labels_from_file`. The method then delays YOLO label conversion until after loading in the labels and images. It will then automatically patch the input images and convert the labels to YOLO labels for each patch. The image patches and a JSON file of their labels will be saved in a folder (`tmp_train`) in the data directory given to `load_yolo_dataset_from_directory`; this allows it to reuse the patched images and perform this process only once on a given dataset.

## Full Example

An example of all the code needed to train a YOLO-based object detector is available in the `examples` folder of DPP as `yolo_object_detector.py`. 