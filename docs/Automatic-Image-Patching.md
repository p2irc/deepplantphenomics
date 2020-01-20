## Automatic Image Patching

Training models with large images can be difficult if the model and images can't fit into memory together. In these cases, splitting large images into smaller patches and training on them can alleviate memory issues.

An image dataset can be patched separately and then loaded into DPP. DPP, however, is capable of automatically splitting images for training and inference for certain problems types. This is currently supported for semantic segmentation, heatmap object counting, and object detection.

Usually, the only extra setting required to invoke this capability is

```
model.set_patch_size(height=128, width=128)
```

#### Patching Training Images

When loading a dataset to train a model, certain loader functions should be used in order to invoke the automatic image patcher. The loaders required for this are:

- Semantic Segmentation: 
    - `load_dataset_from_directory_with_segmentation_masks(dirname, seg_dirname)`
- Heatmap Object Counting: 
    - `load_dataset_from_directory_with_segmentation_masks(dirname, seg_dirname)`
    - `load_heatmap_dataset_with_csv_from_directory(dirname, label_file)`
    - `load_heatmap_dataset_with_json_files_from_directory(dirname)`
- Object Detection: 
    - `load_yolo_dataset_from_directory(label_file, image_dir)`

These loaders will load the original dataset first and then split the images up into patches, saving them and the adjusted labels in a folder adjacent to the original dataset. Generally, using this requires that the patch size and the image size specified by `set_image_dimensions` match, since the patches are what the model will see.

For semantic segmentation and heatmap object counting, the patching simply splits each image into as many patches as are necessary to capture the original image. If an image dimension isn't evenly divisible by the corresponding patch dimension, the image is padded on the bottom and right sides with black pixels.

The patches for object detection are generated with a different approach. It first tries to generate patches such that every grid cell will have an object in it for at least one patch (see the [object detection tutorial](Tutorial-Training-An-Object-Detector.md) for clarification on grid cells). It then generates augmented random patches with objects in them and then doubles the dataset with totally random patches from the images.

The auto-patching can also see previously generated patches and load them directly instead of repeating the patching process.

#### Patching Inference Images

When performing inference on images after training a model, the images will be split up into patches internally before performing the forward pass on them. The returned predictions then vary with the problem type.

For semantic segmentation and heatmap object counting, the patches are stitched back together and output corresponding to the padded portion of the image are cropped out. For object detection, however, the predictions for each patch are returned.
