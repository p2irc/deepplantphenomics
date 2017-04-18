This guide details the available preprocessing steps which can be added via the `add_preprocessing_step` function.

All preprocessing steps are executed as soon as images are loaded using one of the image or dataset loading functions. They are run in the order they were added. The preprocessed images are saved in a separate folder and are cached there to avoid re-processing the images on subsequent runs. The location of this folder defaults to `./DPP-Processed` but can be specified with `set_processed_images_dir()`.

**In general, pre-processing is unnecessary.** Convolutional neural networks typically take raw RGB input, without the need for segmentation, fitlering, or any such steps. The pre-processors listed here are only provided as a courtesy and an example of how deep neural networks can be applied in image processing.

## auto-segmentation

The `auto-segmentation` preprocessing step performs automatic segmentation on shoot images against a white background. It utilizes PlantCV for pixel thresholding and masking, and a pre-trained bounding box regressor (`bbox-regressor-lemnatec`) to localize the region of interest containing the plant.

```
model.add_preprocessing_step('auto-segmentation')
```

This preprocessor should only be used for **single-plant side-view shoot images** with a white background. It was designed to be used with the side-view RGB images which originate from [Lemnatec](http://www.lemnatec.com/) plant scanners.
