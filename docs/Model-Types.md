## Model Types

DPP is mainly used through several model objects. These objects contain the methods used to load image & label datasets, control training hyperparameters and data augmentation, build and connect model layers, and train models.

There are 6 `Model` objects is DPP for each problem type that it supports: 

- `ClassificationModel`, for classifying images into different classes (like objects or biotic stress)
- `RegressionModel`, for determining parameter values (like leaf counts) from images
- `SemanticSegmentationModel`, for determining binary and multi-class segmentation masks for images
- `ObjectDetectionModel`, for detecting objects in images using the YOLO model
- `CountCeptionModel`, for counting objects in images using the Countception model
- `HeatmapObjectCountingModel`, for counting objects in images by determining heatmaps of their locations. This works similarly to semantic segmentation otherwise.

#### Model Creation

All of the models have the same interface for creating them (using `RegressionModel` as an example):

```python
import deepplantphenomics as dpp
model = dpp.RegressionModel(debug=False, load_from_saved=False, save_checkpoints=True, 
                            initialize=True, tensorboard_dir=None, report_rate=100, save_dir=None)
```

- `debug` controls the printing of extra debugging information during model construction, data loading, and model training.
- `load_from_saved` is an optional string with a Tensorflow checkpoint file to load model variables from.
- `save_checkpoints` is a flag for whether to periodically save checkpoint files during training instead of just at the end of training.
- `initialize` toggles the creation of a new Tensorflow session with an empty graph with the model object. This should almost always be left at `True`.
- `tensorboard_dir` is an optional string with a directory to place Tensorboard summary files to during training.
- `report_rate` controls how often console output and Tensorboard summaries on training results are produced.
- `save_dir` is an optional string with a directory to save checkpoint files to.

#### Model Methods

Most of the hyperparameter setting methods, all of the layer creation methods, and some of the more general data loaders are shared between all of the `Model` objects. See [Model Options](Model-Options.md), [Neural Network Layers](Neural-Network-Layers.md), and [Loaders](Loaders.md) for more info about those shared methods and methods unique to certain `Model` objects.