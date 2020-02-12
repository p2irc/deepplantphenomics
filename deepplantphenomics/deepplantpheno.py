from . import layers, loaders, definitions
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.contrib
from tensorflow.python.client import device_lib
import os
import json
import datetime
import time
import warnings
import copy
import math
import random
from abc import ABC, abstractmethod
from tqdm import tqdm


class DPPModel(ABC):
    """
    The DPPModel class represents a model which can either be trained, or loaded from an existing checkpoint file. It
    provides common functionality and parameters for models of all problem types. Subclasses of DPPModel implement any
    changes and extra methods required to support that particular problem.
    """
    # Class variables with the supported implementations for various network components; subclasses should override
    # these
    _supported_optimizers = ['adam', 'adagrad', 'adadelta', 'sgd', 'sgd_momentum']
    _supported_weight_initializers = ['normal', 'xavier']
    _supported_activation_functions = ['relu', 'tanh', 'lrelu', 'selu']
    _supported_pooling_types = ['max', 'avg']
    _supported_loss_fns = ['softmax cross entropy', 'l2', 'l1', 'smooth l1', 'sigmoid cross entropy',
                           'yolo']
    _supported_predefined_models = ['vgg-16', 'alexnet', 'resnet-18', 'yolov2', 'xsmall', 'small', 'medium', 'large',
                                    'countception', 'u-net', 'fcn-18']
    _supported_augmentations = [definitions.AugmentationType.FLIP_HOR,
                                definitions.AugmentationType.FLIP_VER,
                                definitions.AugmentationType.CROP,
                                definitions.AugmentationType.CONTRAST_BRIGHT,
                                definitions.AugmentationType.ROTATE]
    _supports_standardization = True

    def __init__(self, debug=False, load_from_saved=False, save_checkpoints=True, initialize=True, tensorboard_dir=None,
                 report_rate=100, save_dir=None):
        """
        Create a new model object

        :param debug: If True, debug messages are printed to the console.
        :param load_from_saved: Optionally, pass the name of a directory containing the checkpoint file.
        :param save_checkpoints: If True, trainable parameters will be saved at intervals during training.
        :param initialize: If False, a new Tensorflow session will not be initialized with the instance.
        :param tensorboard_dir: Optionally, provide the path to your Tensorboard logs directory.
        :param report_rate: Set the frequency at which progress is reported during training (also the rate at which new
        timepoints are recorded to Tensorboard).
        """
        # Set instance variables, which is most of them since models shouldn't share most of their attributes
        # Operation settings
        self._with_patching = False
        self._has_trained = False
        self._save_checkpoints = save_checkpoints
        self._save_dir = save_dir
        self._validation = True
        self._testing = True
        self._hyper_param_search = False

        # Input options
        self._total_classes = 0
        self._total_raw_samples = 0
        self._total_training_samples = 0
        self._total_validation_samples = 0
        self._total_testing_samples = 0

        self._image_width = None
        self._image_height = None
        self._image_width_original = None
        self._image_height_original = None
        self._image_depth = None
        self._patch_height = None
        self._patch_width = None
        self._resize_bbox_coords = False

        self._crop_or_pad_images = False
        self._resize_images = False

        # Augmentation options
        self._augmentation_flip_horizontal = False
        self._augmentation_flip_vertical = False
        self._augmentation_crop = False
        self._crop_amount = 0.75
        self._augmentation_contrast = False
        self._augmentation_rotate = False
        self._rotate_crop_borders = False

        # Dataset storage
        self._all_ids = None
        self._gen_data_overwrite = False

        self._train_dataset = None
        self._test_dataset = None
        self._val_dataset = None

        self._all_images = None
        self._train_images = None
        self._test_images = None
        self._val_images = None

        self._all_labels = None
        self._train_labels = None
        self._test_labels = None
        self._val_labels = None
        self._split_labels = True

        self._images_only = False

        self._raw_image_files = None
        self._raw_test_image_files = None
        self._raw_train_image_files = None
        self._raw_val_image_files = None

        self._raw_labels = None
        self._raw_test_labels = None
        self._raw_train_labels = None
        self._raw_val_labels = None

        self._all_moderation_features = None
        self._has_moderation = False
        self._moderation_features_size = None
        self._train_moderation_features = None
        self._test_moderation_features = None
        self._val_moderation_features = None

        self._training_augmentation_images = None
        self._training_augmentation_labels = None

        # Network internal representation
        self._session = None
        self._graph = None
        self._graph_ops = {}
        self._layers = []
        self._global_epoch = 0

        self._num_layers_norm = 0
        self._num_layers_conv = 0
        self._num_layers_upsample = 0
        self._num_layers_pool = 0
        self._num_layers_fc = 0
        self._num_layers_dropout = 0
        self._num_layers_batchnorm = 0
        self._num_blocks_paral_conv = 0
        self._num_skip_connections = 0
        self._num_copy_connections = 0

        # Network options
        self._batch_size = 1
        self._test_split = 0.10
        self._validation_split = 0.10
        self._force_split_partition = False
        self._maximum_training_batches = None
        self._reg_coeff = None
        self._optimizer = 'adam'
        self._weight_initializer = 'xavier'
        self._loss_fn = None

        self._learning_rate = 0.001
        self._lr_decay_factor = None
        self._lr_decay_epochs = None
        self._lr_epoch = None

        # Wrapper options
        self._debug = debug
        self._load_from_saved = load_from_saved
        self._tb_dir = tensorboard_dir
        self._report_rate = report_rate

        # Multi-threading and GPU
        self._num_threads = 1
        self._num_gpus = 1
        self._max_gpus = 1  # Set this properly below
        self._subbatch_size = self._batch_size

        # Now do actual initialization stuff
        # Add the run level to the tensorboard path
        if self._tb_dir is not None:
            self._tb_dir = "{0}/{1}".format(self._tb_dir, datetime.datetime.now().strftime("%d%B%Y%I:%M%p"))

        # Determine the maximum number of GPUs we can use (using code from https://stackoverflow.com/a/38580201 to find
        # out how many we can actually reach). If this ends up as 0, then other code knows to construct CPU-only graphs.
        gpu_list = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
        self._max_gpus = len(gpu_list)

        if initialize:
            self._log('TensorFlow loaded...')
            self._reset_graph()
            self._reset_session()

    def _log(self, message):
        if self._debug:
            print('{0}: {1}'.format(datetime.datetime.now().strftime("%I:%M%p"), message))

    def _last_layer(self):
        return self._layers[-1]

    def _last_layer_outputs_volume(self):
        return isinstance(self._last_layer().output_size, (list,))

    def _first_layer(self):
        return next(layer for layer in self._layers if
                    isinstance(layer, layers.convLayer) or isinstance(layer, layers.fullyConnectedLayer))

    def _reset_session(self):
        self._session = tf.Session(graph=self._graph,
                                   config=tf.ConfigProto(allow_soft_placement=True))

    def _reset_graph(self):
        self._graph = tf.Graph()

    def set_number_of_threads(self, num_threads):
        """Set number of threads for preprocessing tasks"""
        if not isinstance(num_threads, int):
            raise TypeError("num_threads must be an int")
        if num_threads <= 0:
            raise ValueError("num_threads must be positive")

        self._num_threads = num_threads

    def set_number_of_gpus(self, num_gpus):
        """Set the number of GPUs to use for graph evaluation. Setting this higher than the number of available GPUs
        has the same effect as setting this to exactly that amount (i.e. setting this to 4 with 2 GPUs available will
        still only use 2 GPUs)."""
        if not isinstance(num_gpus, int):
            raise TypeError("num_gpus must be an int")
        if num_gpus <= 0:
            raise ValueError("num_gpus must be positive")

        if self._max_gpus != 0:
            self._num_gpus = num_gpus if (num_gpus <= self._max_gpus) else self._max_gpus
        else:
            self._num_gpus = 1  # So batch-setting code doesn't gobble a goose

        if self._batch_size % self._num_gpus == 0:
            self._subbatch_size = self._batch_size // self._num_gpus
        else:
            raise RuntimeError("{0} GPUs can't evenly distribute a batch size of {1}"
                               .format(self._num_gpus, self._batch_size))

    def set_random_seed(self, seed):
        """
        Sets a random seed for any random operations used during augmentation and training. This is used to help
        reproduce results for debugging purposes.
        :param seed: An integer to use for seeding random operations
        """
        if not isinstance(seed, int):
            raise TypeError("seed must be an int")

        random.seed(seed)
        np.random.seed(seed)
        with self._graph.as_default():
            tf.set_random_seed(seed)

    def set_batch_size(self, size):
        """Set the batch size"""
        if not isinstance(size, int):
            raise TypeError("size must be an int")
        if size <= 0:
            raise ValueError("size must be positive")

        self._batch_size = size

        if size % self._num_gpus == 0:
            self._subbatch_size = size // self._num_gpus
        else:
            raise RuntimeError("{0} GPUs can't evenly distribute a batch size of {1}"
                               .format(self._num_gpus, size))

    def set_test_split(self, ratio):
        """Set a ratio for the total number of samples to use as a testing set"""
        if not isinstance(ratio, float) and ratio != 0:
            raise TypeError("ratio must be a float or 0")
        if ratio < 0 or ratio > 1:
            raise ValueError("ratio must be between 0 and 1")

        if ratio == 0 or ratio is None:
            self._testing = False
            ratio = 0
        else:
            self._testing = True
        self._test_split = ratio
        if self._test_split + self._validation_split > 0.5:
            warnings.warn('WARNING: Less than 50% of data is being used for training. ' +
                          '({test}% testing and {val}% validation)'.format(test=int(self._test_split * 100),
                                                                           val=int(self._validation_split * 100)))

    def set_validation_split(self, ratio):
        """Set a ratio for the total number of samples to use as a validation set"""
        if not isinstance(ratio, float) and ratio != 0:
            raise TypeError("ratio must be a float or 0")
        if ratio < 0 or ratio > 1:
            raise ValueError("ratio must be between 0 and 1")

        if ratio == 0 or ratio is None:
            self._validation = False
            ratio = 0
        else:
            self._validation = True
        self._validation_split = ratio
        if self._test_split + self._validation_split > 0.5:
            warnings.warn('WARNING: Less than 50% of data is being used for training. ' +
                          '({test}% testing and {val}% validation)'.format(test=int(self._test_split * 100),
                                                                           val=int(self._validation_split * 100)))

    def force_split_shuffle(self, force_split):
        """
        Sets whether to force shuffling of a loaded dataset into train, test, and validation partitions. By default,
        this is turned off; these partitions are shuffled and saved the first time a dataset is used for training, and
        subsequent training runs load and reuse this partitioning, making training more reproducible.
        :param force_split: A boolean flag for whether to force
        """
        if not isinstance(force_split, bool):
            raise TypeError("force_split must be a bool")

        self._force_split_partition = force_split

    def set_maximum_training_epochs(self, epochs):
        """Set the max number of training epochs"""
        if not isinstance(epochs, int):
            raise TypeError("epochs must be an int")
        if epochs <= 0:
            raise ValueError("epochs must be positive")

        self._maximum_training_batches = epochs

    def set_learning_rate(self, rate):
        """Set the initial learning rate"""
        if not isinstance(rate, float):
            raise TypeError("rate must be a float")
        if rate <= 0:
            raise ValueError("rate must be positive")

        self._learning_rate = rate

    def set_crop_or_pad_images(self, crop_or_pad):
        """Apply padding or cropping images to, which is required if the dataset has images of different sizes"""
        if not isinstance(crop_or_pad, bool):
            raise TypeError("crop_or_pad must be a bool")

        self._crop_or_pad_images = crop_or_pad

    def set_resize_images(self, resize):
        """Up-sample or down-sample images to specified size"""
        if not isinstance(resize, bool):
            raise TypeError("resize must be a bool")

        self._resize_images = resize

    def set_augmentation_flip_horizontal(self, flip):
        """Randomly flip training images horizontally"""
        if not isinstance(flip, bool):
            raise TypeError("flip must be a bool")
        if definitions.AugmentationType.FLIP_HOR not in self._supported_augmentations:
            raise RuntimeError("Flip augmentations are incompatible with the current model type")

        self._augmentation_flip_horizontal = flip

    def set_augmentation_flip_vertical(self, flip):
        """Randomly flip training images vertically"""
        if not isinstance(flip, bool):
            raise TypeError("flip must be a bool")
        if definitions.AugmentationType.FLIP_VER not in self._supported_augmentations:
            raise RuntimeError("Flip augmentations are incompatible with the current model type")

        self._augmentation_flip_vertical = flip

    def set_augmentation_crop(self, resize, crop_ratio=0.75):
        """Randomly crop images during training, and crop images to center during testing"""
        if not isinstance(resize, bool):
            raise TypeError("resize must be a bool")
        if not isinstance(crop_ratio, float):
            raise TypeError("crop_ratio must be a float")
        if crop_ratio <= 0 or crop_ratio > 1:
            raise ValueError("crop_ratio must be in (0, 1]")
        if definitions.AugmentationType.CROP not in self._supported_augmentations:
            raise RuntimeError("Crop augmentations are incompatible with the current model type")

        self._augmentation_crop = resize
        self._crop_amount = crop_ratio

    def set_augmentation_brightness_and_contrast(self, contr):
        """Randomly adjust contrast and/or brightness on training images"""
        if not isinstance(contr, bool):
            raise TypeError("contr must be a bool")
        if definitions.AugmentationType.CONTRAST_BRIGHT not in self._supported_augmentations:
            raise RuntimeError("Contrast and brightness augmentations are incompatible with the current model type")

        self._augmentation_contrast = contr

    def set_augmentation_rotation(self, rot, crop_borders=False):
        """Randomly rotate training images"""
        if not isinstance(rot, bool):
            raise TypeError("rot must be a bool")
        if not isinstance(crop_borders, bool):
            raise TypeError("crop_borders must be a bool")
        if definitions.AugmentationType.ROTATE not in self._supported_augmentations:
            raise RuntimeError("Rotation augmentations are incompatible with the current model type")

        self._augmentation_rotate = rot
        self._rotate_crop_borders = crop_borders

    def set_regularization_coefficient(self, lamb):
        """Set lambda for L2 weight decay"""
        if not isinstance(lamb, float):
            raise TypeError("lamb must be a float")
        if lamb <= 0:
            raise ValueError("lamb must be positive")

        self._reg_coeff = lamb

    def set_learning_rate_decay(self, decay_factor, batches_per_decay):
        """Set learning rate decay"""
        if not isinstance(decay_factor, float):
            raise TypeError("decay_factor must be a float")
        if decay_factor <= 0:
            raise ValueError("decay_factor must be positive")
        if not isinstance(batches_per_decay, int):
            raise TypeError("epochs_per_day must be an int")
        if batches_per_decay <= 0:
            raise ValueError("epochs_per_day must be positive")

        self._lr_decay_factor = decay_factor
        self._lr_decay_epochs = batches_per_decay

    def set_optimizer(self, optimizer):
        """Set the optimizer to use"""
        if not isinstance(optimizer, str):
            raise TypeError("optimizer must be a str")
        if optimizer.lower() in self._supported_optimizers:
            optimizer = optimizer.lower()
        else:
            raise ValueError("'" + optimizer + "' is not one of the currently supported optimizers. Choose one of " +
                             " ".join("'" + x + "'" for x in self._supported_optimizers))

        self._optimizer = optimizer

    def set_loss_function(self, loss_fn):
        """Set the loss function to use"""
        if not isinstance(loss_fn, str):
            raise TypeError("loss_fn must be a str")
        loss_fn = loss_fn.lower()

        if loss_fn not in self._supported_loss_fns:
            raise ValueError("'" + loss_fn + "' is not a supported loss function for the current model type. Make " +
                             "sure you're using the correct model class for the problem or selecting one of these " +
                             "loss functions: " +
                             " ".join("'" + x + "'" for x in self._supported_loss_fns))

        self._loss_fn = loss_fn

    def set_weight_initializer(self, initializer):
        """Set the initialization scheme used by convolutional and fully connected layers"""
        if not isinstance(initializer, str):
            raise TypeError("initializer must be a str")
        initializer = initializer.lower()
        if initializer not in self._supported_weight_initializers:
            raise ValueError("'" + initializer + "' is not one of the currently supported weight initializers." +
                             " Choose one of: " + " ".join("'"+x+"'" for x in self._supported_weight_initializers))

        self._weight_initializer = initializer

    def set_image_dimensions(self, image_height, image_width, image_depth):
        """Specify the image dimensions for images in the dataset (depth is the number of channels)"""
        if not isinstance(image_height, int):
            raise TypeError("image_height must be an int")
        if image_height <= 0:
            raise ValueError("image_height must be positive")
        if not isinstance(image_width, int):
            raise TypeError("image_width must be an int")
        if image_width <= 0:
            raise ValueError("image_width must be positive")
        if not isinstance(image_depth, int):
            raise TypeError("image_depth must be an int")
        if image_depth <= 0:
            raise ValueError("image_depth must be positive")

        self._image_width = image_width
        self._image_height = image_height
        self._image_depth = image_depth

    def set_original_image_dimensions(self, image_height, image_width):
        """
        Specify the original size of the image, before resizing.
        This is only needed in special cases, for instance, if you are resizing input images but using image coordinate
        labels which reference the original size.
        """
        if not isinstance(image_height, int):
            raise TypeError("image_height must be an int")
        if image_height <= 0:
            raise ValueError("image_height must be positive")
        if not isinstance(image_width, int):
            raise TypeError("image_width must be an int")
        if image_width <= 0:
            raise ValueError("image_width must be positive")

        self._image_width_original = image_width
        self._image_height_original = image_height

    def add_moderation_features(self, moderation_features):
        """Specify moderation features for examples in the dataset"""
        self._has_moderation = True
        self._moderation_features_size = moderation_features.shape[1]
        self._all_moderation_features = moderation_features

    def set_patch_size(self, height, width):
        """Sets the size of patches generated from larger input images and turns on automatic patching"""
        if not isinstance(height, int):
            raise TypeError("height must be an int")
        if height <= 0:
            raise ValueError("height must be positive")
        if not isinstance(width, int):
            raise TypeError("width must be an int")
        if width <= 0:
            raise ValueError("width must be positive")

        self._patch_height = height
        self._patch_width = width
        self._with_patching = True

    def set_gen_data_overwrite(self, overwrite):
        """Sets whether to overwrite generated data like patches and object heatmaps when loading data or to load any
        previous generated data that exists"""
        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be a bool")

        self._gen_data_overwrite = overwrite

    def _get_device_list(self):
        """Returns the list of CPU and/or GPU devices to construct and evaluate graphs for"""
        if not tf.test.is_gpu_available():
            return ['/device:cpu:0']
        else:
            return ['/device:gpu:' + str(x) for x in range(self._num_gpus)]

    def _add_layers_to_graph(self):
        """
        Adds the layers in self.layers to the computational graph.
        """
        # Adding layers to the graph mostly involves setting up the required variables. Those variables should be on
        # the CPU if we are using multiple GPUs and need to share them across multiple graph towers. Otherwise, they
        # can go on whatever device Tensorflow deems sensible.
        if tf.test.is_gpu_available() and self._num_gpus > 1:
            d = '/device:cpu:0'
        else:
            d = None  # Effectively /device:cpu:0 for CPU-only or /device:gpu:0 for 1 GPU

        for layer in self._layers:
            if callable(getattr(layer, 'add_to_graph', None)):
                with tf.device(d):
                    layer.add_to_graph()

    def _graph_parse_data(self):
        """
        Add graph components that parse the input images and labels into tensors and split them into training,
        validation, and testing sets
        """
        if self._raw_test_labels is not None:
            # currently think of moderation features as None so they are passed in hard-coded
            self._parse_dataset(self._raw_train_image_files, self._raw_train_labels, None,
                                self._raw_test_image_files, self._raw_test_labels, None,
                                self._raw_val_image_files, self._raw_val_labels, None)
        elif self._images_only:
            self._parse_images(self._raw_image_files)
        else:
            # Split the data into training, validation, and testing sets. If there is no validation set or no
            # moderation features being used they will be returned as 0 (for validation) or None (for moderation
            # features)
            train_images, train_labels, train_mf, test_images, test_labels, test_mf, val_images, val_labels, val_mf = \
                loaders.split_raw_data(self._raw_image_files, self._raw_labels,
                                       self._test_split, self._validation_split, self._all_moderation_features,
                                       self._training_augmentation_images, self._training_augmentation_labels,
                                       self._split_labels,
                                       force_mask_creation=self._force_split_partition)
            # Parse the images and set the appropriate environment variables
            self._parse_dataset(train_images, train_labels, train_mf,
                                test_images, test_labels, test_mf,
                                val_images, val_labels, val_mf)

    def _graph_extract_patch(self, x, offsets=None):
        """
        Adds graph components to extract patches from input images
        :param x: Tensor, an image to extract a patch from
        :param offsets: An optional list of (height, width) tuples for where to extract patches from in the images. When
        this isn't given, offsets will be generated at random and returned
        :return: The extracted image patches and the offsets used to get them
        """
        if not offsets:
            offset_h = np.random.randint(self._patch_height // 2,
                                         self._image_height - (self._patch_height // 2),
                                         self._batch_size)
            offset_w = np.random.randint(self._patch_width // 2,
                                         self._image_width - (self._patch_width // 2),
                                         self._batch_size)
            offsets = [x for x in zip(offset_h, offset_w)]
        x = tf.image.extract_glimpse(x, [self._patch_height, self._patch_width], offsets,
                                     normalized=False, centered=False)
        return x, offsets

    def _graph_make_optimizer(self):
        """Generate a new optimizer object for computing and applying gradients"""
        if self._optimizer == 'adagrad':
            self._log('Using Adagrad optimizer')
            return tf.train.AdagradOptimizer(self._learning_rate)
        elif self._optimizer == 'adadelta':
            self._log('Using Adadelta optimizer')
            return tf.train.AdadeltaOptimizer(self._learning_rate)
        elif self._optimizer == 'sgd':
            self._log('Using SGD optimizer')
            return tf.train.GradientDescentOptimizer(self._learning_rate)
        elif self._optimizer == 'adam':
            self._log('Using Adam optimizer')
            return tf.train.AdamOptimizer(self._learning_rate)
        elif self._optimizer == 'sgd_momentum':
            self._log('Using SGD with momentum optimizer')
            return tf.train.MomentumOptimizer(self._learning_rate, 0.9, use_nesterov=True)
        else:
            warnings.warn('Unrecognized optimizer requested')
            exit()

    def _graph_get_gradients(self, loss, optimizer):
        """
        Add graph components for getting gradients given an optimizer some losses
        :param loss: The loss value to use when computing the gradients
        :param optimizer: The optimizer object used to generate the gradients
        :return: The graph's gradients, variables, and the global gradient norm from clipping
        """
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, global_grad_norm = tf.clip_by_global_norm(gradients, 5.0)
        return gradients, variables, global_grad_norm

    def _graph_average_gradients(self, graph_gradients):
        """
        Add graph components for averaging the computed gradients from multiple runs of a graph (i.e. over gradients
        from multiple GPUs)
        :param graph_gradients: A list of the computed gradient lists from each (GPU) run
        :return: A list of the averaged gradients across each run
        """
        # No averaging needed if there's only gradients from one run (because a single device, CPU or GPU, was used)
        if len(graph_gradients) == 1:
            return graph_gradients[0]

        averaged_gradients = []
        for gradients in zip(*graph_gradients):
            grads = [tf.expand_dims(g, 0) for g in gradients]
            grads = tf.concat(grads, axis=0)
            grads = tf.reduce_mean(grads, axis=0)
            averaged_gradients.append(grads)

        return averaged_gradients

    def _graph_apply_gradients(self, gradients, variables, optimizer):
        """
        Add graph components for using an optimizer applying gradients to variables
        :param gradients: The gradients to be applied
        :param variables: The variables to apply the gradients to
        :param optimizer: The optimizer object used to apply the gradients
        :return: An operation for applying gradients to the graph variables
        """
        return optimizer.apply_gradients(zip(gradients, variables), global_step=self._lr_epoch)

    def _graph_layer_loss(self):
        """Calculates and returns the total L2 loss from the weights of fully connected layers. This is 0 if a
        regularization coefficient isn't specified."""
        if self._reg_coeff is not None:
            return tf.squeeze(tf.reduce_sum(
                [layer.regularization_coefficient * tf.nn.l2_loss(layer.weights) for layer in self._layers
                 if isinstance(layer, layers.fullyConnectedLayer)]))
        else:
            return 0.0

    @abstractmethod
    def _graph_problem_loss(self, pred, lab):
        """
        Calculates the loss function for each item in a batch with a given pairing of predictions and labels. This is
        specific to each problem type.
        :param pred: A Tensor with Model predictions. The shape depends on the model and problem.
        :param lab: A Tensor Labels to compare the predictions to. Most problems expect this to be the same shape as
        pred, but exceptions exist.
        :return: Loss values for each item in a batch
        """
        pass

    def _graph_tensorboard_common_summary(self, l2_cost, gradients, variables, global_grad_norm):
        """
        Adds graph components common to every problem type related to outputting losses and other summary variables to
        Tensorboard.
        :param l2_cost: The L2 loss component of the computed cost
        :param gradients: The gradients for the variables in the graph
        :param global_grad_norm: The global norm used to normalize the gradients
        """
        self._log('Creating Tensorboard summaries...')

        # Summaries for any problem type
        tf.summary.scalar('train/loss', self._graph_ops['cost'], collections=['custom_summaries'])
        tf.summary.scalar('train/learning_rate', self._learning_rate, collections=['custom_summaries'])
        tf.summary.scalar('train/l2_loss', l2_cost, collections=['custom_summaries'])
        filter_summary = self._get_weights_as_image(self._first_layer().weights)
        tf.summary.image('filters/first', filter_summary, collections=['custom_summaries'])

        def _add_layer_histograms(net_layer):
            tf.summary.histogram('weights/' + net_layer.name, net_layer.weights, collections=['custom_summaries'])
            if not ((isinstance(net_layer, layers.convLayer) or isinstance(net_layer, layers.upsampleLayer)) and net_layer.use_bias is False):
                tf.summary.histogram('biases/' + net_layer.name, net_layer.biases, collections=['custom_summaries'])

            # At one point the graph would hang on session.run(graph_ops['merged']) inside of begin_training
            # and it was found that if you commented the below line then the code wouldn't hang. Never
            # fully understood why, as it only happened if you tried running with train/test and no
            # validation. But after adding more features and just randomly trying to uncomment the below
            # line to see if it would work, it appears to now be working, but still don't know why...
            tf.summary.histogram('activations/' + net_layer.name, net_layer.activations,
                                 collections=['custom_summaries'])

        # Summaries for each net_layer
        for layer in self._layers:
            if hasattr(layer, 'name'):
                if isinstance(layer, layers.paralConvBlock):
                    _add_layer_histograms(layer.conv1)
                    _add_layer_histograms(layer.conv2)
                elif not isinstance(layer, layers.batchNormLayer) and not isinstance(layer, layers.copyConnection) \
                        and not isinstance(layer, layers.skipConnection):
                    _add_layer_histograms(layer)

        # Summaries for gradients
        # We use variables[index].name[:-2] because variables[index].name will have a ':0' at the end of
        # the name and tensorboard does not like this so we remove it with the [:-2]
        # We also currently seem to get None's for gradients when performing a hyper-parameter search
        # and as such it is simply left out for hyper-param searches, needs to be fixed
        if not self._hyper_param_search:
            for index, grad in enumerate(gradients):
                tf.summary.histogram("gradients/" + variables[index].name[:-2], gradients[index],
                                     collections=['custom_summaries'])

            tf.summary.histogram("gradient_global_norm/", global_grad_norm, collections=['custom_summaries'])

    def _graph_tensorboard_summary(self, l2_cost, gradients, variables, global_grad_norm):
        """
        Adds graph components related to outputting losses and other summary variables to Tensorboard.
        :param l2_cost: ...
        :param gradients: ...
        :param global_grad_norm: ...
        """
        self._graph_tensorboard_common_summary(l2_cost, gradients, variables, global_grad_norm)
        self._graph_ops['merged'] = tf.summary.merge_all(key='custom_summaries')

    @abstractmethod
    def _assemble_graph(self):
        """
        Constructs the Tensorflow graph that defines the network. This includes splitting the input data into
        train/validation/test partitions, parsing it into Tensors, performing the forward pass and optimization steps,
        returning test and validation losses, and outputting losses and other variables to Tensorboard if necessary.
        Parts of the graph should be exposed by adding graph nodes to the `_graph_ops` variable; which nodes and their
        names will vary with the problem type.
        """
        pass

    def _batch_and_iterate(self, dataset, shuffle=False):
        """
        Sets up batching and prefetching for a Dataset, with optional shuffling (for training), and returns an iterator
        for the final Dataset.
        :param dataset: The Dataset to prepare with batching and prefetching
        :param shuffle: A flag for whether to shuffle the Dataset items
        :return: A one-shot iterator for the prepared Dataset
        """
        if shuffle:
            dataset = dataset.shuffle(10000)
        dataset = dataset.batch(self._subbatch_size)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(self._num_gpus)
        data_iter = dataset.make_one_shot_iterator()
        return data_iter

    def _training_batch_results(self, batch_num, start_time, tqdm_range, train_writer=None):
        """
        Calculates and reports mid-training losses and other statistics, both through the console and through writing
        Tensorboard log files
        :param batch_num: The batch number for the mid-training results
        :param start_time: The start time to use for calculating the processing rate
        :param tqdm_range: A `tqdm` object for displaying training results to the console
        :param train_writer: A `tf.summary.FileWriter` for writing Tensorboard log files
        """
        elapsed = time.time() - start_time

        if train_writer is not None:
            summary = self._session.run(self._graph_ops['merged'])
            train_writer.add_summary(summary, batch_num)

        if self._validation:
            loss, epoch_test_loss = self._session.run([self._graph_ops['cost'], self._graph_ops['val_cost']])
            samples_per_sec = self._batch_size / elapsed

            desc_str = "{}: Results for batch {} (epoch {:.1f}) - Loss: {}, Validation Loss: {}, samples/sec: {:.2f}"
            tqdm_range.set_description(
                desc_str.format(datetime.datetime.now().strftime("%I:%M%p"),
                                batch_num,
                                batch_num / (self._total_training_samples / self._batch_size),
                                loss,
                                epoch_test_loss,
                                samples_per_sec))
        else:
            loss = self._session.run([self._graph_ops['cost']])
            samples_per_sec = self._batch_size / elapsed

            desc_str = "{}: Results for batch {} (epoch {:.1f}) - Loss: {}, samples/sec: {:.2f}"
            tqdm_range.set_description(
                desc_str.format(datetime.datetime.now().strftime("%I:%M%p"),
                                batch_num,
                                batch_num / (self._total_training_samples / self._batch_size),
                                loss,
                                samples_per_sec))

    def begin_training(self, return_test_loss=False):
        """
        Initialize the network and either run training to the specified max epoch, or load trainable variables. The
        full test accuracy is calculated immediately afterward and the trainable parameters are saved before the
        session is shut down. Before calling this function, the images and labels should be loaded, as well as all
        relevant hyper-parameters.
        """
        with self._graph.as_default():
            self._lr_epoch = tf.Variable(0, trainable=False)
            self._set_learning_rate()
            self._assemble_graph()
            self._log('Assembled the graph')

            # Either load the network parameters from a checkpoint file or start training
            if self._load_from_saved:
                self._has_trained = True
                self.load_state()
                self.compute_full_test_accuracy()
                self.shut_down()
            else:
                if self._tb_dir is not None:
                    train_writer = tf.summary.FileWriter(self._tb_dir, self._session.graph)

                self._log('Initializing parameters...')
                self._session.run(tf.global_variables_initializer())

                self._log('Beginning training...')

                # Needed for batch norm
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                self._graph_ops['optimizer'] = tf.group([self._graph_ops['optimizer'], update_ops])

                # Weight decay
                if False:
                    decay_ops = [l.decay_weights() for l in self._layers if callable(getattr(l, 'decay_weights', None))]

                tqdm_range = tqdm(range(self._maximum_training_batches))
                for i in tqdm_range:
                    start_time = time.time()
                    self._global_epoch = i
                    self._session.run(self._graph_ops['optimizer'])

                    if self._global_epoch > 0 and self._global_epoch % self._report_rate == 0:
                        if self._tb_dir is not None:
                            self._training_batch_results(i, start_time, tqdm_range, train_writer)
                        else:
                            self._training_batch_results(i, start_time, tqdm_range)

                        if self._save_checkpoints and self._global_epoch % (self._report_rate * 100) == 0:
                            self.save_state(self._save_dir)
                    else:
                        loss = self._session.run([self._graph_ops['cost']])

                        if False:
                            self._session.run(decay_ops)

                    if loss == 0.0:
                        self._log('Stopping due to zero loss')
                        break

                    if i == self._maximum_training_batches - 1:
                        self._log('Stopping due to maximum epochs')

                self.save_state(self._save_dir)

                final_test_loss = None
                if self._testing:
                    final_test_loss = self.compute_full_test_accuracy()

                self.shut_down()

                if return_test_loss:
                    return final_test_loss
                else:
                    return

    def begin_training_with_hyperparameter_search(self, l2_reg_limits=None, lr_limits=None, num_steps=3):
        """
        Performs grid-based hyper-parameter search given the ranges passed. Parameters are optional.

        :param l2_reg_limits: array representing a range of L2 regularization coefficients in the form [low, high]
        :param lr_limits: array representing a range of learning rates in the form [low, high]
        :param num_steps: the size of the grid. Larger numbers are exponentially slower.
        """
        self._hyper_param_search = True

        base_tb_dir = self._tb_dir

        unaltered_image_height = self._image_height
        unaltered_image_width = self._image_width
        unaltered_epochs = self._maximum_training_batches

        if l2_reg_limits is None:
            all_l2_reg = [self._reg_coeff]
        else:
            step_size = (l2_reg_limits[1] - l2_reg_limits[0]) / np.float32(num_steps-1)
            all_l2_reg = np.arange(l2_reg_limits[0], l2_reg_limits[1], step_size)
            all_l2_reg = np.append(all_l2_reg, l2_reg_limits[1])

        if lr_limits is None:
            all_lr = [self._learning_rate]
        else:
            step_size = (lr_limits[1] - lr_limits[0]) / np.float32(num_steps-1)
            all_lr = np.arange(lr_limits[0], lr_limits[1], step_size)
            all_lr = np.append(all_lr, lr_limits[1])

        all_loss_results = np.empty([len(all_l2_reg), len(all_lr)])

        for i, current_l2 in enumerate(all_l2_reg):
            for j, current_lr in enumerate(all_lr):
                self._log('HYPERPARAMETER SEARCH: Doing l2reg=%f, lr=%f' % (current_l2, current_lr))

                # Make a new graph, associate a new session with it.
                self._reset_graph()
                self._reset_session()

                self._learning_rate = current_lr
                self._reg_coeff = current_l2

                # Set calculated variables back to their unaltered form
                self._image_height = unaltered_image_height
                self._image_width = unaltered_image_width
                self._maximum_training_batches = unaltered_epochs

                # Reset the reg. coef. for all fc layers.
                with self._graph.as_default():
                    for layer in self._layers:
                        if isinstance(layer, layers.fullyConnectedLayer):
                            layer.regularization_coefficient = current_l2

                if base_tb_dir is not None:
                    self._tb_dir = base_tb_dir + '_lr:' + current_lr.astype('str') + '_l2:' + current_l2.astype('str')

                try:
                    current_loss = self.begin_training(return_test_loss=True)
                    all_loss_results[i][j] = current_loss
                except Exception as e:
                    self._log('HYPERPARAMETER SEARCH: Run threw an exception, this result will be NaN.')
                    print("Exception message: "+str(e))
                    all_loss_results[i][j] = np.nan

        self._log('Finished hyperparameter search, failed runs will appear as NaN.')
        self._log('All l2 coef. tested:')
        self._log('\n'+np.array2string(np.transpose(all_l2_reg)))
        self._log('All learning rates tested:')
        self._log('\n'+np.array2string(all_lr))
        self._log('Loss/error grid:')
        self._log('\n'+np.array2string(all_loss_results, precision=4))

    @abstractmethod
    def compute_full_test_accuracy(self):
        """
        Prints to console and returns accuracy and loss statistics for the trained network. The applicable statistics
        will depend on the problem type.
        """
        pass

    def shut_down(self):
        """End the current session. The model cannot be used anymore after this is done."""
        self._log('Shutdown requested, ending session...')
        self._session.close()

    def _get_weights_as_image(self, kernel, size=None):
        """Filter visualization, adapted with permission from https://gist.github.com/kukuruza/03731dc494603ceab0c5"""
        with self._graph.as_default():
            pad = 1

            # pad x and y
            x1 = tf.pad(kernel, tf.constant([[pad, 0], [pad, 0], [0, 0], [0, 0]]))

            # when kernel is dynamically shaped at runtime it has [?,?,?,?] dimensions which result in None's
            # thus size needs to be passed in so we have actual dimensions to work with (this is mostly from the
            # upsampling layer) and grid_y will be determined by batch size as we want to see each img in the batch
            # However, for visualizing the weights we wont pass in a size parameter and as a result we need to
            # compute grid_y based off what is passed in and not the batch size because we want to see the
            # convolution grid for each layer, not each batch.
            if size is not None:
                # this is when visualizing the actual images
                grid_y_prelim = int(np.ceil(self._batch_size))
                # x and y dimensions, w.r.t. padding
                y = size[1] + pad
                x = size[2] + pad
                num_channels = size[-1]
            else:
                # this is when visualizing the weights
                grid_y_prelim = (kernel.get_shape().as_list()[-1])
                # x and y dimensions, w.r.t. padding
                y = kernel.get_shape()[0] + pad
                x = kernel.get_shape()[1] + pad
                num_channels = kernel.get_shape().as_list()[2]

            # we then want to set grid_x somewhat dynamically based on grid_y, making it the largest possible out of
            # 4, 2, or 1
            grid_x = 4 if grid_y_prelim % 4 == 0 else (2 if grid_y_prelim % 2 == 0 else 1)
            grid_y = grid_y_prelim // grid_x

            # pack into image with proper dimensions for tf.image_summary
            x2 = tf.transpose(x1, (3, 0, 1, 2))
            x3 = tf.reshape(x2, tf.stack([grid_x, y * grid_y, x, num_channels]))
            x4 = tf.transpose(x3, (0, 2, 1, 3))
            x5 = tf.reshape(x4, tf.stack([1, x * grid_x, y * grid_y, num_channels]))
            x6 = tf.transpose(x5, (2, 1, 3, 0))
            x7 = tf.transpose(x6, (3, 0, 1, 2))

            # scale to [0, 1]
            x_min = tf.reduce_min(x7)
            x_max = tf.reduce_max(x7)
            x8 = (x7 - x_min) / (x_max - x_min)

        return x8

    def save_state(self, directory=None):
        """Save all trainable variables as a checkpoint in the current working path"""
        self._log('Saving parameters...')

        if directory is None:
            state_dir = './saved_state'
        else:
            state_dir = directory + '/saved_state'

        if not os.path.isdir(state_dir):
            os.mkdir(state_dir)

        with self._graph.as_default():
            saver = tf.train.Saver(tf.global_variables())
            saver.save(self._session, state_dir + '/tfhSaved')

        self._has_trained = True

    def load_state(self):
        """
        Load all trainable variables from a checkpoint file specified from the load_from_saved parameter in the
        class constructor.
        """
        if not self._has_trained:
            self._add_layers_to_graph()

        if self._load_from_saved is not False:
            self._log('Loading from checkpoint file...')

            with self._graph.as_default():
                saver = tf.train.Saver(tf.global_variables())
                saver.restore(self._session, tf.train.latest_checkpoint(self._load_from_saved))

            self._has_trained = True
        else:
            warnings.warn('Tried to load state with no file given. Make sure load_from_saved is set in constructor.')
            exit()

    def _set_learning_rate(self):
        if self._lr_decay_factor is not None:
            self._log('Setting learning rate decay to every {0} steps'.format(self._lr_decay_epochs))

            self._learning_rate = tf.train.exponential_decay(self._learning_rate,
                                                             self._lr_epoch,
                                                             self._lr_decay_epochs,
                                                             self._lr_decay_factor,
                                                             staircase=True)

    def forward_pass(self, x, deterministic=False, moderation_features=None):
        """
        Perform a forward pass of the network with an input tensor. In general, this is only used when the model is
        integrated into a Tensorflow graph. See forward_pass_with_file_inputs for a version that returns network
        outputs detached from a graph.

        :param x: input tensor where the first dimension is batch
        :param deterministic: if True, performs inference-time operations on stochastic layers e.g. DropOut layers
        :param moderation_features: ???
        :return: output tensor where the first dimension is batch
        """
        residual = None
        copy_stack = []

        with self._graph.as_default():
            for layer in self._layers:
                if isinstance(layer, layers.skipConnection):
                    # The first skip only sends its residual value down to later layers. Further skips have to receive
                    # that, possibly downsample it, and add it to the latest output before setting the next residual.
                    if residual is None:
                        residual = x
                    else:
                        x = x + layer.forward_pass(residual, False)
                        residual = x
                elif isinstance(layer, layers.moderationLayer) and moderation_features is not None:
                    x = layer.forward_pass(x, deterministic, moderation_features)
                elif isinstance(layer, layers.copyConnection):
                    if layer.mode == 'save':
                        copy_stack.append(x)
                    else:
                        x = tf.concat([x, copy_stack.pop()], -1)
                else:
                    x = layer.forward_pass(x, deterministic)

        return x

    @abstractmethod
    def forward_pass_with_file_inputs(self, x):
        """
        Get network outputs with a list of filenames of images as input. Handles all the loading and batching
        automatically, so the size of the input can exceed the available memory without any problems.

        :param x: list of strings representing image filenames
        :return: ndarray representing network outputs corresponding to inputs in the same order
        """
        pass

    @abstractmethod
    def forward_pass_with_interpreted_outputs(self, x):
        """
        Performs the forward pass of the network and then interprets the raw outputs into the desired format based on
        the problem type and whether patching is being used.

        :param x: list of strings representing image filenames
        :return: ndarray representing network outputs corresponding to inputs in the same order
        """
        pass

    def add_input_layer(self):
        """Add an input layer to the network"""
        if len(self._layers) > 0:
            raise RuntimeError("Trying to add an input layer to a model that already contains other layers. " +
                               " The input layer need to be the first layer added to the model.")

        self._log('Adding the input layer...')

        apply_crop = (self._augmentation_crop and self._all_images is None and self._train_images is None)

        if apply_crop:
            size = [self._subbatch_size, int(self._image_height * self._crop_amount),
                    int(self._image_width * self._crop_amount), self._image_depth]
        else:
            size = [self._subbatch_size, self._image_height, self._image_width, self._image_depth]

        if self._with_patching:
            size = [self._subbatch_size, self._patch_height, self._patch_width, self._image_depth]

        with self._graph.as_default():
            layer = layers.inputLayer(size)

        self._layers.append(layer)

    def add_moderation_layer(self):
        """Add a moderation layer to the network"""
        self._log('Adding moderation layer...')

        reshape = self._last_layer_outputs_volume()

        feat_size = self._moderation_features_size

        with self._graph.as_default():
            layer = layers.moderationLayer(copy.deepcopy(self._last_layer().output_size),
                                           feat_size, reshape, self._subbatch_size)

        self._layers.append(layer)

    def add_convolutional_layer(self, filter_dimension, stride_length, activation_function,
                                padding=None, batch_norm=False, use_bias=True, epsilon=1e-5, decay=0.9):
        """
        Add a convolutional layer to the model.

        :param filter_dimension: array of dimensions in the format [x_size, y_size, depth, num_filters]
        :param stride_length: convolution stride length
        :param activation_function: the activation function to apply to the activation map
        :param padding: An optional amount of padding for the layer to add to the edges of inputs before convolving
        them. Defaults to using enough padding to keep the image size unchanged after convolution.
        :param batch_norm: A flag for including a batch norm layer immediately after the convolution layer but before
        the activation layer. Defaults to False (i.e. no intermediate batch norm layer)
        :param epsilon: The epsilon value to use for an intermediate batch norm layer
        :param decay: The decay value to use for an intermediate batch norm layer
        """
        if len(self._layers) < 1:
            raise RuntimeError("A convolutional layer cannot be the first layer added to the model. " +
                               "Add an input layer with DPPModel.add_input_layer() first.")
        try:
            # try to iterate through filter_dimension, checking it has 4 ints
            idx = 0
            for idx, dim in enumerate(filter_dimension):
                if not (isinstance(dim, int) or isinstance(dim, np.int64)):  # np.int64 numpy default int
                    raise TypeError()
            if idx != 3:
                raise TypeError()
        except Exception:
            raise TypeError("filter_dimension must be a list or array of 4 ints")
        if not isinstance(stride_length, int):
            raise TypeError("stride_length must be an int")
        if stride_length <= 0:
            raise ValueError("stride_length must be positive")
        if not isinstance(activation_function, str):
            raise TypeError("activation_function must be a str")

        activation_function = activation_function.lower()
        if activation_function not in self._supported_activation_functions:
            raise ValueError(
                "'" + activation_function + "' is not one of the currently supported activation functions." +
                " Choose one of: " +
                " ".join("'" + x + "'" for x in self._supported_activation_functions))

        if padding is not None:
            if not isinstance(padding, int):
                raise TypeError("padding must be an int")
            if padding < 0:
                raise ValueError("padding can't be negative")
        if not isinstance(batch_norm, bool):
            raise TypeError("batch_norm must be a boolean")
        if not isinstance(epsilon, float):
            raise TypeError("epsilon must be a float")
        if epsilon < 0:
            raise TypeError("epsilon can't be negative")
        if not isinstance(decay, float):
            raise TypeError("decay must be a float")
        if decay < 0 or decay > 1:
            raise TypeError("decay must be between 0 and 1")

        self._num_layers_conv += 1
        layer_name = 'conv%d' % self._num_layers_conv
        self._log('Adding convolutional layer %s...' % layer_name)

        with self._graph.as_default():
            filter_dimension[2] = self._last_layer().output_size[-1]
            layer = layers.convLayer(layer_name,
                                     copy.deepcopy(self._last_layer().output_size),
                                     filter_dimension,
                                     stride_length,
                                     activation_function,
                                     self._weight_initializer,
                                     padding,
                                     batch_norm,
                                     use_bias,
                                     epsilon,
                                     decay)

        self._log('Filter dimensions: {0} Outputs: {1}'.format(filter_dimension, layer.output_size))

        self._layers.append(layer)

    def add_upsampling_layer(self, filter_size, num_filters, upscale_factor=2,
                             activation_function=None, use_bias=True, regularization_coefficient=None):
        """
        Add a 2d upsampling layer to the model.

        :param filter_size: an int, representing the dimension of the square filter to be used
        :param num_filters: an int, representing the number of filters that will be outputted (the output tensor depth)
        :param upscale_factor: an int, or tuple of ints, representing the upsampling factor for rows and columns
        :param activation_function: the activation function to apply to the activation map
        :param regularization_coefficient: optionally, an L2 decay coefficient for this layer (overrides the coefficient
         set by set_regularization_coefficient)
        """
        self._num_layers_upsample += 1
        layer_name = 'upsample%d' % self._num_layers_upsample
        self._log('Adding upsampling layer %s...' % layer_name)

        if regularization_coefficient is None and self._reg_coeff is not None:
            regularization_coefficient = self._reg_coeff
        elif regularization_coefficient is None and self._reg_coeff is None:
            regularization_coefficient = 0.0

        if self._with_patching:
            patches_horiz = self._image_width // self._patch_width
            patches_vert = self._image_height // self._patch_height
            batch_multiplier = patches_horiz * patches_vert
        else:
            batch_multiplier = 1

        last_layer_dims = copy.deepcopy(self._last_layer().output_size)
        with self._graph.as_default():
            layer = layers.upsampleLayer(layer_name,
                                         last_layer_dims,
                                         filter_size,
                                         num_filters,
                                         upscale_factor,
                                         activation_function,
                                         batch_multiplier,
                                         self._weight_initializer,
                                         use_bias,
                                         regularization_coefficient)

        self._log('Filter dimensions: {0} Outputs: {1}'.format(layer.weights_shape, layer.output_size))

        self._layers.append(layer)

    def add_pooling_layer(self, kernel_size, stride_length, pooling_type='max'):
        """
        Add a pooling layer to the model.

        :param kernel_size: an integer representing the width and height dimensions of the pooling operation
        :param stride_length: convolution stride length
        :param pooling_type: optional, the type of pooling operation
        """
        if len(self._layers) < 1:
            raise RuntimeError("A pooling layer cannot be the first layer added to the model. " +
                               "Add an input layer with DPPModel.add_input_layer() first.")
        if not isinstance(kernel_size, int):
            raise TypeError("kernel_size must be an int")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if not isinstance(stride_length, int):
            raise TypeError("stride_length must be an int")
        if stride_length <= 0:
            raise ValueError("stride_length must be positive")
        if not isinstance(pooling_type, str):
            raise TypeError("pooling_type must be a str")
        pooling_type = pooling_type.lower()
        if pooling_type not in self._supported_pooling_types:
            raise ValueError("'" + pooling_type + "' is not one of the currently supported pooling types." +
                             " Choose one of: " +
                             " ".join("'"+x+"'" for x in self._supported_pooling_types))

        self._num_layers_pool += 1
        layer_name = 'pool%d' % self._num_layers_pool
        self._log('Adding pooling layer %s...' % layer_name)

        with self._graph.as_default():
            layer = layers.poolingLayer(copy.deepcopy(
                self._last_layer().output_size), kernel_size, stride_length, pooling_type)

        self._log('Outputs: %s' % layer.output_size)

        self._layers.append(layer)

    def add_normalization_layer(self):
        """Add a local response normalization layer to the model"""
        if len(self._layers) < 1:
            raise RuntimeError("A normalization layer cannot be the first layer added to the model. " +
                               "Add an input layer with DPPModel.add_input_layer() first.")

        self._num_layers_norm += 1
        layer_name = 'norm%d' % self._num_layers_pool
        self._log('Adding pooling layer %s...' % layer_name)

        with self._graph.as_default():
            layer = layers.normLayer(copy.deepcopy(self._last_layer().output_size))

        self._layers.append(layer)

    def add_dropout_layer(self, p):
        """
        Add a DropOut layer to the model.

        :param p: the keep-probability parameter for the DropOut operation
        """
        if len(self._layers) < 1:
            raise RuntimeError("A dropout layer cannot be the first layer added to the model. " +
                               "Add an input layer with DPPModel.add_input_layer() first.")
        if not isinstance(p, float):
            raise TypeError("p must be a float")
        if p < 0 or p >= 1:
            raise ValueError("p must be in range [0, 1)")

        self._num_layers_dropout += 1
        layer_name = 'drop%d' % self._num_layers_dropout
        self._log('Adding dropout layer %s...' % layer_name)

        with self._graph.as_default():
            layer = layers.dropoutLayer(copy.deepcopy(self._last_layer().output_size), p)

        self._layers.append(layer)

    def add_batch_norm_layer(self):
        """Add a batch normalization layer to the model."""
        if len(self._layers) < 1:
            raise RuntimeError("A batch norm layer cannot be the first layer added to the model.")

        self._num_layers_batchnorm += 1
        layer_name = 'bn%d' % self._num_layers_batchnorm
        self._log('Adding batch norm layer %s...' % layer_name)

        with self._graph.as_default():
            layer = layers.batchNormLayer(layer_name, copy.deepcopy(self._last_layer().output_size))

        self._layers.append(layer)

    def add_fully_connected_layer(self, output_size, activation_function, regularization_coefficient=None):
        """
        Add a fully connected layer to the model.

        :param output_size: the number of units in the layer
        :param activation_function: optionally, the activation function to use
        :param regularization_coefficient: optionally, an L2 decay coefficient for this layer (overrides the coefficient
         set by set_regularization_coefficient)
        """
        if len(self._layers) < 1:
            raise RuntimeError("A fully connected layer cannot be the first layer added to the model. " +
                               "Add an input layer with DPPModel.add_input_layer() first.")
        if not isinstance(output_size, int):
            raise TypeError("output_size must be an int")
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        if not isinstance(activation_function, str):
            raise TypeError("activation_function must be a str")

        activation_function = activation_function.lower()
        if activation_function not in self._supported_activation_functions:
            raise ValueError("'" + activation_function + "' is not one of the currently supported activation " +
                             "functions. Choose one of: " +
                             " ".join("'"+x+"'" for x in self._supported_activation_functions))

        if regularization_coefficient is not None:
            if not isinstance(regularization_coefficient, float):
                raise TypeError("regularization_coefficient must be a float or None")
            if regularization_coefficient < 0:
                raise ValueError("regularization_coefficient must be non-negative")

        self._num_layers_fc += 1
        layer_name = 'fc%d' % self._num_layers_fc
        self._log('Adding fully connected layer %s...' % layer_name)

        reshape = self._last_layer_outputs_volume()

        if regularization_coefficient is None and self._reg_coeff is not None:
            regularization_coefficient = self._reg_coeff
        if regularization_coefficient is None and self._reg_coeff is None:
            regularization_coefficient = 0.0

        with self._graph.as_default():
            layer = layers.fullyConnectedLayer(layer_name, copy.deepcopy(self._last_layer().output_size), output_size,
                                               reshape, activation_function, self._weight_initializer,
                                               regularization_coefficient)

        self._log('Inputs: {0} Outputs: {1}'.format(layer.input_size, layer.output_size))

        self._layers.append(layer)

    def add_paral_conv_block(self, filter_dimension_1, filter_dimension_2):
        """
        Add a layer(block) consisting of two parallel convolutional layers to the network

        :param filter_dimension_1: filter dimension for the first convolutional layer.
        :param filter_dimension_2: filter dimension for the second convolutional layer.
        """
        if len(self._layers) < 1:
            raise RuntimeError("An output layer cannot be the first layer added to the model. " +
                               "Add an input layer with DPPModel.add_input_layer() first.")

        def check_filter_dimension(filter_dimension):
            try:
                # try to iterate through filter_dimension, checking it has 4 ints
                idx = 0
                for idx, dim in enumerate(filter_dimension):
                    if not (isinstance(dim, int) or isinstance(dim, np.int64)):  # np.int64 numpy default int
                        raise TypeError()
                if idx != 3:
                    raise TypeError()
            except Exception:
                raise TypeError("filter_dimension {} is not a list or array of 4 ints".format(filter_dimension))

        check_filter_dimension(filter_dimension_1)
        check_filter_dimension(filter_dimension_2)
        filter_dimension_1[2] = self._last_layer().output_size[-1]
        filter_dimension_2[2] = self._last_layer().output_size[-1]

        self._num_blocks_paral_conv += 1
        block_name = 'paral_conv_block%d' % self._num_blocks_paral_conv
        self._log('Adding parallel convolutional block %s...' % block_name)

        with self._graph.as_default():
            block = layers.paralConvBlock(block_name,
                                          copy.deepcopy(self._last_layer().output_size),
                                          filter_dimension_1,
                                          filter_dimension_2)

        self._log('Filter dimensions: {0}, {1} Outputs: {2}'.format(filter_dimension_1, filter_dimension_2,
                                                                    block.output_size))

        self._layers.append(block)

    def add_skip_connection(self, downsampled=False):
        """Adds a residual connection between this point and the last residual connection"""
        if len(self._layers) < 1:
            raise RuntimeError("A skip connection cannot be the first layer added to the model.")

        self._num_skip_connections += 1
        layer_name = 'skip%d' % self._num_skip_connections
        self._log('Adding skip connection...')

        layer = layers.skipConnection(layer_name, self._last_layer().output_size, downsampled)

        self._log('Inputs: {0} Outputs: {1}'.format(layer.input_size, layer.output_size))

        self._layers.append(layer)

    def add_copy_connection(self, mode):
        if len(self._layers) < 1:
            raise RuntimeError("A copy connection cannot be the first layer added to the model.")

        self._num_copy_connections += 1
        layer_name = 'copy%d' % self._num_copy_connections
        self._log('Adding copy connection ({0})...'.format(mode))

        layer = layers.copyConnection(layer_name, self._last_layer().output_size, mode)

        self._log('Inputs: {0} Outputs: {1}'.format(layer.input_size, layer.output_size))

        self._layers.append(layer)

    def add_global_average_pooling_layer(self):
        """Adds a global average pooling layer"""
        if len(self._layers) < 1:
            raise RuntimeError("A GAP layer cannot be the first layer added to the model.")

        self._num_skip_connections += 1
        layer_name = 'GAP'
        self._log('Adding global average pooling layer...')

        layer = layers.globalAveragePoolingLayer(layer_name, self._last_layer().output_size)

        self._log('Inputs: {0} Outputs: {1}'.format(layer.input_size, layer.output_size))

        self._layers.append(layer)

    @abstractmethod
    def add_output_layer(self, regularization_coefficient=None, output_size=None):
        """
        Add an output layer to the network (affine layer where the number of units equals the number of network outputs)

        :param regularization_coefficient: optionally, an L2 decay coefficient for this layer (overrides the coefficient
         set by set_regularization_coefficient)
        :param output_size: optionally, override the output size of this layer. Typically not needed, but required for
        use cases such as creating the output layer before loading data.
        """
        pass

    def use_predefined_model(self, model_name):
        """
        Add network layers to build a predefined network model
        :param model_name: The predefined model name
        """
        if model_name not in self._supported_predefined_models:
            raise ValueError("'" + model_name + "' is not one of the currently supported predefined models." +
                             " Make sure you have the correct problem type set with DPPModel.set_problem_type() " +
                             "first, or choose one of " +
                             " ".join("'" + x + "'" for x in self._supported_predefined_models))

        if model_name == 'u-net':
            bn = False

            self.add_input_layer()

            self.add_convolutional_layer(filter_dimension=[3, 3, self._image_depth, 64], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_convolutional_layer(filter_dimension=[3, 3, 64, 64], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_copy_connection('save')
            self.add_pooling_layer(kernel_size=2, stride_length=2)

            self.add_convolutional_layer(filter_dimension=[3, 3, 64, 128], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_convolutional_layer(filter_dimension=[3, 3, 128, 128], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_copy_connection('save')
            self.add_pooling_layer(kernel_size=2, stride_length=2)

            self.add_convolutional_layer(filter_dimension=[3, 3, 128, 256], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_convolutional_layer(filter_dimension=[3, 3, 256, 256], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_copy_connection('save')
            self.add_pooling_layer(kernel_size=2, stride_length=2)

            self.add_convolutional_layer(filter_dimension=[3, 3, 256, 512], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 512], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_copy_connection('save')
            self.add_pooling_layer(kernel_size=2, stride_length=2)

            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 1024], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_convolutional_layer(filter_dimension=[3, 3, 1024, 1024], stride_length=1, activation_function='relu', batch_norm=bn)

            self.add_upsampling_layer(filter_size=2, num_filters=512, activation_function='relu')

            self.add_copy_connection('load')
            self.add_convolutional_layer(filter_dimension=[3, 3, 1024, 512], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 512], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_upsampling_layer(filter_size=2, num_filters=256, activation_function='relu')

            self.add_copy_connection('load')
            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 256], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_convolutional_layer(filter_dimension=[3, 3, 256, 256], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_upsampling_layer(filter_size=2, num_filters=128, activation_function='relu')

            self.add_copy_connection('load')
            self.add_convolutional_layer(filter_dimension=[3, 3, 256, 128], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_convolutional_layer(filter_dimension=[3, 3, 128, 128], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_upsampling_layer(filter_size=2, num_filters=64, activation_function='relu')

            self.add_copy_connection('load')
            self.add_convolutional_layer(filter_dimension=[3, 3, 128, 64], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_convolutional_layer(filter_dimension=[3, 3, 64, 64], stride_length=1, activation_function='relu', batch_norm=bn)

            self.add_output_layer()

        if model_name == 'fcn-18':
            bn = False

            self.add_input_layer()

            self.add_convolutional_layer(filter_dimension=[3, 3, self._image_depth, 64], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_convolutional_layer(filter_dimension=[3, 3, 64, 64], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_pooling_layer(kernel_size=2, stride_length=2)
            self.add_skip_connection()

            self.add_convolutional_layer(filter_dimension=[3, 3, 64, 128], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_convolutional_layer(filter_dimension=[3, 3, 128, 128], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_pooling_layer(kernel_size=2, stride_length=2)
            self.add_skip_connection(downsampled=True)

            self.add_convolutional_layer(filter_dimension=[3, 3, 128, 256], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_convolutional_layer(filter_dimension=[3, 3, 256, 256], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_pooling_layer(kernel_size=2, stride_length=2)
            self.add_skip_connection(downsampled=True)

            self.add_convolutional_layer(filter_dimension=[3, 3, 256, 512], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 512], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_pooling_layer(kernel_size=2, stride_length=2)
            self.add_skip_connection(downsampled=True)

            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 1024], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_convolutional_layer(filter_dimension=[3, 3, 1024, 1024], stride_length=1, activation_function='relu', batch_norm=bn)

            self.add_upsampling_layer(filter_size=2, num_filters=512, activation_function='relu')

            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 512], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 512], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_upsampling_layer(filter_size=2, num_filters=256, activation_function='relu')

            self.add_convolutional_layer(filter_dimension=[3, 3, 256, 256], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_convolutional_layer(filter_dimension=[3, 3, 256, 256], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_upsampling_layer(filter_size=2, num_filters=128, activation_function='relu')

            self.add_convolutional_layer(filter_dimension=[3, 3, 128, 128], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_convolutional_layer(filter_dimension=[3, 3, 128, 128], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_upsampling_layer(filter_size=2, num_filters=64, activation_function='relu')

            self.add_convolutional_layer(filter_dimension=[3, 3, 64, 64], stride_length=1, activation_function='relu', batch_norm=bn)
            self.add_convolutional_layer(filter_dimension=[3, 3, 64, 64], stride_length=1, activation_function='relu', batch_norm=bn)

            self.add_output_layer()

        if model_name == 'vgg-16':
            self.add_input_layer()

            self.add_convolutional_layer(filter_dimension=[3, 3, self._image_depth, 64],
                                         stride_length=1, activation_function='relu')
            self.add_convolutional_layer(filter_dimension=[3, 3, 64, 64], stride_length=1, activation_function='relu')
            self.add_pooling_layer(kernel_size=2, stride_length=2)

            self.add_convolutional_layer(filter_dimension=[3, 3, 64, 128], stride_length=1, activation_function='relu')
            self.add_convolutional_layer(filter_dimension=[3, 3, 128, 128], stride_length=1, activation_function='relu')
            self.add_pooling_layer(kernel_size=2, stride_length=2)

            self.add_convolutional_layer(filter_dimension=[3, 3, 128, 256], stride_length=1, activation_function='relu')
            self.add_convolutional_layer(filter_dimension=[3, 3, 256, 256], stride_length=1, activation_function='relu')
            self.add_pooling_layer(kernel_size=2, stride_length=2)

            self.add_convolutional_layer(filter_dimension=[3, 3, 256, 512], stride_length=1, activation_function='relu')
            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 512], stride_length=1, activation_function='relu')
            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 512], stride_length=1, activation_function='relu')
            self.add_pooling_layer(kernel_size=2, stride_length=2)

            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 512], stride_length=1, activation_function='relu')
            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 512], stride_length=1, activation_function='relu')
            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 512], stride_length=1, activation_function='relu')
            self.add_pooling_layer(kernel_size=2, stride_length=2)

            self.add_fully_connected_layer(output_size=4096, activation_function='relu')
            self.add_dropout_layer(0.5)
            self.add_fully_connected_layer(output_size=4096, activation_function='relu')
            self.add_dropout_layer(0.5)

            self.add_output_layer()

        if model_name == 'alexnet':
            self.add_input_layer()

            self.add_convolutional_layer(filter_dimension=[11, 11, self._image_depth, 48],
                                         stride_length=4, activation_function='relu')
            self.add_normalization_layer()
            self.add_pooling_layer(kernel_size=3, stride_length=2)

            self.add_convolutional_layer(filter_dimension=[5, 5, 48, 256], stride_length=1, activation_function='relu')
            self.add_normalization_layer()
            self.add_pooling_layer(kernel_size=3, stride_length=2)

            self.add_convolutional_layer(filter_dimension=[3, 3, 256, 384], stride_length=1, activation_function='relu')
            self.add_convolutional_layer(filter_dimension=[3, 3, 384, 384], stride_length=1, activation_function='relu')
            self.add_convolutional_layer(filter_dimension=[3, 3, 384, 256], stride_length=1, activation_function='relu')
            self.add_pooling_layer(kernel_size=3, stride_length=2)

            self.add_fully_connected_layer(output_size=4096, activation_function='relu')
            self.add_dropout_layer(0.5)
            self.add_fully_connected_layer(output_size=4096, activation_function='relu')
            self.add_dropout_layer(0.5)

            self.add_output_layer()

        if model_name == 'resnet-18':
            self.add_input_layer()

            self.add_convolutional_layer(filter_dimension=[7, 7, self._image_depth, 64], stride_length=2,
                                         activation_function='relu', batch_norm=True)
            self.add_pooling_layer(kernel_size=3, stride_length=2)

            self.add_skip_connection()
            self.add_convolutional_layer(filter_dimension=[3, 3, 64, 64], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_convolutional_layer(filter_dimension=[3, 3, 64, 64], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_skip_connection()
            self.add_convolutional_layer(filter_dimension=[3, 3, 64, 64], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_convolutional_layer(filter_dimension=[3, 3, 64, 128], stride_length=2,
                                         activation_function='relu', batch_norm=True)
            self.add_skip_connection(downsampled=True)

            self.add_convolutional_layer(filter_dimension=[3, 3, 128, 128], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_convolutional_layer(filter_dimension=[3, 3, 128, 128], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_skip_connection()
            self.add_convolutional_layer(filter_dimension=[3, 3, 128, 128], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_convolutional_layer(filter_dimension=[3, 3, 128, 256], stride_length=2,
                                         activation_function='relu', batch_norm=True)
            self.add_skip_connection(downsampled=True)

            self.add_convolutional_layer(filter_dimension=[3, 3, 256, 256], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_convolutional_layer(filter_dimension=[3, 3, 256, 256], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_skip_connection()
            self.add_convolutional_layer(filter_dimension=[3, 3, 256, 256], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_convolutional_layer(filter_dimension=[3, 3, 256, 512], stride_length=2,
                                         activation_function='relu', batch_norm=True)
            self.add_skip_connection(downsampled=True)

            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 512], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 512], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_skip_connection()
            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 512], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 512], stride_length=2,
                                         activation_function='relu', batch_norm=True)
            self.add_skip_connection()

            self.add_global_average_pooling_layer()

            self.add_fully_connected_layer(output_size=1000, activation_function='relu')

            self.add_output_layer()

        if model_name == 'xsmall':
            self.add_input_layer()

            self.add_convolutional_layer(filter_dimension=[3, 3, self._image_depth, 16],
                                         stride_length=1, activation_function='relu')
            self.add_pooling_layer(kernel_size=2, stride_length=2)

            self.add_convolutional_layer(filter_dimension=[3, 3, 16, 32], stride_length=1, activation_function='relu')
            self.add_pooling_layer(kernel_size=2, stride_length=2)

            self.add_convolutional_layer(filter_dimension=[3, 3, 32, 32], stride_length=1, activation_function='relu')
            self.add_pooling_layer(kernel_size=2, stride_length=2)

            self.add_fully_connected_layer(output_size=64, activation_function='relu')

            self.add_output_layer()

        if model_name == 'small':
            self.add_input_layer()

            self.add_convolutional_layer(filter_dimension=[3, 3, self._image_depth, 64],
                                         stride_length=1, activation_function='relu', batch_norm=True)
            self.add_pooling_layer(kernel_size=2, stride_length=2)

            self.add_convolutional_layer(filter_dimension=[3, 3, 64, 128], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_convolutional_layer(filter_dimension=[3, 3, 128, 128], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_pooling_layer(kernel_size=2, stride_length=2)

            self.add_convolutional_layer(filter_dimension=[3, 3, 128, 128], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_convolutional_layer(filter_dimension=[3, 3, 128, 128], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_pooling_layer(kernel_size=2, stride_length=2)

            self.add_fully_connected_layer(output_size=64, activation_function='relu')

            self.add_output_layer()

        if model_name == 'medium':
            self.add_input_layer()

            self.add_convolutional_layer(filter_dimension=[3, 3, self._image_depth, 64],
                                         stride_length=1, activation_function='relu', batch_norm=True)
            self.add_convolutional_layer(filter_dimension=[3, 3, 64, 64], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_pooling_layer(kernel_size=2, stride_length=2)

            self.add_convolutional_layer(filter_dimension=[3, 3, 64, 128], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_convolutional_layer(filter_dimension=[3, 3, 128, 128], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_pooling_layer(kernel_size=2, stride_length=2)

            self.add_convolutional_layer(filter_dimension=[3, 3, 128, 256], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_convolutional_layer(filter_dimension=[3, 3, 256, 256], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_pooling_layer(kernel_size=2, stride_length=2)

            self.add_convolutional_layer(filter_dimension=[3, 3, 256, 512], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 512], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 512], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_pooling_layer(kernel_size=2, stride_length=2)

            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 512], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 512], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 512], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_pooling_layer(kernel_size=2, stride_length=2)

            self.add_fully_connected_layer(output_size=256, activation_function='relu')

            self.add_output_layer()

        if model_name == 'large':
            self.add_input_layer()

            self.add_convolutional_layer(filter_dimension=[3, 3, self._image_depth, 64], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_convolutional_layer(filter_dimension=[3, 3, 64, 64], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_pooling_layer(kernel_size=2, stride_length=2)

            self.add_convolutional_layer(filter_dimension=[3, 3, 64, 128], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_convolutional_layer(filter_dimension=[3, 3, 128, 128], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_pooling_layer(kernel_size=2, stride_length=2)

            self.add_convolutional_layer(filter_dimension=[3, 3, 128, 256], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_convolutional_layer(filter_dimension=[3, 3, 256, 256], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_pooling_layer(kernel_size=2, stride_length=2)

            self.add_convolutional_layer(filter_dimension=[3, 3, 256, 512], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 512], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 512], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_pooling_layer(kernel_size=2, stride_length=2)

            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 512], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 512], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 512], stride_length=1,
                                         activation_function='relu', batch_norm=True)
            self.add_pooling_layer(kernel_size=2, stride_length=2)

            self.add_fully_connected_layer(output_size=512, activation_function='relu')
            self.add_fully_connected_layer(output_size=384, activation_function='relu')

            self.add_output_layer()

        if model_name == 'yolov2':
            self.add_input_layer()

            self.add_convolutional_layer(filter_dimension=[3, 3, self._image_depth, 32],  stride_length=1,
                                         activation_function='lrelu')
            self.add_pooling_layer(kernel_size=3, stride_length=2)

            self.add_convolutional_layer(filter_dimension=[3, 3, 32, 64], stride_length=1, activation_function='lrelu')
            self.add_pooling_layer(kernel_size=3, stride_length=2)

            self.add_convolutional_layer(filter_dimension=[3, 3, 64, 128], stride_length=1, activation_function='lrelu')
            self.add_convolutional_layer(filter_dimension=[1, 1, 128, 64], stride_length=1, activation_function='lrelu')
            self.add_convolutional_layer(filter_dimension=[3, 3, 64, 128], stride_length=1, activation_function='lrelu')
            self.add_pooling_layer(kernel_size=3, stride_length=2)

            self.add_convolutional_layer(filter_dimension=[3, 3, 128, 256],
                                         stride_length=1, activation_function='lrelu')
            self.add_convolutional_layer(filter_dimension=[1, 1, 256, 128],
                                         stride_length=1, activation_function='lrelu')
            self.add_convolutional_layer(filter_dimension=[3, 3, 128, 256],
                                         stride_length=1, activation_function='lrelu')
            self.add_pooling_layer(kernel_size=3, stride_length=2)

            self.add_convolutional_layer(filter_dimension=[3, 3, 256, 512],
                                         stride_length=1, activation_function='lrelu')
            self.add_convolutional_layer(filter_dimension=[1, 1, 512, 256],
                                         stride_length=1, activation_function='lrelu')
            self.add_convolutional_layer(filter_dimension=[3, 3, 256, 512],
                                         stride_length=1, activation_function='lrelu')
            self.add_convolutional_layer(filter_dimension=[1, 1, 512, 256],
                                         stride_length=1, activation_function='lrelu')
            self.add_convolutional_layer(filter_dimension=[3, 3, 256, 512],
                                         stride_length=1, activation_function='lrelu')
            self.add_pooling_layer(kernel_size=3, stride_length=2)

            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 1024],
                                         stride_length=1, activation_function='lrelu')
            self.add_convolutional_layer(filter_dimension=[1, 1, 1024, 512],
                                         stride_length=1, activation_function='lrelu')
            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 1024],
                                         stride_length=1, activation_function='lrelu')
            self.add_convolutional_layer(filter_dimension=[1, 1, 1024, 512],
                                         stride_length=1, activation_function='lrelu')
            self.add_convolutional_layer(filter_dimension=[3, 3, 512, 1024],
                                         stride_length=1, activation_function='lrelu')
            self.add_pooling_layer(kernel_size=3, stride_length=2)

            self.add_convolutional_layer(filter_dimension=[3, 3, 1024, 1024],
                                         stride_length=1, activation_function='lrelu')
            self.add_convolutional_layer(filter_dimension=[3, 3, 1024, 1024],
                                         stride_length=1, activation_function='lrelu')
            self.add_convolutional_layer(filter_dimension=[3, 3, 1024, 1024],
                                         stride_length=1, activation_function='lrelu')

            self.add_output_layer()

        if model_name == 'countception':
            patch_size = 32
            self.add_input_layer()
            self.add_convolutional_layer(filter_dimension=[3, 3, 3, 64],
                                         stride_length=1,
                                         activation_function='lrelu',
                                         padding=patch_size,
                                         batch_norm=True,
                                         epsilon=1e-5,
                                         decay=0.9)
            self.add_paral_conv_block(filter_dimension_1=[1, 1, 0, 16],
                                      filter_dimension_2=[3, 3, 0, 16])
            self.add_paral_conv_block(filter_dimension_1=[1, 1, 0, 16],
                                      filter_dimension_2=[3, 3, 0, 32])
            self.add_convolutional_layer(filter_dimension=[14, 14, 0, 16],
                                         stride_length=1,
                                         activation_function='lrelu',
                                         padding=0,
                                         batch_norm=True,
                                         epsilon=1e-5,
                                         decay=0.9)
            self.add_paral_conv_block(filter_dimension_1=[1, 1, 0, 112],
                                      filter_dimension_2=[3, 3, 0, 48])
            self.add_paral_conv_block(filter_dimension_1=[1, 1, 0, 64],
                                      filter_dimension_2=[3, 3, 0, 32])
            self.add_paral_conv_block(filter_dimension_1=[1, 1, 0, 40],
                                      filter_dimension_2=[3, 3, 0, 40])
            self.add_paral_conv_block(filter_dimension_1=[1, 1, 0, 32],
                                      filter_dimension_2=[3, 3, 0, 96])
            self.add_convolutional_layer(filter_dimension=[18, 18, 0, 32],
                                         stride_length=1,
                                         activation_function='lrelu',
                                         padding=0,
                                         batch_norm=True,
                                         epsilon=1e-5,
                                         decay=0.9)
            self.add_convolutional_layer(filter_dimension=[1, 1, 0, 64],
                                         stride_length=1,
                                         activation_function='lrelu',
                                         padding=0,
                                         batch_norm=True,
                                         epsilon=1e-5,
                                         decay=0.9)
            self.add_convolutional_layer(filter_dimension=[1, 1, 0, 64],
                                         stride_length=1,
                                         activation_function='lrelu',
                                         padding=0,
                                         batch_norm=True,
                                         epsilon=1e-5,
                                         decay=0.9)
            self.add_convolutional_layer(filter_dimension=[1, 1, 0, 1],
                                         stride_length=1,
                                         activation_function='lrelu',
                                         padding=0,
                                         batch_norm=True,
                                         epsilon=1e-5,
                                         decay=0.9)

    def load_dataset_from_directory_with_csv_labels(self, dirname, labels_file, column_number=False):
        """
        Loads the png images in the given directory into an internal representation, using the labels provided in a CSV
        file.

        :param dirname: the path of the directory containing the images
        :param labels_file: the path of the .csv file containing the labels
        :param column_number: the column number (zero-indexed) of the column in the csv file representing the label
        """
        if not isinstance(dirname, str):
            raise TypeError("dirname must be a str")
        if not os.path.isdir(dirname):
            raise ValueError("'"+dirname+"' does not exist")
        if not isinstance(labels_file, str):
            raise TypeError("labels_file must be a str")

        image_files = [os.path.join(dirname, name) for name in os.listdir(dirname) if
                       os.path.isfile(os.path.join(dirname, name)) & name.endswith('.png')]

        labels = loaders.read_csv_labels(labels_file, column_number)

        self._total_raw_samples = len(image_files)
        self._total_classes = len(set(labels))

        self._log('Total raw examples is %d' % self._total_raw_samples)
        self._log('Total classes is %d' % self._total_classes)

        self._raw_image_files = image_files
        self._raw_labels = labels
        self._split_labels = False  # Band-aid fix

    def load_ippn_tray_dataset_from_directory(self, dirname):
        """
        Loads the RGB tray images and plant bounding box labels from the International Plant Phenotyping Network
        dataset.
        """
        self._resize_bbox_coords = True

        images = [os.path.join(dirname, name) for name in sorted(os.listdir(dirname)) if
                  os.path.isfile(os.path.join(dirname, name)) & name.endswith('_rgb.png')]

        label_files = [os.path.join(dirname, name) for name in sorted(os.listdir(dirname)) if
                       os.path.isfile(os.path.join(dirname, name)) & name.endswith('_bbox.csv')]

        # currently reads columns, need to read rows instead!!!
        labels = [loaders.read_csv_rows(label_file) for label_file in label_files]

        self._all_labels = []
        for label in labels:
            curr_label = []
            for nums in label:
                curr_label.extend(loaders.box_coordinates_to_pascal_voc_coordinates(nums))
            self._all_labels.append(curr_label)

        self._total_raw_samples = len(images)

        self._log('Total raw examples is %d' % self._total_raw_samples)
        self._log('Parsing dataset...')

        self._raw_image_files = images
        self._raw_labels = self._all_labels

    def load_ippn_leaf_count_dataset_from_directory(self, dirname):
        """Loads the RGB images and species labels from the International Plant Phenotyping Network dataset."""
        if self._image_height is None or self._image_width is None or self._image_depth is None:
            raise RuntimeError("Image dimensions need to be set before loading data." +
                               " Try using DPPModel.set_image_dimensions() first.")
        if self._maximum_training_batches is None:
            raise RuntimeError("The number of maximum training epochs needs to be set before loading data." +
                               " Try using DPPModel.set_maximum_training_epochs() first.")

        labels, ids = loaders.read_csv_labels_and_ids(os.path.join(dirname, 'Leaf_counts.csv'), 1, 0)

        # labels must be lists
        labels = [[label] for label in labels]

        image_files = [os.path.join(dirname, im_id + '_rgb.png') for im_id in ids]

        self._total_raw_samples = len(image_files)

        self._log('Total raw examples is %d' % self._total_raw_samples)
        self._log('Parsing dataset...')

        self._raw_image_files = image_files
        self._raw_labels = labels

    def load_inra_dataset_from_directory(self, dirname):
        """Loads the RGB images and labels from the INRA dataset."""

        labels, ids = loaders.read_csv_labels_and_ids(os.path.join(dirname, 'AutomatonImages.csv'), 1, 3, character=';')

        # Remove the header line
        labels.pop(0)
        ids.pop(0)

        image_files = [os.path.join(dirname, im_id) for im_id in ids]

        self._total_raw_samples = len(image_files)
        self._total_classes = len(set(labels))

        # transform into numerical one-hot labels
        labels = loaders.string_labels_to_sequential(labels)
        labels = tf.one_hot(labels, self._total_classes)

        self._log('Total raw examples is %d' % self._total_raw_samples)
        self._log('Total classes is %d' % self._total_classes)
        self._log('Parsing dataset...')

        self._raw_image_files = image_files
        self._raw_labels = labels

    def load_cifar10_dataset_from_directory(self, dirname):
        """
        Loads the images and labels from a directory containing the CIFAR-10 image classification dataset as
        downloaded by nvidia DIGITS.
        """

        train_dir = os.path.join(dirname, 'train')
        test_dir = os.path.join(dirname, 'test')
        self._total_classes = 10

        train_labels, train_images = loaders.read_csv_labels_and_ids(os.path.join(train_dir, 'train.txt'), 1, 0,
                                                                     character=' ')

        def one_hot(labels, num_classes):
            return [[1 if i == label else 0 for i in range(num_classes)] for label in labels]

        # transform into numerical one-hot labels
        train_labels = [int(label) for label in train_labels]
        train_labels = one_hot(train_labels, self._total_classes)

        test_labels, test_images = loaders.read_csv_labels_and_ids(os.path.join(test_dir, 'test.txt'), 1, 0,
                                                                   character=' ')

        # transform into numerical one-hot labels
        test_labels = [int(label) for label in test_labels]
        test_labels = one_hot(test_labels, self._total_classes)

        self._total_raw_samples = len(train_images) + len(test_images)
        self._test_split = len(test_images) / self._total_raw_samples

        self._log('Total raw examples is %d' % self._total_raw_samples)
        self._log('Total classes is %d' % self._total_classes)

        self._raw_test_image_files = test_images
        self._raw_train_image_files = train_images
        self._raw_test_labels = test_labels
        self._raw_train_labels = train_labels
        if not self._testing:
            self._raw_train_image_files.extend(self._raw_test_image_files)
            self._raw_test_image_files = []
            self._raw_train_labels.extend(self._raw_test_labels)
            self._raw_test_labels = []
            self._test_split = 0
        if self._validation:
            num_val_samples = int(self._total_raw_samples * self._validation_split)
            self._raw_val_image_files = self._raw_train_image_files[:num_val_samples]
            self._raw_train_image_files = self._raw_train_image_files[num_val_samples:]
            self._raw_val_labels = self._raw_train_labels[:num_val_samples]
            self._raw_train_labels = self._raw_train_labels[num_val_samples:]

    def load_lemnatec_images_from_directory(self, dirname):
        """
        Loads the RGB (VIS) images from a Lemnatec plant scanner image dataset. Unless you only want to do
        preprocessing, regression or classification labels MUST be loaded first.
        """

        # Load all snapshot subdirectories
        subdirs = list(filter(lambda item: os.path.isdir(item) & (item != '.DS_Store'),
                              [os.path.join(dirname, f) for f in os.listdir(dirname)]))

        image_files = []

        # Load the VIS images in each subdirectory
        for sd in subdirs:
            image_paths = [os.path.join(sd, name) for name in os.listdir(sd) if
                           os.path.isfile(os.path.join(sd, name)) & name.startswith('VIS_SV_')]

            image_files = image_files + image_paths

        # Put the image files in the order of the IDs (if there are any labels loaded)
        sorted_paths = []

        if self._all_labels is not None:
            for image_id in self._all_ids:
                path = list(filter(lambda item: item.endswith(image_id), [p for p in image_files]))
                assert len(path) == 1, 'Found no image or multiple images for %r' % image_id
                sorted_paths.append(path[0])
        else:
            sorted_paths = image_files

        self._total_raw_samples = len(sorted_paths)

        self._log('Total raw examples is %d' % self._total_raw_samples)
        self._log('Parsing dataset...')

        images = sorted_paths

        # prepare images for training (if there are any labels loaded)
        if self._all_labels is not None:
            labels = self._all_labels

            self._raw_image_files = images
            self._raw_labels = labels

    def load_images_from_list(self, image_files):
        """
        Loads images from a list of file names (strings). Regression or classification labels MUST be loaded first.
        """

        self._total_raw_samples = len(image_files)

        self._log('Total raw examples is %d' % self._total_raw_samples)
        self._log('Parsing dataset...')

        images = image_files

        self._raw_image_files = images
        if self._all_labels is not None:
            self._raw_labels = self._all_labels
        else:
            self._images_only = True

    def load_multiple_labels_from_csv(self, filepath, id_column=0):
        """
        Load multiple labels from a CSV file, for instance values for regression.
        Parameter id_column is the column number specifying the image file name.
        """

        self._all_labels, self._all_ids = loaders.read_csv_multi_labels_and_ids(filepath, id_column)

    def load_images_with_ids_from_directory(self, im_dir):
        """Loads images from a directory, relating them to labels by the IDs which were loaded from a CSV file"""

        # Load all images in directory
        image_files = [os.path.join(im_dir, name) for name in os.listdir(im_dir) if
                       os.path.isfile(os.path.join(im_dir, name)) & name.endswith('.png')]

        # Put the image files in the order of the IDs (if there are any labels loaded)
        sorted_paths = []

        if self._all_labels is not None:
            for image_id in self._all_ids:
                path = list(filter(lambda item: item.endswith('/' + image_id), [p for p in image_files]))
                assert len(path) == 1, 'Found no image or multiple images for %r' % image_id
                sorted_paths.append(path[0])
        else:
            sorted_paths = image_files

        self._total_raw_samples = len(sorted_paths)

        self._log('Total raw examples is %d' % self._total_raw_samples)
        self._log('Parsing dataset...')

        processed_images = sorted_paths

        # prepare images for training (if there are any labels loaded)
        if self._all_labels is not None:
            self._raw_image_files = processed_images
            self._raw_labels = self._all_labels

    def load_training_augmentation_dataset_from_directory_with_csv_labels(self, dirname, labels_file, column_number=1,
                                                                          id_column_number=0):
        """
        Loads the png images from a directory as training augmentation images, using the labels provided in a CSV file.

        :param dirname: the path of the directory containing the images
        :param labels_file: the path of the .csv file containing the labels
        :param column_number: the column number (zero-indexed) of the column in the csv file representing the label
        :param id_column_number: the column number (zero-indexed) representing the file ID
        """

        image_files = [os.path.join(dirname, name) for name in os.listdir(dirname) if
                       os.path.isfile(os.path.join(dirname, name)) & name.endswith('.png')]

        labels, ids = loaders.read_csv_labels_and_ids(labels_file, column_number, id_column_number)

        sorted_paths = []

        for image_id in ids:
            path = list(filter(lambda item: item.endswith('/' + image_id), [p for p in image_files]))
            assert len(path) == 1, 'Found no image or multiple images for %r' % image_id
            sorted_paths.append(path[0])

        self._training_augmentation_images = sorted_paths
        self._training_augmentation_labels = labels

    def load_pascal_voc_labels_from_directory(self, data_dir):
        """Loads single per-image bounding boxes from XML files in Pascal VOC format."""

        self._all_ids = []
        self._all_labels = []

        file_paths = [os.path.join(data_dir, name) for name in os.listdir(data_dir) if
                      os.path.isfile(os.path.join(data_dir, name)) & name.endswith('.xml')]

        for voc_file in file_paths:
            im_id, x_min, x_max, y_min, y_max = loaders.read_single_bounding_box_from_pascal_voc(voc_file)

            # re-scale coordinates if images are being resized
            if self._resize_images:
                x_min = int(x_min * (float(self._image_width) / self._image_width_original))
                x_max = int(x_max * (float(self._image_width) / self._image_width_original))
                y_min = int(y_min * (float(self._image_height) / self._image_height_original))
                y_max = int(y_max * (float(self._image_height) / self._image_height_original))

            self._all_ids.append(im_id)
            self._all_labels.append([x_min, x_max, y_min, y_max])

    def load_json_labels_from_file(self, filename):
        """Loads bounding boxes for multiple images from a single json file."""

        self._all_ids = []
        self._all_labels = []

        with open(filename, 'r', encoding='utf-8-sig') as f:
            box_data = json.load(f)
        for box in sorted(box_data.items()):
            self._all_ids.append(box[0])  # Name of corresponding image
            w_original = box[1]['width']
            h_original = box[1]['height']
            boxes = []
            for plant in box[1]['plants']:
                x_min = plant['all_points_x'][0]
                x_max = plant['all_points_x'][1]
                y_min = plant['all_points_y'][0]
                y_max = plant['all_points_y'][1]

                # re-scale coordinates if images are being resized
                if self._resize_images:
                    x_min = int(x_min * (float(self._image_width) / w_original))
                    x_max = int(x_max * (float(self._image_width) / w_original))
                    y_min = int(y_min * (float(self._image_height) / h_original))
                    y_max = int(y_max * (float(self._image_height) / h_original))

                boxes.append([x_min, x_max, y_min, y_max])
            self._all_labels.append(boxes)

    def _parse_dataset(self, train_images, train_labels, train_mf,
                       test_images, test_labels, test_mf,
                       val_images, val_labels, val_mf):
        """Parses training & testing images and labels, creating input pipelines internal to this instance"""
        with self._graph.as_default():
            # Get the number of training, testing, and validation samples
            self._parse_get_sample_counts(train_images, test_images, val_images)

            # Logging verbosity
            self._log('Total training samples is {0}'.format(self._total_training_samples))
            self._log('Total validation samples is {0}'.format(self._total_validation_samples))
            self._log('Total testing samples is {0}'.format(self._total_testing_samples))

            # Calculate number of batches to run
            batches_per_epoch = self._total_training_samples / float(self._batch_size)
            self._maximum_training_batches = int(self._maximum_training_batches * batches_per_epoch)

            if self._batch_size > self._total_training_samples:
                self._log('Less than one batch in training set, exiting now')
                exit()
            self._log('Batches per epoch: {:f}'.format(batches_per_epoch))
            self._log('Running to {0} batches'.format(self._maximum_training_batches))

            # Create datasets for moderation features
            def _make_mod_features_dataset(mod):
                mod_dataset = tf.data.Dataset.from_tensor_slices(mod)
                mod_dataset = mod_dataset.map(lambda x: tf.cast(x, tf.float32), num_parallel_calls=self._num_threads)
                return mod_dataset

            if train_mf is not None:
                self._train_moderation_features = _make_mod_features_dataset(train_mf)
            if test_mf is not None:
                self._test_moderation_features = _make_mod_features_dataset(test_mf)
            if val_mf is not None:
                self._val_moderation_features = _make_mod_features_dataset(val_mf)

            # Create datasets for the input data
            self._train_dataset = self._make_input_dataset(train_images, train_labels, True)
            if self._testing:
                self._test_dataset = self._make_input_dataset(test_images, test_labels, False)
            if self._validation:
                self._val_dataset = self._make_input_dataset(val_images, val_labels, False)

            # Set the image size to cropped values if crop augmentation was used
            if self._augmentation_crop:
                self._image_height = int(self._image_height * self._crop_amount)
                self._image_width = int(self._image_width * self._crop_amount)

    def _make_input_dataset(self, images, labels, train_set):
        """
        Create Tensorflow datasets and construct an input and augmentation pipeline given paired images and labels
        :param images: A list of image names for the dataset
        :param labels: The labels corresponding to the images
        :param train_set: A flag for whether this is the training dataset; certain augmentations only occur or change
        for training data specifically
        :return: A tf.data.Dataset object that encapsulates the data input and augmentation pipeline
        """
        def _with_labels(fn):
            """Takes a function on images only and appends its labels to the output"""
            return lambda im, lab: (fn(im), lab)

        data_height = self._image_height
        data_width = self._image_width

        # Create the dataset and load in the images
        input_dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        input_dataset = input_dataset.map(self._parse_apply_preprocessing, num_parallel_calls=self._num_threads)
        if self._resize_images:
            input_dataset = input_dataset.map(lambda x, y: self._parse_resize_images(x, y, data_height, data_width),
                                              num_parallel_calls=self._num_threads)

        # Augmentations that we should do to every dataset (training, testing, and validation)
        if self._augmentation_crop:  # Apply random crops to images
            data_height = int(data_height * self._crop_amount)
            data_width = int(data_width * self._crop_amount)
            if train_set:
                input_dataset = input_dataset.map(
                    _with_labels(lambda x: tf.random_crop(x, [data_height, data_width, self._image_depth])),
                    num_parallel_calls=self._num_threads)
            else:
                input_dataset = input_dataset.map(lambda x, y: self._parse_crop_or_pad(x, y, data_height, data_width),
                                                  num_parallel_calls=self._num_threads)

        if self._crop_or_pad_images:  # Apply padding or cropping to deal with images of different sizes
            input_dataset = input_dataset.map(lambda x, y: self._parse_crop_or_pad(x, y, data_height, data_width),
                                              num_parallel_calls=self._num_threads)

        if train_set:
            # Augmentations that we should only do to the training dataset
            if self._augmentation_flip_horizontal:  # Apply random horizontal flips
                input_dataset = input_dataset.map(_with_labels(tf.image.random_flip_left_right),
                                                  num_parallel_calls=self._num_threads)

            if self._augmentation_flip_vertical:  # Apply random vertical flips
                input_dataset = input_dataset.map(_with_labels(tf.image.random_flip_up_down),
                                                  num_parallel_calls=self._num_threads)

            if self._augmentation_contrast:  # Apply random contrast and brightness adjustments
                def contrast_fn(x):
                    x = tf.image.random_brightness(x, max_delta=63)
                    x = tf.image.random_contrast(x, lower=0.2, upper=1.8)
                    return x

                input_dataset = input_dataset.map(_with_labels(contrast_fn), num_parallel_calls=self._num_threads)

            if self._augmentation_rotate:  # Apply random rotations, then optionally border crop and resize
                input_dataset = input_dataset.map(_with_labels(self._parse_rotate),
                                                  num_parallel_calls=self._num_threads)
                if self._rotate_crop_borders:
                    crop_fraction = self._smallest_crop_fraction(data_height, data_width)
                    input_dataset = input_dataset.map(
                        _with_labels(lambda x: self._parse_rotation_crop(x, crop_fraction, data_height, data_width)),
                        num_parallel_calls=self._num_threads)

        # Mean-center all inputs
        if self._supports_standardization:
            input_dataset = input_dataset.map(_with_labels(tf.image.per_image_standardization),
                                              num_parallel_calls=self._num_threads)

        # Manually set the shape of the image tensors so it matches the shape of the images
        input_dataset = input_dataset.map(
            lambda x, y: self._parse_force_set_shape(x, y, data_height, data_width, self._image_depth),
            num_parallel_calls=self._num_threads)

        return input_dataset

    def _parse_images(self, images):
        """
        Convert a list of image names into an internal Dataset of processed images
        :param images: A list of image names to parse
        """
        with self._graph.as_default():
            input_dataset = tf.data.Dataset.from_tensor_slices(images)
            input_dataset = input_dataset.map(lambda x: self._parse_read_images(x, channels=self._image_depth),
                                              num_parallel_calls=self._num_threads)
            input_dataset = input_dataset.map(
                lambda x: tf.image.resize_images(x, [self._image_height, self._image_width]),
                num_parallel_calls=self._num_threads)

            if self._augmentation_crop or self._crop_or_pad_images:
                if self._augmentation_crop:
                    self._image_height = int(self._image_height * self._crop_amount)
                    self._image_width = int(self._image_width * self._crop_amount)
                input_dataset = input_dataset.map(
                    lambda x: tf.image.resize_image_with_crop_or_pad(x, self._image_height, self._image_width),
                    num_parallel_calls=self._num_threads)

            # Mean-center all inputs
            if self._supports_standardization:
                input_dataset = input_dataset.map(tf.image.per_image_standardization,
                                                  num_parallel_calls=self._num_threads)

            # Manually set the shape of the image tensors so it matches the shape of the images
            def force_set(x):
                x.set_shape([self._image_height, self._image_width, self._image_depth])
                return x

            input_dataset = input_dataset.map(force_set, num_parallel_calls=self._num_threads)

            self._all_images = input_dataset

    def _parse_get_sample_counts(self, train_images, test_images, val_images):
        """
        Determines the number of training, testing, and validation samples in a dataset while parsing it
        :param train_images: A tensor or list of tensors with the training images
        :param test_images: A tensor or list of tensors with the testing images
        :param val_images: A tensor or list of tensors with the validation images
        """
        # Try to get the number of samples the normal way
        if isinstance(train_images, tf.Tensor):
            self._total_training_samples = train_images.get_shape().as_list()[0]
            if self._testing:
                self._total_testing_samples = test_images.get_shape().as_list()[0]
            if self._validation:
                self._total_validation_samples = val_images.get_shape().as_list()[0]
        elif isinstance(train_images[0], tf.Tensor):
            self._total_training_samples = train_images[0].get_shape().as_list()[0]
        else:
            self._total_training_samples = len(train_images)
            if self._testing:
                self._total_testing_samples = len(test_images)
            if self._validation:
                self._total_validation_samples = len(val_images)

        # Most often train/test/val_images will be a tensor with shape (?,), from tf.dynamic_partition, which
        # will have None for its size, so the above won't work and we manually calculate it here
        if self._total_training_samples is None:
            self._total_training_samples = int(self._total_raw_samples)
            if self._testing:
                self._total_testing_samples = int(self._total_raw_samples * self._test_split)
                self._total_training_samples = self._total_training_samples - self._total_testing_samples
            if self._validation:
                self._total_validation_samples = int(self._total_raw_samples * self._validation_split)
                self._total_training_samples = self._total_training_samples - self._total_validation_samples

    def _parse_apply_preprocessing(self, images, labels):
        """
        Applies input loading and preprocessing to images and labels from a dataset
        :param images: Image names to load and preprocess
        :param labels: The accompanying labels; normally passed through unchanged
        :return: The preprocessed versions of the images and the passed-through labels
        """
        images = self._parse_read_images(images, channels=self._image_depth)
        return images, labels

    def _parse_read_images(self, images, channels=1, image_type=tf.float32):
        """
        Read in input images during dataset parsing. This involves reading from disk, decoding the images, and
        converting them to 0-1 float images.
        :param images: Strings with the names of the images to preprocess
        :param channels: The number of channels in the image. Defaults to 1
        :param image_type: The desired Tensorflow type for the image after reading it in. Defaults to tf.float32
        (32-bit float images).
        :return: The preprocessed versions of the images
        """
        # decode_png and decode_jpeg apparently both accept JPEG and PNG. We're using one of them because decode_image
        # also accepts GIF, preventing the return of a static shape and preventing resize_images from running. See this
        # Github issue for Tensorflow: https://github.com/tensorflow/tensorflow/issues/9356
        images = tf.io.read_file(images)
        images = tf.io.decode_png(images, channels=channels)
        images = tf.image.convert_image_dtype(images, dtype=image_type)
        return images

    def _parse_resize_images(self, images, labels, height, width):
        """
        Resize images to a consistent size during dataset parsing
        :param images: The images to resize
        :param labels: The accompanying labels; normally passed through unchanged
        :param height: The new height for the images
        :param width: The new width for the images
        :return: The resized images and passed through labels
        """
        images = tf.image.resize_images(images, [height, width])
        return images, labels

    def _parse_crop_or_pad(self, images, labels, height, width):
        """
        Applies a crop/pad resizing to input images to standardize their size during dataset parsing
        :param images: The images to resize with crop/pad
        :param labels: The accompanying labels; normally passed through unchanged
        :param height: The new height for the images
        :param width: The new width for the images
        :return: The resized images
        """
        images = tf.image.resize_image_with_crop_or_pad(images, height, width)
        return images, labels

    def _parse_rotate(self, images):
        """
        Applies random rotation augmentation to input images during dataset parsing
        :param images: The images to rotate
        :return: The randomly rotated images
        """
        angle = tf.random_uniform([], maxval=2 * math.pi)
        images = tensorflow.contrib.image.rotate(images, angle, interpolation='BILINEAR')
        return images

    def _parse_rotation_crop(self, images, crop_fraction, height, width):
        """
        Applies optional centre cropping for random rotation augmentation
        :param images: Rotated images to centre crop
        :param crop_fraction: The fraction of the image to keep with the crop
        :param height: The original height to maintain after cropping the images
        :param width: The original width to maintain after cropping the images
        :return: The centre cropped images
        """
        # Cropping is done using the smallest fraction possible for the image's aspect ratio to maintain a consistent
        # scale across the images
        images = tf.image.central_crop(images, crop_fraction)
        images = tf.image.resize_images(images, [height, width])
        return images

    def _parse_force_set_shape(self, images, labels, height, width, depth):
        """
        Force set the shapes of image tensors, since we know what their sizes should be but Tensorflow can't properly
        infer them (unless image resizing occurs)
        :param images: The images to force-set shapes for
        :param labels: The accompanying labels; passed through unchanged
        :param height: The height to force for the image tensors
        :param width: The width to force for the image tensors
        :param depth: The depth/channels to force for the image tensors
        :return: The shape-defined images and passed through labels
        """
        images.set_shape([height, width, depth])
        return images, labels

    def _smallest_crop_fraction(self, height, width):
        """
        Determine the angle and crop fraction for rotated images that gives the maximum border-less crop area for a
        given angle but the smallest such area among all angles from 0-90 degrees. This is used during rotation
        augmentation to apply a consistent crop and maintain similar scale across all images. Using larger crop
        fractions based on the rotation angle would result in different scales.
        :param height: The original height of the rotated image
        :param width: The original width of the rotated image
        :return: The crop fraction that achieves the smallest area among border-less crops for rotated images
        """
        # Regardless of the aspect ratio, the smallest crop fraction always corresponds to the required crop for a 45
        # degree or pi/4 radian rotation
        angle = math.pi / 4

        # Determine which sides of the original image are the shorter and longer sides
        width_is_longer = width >= height
        if width_is_longer:
            (short_length, long_length) = (height, width)
        else:
            (short_length, long_length) = (width, height)

        # Get the absolute sin and cos of the angle, since the quadrant doesn't affect us
        sin_a = abs(math.sin(angle))
        cos_a = abs(math.cos(angle))

        # There are 2 possible solutions for the width and height in general depending on the angle and aspect ratio,
        # but 45 degree rotations always fall into the solution below. This corresponds to a rectangle with one corner
        # at the midpoint of an edge and the other corner along the centre line of the rotated image, although this
        # cropped rectangle will ultimately be slid up so that it's centered inside the rotated image.
        x = 0.5 * short_length
        if width_is_longer:
            (crop_width, crop_height) = (x / sin_a, x / cos_a)
        else:
            (crop_width, crop_height) = (x / cos_a, x / sin_a)

        # Use the crop width and height to calculate the required crop ratio
        return (crop_width * crop_height) / (width * height)
