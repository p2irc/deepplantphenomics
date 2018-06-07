from . import layers
from . import loaders
from . import preprocessing
from . import definitions
from . import networks
import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed
import os
import datetime
import time
import warnings
import copy
import matplotlib.pyplot as plt

class DPPModel(object):
    # Operation settings
    __problem_type = definitions.ProblemType.CLASSIFICATION
    __loss_fn = 'softmax cross entropy'
    __with_patching = False
    __has_trained = False
    __save_checkpoints = None
    __save_dir = None

    # Input options
    __total_classes = 0
    __total_raw_samples = 0
    __total_training_samples = 0

    __image_width = None
    __image_height = None
    __image_width_original = None
    __image_height_original = None
    __image_depth = None
    __patch_height = None
    __patch_width = None

    __crop_or_pad_images = False
    __resize_images = False
    __preprocessing_steps = []

    __processed_images_dir = './DPP-Processed'

    # supported implementations, we may add more to in future
    __supported_problem_types = ['classification', 'regression', 'semantic_segmentation']
    __supported_preprocessing_steps = ['auto-segmentation']
    __supported_optimizers = ['Adam', 'Adagrad', 'Adadelta', 'SGD']
    __supported_weight_initializers = ['normal', 'xavier']
    __supported_activation_functions = ['relu', 'tanh']
    __supported_pooling_types = ['max', 'avg']
    __supported_loss_fns_cls = ['softmax cross entropy']
    __supported_loss_fns_reg = ['l2', 'l1', 'smooth l1', 'log loss']
    __supported_loss_fns_ss = ['sigmoid cross entropy']

    # Augmentation options
    __augmentation_flip_horizontal = False
    __augmentation_flip_vertical = False
    __augmentation_crop = False
    __augmentation_contrast = False
    __crop_amount = 0.75

    # Dataset storage
    __all_ids = None

    __all_images = None
    __train_images = None
    __test_images = None

    __all_labels = None
    __train_labels = None
    __test_labels = None
    __split_labels = True

    __images_only = False

    __raw_image_files = None
    __raw_labels = None

    __raw_test_image_files = None
    __raw_train_image_files = None
    __raw_test_labels = None
    __raw_train_labels = None

    __all_moderation_features = None
    __has_moderation = False
    __moderation_features_size = None
    __train_moderation_features = None
    __test_moderation_features = None

    __training_augmentation_images = None
    __training_augmentation_labels = None

    # Network internal representation
    __session = None
    __graph = None
    __graph_ops = {}
    __layers = []
    __global_epoch = 0

    __num_layers_norm = 0
    __num_layers_conv = 0
    __num_layers_upsample = 0
    __num_layers_pool = 0
    __num_layers_fc = 0
    __num_layers_dropout = 0
    __num_layers_batchnorm = 0

    # Network options
    __batch_size = 1
    __train_test_split = 0.7
    __maximum_training_batches = None
    __reg_coeff = None
    __optimizer = 'Adam'
    __weight_initializer = 'normal'

    __learning_rate = 0.01
    __lr_decay_factor = None
    __lr_decay_epochs = None

    __num_regression_outputs = 1

    # Wrapper options
    __debug = None
    __load_from_saved = None
    __tb_dir = None
    __queue_capacity = 50
    __report_rate = None

    # Multithreading
    __num_threads = 1
    __coord = None
    __threads = None

    def __init__(self, debug=False, load_from_saved=False, save_checkpoints=True, initialize=True, tensorboard_dir=None,
                 report_rate=100, save_dir=None):
        """
        The DPPModel class represents a model which can either be trained, or loaded from an existing checkpoint file.
        This class is the singular point of contact for the DPP module.

        :param debug: If True, debug messages are printed to the console.
        :param load_from_saved: Optionally, pass the name of a directory containing the checkpoint file.
        :param save_checkpoints: If True, trainable parameters will be saved at intervals during training.
        :param initialize: If False, a new Tensorflow session will not be initialized with the instance. This is useful,
         for example, if you want to perform preprocessing only and will not be using a Tensorflow graph.
        :param tensorboard_dir: Optionally, provide the path to your Tensorboard logs directory.
        :param report_rate: Set the frequency at which progress is reported during training (also the rate at which new
        timepoints are recorded to Tensorboard).
        """

        self.__debug = debug
        self.__load_from_saved = load_from_saved
        self.__tb_dir = tensorboard_dir
        self.__report_rate = report_rate
        self.__save_checkpoints = save_checkpoints
        self.__save_dir = save_dir

        # Add the run level to the tensorboard path
        if self.__tb_dir is not None:
            self.__tb_dir = "{0}/{1}".format(self.__tb_dir, datetime.datetime.now().strftime("%d%B%Y%I:%M%p"))

        if initialize:
            self.__log('TensorFlow loaded...')

            self.__reset_graph()
            self.__reset_session()

    def __log(self, message):
        if self.__debug:
            print('{0}: {1}'.format(datetime.datetime.now().strftime("%I:%M%p"), message))

    def __last_layer(self):
        return self.__layers[-1]

    def __last_layer_outputs_volume(self):
        return isinstance(self.__last_layer().output_size, (list,))

    def __first_layer(self):
        return next(layer for layer in self.__layers if
                    isinstance(layer, layers.convLayer) or isinstance(layer, layers.fullyConnectedLayer))

    def __reset_session(self):
        self.__session = tf.Session(graph=self.__graph)

    def __reset_graph(self):
        self.__graph = tf.Graph()

    def __initialize_queue_runners(self):
        self.__log('Initializing queue runners...')
        self.__coord = tf.train.Coordinator()
        self.__threads = tf.train.start_queue_runners(sess=self.__session, coord=self.__coord)

    def set_number_of_threads(self, num_threads):
        """Set number of threads for input queue runners and preprocessing tasks"""
        if not isinstance(num_threads, int):
            raise TypeError("num_threads must be an int")
        if num_threads <= 0:
            raise ValueError("num_threads must be positive")

        self.__num_threads = num_threads

    def set_processed_images_dir(self, dir):
        """Set the directory for storing processed images when pre-processing is used"""
        if not isinstance(dir, str):
            raise TypeError("dir must be a str")

        self.__processed_images_dir = dir

    def set_batch_size(self, size):
        """Set the batch size"""
        if not isinstance(size, int):
            raise TypeError("size must be an int")
        if size <= 0:
            raise ValueError("size must be positive")

        self.__batch_size = size
        self.__queue_capacity = size * 5

    def set_num_regression_outputs(self, num):
        """Set the number of regression response variables"""
        if self.__problem_type != definitions.ProblemType.REGRESSION:
            raise RuntimeError("The problem type needs to be set to 'regression' before setting the number "+
                               "of regression outputs. Try using DPPModel.set_problem_type() first.")
        if not isinstance(num, int):
            raise TypeError("num must be an int")
        if num <= 0:
            raise ValueError("num must be positive")

        self.__num_regression_outputs = num

    def set_train_test_split(self, ratio):
        """Set a ratio for the number of samples to use as training set"""
        if not isinstance(ratio, float):
            raise TypeError("ratio must be a float")
        if ratio <= 0 or ratio > 1:
            raise ValueError("ratio must be between 0 and 1")

        self.__train_test_split = ratio

    def set_maximum_training_epochs(self, epochs):
        """Set the max number of training epochs"""
        if not isinstance(epochs, int):
            raise TypeError("epochs must be an int")
        if epochs <= 0:
            raise ValueError("epochs must be positive")

        self.__maximum_training_batches = epochs

    def set_learning_rate(self, rate):
        """Set the initial learning rate"""
        if not isinstance(rate, float):
            raise TypeError("rate must be a float")
        if rate <= 0:
            raise ValueError("rate must be positive")

        self.__learning_rate = rate

    def set_crop_or_pad_images(self, crop_or_pad):
        """Apply padding or cropping images to, which is required if the dataset has images of different sizes"""
        if not isinstance(crop_or_pad, bool):
            raise TypeError("crop_or_pad must be a bool")

        self.__crop_or_pad_images = crop_or_pad

    def set_resize_images(self, resize):
        """Up-sample or down-sample images to specified size"""
        if not isinstance(resize, bool):
            raise TypeError("resize must be a bool")

        self.__resize_images = resize

    def set_augmentation_flip_horizontal(self, flip):
        """Randomly flip training images horizontally"""
        if not isinstance(flip, bool):
            raise TypeError("flip must be a bool")

        self.__augmentation_flip_horizontal = flip

    def set_augmentation_flip_vertical(self, flip):
        """Randomly flip training images vertically"""
        if not isinstance(flip, bool):
            raise TypeError("flip must be a bool")

        self.__augmentation_flip_vertical = flip

    def set_augmentation_crop(self, resize, crop_ratio=0.75):
        """Randomly crop images during training, and crop images to center during testing"""
        if not isinstance(resize, bool):
            raise TypeError("resize must be a bool")
        if not isinstance(crop_ratio, float):
            raise TypeError("crop_ratio must be a float")
        if crop_ratio <= 0 or crop_ratio > 1:
            raise ValueError("crop_ratio must be in (0, 1]")

        self.__augmentation_crop = resize
        self.__crop_amount = crop_ratio

    def set_augmentation_brightness_and_contrast(self, contr):
        """Randomly adjust contrast and/or brightness on training images"""
        if not isinstance(contr, bool):
            raise TypeError("contr must be a bool")

        self.__augmentation_contrast = contr

    def set_regularization_coefficient(self, lamb):
        """Set lambda for L2 weight decay"""
        if not isinstance(lamb, float):
            raise TypeError("lamb must be a float")
        if lamb <= 0:
            raise ValueError("lamb must be positive")

        self.__reg_coeff = lamb

    def set_learning_rate_decay(self, decay_factor, epochs_per_decay):
        """Set learning rate decay"""
        if not isinstance(decay_factor, float):
            raise TypeError("decay_factor must be a float")
        if decay_factor <= 0:
            raise ValueError("decay_factor must be positive")
        if not isinstance(epochs_per_decay, int):
            raise TypeError("epochs_per_day must be an int")
        if epochs_per_decay <= 0:
            raise ValueError("epochs_per_day must be positive")
        if self.__total_training_samples == 0:
            raise RuntimeError("Data needs to be loaded before learning rate decay can be set.")

        self.__lr_decay_factor = decay_factor
        self.__lr_decay_epochs = epochs_per_decay * (self.__total_training_samples * self.__train_test_split)

    def set_optimizer(self, optimizer):
        """Set the optimizer to use"""
        if not isinstance(optimizer, str):
            raise TypeError("optimizer must be a str")
        if optimizer.lower().capitalize() in self.__supported_optimizers:
            optimizer = optimizer.lower().capitalize()
        elif optimizer.upper() in self.__supported_optimizers:  # 'SGD' doesn't work with the above if statement
            optimizer = optimizer.upper()
        else:
            raise ValueError("'" + optimizer + "' is not one of the currently supported optimizers. Choose one of " +
                             " ".join("'" + x + "'" for x in self.__supported_optimizers))

        self.__optimizer = optimizer

    def set_loss_function(self, loss_fn):
        """Set the loss function to use"""
        if not isinstance(loss_fn, str):
            raise TypeError("loss_fn must be a str")
        loss_fn = loss_fn.lower()
        if self.__problem_type == definitions.ProblemType.CLASSIFICATION and loss_fn not in self.__supported_loss_fns_cls:
            raise ValueError("'" + loss_fn + "' is not one of the currently supported loss functions for classification."+
                             " Make sure you have the correct problem type set with DPPModel.set_problem_type() first,"+
                             " or choose one of " + " ".join("'" + x + "'" for x in self.__supported_loss_fns_reg))
        elif self.__problem_type == definitions.ProblemType.REGRESSION and loss_fn not in self.__supported_loss_fns_reg:
            raise ValueError("'" + loss_fn + "' is not one of the currently supported loss functions for regression."+
                             " Make sure you have the correct problem type set with DPPModel.set_problem_type() first,"+
                             " or choose one of " + " ".join("'" + x + "'" for x in self.__supported_loss_fns_reg))
        elif self.__problem_type == definitions.ProblemType.SEMANTICSEGMETNATION and loss_fn not in self.__supported_loss_fns_ss:
            raise ValueError("'" + loss_fn + "' is not one of the currently supported loss functions for semantic segmentation."+
                             " Make sure you have the correct problem type set with DPPModel.set_problem_type() first,"+
                             " or choose one of " + " ".join("'" + x + "'" for x in self.__supported_loss_fns_reg))

        self.__loss_fn = loss_fn

    def set_weight_initializer(self, initializer):
        """Set the initialization scheme used by convolutional and fully connected layers"""
        if not isinstance(initializer, str):
            raise TypeError("initializer must be a str")
        initializer = initializer.lower()
        if not initializer in self.__supported_weight_initializers:
            raise ValueError("'"+initializer+"' is not one of the currently supported weight initializers."+
                             " Choose one of: "+" ".join("'"+x+"'" for x in self.__supported_weight_initializers))

        self.__weight_initializer = initializer

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

        self.__image_width = image_width
        self.__image_height = image_height
        self.__image_depth = image_depth

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

        self.__image_width_original = image_width
        self.__image_height_original = image_height

    def add_moderation_features(self, moderation_features):
        """Specify moderation features for examples in the dataset"""
        self.__has_moderation = True
        self.__moderation_features_size = moderation_features.shape[1]
        self.__all_moderation_features = moderation_features

    def add_preprocessor(self, selection):
        """Add a data preprocessing step"""
        if not isinstance(selection, str):
            raise TypeError("selection must be a str")
        if not selection in self.__supported_preprocessing_steps:
            raise ValueError("'"+selection+"' is not one of the currently supported preprocessing steps."+
                             " Choose one of: "+" ".join("'"+x+"'" for x in self.__supported_preprocessing_steps))

        self.__preprocessing_steps.append(selection)

    def clear_preprocessors(self):
        """Clear all preprocessing steps"""
        self.__preprocessing_steps = []

    def set_problem_type(self, type):
        """Set the problem type to be solved, either classification or regression"""
        if not isinstance(type, str):
            raise TypeError("type must be a str")
        if not type in self.__supported_problem_types:
            raise ValueError("'"+type+"' is not one of the currently supported problem types."+
                             " Choose one of: "+" ".join("'"+x+"'" for x in self.__supported_problem_types))

        if type == 'classification':
            self.__problem_type = definitions.ProblemType.CLASSIFICATION
            self.__loss_fn = self.__supported_loss_fns_cls[0]
        elif type == 'regression':
            self.__problem_type = definitions.ProblemType.REGRESSION
            self.__loss_fn = self.__supported_loss_fns_reg[0]
        elif type == 'semantic_segmentation':
            self.__problem_type = definitions.ProblemType.SEMANTICSEGMETNATION
            self.__loss_fn = self.__supported_loss_fns_ss[0]
        else:
            warnings.warn('Problem type specified not supported')
            exit()

    def set_patch_size(self, height, width):
        # if self.__problem_type != definitions.ProblemType.SEMANTICSEGMETNATION:
        #     throw RuntimeError
        if not isinstance(height, int):
            raise TypeError("height must be an int")
        if height <= 0:
            raise ValueError("height must be positive")
        if not isinstance(width, int):
            raise TypeError("width must be an int")
        if width <= 0:
            raise ValueError("width must be positive")

        self.__patch_height = height
        self.__patch_width = width
        self.__with_patching = True

    def __add_layers_to_graph(self):
        """
        Adds the layers in self.layers to the computational graph.

        Currently __assemble_graph is doing too many things, so this is needed as a separate function so that other
        functions such as load_state can add layers to the graph without performing everything else in asseble_graph
        """
        for layer in self.__layers:
            if callable(getattr(layer, 'add_to_graph', None)):
                layer.add_to_graph()

    def __assemble_graph(self):
        with self.__graph.as_default():

            self.__log('Parsing dataset...')

            if self.__images_only:
                self.__parse_images(self.__raw_image_files)
            elif self.__raw_test_labels is not None:
                self.__parse_dataset(self.__raw_train_image_files, self.__raw_train_labels,
                                     self.__raw_test_image_files, self.__raw_test_labels)
            else:
                train_images, train_labels, test_images, test_labels, train_mf, test_mf = \
                    loaders.split_raw_data(self.__raw_image_files, self.__raw_labels, self.__train_test_split,
                                           self.__all_moderation_features, self.__training_augmentation_images,
                                           self.__training_augmentation_labels, self.__split_labels)
                self.__parse_dataset(train_images, train_labels, test_images, test_labels, train_mf, test_mf)

            self.__log('Creating layer parameters...')

            self.__add_layers_to_graph()

            self.__log('Assembling graph...')

            # Define batches
            if self.__has_moderation:
                x, y, mod_w = tf.train.shuffle_batch(
                    [self.__train_images, self.__train_labels, self.__train_moderation_features],
                    batch_size=self.__batch_size,
                    num_threads=self.__num_threads,
                    capacity=self.__queue_capacity,
                    min_after_dequeue=self.__batch_size)
            else:
                x, y = tf.train.shuffle_batch([self.__train_images, self.__train_labels],
                                              batch_size=self.__batch_size,
                                              num_threads=self.__num_threads,
                                              capacity=self.__queue_capacity,
                                              min_after_dequeue=self.__batch_size)

            # Reshape input to the expected image dimensions
            x = tf.reshape(x, shape=[-1, self.__image_height, self.__image_width, self.__image_depth])
            if self.__problem_type == definitions.ProblemType.SEMANTICSEGMETNATION:
                y = tf.reshape(y, shape=[-1, self.__image_height, self.__image_width, 1])

            # if using patching we extract a patch of image here
            if self.__with_patching:
                # Take a slice
                patch_width = self.__patch_width
                patch_height = self.__patch_height
                offset_h = np.random.randint(patch_height/2, self.__image_height - (patch_height/2),
                                             self.__batch_size)
                offset_w = np.random.randint(patch_width/2, self.__image_width - (patch_width/2),
                                             self.__batch_size)
                offsets = [x for x in zip(offset_h, offset_w)]
                x = tf.image.extract_glimpse(x, [patch_height, patch_width], offsets,
                                             normalized=False, centered=False)
                if self.__problem_type == definitions.ProblemType.SEMANTICSEGMETNATION:
                    y = tf.image.extract_glimpse(y, [patch_height, patch_width], offsets, normalized=False,
                                                 centered=False)

            # If this is a regression problem, unserialize the label
            if self.__problem_type == definitions.ProblemType.REGRESSION:
                y = loaders.label_string_to_tensor(y, self.__batch_size, self.__num_regression_outputs)

            # Run the network operations
            if self.__has_moderation:
                xx = self.forward_pass(x, deterministic=False, moderation_features=mod_w)
            else:
                xx = self.forward_pass(x, deterministic=False)

            # Define regularization cost
            if self.__reg_coeff is not None:
                l2_cost = tf.squeeze(tf.reduce_sum(
                    [layer.regularization_coefficient * tf.nn.l2_loss(layer.weights) for layer in self.__layers
                     if isinstance(layer, layers.fullyConnectedLayer) or isinstance(layer, layers.convLayer)]))
            else:
                l2_cost = 0.0

            # Define cost function
            if self.__problem_type == definitions.ProblemType.CLASSIFICATION:
                if self.__loss_fn == 'softmax cross entropy':
                    sf_logits = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=xx, labels=tf.argmax(y, 1))

                self.__graph_ops['cost'] = tf.add(tf.reduce_mean(tf.concat([sf_logits], axis=0)), l2_cost)
            elif self.__problem_type == definitions.ProblemType.REGRESSION:
                if self.__loss_fn == 'l2':
                    regression_loss = self.__batch_mean_l2_loss(tf.subtract(xx, y))
                elif self.__loss_fn == 'l1':
                    regression_loss = self.__batch_mean_l1_loss(tf.subtract(xx, y))
                elif self.__loss_fn == 'smooth l1':
                    regression_loss = self.__batch_mean_smooth_l1_loss(tf.subtract(xx, y))
                elif self.__loss_fn == 'log loss':
                    regression_loss = self.__batch_mean_log_loss(tf.subtract(xx, y))

                self.__graph_ops['cost'] = tf.add(regression_loss, l2_cost)
            elif self.__problem_type == definitions.ProblemType.SEMANTICSEGMETNATION:
                if self.__loss_fn == 'sigmoid cross entropy':
                    pixel_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=xx, labels=y[:,:,:,0]))

                self.__graph_ops['cost'] = tf.squeeze(tf.add(pixel_loss, l2_cost))

            # Identify which optimizer we are using
            if self.__optimizer == 'Adagrad':
                self.__graph_ops['optimizer'] = tf.train.AdagradOptimizer(self.__learning_rate)
                self.__log('Using Adagrad optimizer')
            elif self.__optimizer == 'Adadelta':
                self.__graph_ops['optimizer'] = tf.train.AdadeltaOptimizer(self.__learning_rate)
                self.__log('Using Adadelta optimizer')
            elif self.__optimizer == 'SGD':
                self.__graph_ops['optimizer'] = tf.train.GradientDescentOptimizer(self.__learning_rate)
                self.__log('Using SGD optimizer')
            elif self.__optimizer == 'Adam':
                self.__graph_ops['optimizer'] = tf.train.AdamOptimizer(self.__learning_rate)
                self.__log('Using Adam optimizer')
            else:
                warnings.warn('Unrecognized optimizer requested')
                exit()

            # Compute gradients, clip them, the apply the clipped gradients
            gradients, variables = zip(*self.__graph_ops['optimizer'].compute_gradients(self.__graph_ops['cost']))
            gradients, global_grad_norm = tf.clip_by_global_norm(gradients, 5.0)  # need to make this 5.0 an adjustable hyperparameter
            self.__graph_ops['optimizer'] = self.__graph_ops['optimizer'].apply_gradients(zip(gradients, variables))

            if self.__problem_type == definitions.ProblemType.CLASSIFICATION:
                class_predictions = tf.argmax(tf.nn.softmax(xx), 1)
                correct_predictions = tf.equal(class_predictions, tf.argmax(y, 1))
                self.__graph_ops['accuracy'] = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

            # Calculate test accuracy
            if self.__has_moderation:
                x_test, self.__graph_ops['y_test'], mod_w_test = tf.train.batch(
                    [self.__test_images, self.__test_labels, self.__test_moderation_features],
                    batch_size=self.__batch_size,
                    num_threads=self.__num_threads,
                    capacity=self.__queue_capacity)
            else:
                x_test, self.__graph_ops['y_test'] = tf.train.batch([self.__test_images, self.__test_labels],
                                                batch_size=self.__batch_size,
                                                num_threads=self.__num_threads,
                                                capacity=self.__queue_capacity)


            if self.__problem_type == definitions.ProblemType.REGRESSION:
                self.__graph_ops['y_test'] = loaders.label_string_to_tensor(self.__graph_ops['y_test'], self.__batch_size, self.__num_regression_outputs)

            x_test = tf.reshape(x_test, shape=[-1, self.__image_height, self.__image_width, self.__image_depth])
            if self.__problem_type == definitions.ProblemType.SEMANTICSEGMETNATION:
                self.__graph_ops['y_test'] = tf.reshape(self.__graph_ops['y_test'], shape=[-1, self.__image_height, self.__image_width, 1])

            if self.__with_patching:
                # Take a slice of image. Same size and location as the slice from training.
                patch_width = self.__patch_width
                patch_height = self.__patch_height
                x_test = tf.image.extract_glimpse(x_test, [patch_height, patch_width], offsets,
                                                  normalized=False, centered=False)
                if self.__problem_type == definitions.ProblemType.SEMANTICSEGMETNATION:
                    self.__graph_ops['y_test'] = tf.image.extract_glimpse(self.__graph_ops['y_test'],
                                                                          [patch_height, patch_width], offsets,
                                                                          normalized=False, centered=False)

            if self.__has_moderation:
                self.__graph_ops['x_test_predicted'] = self.forward_pass(x_test, deterministic=True, moderation_features=mod_w_test)
            else:
                self.__graph_ops['x_test_predicted'] = self.forward_pass(x_test, deterministic=True)

            if self.__problem_type == definitions.ProblemType.CLASSIFICATION:
                test_class_predictions = tf.argmax(tf.nn.softmax(self.__graph_ops['x_test_predicted']), 1)
                test_correct_predictions = tf.equal(test_class_predictions, tf.argmax(self.__graph_ops['y_test'], 1))
                self.__graph_ops['test_losses'] = test_correct_predictions
                self.__graph_ops['test_accuracy'] = tf.reduce_mean(tf.cast(test_correct_predictions, tf.float32))
            elif self.__problem_type == definitions.ProblemType.REGRESSION:
                if self.__num_regression_outputs == 1:
                    self.__graph_ops['test_losses'] = tf.squeeze(tf.stack(tf.subtract(self.__graph_ops['x_test_predicted'], self.__graph_ops['y_test'])))
                else:
                    self.__graph_ops['test_losses'] = self.__l2_norm(tf.subtract(self.__graph_ops['x_test_predicted'], self.__graph_ops['y_test']))

                self.__graph_ops['test_cost'] = tf.reduce_mean(tf.abs(self.__graph_ops['test_losses']))
            elif self.__problem_type == definitions.ProblemType.SEMANTICSEGMETNATION:
                self.__graph_ops['test_losses'] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.__graph_ops['x_test_predicted'], labels=self.__graph_ops['y_test'][:, :, :, 0]), axis=2)
                self.__graph_ops['test_losses'] = tf.transpose(tf.reduce_mean(self.__graph_ops['test_losses'], axis=1))
                self.__graph_ops['test_cost'] = tf.reduce_mean(self.__graph_ops['test_losses'])

            # Epoch summaries for Tensorboard
            if self.__tb_dir is not None:
                self.__log('Creating Tensorboard summaries...')
                # Summaries for any problem type
                tf.summary.scalar('train/loss', self.__graph_ops['cost'], collections=['custom_summaries'])
                tf.summary.scalar('train/learning_rate', self.__learning_rate, collections=['custom_summaries'])
                tf.summary.scalar('train/l2_loss', l2_cost, collections=['custom_summaries'])
                filter_summary = self.__get_weights_as_image(self.__first_layer().weights)
                tf.summary.image('filters/first', filter_summary, collections=['custom_summaries'])

                # Summaries for classification problems
                if self.__problem_type == definitions.ProblemType.CLASSIFICATION:
                    tf.summary.scalar('train/accuracy', self.__graph_ops['accuracy'], collections=['custom_summaries'])
                    tf.summary.scalar('test/accuracy', self.__graph_ops['test_accuracy'], collections=['custom_summaries'])
                    tf.summary.histogram('train/class_predictions', class_predictions, collections=['custom_summaries'])
                    tf.summary.histogram('test/class_predictions', test_class_predictions,
                                         collections=['custom_summaries'])

                # Summaries for regression
                if self.__problem_type == definitions.ProblemType.REGRESSION:
                    if self.__num_regression_outputs == 1:
                        tf.summary.scalar('train/regression_loss', regression_loss, collections=['custom_summaries'])
                        tf.summary.scalar('test/loss', self.__graph_ops['test_cost'], collections=['custom_summaries'])
                        tf.summary.histogram('test/batch_losses', self.__graph_ops['test_losses'], collections=['custom_summaries'])

                # Summaries for semantic segmentation
                # we send in the last layer's output size (i.e. the final image dimensions) to get_weights_as_image
                # because xx and x_test_predicted have dynamic dims [?,?,?,?], so we need actual numbers passed in
                if self.__problem_type == definitions.ProblemType.SEMANTICSEGMETNATION:
                    tf.summary.scalar('test/loss', self.__graph_ops['test_cost'], collections=['custom_summaries'])
                    train_images_summary = self.__get_weights_as_image(
                        tf.transpose(tf.expand_dims(xx, -1), (1, 2, 3, 0)),
                        self.__layers[-1].output_size)
                    tf.summary.image('masks/train', train_images_summary, collections=['custom_summaries'])
                    test_images_summary = self.__get_weights_as_image(
                        tf.transpose(tf.expand_dims(self.__graph_ops['x_test_predicted'], -1), (1, 2, 3, 0)),
                        self.__layers[-1].output_size)
                    tf.summary.image('masks/test', test_images_summary, collections=['custom_summaries'])

                # Summaries for each layer
                for layer in self.__layers:
                    if hasattr(layer, 'name') and not isinstance(layer, layers.batchNormLayer):
                        tf.summary.histogram('weights/' + layer.name, layer.weights, collections=['custom_summaries'])
                        tf.summary.histogram('biases/' + layer.name, layer.biases, collections=['custom_summaries'])
                        tf.summary.histogram('activations/' + layer.name, layer.activations,
                                             collections=['custom_summaries'])

                # Summaries for gradients
                for index, grad in enumerate(gradients):
                    tf.summary.histogram("gradients/" + variables[index].name, gradients[index],
                                         collections=['custom_summaries'])
                tf.summary.histogram("gradient_global_norm/", global_grad_norm, collections=['custom_summaries'])

                self.__graph_ops['merged'] = tf.summary.merge_all(key='custom_summaries')

    def begin_training(self, return_test_loss=False):
        """
        Initialize the network and either run training to the specified max epoch, or load trainable variables.
        The full test accuracy is calculated immediately afterward. Finally, the trainable parameters are saved and
        the session is shut down.
        Before calling this function, the images and labels should be loaded, as well as all relevant hyperparameters.
        """
        # if None in [self.__train_images, self.__test_images,
        #             self.__train_labels, self.__test_labels]:
        #     raise RuntimeError("Images and Labels need to be loaded before you can begin training. " +
        #                        "Try first using one of the methods starting with 'load_...' such as " +
        #                        "'DPPModel.load_dataset_from_directory_with_csv_labels()'")
        # if (len(self.__layers) < 1):
        #     raise RuntimeError("There are no layers currently added to the model when trying to begin training. " +
        #                        "Add layers first by using functions such as 'DPPModel.add_input_layer()' or " +
        #                        "'DPPModel.add_convolutional_layer()'. See documentation for a complete list of layers.")

        with self.__graph.as_default():
            self.__assemble_graph()

            # Either load the network parameters from a checkpoint file or start training
            if self.__load_from_saved is not False:
                self.load_state()

                self.__initialize_queue_runners()

                self.compute_full_test_accuracy()

                self.shut_down()
            else:
                if self.__tb_dir is not None:
                    train_writer = tf.summary.FileWriter(self.__tb_dir, self.__session.graph)

                self.__log('Initializing parameters...')
                init_op = tf.global_variables_initializer()
                self.__session.run(init_op)

                self.__initialize_queue_runners()

                self.__log('Beginning training...')

                self.__set_learning_rate()

                for i in range(self.__maximum_training_batches):
                    start_time = time.time()
                    self.__global_epoch = i

                    self.__session.run(self.__graph_ops['optimizer'])

                    if self.__global_epoch > 0 and self.__global_epoch % self.__report_rate == 0:
                        elapsed = time.time() - start_time

                        if self.__tb_dir is not None:
                            summary = self.__session.run(self.__graph_ops['merged'])
                            train_writer.add_summary(summary, i)

                        if self.__problem_type == definitions.ProblemType.CLASSIFICATION:
                            loss, epoch_accuracy, epoch_test_accuracy = self.__session.run(
                                [self.__graph_ops['cost'],
                                 self.__graph_ops['accuracy'],
                                 self.__graph_ops['test_accuracy']])

                            samples_per_sec = self.__batch_size / elapsed

                            self.__log(
                                'Results for batch {} (epoch {}) - Loss: {:.5f}, Training Accuracy: {:.4f}, samples/sec: {:.2f}'
                                    .format(i,
                                            i / (self.__total_training_samples / self.__batch_size),
                                            loss,
                                            epoch_accuracy,
                                            samples_per_sec))
                        elif self.__problem_type == definitions.ProblemType.REGRESSION or \
                                        self.__problem_type == definitions.ProblemType.SEMANTICSEGMETNATION:
                            loss, epoch_test_loss = self.__session.run([self.__graph_ops['cost'],
                                                                        self.__graph_ops['test_cost']])

                            samples_per_sec = self.__batch_size / elapsed

                            self.__log(
                                'Results for batch {} (epoch {}) - Loss: {}, samples/sec: {:.2f}'
                                    .format(i,
                                            i / (self.__total_training_samples / self.__batch_size),
                                            loss,
                                            samples_per_sec))

                        if self.__save_checkpoints and self.__global_epoch % (self.__report_rate * 100) == 0:
                            self.save_state(self.__save_dir)
                    else:
                        loss = self.__session.run([self.__graph_ops['cost']])

                    if loss == 0.0:
                        self.__log('Stopping due to zero loss')
                        break

                    if i == self.__maximum_training_batches - 1:
                        self.__log('Stopping due to maximum epochs')

                self.save_state(self.__save_dir)

                final_test_loss = self.compute_full_test_accuracy()

                # Commented out because I wanted to test on the model directly after training on another dataset.
                self.shut_down()

                if return_test_loss:
                    return final_test_loss
                else:
                    return

    def begin_training_with_hyperparameter_search(self, l2_reg_limits=None, lr_limits=None, num_steps=3):
        """
        Performs grid-based hyperparameter search given the ranges passed. Parameters are optional.

        :param l2_reg_limits: array representing a range of L2 regularization coefficients in the form [low, high]
        :param lr_limits: array representing a range of learning rates in the form [low, high]
        :param num_steps: the size of the grid. Larger numbers are exponentially slower.
        """

        all_l2_reg = []
        all_lr = []
        base_tb_dir = self.__tb_dir

        unaltered_image_height = self.__image_height
        unaltered_image_width = self.__image_width
        unaltered_epochs = self.__maximum_training_batches

        if l2_reg_limits is None:
            all_l2_reg = [self.__reg_coeff]
        else:
            step_size = (l2_reg_limits[1] - l2_reg_limits[0]) / np.float32(num_steps-1)
            all_l2_reg = np.arange(l2_reg_limits[0], l2_reg_limits[1], step_size)
            all_l2_reg = np.append(all_l2_reg, l2_reg_limits[1])

        if lr_limits is None:
            all_lr = [self.__learning_rate]
        else:
            step_size = (lr_limits[1] - lr_limits[0]) / np.float32(num_steps-1)
            all_lr = np.arange(lr_limits[0], lr_limits[1], step_size)
            all_lr = np.append(all_lr, lr_limits[1])

        all_loss_results = np.empty([len(all_l2_reg), len(all_lr)])

        for i, current_l2 in enumerate(all_l2_reg):
            for j, current_lr in enumerate(all_lr):
                self.__log('HYPERPARAMETER SEARCH: Doing l2reg=%f, lr=%f' % (current_l2, current_lr))

                # Make a new graph, associate a new session with it.
                self.__reset_graph()
                self.__reset_session()

                self.__learning_rate = current_lr
                self.__reg_coeff = current_l2

                # Set calculated variables back to their unaltered form
                self.__image_height = unaltered_image_height
                self.__image_width = unaltered_image_width
                self.__maximum_training_batches = unaltered_epochs

                # Reset the reg. coef. for all fc layers.
                with self.__graph.as_default():
                    for layer in self.__layers:
                        if isinstance(layer, layers.fullyConnectedLayer):
                            layer.regularization_coefficient = current_l2

                if base_tb_dir is not None:
                    self.__tb_dir = base_tb_dir+'_lr:'+current_lr.astype('str')+'_l2:'+current_l2.astype('str')

                try:
                    current_loss = self.begin_training(return_test_loss=True)
                    all_loss_results[i][j] = current_loss
                except Exception as e:
                    self.__log('HYPERPARAMETER SEARCH: Run threw an exception, this result will be NaN.')
                    print("Exception message: "+str(e))
                    all_loss_results[i][j] = np.nan

        self.__log('Finished hyperparameter search, failed runs will appear as NaN.')
        self.__log('All l2 coef. tested:')
        self.__log('\n'+np.array2string(np.transpose(all_l2_reg)))
        self.__log('All learning rates tested:')
        self.__log('\n'+np.array2string(all_lr))
        self.__log('Loss/error grid:')
        self.__log('\n'+np.array2string(all_loss_results, precision=4))

    def compute_full_test_accuracy(self):
        """Returns statistics of the test losses depending on the type of task"""

        self.__log('Computing total test accuracy/regression loss...')

        with self.__graph.as_default():
            num_test = self.__total_raw_samples - self.__total_training_samples
            num_batches = int(num_test / self.__batch_size) + 1

            if num_batches == 0:
                warnings.warn('Less than a batch of testing data')
                exit()

            sum = 0.0
            all_losses = np.empty(shape=(self.__num_regression_outputs))
            all_y = np.empty(shape=(self.__num_regression_outputs))
            all_predictions = np.empty(shape=(self.__num_regression_outputs))

            # Main test loop
            for i in range(num_batches):
                if self.__problem_type == definitions.ProblemType.CLASSIFICATION:
                    batch_mean = self.__session.run([self.__graph_ops['test_losses']])
                    sum = sum + np.mean(batch_mean)
                elif self.__problem_type == definitions.ProblemType.REGRESSION:
                    r_losses, r_y, r_predicted = self.__session.run([self.__graph_ops['test_losses'],
                                                                     self.__graph_ops['y_test'],
                                                                     self.__graph_ops['x_test_predicted']])

                    all_losses = np.concatenate((all_losses, r_losses), axis=0)
                    all_y = np.concatenate((all_y, np.squeeze(r_y)), axis=0)
                    all_predictions = np.concatenate((all_predictions, np.squeeze(r_predicted)), axis=0)
                elif self.__problem_type == definitions.ProblemType.SEMANTICSEGMETNATION:
                    r_losses = self.__session.run([self.__graph_ops['test_losses']])

                    all_losses = np.concatenate((all_losses, r_losses[0]), axis=0)

            # Delete the weird first entries
            all_losses = np.delete(all_losses, 0)
            all_y = np.delete(all_y, 0)
            all_predictions = np.delete(all_predictions, 0)

            if self.__problem_type == definitions.ProblemType.CLASSIFICATION:
                # For classification problems (assumed to be multi-class), we want accuracy and confusion matrix
                mean = (sum / num_batches)

                self.__log('Average test accuracy: {:.5f}'.format(mean))

                return 1.0-mean.astype(np.float32)
            elif self.__problem_type == definitions.ProblemType.REGRESSION or \
                            self.__problem_type == definitions.ProblemType.SEMANTICSEGMETNATION:
                # For regression problems we want relative and abs mean, std of L2 norms, plus a histogram of errors
                abs_mean = np.mean(np.abs(all_losses))
                abs_var = np.var(np.abs(all_losses))
                abs_std = np.sqrt(abs_var)

                mean = np.mean(all_losses)
                var = np.var(all_losses)
                mse = np.mean(np.square(all_losses))
                std = np.sqrt(var)
                max = np.amax(all_losses)
                min = np.amin(all_losses)

                hist, _ = np.histogram(all_losses, bins=100)

                self.__log('Mean loss: {}'.format(mean))
                self.__log('Loss standard deviation: {}'.format(std))
                self.__log('Mean absolute loss: {}'.format(abs_mean))
                self.__log('Absolute loss standard deviation: {}'.format(abs_std))
                self.__log('Min error: {}'.format(min))
                self.__log('Max error: {}'.format(max))
                self.__log('MSE: {}'.format(mse))

                if len(all_y) > 0:
                    all_y_mean = np.mean(all_y)
                    total_error = np.sum(np.square(all_y - all_y_mean))
                    unexplained_error = np.sum(np.square(all_losses))
                    R2 = 1. - (unexplained_error / total_error)

                    self.__log('R^2: {}'.format(R2))
                    self.__log('All test labels:')
                    self.__log(all_y)

                if len(all_predictions) > 0:
                    self.__log('All predictions:')
                    self.__log(all_predictions)

                self.__log('Histogram of L2 losses:')
                self.__log(hist)

                return abs_mean.astype(np.float32)

            return

    def shut_down(self):
        """Stop all queues and end session. The model cannot be used anymore after a shut down is completed."""
        self.__log('Shutdown requested, ending session...')

        self.__coord.request_stop()
        self.__coord.join(self.__threads)

        self.__session.close()

    def __get_weights_as_image(self, kernel, size=None):
        """Filter visualization, adapted with permission from https://gist.github.com/kukuruza/03731dc494603ceab0c5"""
        with self.__graph.as_default():
            pad = 1
            grid_X = 4
            grid_Y = (self.__batch_size // 4) + 1
            # pad X and Y
            x1 = tf.pad(kernel, tf.constant([[pad, 0], [pad, 0], [0, 0], [0, 0]]))

            # when kernel is dynamically shaped at runtime it has [?,?,?,?] dimensions, thus size needs to be
            # passed in so we have actual dimensions to work with
            if size is not None:
                # X and Y dimensions, w.r.t. padding
                Y = size[1] + pad
                X = size[2] + pad
                num_channels = size[-1]
            else:
                # X and Y dimensions, w.r.t. padding
                Y = kernel.get_shape()[0] + pad
                X = kernel.get_shape()[1] + pad
                num_channels = kernel.get_shape().as_list()[2]

            # pack into image with proper dimensions for tf.image_summary
            x2 = tf.transpose(x1, (3, 0, 1, 2))
            x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, num_channels]))
            x4 = tf.transpose(x3, (0, 2, 1, 3))
            x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, num_channels]))
            x6 = tf.transpose(x5, (2, 1, 3, 0))
            x7 = tf.transpose(x6, (3, 0, 1, 2))

            # scale to [0, 1]
            x_min = tf.reduce_min(x7)
            x_max = tf.reduce_max(x7)
            x8 = (x7 - x_min) / (x_max - x_min)

        return x8

    def save_state(self, directory=None):
        """Save all trainable variables as a checkpoint in the current working path"""
        self.__log('Saving parameters...')

        if directory is None:
            dir = './saved_state'
        else:
            dir = directory + '/saved_state'

        if not os.path.isdir(dir):
            os.mkdir(dir)

        with self.__graph.as_default():
            saver = tf.train.Saver(tf.trainable_variables())
            saver.save(self.__session, dir + '/tfhSaved')

        self.__has_trained = True

    def load_state(self):
        """
        Load all trainable variables from a checkpoint file specified from the load_from_saved parameter in the
        class constructor.
        """
        if not self.__has_trained:
            self.__add_layers_to_graph()

        if self.__load_from_saved is not False:
            self.__log('Loading from checkpoint file...')

            with self.__graph.as_default():
                saver = tf.train.Saver(tf.trainable_variables())
                saver.restore(self.__session, tf.train.latest_checkpoint(self.__load_from_saved))

            self.__has_trained = True
        else:
            warnings.warn('Tried to load state with no file given. Make sure load_from_saved is set in constructor.')
            exit()

    def __set_learning_rate(self):
        if self.__lr_decay_factor is not None:
            self.__learning_rate = tf.train.exponential_decay(self.__learning_rate,
                                                              self.__global_epoch,
                                                              self.__lr_decay_epochs,
                                                              self.__lr_decay_factor,
                                                              staircase=True)

    def forward_pass(self, x, deterministic=False, moderation_features=None):
        """
        Perform a forward pass of the network with an input tensor.
        In general, this is only used when the model is integrated into a Tensorflow graph.
        See also forward_pass_with_file_inputs.

        :param x: input tensor where the first dimension is batch
        :param deterministic: if True, performs inference-time operations on stochastic layers e.g. DropOut layers
        :return: output tensor where the first dimension is batch
        """
        with self.__graph.as_default():
            for layer in self.__layers:
                if isinstance(layer, layers.moderationLayer) and moderation_features is not None:
                    x = layer.forward_pass(x, deterministic, moderation_features)
                else:
                    x = layer.forward_pass(x, deterministic)

        return x


    def forward_pass_with_file_inputs(self, x):
        """
        Get network outputs with a list of filenames of images as input.
        Handles all the loading and batching automatically, so the size of the input can exceed the available memory
        without any problems.

        :param x: list of strings representing image filenames
        :return: ndarray representing network outputs corresponding to inputs in the same order
        """
        with self.__graph.as_default():
            if self.__problem_type == definitions.ProblemType.CLASSIFICATION:
                total_outputs = np.empty([1, self.__last_layer().output_size])
            elif self.__problem_type == definitions.ProblemType.REGRESSION:
                total_outputs = np.empty([1, self.__num_regression_outputs])
            elif self.__problem_type == definitions.ProblemType.SEMANTICSEGMETNATION:
                # we want the largest multiple of of patch height/width that is smaller than the original
                # image height/width, for the final image dimensions
                if self.__with_patching:
                    patch_height = self.__patch_height
                    patch_width = self.__patch_width
                    final_height = (self.__image_height // patch_height) * patch_height
                    final_width = (self.__image_width // patch_width) * patch_width
                    # find image differencees to determine recentering crop coords, we divide by 2 so that the leftover
                    # is equal on all sides of image
                    offset_height = (self.__image_height - final_height) // 2
                    offset_width = (self.__image_width - final_width) // 2
                    # pre-allocate output dimensions
                    total_outputs = np.empty([1, final_height, final_width])
                else:
                    total_outputs = np.empty([1, self.__image_height, self.__image_width])
            else:
                warnings.warn('Problem type is not recognized')
                exit()

            num_batches = len(x) // self.__batch_size
            remainder = len(x) % self.__batch_size

            if remainder != 0:
                num_batches += 1
                remainder = self.__batch_size - remainder

            # self.load_images_from_list(x) no longer calls following 2 lines so we needed to force them here
            images = self.__apply_preprocessing(x)
            self.__parse_images(images)

            x_test = tf.train.batch([self.__all_images], batch_size=self.__batch_size, num_threads=self.__num_threads)
            x_test = tf.reshape(x_test, shape=[-1, self.__image_height, self.__image_width, self.__image_depth])
            if self.__with_patching:
                x_test = tf.image.crop_to_bounding_box(x_test, offset_height, offset_width, final_height, final_width)

            if self.__with_patching:
                # Split the images up into the multiple slices of size patch_height x patch_width
                ksizes = [1, patch_height, patch_width, 1]
                strides = [1, patch_height, patch_width, 1]
                rates = [1, 1, 1, 1]
                x_test = tf.extract_image_patches(x_test, ksizes, strides, rates, "VALID")
                x_test = tf.reshape(x_test, shape=[-1, patch_height, patch_width, self.__image_depth])

            if self.__load_from_saved:
                self.load_state()

            self.__initialize_queue_runners()

            # Run model on them
            x_pred = self.forward_pass(x_test, deterministic=True)

            if self.__with_patching:
                num_patch_rows = final_height // patch_height
                num_patch_cols = final_width // patch_width
                for i in range(num_batches):
                    xx = self.__session.run(x_pred)

                    # generalized image stitching
                    for img in np.array_split(xx, self.__batch_size): # for each img in current batch
                        # we are going to build a list of rows of imgs called img_rows, where each element
                        # of img_rows is a row of img's concatenated together horizontally (axis=1), then we will
                        # iterate through img_rows concatenating the rows vertically (axis=0) to build
                        # the full img

                        img_rows = []
                        # for each row
                        for j in range(num_patch_rows):
                            curr_row = img[j*num_patch_cols] # start new row with first img
                            # iterate through the rest of the row, concatenating img's together
                            for k in range(1, num_patch_cols):
                                curr_row = np.concatenate((curr_row, img[k+(j*num_patch_cols)]), axis=1) # horizontal cat
                            img_rows.append(curr_row) # add row of img's to the list

                        # start full img with the first full row of imgs
                        full_img = img_rows[0]
                        # iterate through rest of rows, concatenating rows together
                        for row_num in range(1, num_patch_rows):
                            full_img = np.concatenate((full_img, img_rows[row_num]), axis=0) # vertical cat

                        # need to match total_outputs dimensions, so we add a dimension to the shape to match
                        full_img = np.array([full_img]) # shape transformation: (x,y) --> (1,x,y)
                        total_outputs = np.append(total_outputs, full_img, axis=0) # add the final img to the list of imgs


            else:
                for i in range(int(num_batches)):
                    xx = self.__session.run(x_pred)
                    for img in np.array_split(xx, self.__batch_size):
                        total_outputs = np.append(total_outputs, img, axis=0)

            # delete weird first row
            total_outputs = np.delete(total_outputs, 0, 0)

            # delete any outputs which are overruns from the last batch
            if remainder != 0:
                for i in range(remainder):
                    total_outputs = np.delete(total_outputs, -1, 0)

        return total_outputs

    def __batch_mean_l2_loss(self, x):
        """Given a batch of vectors, calculates the mean per-vector L2 norm"""
        with self.__graph.as_default():
            agg = self.__l2_norm(x)
            mean = tf.reduce_mean(agg)

        return mean

    def __l2_norm(self, x):
        """Returns the L2 norm of a tensor"""
        with self.__graph.as_default():
            y = tf.map_fn(lambda ex: tf.norm(ex, ord=2), x)

        return y

    def __batch_mean_l1_loss(self, x):
        """Given a batch of vectors, calculates the mean per-vector L1 norm"""
        with self.__graph.as_default():
            agg = self.__l1_norm(x)
            mean = tf.reduce_mean(agg)

        return mean

    def __l1_norm(self, x):
        """Returns the L1 norm of a tensor"""
        with self.__graph.as_default():
            y = tf.map_fn(lambda ex: tf.norm(ex, ord=1), x)

        return y

    def __batch_mean_smooth_l1_loss(self, x):
        """Given a batch of vectors, calculates the mean per-vector smooth L1 norm"""
        with self.__graph.as_default():
            agg = self.__smooth_l1_norm(x)
            mean = tf.reduce_mean(agg)

        return mean

    def __smooth_l1_norm(self, x):
        """Returns the smooth L1 norm of a tensor"""
        HUBER_DELTA = 1 # may want to make this a tunable hyper parameter in future
        with self.__graph.as_default():
            x = tf.abs(x)
            y = tf.map_fn(lambda ex: tf.where(ex<HUBER_DELTA,
                                              0.5*ex**2,
                                              HUBER_DELTA*(ex-0.5*HUBER_DELTA)), x)

        return y

    def __batch_mean_log_loss(self, x):
        """Given a batch of vectors, calculates the mean per-vector log loss"""
        with self.__graph.as_default():
            x = tf.abs(x)
            x = tf.clip_by_value(x, 0, 0.9999999)
            agg = -tf.log(1-x)
            mean = tf.reduce_mean(agg)

        return mean

    def add_input_layer(self):
        """Add an input layer to the network"""
        if len(self.__layers) > 0:
            raise RuntimeError("Trying to add an input layer to a model that already contains other layers. "+
                               " The input layer need to be the first layer added to the model.")

        self.__log('Adding the input layer...')

        apply_crop = (self.__augmentation_crop and self.__all_images is None and self.__train_images is None)

        if apply_crop:
            size = [self.__batch_size, int(self.__image_height * self.__crop_amount),
                    int(self.__image_width * self.__crop_amount), self.__image_depth]
        else:
            size = [self.__batch_size, self.__image_height, self.__image_width, self.__image_depth]

        if self.__with_patching:
            size = [self.__batch_size, self.__patch_height, self.__patch_width, self.__image_depth]

        with self.__graph.as_default():
            layer = layers.inputLayer(size)

        self.__layers.append(layer)

    def add_moderation_layer(self):
        """Add a moderation layer to the network"""
        self.__log('Adding moderation layer...')

        reshape = self.__last_layer_outputs_volume()

        feat_size = self.__moderation_features_size

        with self.__graph.as_default():
            layer = layers.moderationLayer(copy.deepcopy(self.__last_layer().output_size), feat_size, reshape, self.__batch_size)

        self.__layers.append(layer)

    def add_convolutional_layer(self, filter_dimension, stride_length, activation_function,
                                regularization_coefficient=None):
        """
        Add a convolutional layer to the model.

        :param filter_dimension: array of dimensions in the format [x_size, y_size, depth, num_filters]
        :param stride_length: convolution stride length
        :param activation_function: the activation function to apply to the activation map
        :param regularization_coefficient: optionally, an L2 decay coefficient for this layer (overrides the coefficient
         set by set_regularization_coefficient)
        """
        if len(self.__layers) < 1:
            raise RuntimeError("A convolutional layer cannot be the first layer added to the model. "+
                               "Add an input layer with DPPModel.add_input_layer() first.")
        try:  # try to iterate through filter_dimension, checking it has 4 ints
            for idx, dim in enumerate(filter_dimension):
                if not (isinstance(dim, int) or isinstance(dim, np.int64)): # np.int64 numpy default int
                    raise TypeError()
            if idx != 3:
                raise TypeError()
        except:
            raise TypeError("filter_dimension must be a list or array of 4 ints")
        if not isinstance(stride_length, int):
            raise TypeError("stride_length must be an int")
        if stride_length <= 0:
            raise ValueError("stride_length must be positive")
        if not isinstance(activation_function, str):
            raise TypeError("activation_function must be a str")
        activation_function = activation_function.lower()
        if not activation_function in self.__supported_activation_functions:
            raise ValueError("'"+activation_function+"' is not one of the currently supported activation functions."+
                             " Choose one of: "+
                             " ".join("'"+x+"'" for x in self.__supported_activation_functions))
        if regularization_coefficient is not None:
            if not isinstance(regularization_coefficient, float):
                raise TypeError("regularization_coefficient must be a float or None")
            if regularization_coefficient < 0:
                raise ValueError("regularization_coefficient must be non-negative")

        self.__num_layers_conv += 1
        layer_name = 'conv%d' % self.__num_layers_conv
        self.__log('Adding convolutional layer %s...' % layer_name)

        if regularization_coefficient is None and self.__reg_coeff is not None:
            regularization_coefficient = self.__reg_coeff
        elif regularization_coefficient is None and self.__reg_coeff is None:
            regularization_coefficient = 0.0

        with self.__graph.as_default():
            layer = layers.convLayer(layer_name,
                                     copy.deepcopy(self.__last_layer().output_size),
                                     filter_dimension,
                                     stride_length,
                                     activation_function,
                                     self.__weight_initializer,
                                     regularization_coefficient)

        self.__log('Filter dimensions: {0} Outputs: {1}'.format(filter_dimension, layer.output_size))

        self.__layers.append(layer)

    def add_upsampling_layer(self, filter_size, num_filters, upscale_factor=2,
                             activation_function=None, regularization_coefficient=None):
        """
        Add a 2d upsampling layer to the model.

        :param filter_size: an int, representing the dimension of the square filter to be used
        :param num_filters: an int, representing the number of filters that will be outputted (the output tensor depth)
        :param upscale_factor: an int, or tuple of ints, representing the upsampling factor for rows and columns
        :param activation_function: the activation function to apply to the activation map
        :param regularization_coefficient: optionally, an L2 decay coefficient for this layer (overrides the coefficient
         set by set_regularization_coefficient)
        """
        self.__num_layers_upsample += 1
        layer_name = 'upsample%d' % self.__num_layers_upsample
        self.__log('Adding upsampling layer %s...' % layer_name)

        if regularization_coefficient is None and self.__reg_coeff is not None:
            regularization_coefficient = self.__reg_coeff
        elif regularization_coefficient is None and self.__reg_coeff is None:
            regularization_coefficient = 0.0

        if self.__with_patching:
            patches_horiz = self.__image_width // self.__patch_width
            patches_vert = self.__image_height // self.__patch_height
            batch_multiplier = patches_horiz * patches_vert
        else:
            batch_multiplier = 1

        last_layer_dims = copy.deepcopy(self.__last_layer().output_size)
        with self.__graph.as_default():
            layer = layers.upsampleLayer(layer_name,
                                         last_layer_dims,
                                         filter_size,
                                         num_filters,
                                         upscale_factor,
                                         activation_function,
                                         batch_multiplier,
                                         self.__weight_initializer,
                                         regularization_coefficient)

        self.__log('Filter dimensions: {0} Outputs: {1}'.format(layer.weights_shape, layer.output_size))

        self.__layers.append(layer)

    def add_pooling_layer(self, kernel_size, stride_length, pooling_type='max'):
        """
        Add a pooling layer to the model.

        :param kernel_size: an integer representing the width and height dimensions of the pooling operation
        :param stride_length: convolution stride length
        :param pooling_type: optional, the type of pooling operation
        """
        if len(self.__layers) < 1:
            raise RuntimeError("A pooling layer cannot be the first layer added to the model. "+
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
        if not pooling_type in self.__supported_pooling_types:
            raise ValueError("'"+pooling_type+"' is not one of the currently supported pooling types."+
                             " Choose one of: "+
                             " ".join("'"+x+"'" for x in self.__supported_pooling_types))

        self.__num_layers_pool += 1
        layer_name = 'pool%d' % self.__num_layers_pool
        self.__log('Adding pooling layer %s...' % layer_name)

        with self.__graph.as_default():
            layer = layers.poolingLayer(copy.deepcopy(self.__last_layer().output_size), kernel_size, stride_length, pooling_type)

        self.__log('Outputs: %s' % layer.output_size)

        self.__layers.append(layer)

    def add_normalization_layer(self):
        """Add a local response normalization layer to the model"""
        if len(self.__layers) < 1:
            raise RuntimeError("A normalization layer cannot be the first layer added to the model. "+
                               "Add an input layer with DPPModel.add_input_layer() first.")

        self.__num_layers_norm += 1
        layer_name = 'norm%d' % self.__num_layers_pool
        self.__log('Adding pooling layer %s...' % layer_name)

        with self.__graph.as_default():
            layer = layers.normLayer(copy.deepcopy(self.__last_layer().output_size))

        self.__layers.append(layer)

    def add_dropout_layer(self, p):
        """
        Add a DropOut layer to the model.

        :param p: the keep-probability parameter for the DropOut operation
        """
        if len(self.__layers) < 1:
            raise RuntimeError("A dropout layer cannot be the first layer added to the model. "+
                               "Add an input layer with DPPModel.add_input_layer() first.")
        if not isinstance(p, float):
            raise TypeError("p must be a float")
        if p < 0 or p >= 1:
            raise ValueError("p must be in range [0, 1)")

        self.__num_layers_dropout += 1
        layer_name = 'drop%d' % self.__num_layers_dropout
        self.__log('Adding dropout layer %s...' % layer_name)

        with self.__graph.as_default():
            layer = layers.dropoutLayer(copy.deepcopy(self.__last_layer().output_size), p)

        self.__layers.append(layer)

    def add_batch_norm_layer(self):
        """Add a batch normalization layer to the model."""
        if len(self.__layers) < 1:
            raise RuntimeError("A batch norm layer cannot be the first layer added to the model.")

        self.__num_layers_batchnorm += 1
        layer_name = 'bn%d' % self.__num_layers_batchnorm
        self.__log('Adding batch norm layer %s...' % layer_name)

        with self.__graph.as_default():
            layer = layers.batchNormLayer(layer_name, copy.deepcopy(self.__last_layer().output_size))

        self.__layers.append(layer)

    def add_fully_connected_layer(self, output_size, activation_function, regularization_coefficient=None):
        """
        Add a fully connected layer to the model.

        :param output_size: the number of units in the layer
        :param activation_function: optionally, the activation function to use
        :param regularization_coefficient: optionally, an L2 decay coefficient for this layer (overrides the coefficient
         set by set_regularization_coefficient)
        """
        if len(self.__layers) < 1:
            raise RuntimeError("A fully connected layer cannot be the first layer added to the model. "+
                               "Add an input layer with DPPModel.add_input_layer() first.")
        if not isinstance(output_size, int):
            raise TypeError("output_size must be an int")
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        if not isinstance(activation_function, str):
            raise TypeError("activation_function must be a str")
        activation_function = activation_function.lower()
        if not activation_function in self.__supported_activation_functions:
            raise ValueError("'"+activation_function+"' is not one of the currently supported activation functions."+
                             " Choose one of: "+
                             " ".join("'"+x+"'" for x in self.__supported_activation_functions))
        if regularization_coefficient is not None:
            if not isinstance(regularization_coefficient, float):
                raise TypeError("regularization_coefficient must be a float or None")
            if regularization_coefficient < 0:
                raise ValueError("regularization_coefficient must be non-negative")

        self.__num_layers_fc += 1
        layer_name = 'fc%d' % self.__num_layers_fc
        self.__log('Adding fully connected layer %s...' % layer_name)

        reshape = self.__last_layer_outputs_volume()

        if regularization_coefficient is None and self.__reg_coeff is not None:
            regularization_coefficient = self.__reg_coeff
        if regularization_coefficient is None and self.__reg_coeff is None:
            regularization_coefficient = 0.0

        with self.__graph.as_default():
            layer = layers.fullyConnectedLayer(layer_name,
                                               copy.deepcopy(self.__last_layer().output_size),
                                               output_size,
                                               reshape,
                                               self.__batch_size,
                                               activation_function,
                                               self.__weight_initializer,
                                               regularization_coefficient)

        self.__log('Inputs: {0} Outputs: {1}'.format(layer.input_size, layer.output_size))

        self.__layers.append(layer)

    def add_output_layer(self, regularization_coefficient=None, output_size=None):
        """
        Add an output layer to the network (affine layer where the number of units equals the number of network outputs)

        :param regularization_coefficient: optionally, an L2 decay coefficient for this layer (overrides the coefficient
         set by set_regularization_coefficient)
        :param output_size: optionally, override the output size of this layer. Typically not needed, but required for
        use cases such as creating the output layer before loading data.
        """
        if len(self.__layers) < 1:
            raise RuntimeError("An output layer cannot be the first layer added to the model. "+
                               "Add an input layer with DPPModel.add_input_layer() first.")
        if regularization_coefficient is not None:
            if not isinstance(regularization_coefficient, float):
                raise TypeError("regularization_coefficient must be a float or None")
            if regularization_coefficient < 0:
                raise ValueError("regularization_coefficient must be non-negative")
        if output_size is not None:
            if not isinstance(output_size, int):
                raise TypeError("output_size must be an int or None")
            if output_size <= 0:
                raise ValueError("output_size must be positive")
            if self.__problem_type == definitions.ProblemType.SEMANTICSEGMETNATION:
                raise RuntimeError("output_size should be None for problem_type semantic_segmentation")

        self.__log('Adding output layer...')

        reshape = self.__last_layer_outputs_volume()

        if regularization_coefficient is None and self.__reg_coeff is not None:
            regularization_coefficient = self.__reg_coeff
        if regularization_coefficient is None and self.__reg_coeff is None:
            regularization_coefficient = 0.0

        if output_size is None:
            if self.__problem_type == definitions.ProblemType.CLASSIFICATION:
                num_out = self.__total_classes
            elif self.__problem_type == definitions.ProblemType.REGRESSION:
                num_out = self.__num_regression_outputs
            elif self.__problem_type == definitions.ProblemType.SEMANTICSEGMETNATION:
                filter_dimension = [1, 1, copy.deepcopy(self.__last_layer().output_size[3]), 1]
            else:
                warnings.warn('Problem type is not recognized')
                exit()
        else:
            num_out = output_size

        with self.__graph.as_default():
            if self.__problem_type is definitions.ProblemType.SEMANTICSEGMETNATION:
                layer = layers.convLayer('output',
                                         copy.deepcopy(self.__last_layer().output_size),
                                         filter_dimension,
                                         1,
                                         None,
                                         self.__weight_initializer,
                                         regularization_coefficient)
            else:
                layer = layers.fullyConnectedLayer('output',
                                                   copy.deepcopy(self.__last_layer().output_size),
                                                   num_out,
                                                   reshape,
                                                   self.__batch_size,
                                                   None,
                                                   self.__weight_initializer,
                                                   regularization_coefficient)

        self.__log('Inputs: {0} Outputs: {1}'.format(layer.input_size, layer.output_size))

        self.__layers.append(layer)

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

        self.__total_raw_samples = len(image_files)
        self.__total_classes = len(set(labels))

        self.__log('Total raw examples is %d' % self.__total_raw_samples)
        self.__log('Total classes is %d' % self.__total_classes)

        self.__raw_image_files = image_files
        self.__raw_labels = labels
        self.__split_labels = False ### Band-aid fix

    def load_dataset_from_directory_with_segmentation_masks(self, dirname, seg_dirname):
        """
        Loads the png images in the given directory into an internal representation, using binary segmentation
        masks from another file with the same filename as ground truth.

        :param dirname: the path of the directory containing the images
        :param seg_dirname: the path of the directory containing ground-truth binary segmentation masks
        """

        if self.__problem_type is not definitions.ProblemType.SEMANTICSEGMETNATION:
            warnings.warn('Trying to load a segmentation dataset, but the problem type is not properly set.')
            exit()

        image_files = [os.path.join(dirname, name) for name in os.listdir(dirname) if
                       os.path.isfile(os.path.join(dirname, name)) & name.endswith('.png')]

        seg_files = [os.path.join(seg_dirname, name) for name in os.listdir(seg_dirname) if
                     os.path.isfile(os.path.join(seg_dirname, name)) & name.endswith('.png')]

        self.__total_raw_samples = len(image_files)

        self.__log('Total raw examples is %d' % self.__total_raw_samples)

        self.__raw_image_files = image_files
        self.__raw_labels = seg_files
        self.__split_labels = False ### Band-aid fix



    def load_ippn_dataset_from_directory(self, dirname, column='strain'):
        """Loads the RGB images and species labels from the International Plant Phenotyping Network dataset."""

        if column == 'treatment':
            labels, ids = loaders.read_csv_labels_and_ids(os.path.join(dirname, 'Metadata.csv'), 2, 0)
        elif column == 'strain':
            labels, ids = loaders.read_csv_labels_and_ids(os.path.join(dirname, 'Metadata.csv'), 1, 0)
        elif column == 'DAG':
            labels, ids = loaders.read_csv_labels_and_ids(os.path.join(dirname, 'Metadata.csv'), 3, 0)
        else:
            warnings.warn('Unknown column in IPPN dataset')
            exit()

        image_files = [os.path.join(dirname, id + '_rgb.png') for id in ids]

        self.__total_raw_samples = len(image_files)

        if self.__problem_type == definitions.ProblemType.CLASSIFICATION:
            self.__total_classes = len(set(labels))

            # transform into numerical one-hot labels
            with self.__graph.as_default():
                labels = loaders.string_labels_to_sequential(labels)
                labels = tf.one_hot(labels, self.__total_classes)

            self.__log('Total classes is %d' % self.__total_classes)
        elif self.__problem_type == definitions.ProblemType.REGRESSION:
            labels = [[label] for label in labels]

        self.__log('Total raw examples is %d' % self.__total_raw_samples)

        self.__raw_image_files = image_files
        self.__raw_labels = labels


    def load_ippn_tray_dataset_from_directory(self, dirname):
        """
        Loads the RGB tray images and plant bounding box labels from the International Plant Phenotyping Network
        dataset.
        """

        images = [os.path.join(dirname, name) for name in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, name)) & name.endswith('_rgb.png')]

        label_files = [os.path.join(dirname, name) for name in os.listdir(dirname) if
                       os.path.isfile(os.path.join(dirname, name)) & name.endswith('_bbox.csv')]

        # currently reads columns, need to read rows instead!!!
        labels = [loaders.read_csv_rows(label_file) for label_file in label_files]

        self.__all_labels = []

        for label in labels:
            self.__all_labels.append([loaders.box_coordinates_to_pascal_voc_coordinates(l) for l in label])

        self.__total_raw_samples = len(images)

        self.__log('Total raw examples is %d' % self.__total_raw_samples)
        self.__log('Parsing dataset...')

        # do preprocessing
        images = self.__apply_preprocessing(images)

        self.__raw_image_files = images
        self.__raw_labels = self.__all_labels

    def load_ippn_leaf_count_dataset_from_directory(self, dirname):
        """Loads the RGB images and species labels from the International Plant Phenotyping Network dataset."""
        if self.__image_height is None or self.__image_width is None or self.__image_depth is None:
            raise RuntimeError("Image dimensions need to be set before loading data."+
                               " Try using DPPModel.set_image_dimensions() first.")
        if self.__maximum_training_batches is None:
            raise RuntimeError("The number of maximum training epochs needs to be set before loading data."+
                               " Try using DPPModel.set_maximum_training_epochs() first.")

        labels, ids = loaders.read_csv_labels_and_ids(os.path.join(dirname, 'Leaf_counts.csv'), 1, 0)

        # labels must be lists
        labels = [[label] for label in labels]

        image_files = [os.path.join(dirname, id + '_rgb.png') for id in ids]

        self.__total_raw_samples = len(image_files)

        self.__log('Total raw examples is %d' % self.__total_raw_samples)
        self.__log('Parsing dataset...')

        self.__raw_image_files = image_files
        self.__raw_labels = labels

    def load_inra_dataset_from_directory(self, dirname):
        """Loads the RGB images and labels from the INRA dataset."""

        labels, ids = loaders.read_csv_labels_and_ids(os.path.join(dirname, 'AutomatonImages.csv'), 1, 3, character=';')

        # Remove the header line
        labels.pop(0)
        ids.pop(0)

        image_files = [os.path.join(dirname, id) for id in ids]

        self.__total_raw_samples = len(image_files)
        self.__total_classes = len(set(labels))

        # transform into numerical one-hot labels
        labels = loaders.string_labels_to_sequential(labels)
        labels = tf.one_hot(labels, self.__total_classes)

        self.__log('Total raw examples is %d' % self.__total_raw_samples)
        self.__log('Total classes is %d' % self.__total_classes)
        self.__log('Parsing dataset...')

        self.__raw_image_files = image_files
        self.__raw_labels = labels

    def load_cifar10_dataset_from_directory(self, dirname):
        """
        Loads the images and labels from a directory containing the CIFAR-10 image classification dataset as
        downloaded by nvidia DIGITS.
        """

        train_dir = os.path.join(dirname, 'train')
        test_dir = os.path.join(dirname, 'test')
        self.__total_classes = 10
        self.__queue_capacity = 60000

        train_labels, train_images = loaders.read_csv_labels_and_ids(os.path.join(train_dir, 'train.txt'), 1, 0,
                                                                         character=' ')

        def one_hot(labels, num_classes):
            return [[1 if i==label else 0 for i in range(num_classes)] for label in labels]

        # transform into numerical one-hot labels
        train_labels = [int(label) for label in train_labels]
        train_labels = one_hot(train_labels, self.__total_classes)

        test_labels, test_images = loaders.read_csv_labels_and_ids(os.path.join(test_dir, 'test.txt'), 1, 0,
                                                                       character=' ')

        # transform into numerical one-hot labels
        test_labels = [int(label) for label in test_labels]
        test_labels = one_hot(test_labels, self.__total_classes)

        self.__total_raw_samples = len(train_images) + len(test_images)

        self.__log('Total raw examples is %d' % self.__total_raw_samples)
        self.__log('Total classes is %d' % self.__total_classes)

        self.__raw_test_image_files = test_images
        self.__raw_train_image_files = train_images
        self.__raw_test_labels = test_labels
        self.__raw_train_labels = train_labels

    def load_dataset_from_directory_with_auto_labels(self, dirname):
        """Loads the png images in the given directory, using subdirectories to separate classes."""

        # Load all file names and labels into arrays
        subdirs = list(filter(lambda item: os.path.isdir(item) & (item != '.DS_Store'),
                         [os.path.join(dirname, f) for f in os.listdir(dirname)]))

        num_classes = len(subdirs)

        image_files = []
        labels = np.array([])

        for sd in subdirs:
            image_paths = [os.path.join(sd, name) for name in os.listdir(sd) if
                           os.path.isfile(os.path.join(sd, name)) & name.endswith('.png')]
            image_files = image_files + image_paths

            # for one-hot labels
            current_labels = np.zeros((num_classes, len(image_paths)))
            current_labels[self.__total_classes, :] = 1
            labels = np.hstack([labels, current_labels]) if labels.size else current_labels
            self.__total_classes += 1

        labels = tf.transpose(labels)

        self.__total_raw_samples = len(image_files)

        self.__log('Total raw examples is %d' % self.__total_raw_samples)
        self.__log('Total classes is %d' % self.__total_classes)
        self.__log('Parsing dataset...')

        self.__raw_image_files = image_files
        self.__raw_labels = labels

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

        print(image_files)
        # Put the image files in the order of the IDs (if there are any labels loaded)
        sorted_paths = []

        if self.__all_labels is not None:
            for image_id in self.__all_ids:
                path = filter(lambda item: item.endswith(image_id), [p for p in image_files])
                assert len(path) == 1, 'Found no image or multiple images for %r' % image_id
                sorted_paths.append(path[0])
        else:
            sorted_paths = image_files

        self.__total_raw_samples = len(sorted_paths)

        self.__log('Total raw examples is %d' % self.__total_raw_samples)
        self.__log('Parsing dataset...')

        # do preprocessing
        images = self.__apply_preprocessing(sorted_paths)

        # prepare images for training (if there are any labels loaded)

        if self.__all_labels is not None:
            labels = self.__all_labels

            self.__raw_image_files = image_files
            self.__raw_labels = labels

    def load_images_from_list(self, image_files):
        """
        Loads images from a list of file names (strings). Unless you only want to do preprocessing,
        regression or classification labels MUST be loaded first.
        """

        self.__total_raw_samples = len(image_files)

        self.__log('Total raw examples is %d' % self.__total_raw_samples)
        self.__log('Parsing dataset...')

        # do preprocessing
        images = self.__apply_preprocessing(image_files)

        # prepare images for training (if there are any labels loaded)
        if self.__all_labels is not None:
            self.__raw_image_files = images
            self.__raw_labels = self.__all_labels
        else:
            self.__raw_image_files = images
            self.__images_only = True

    def load_multiple_labels_from_csv(self, filepath, id_column=0):
        """
        Load multiple labels from a CSV file, for instance values for regression.
        Parameter id_column is the column number specifying the image file name.
        """

        self.__all_labels, self.__all_ids = loaders.read_csv_multi_labels_and_ids(filepath, id_column)

    def load_images_with_ids_from_directory(self, dir):
        """Loads images from a directroy, relating them to labels by the IDs which were loaded from a CSV file"""

        # Load all images in directory
        image_files = [os.path.join(dir, name) for name in os.listdir(dir) if
                       os.path.isfile(os.path.join(dir, name)) & name.endswith('.png')]

        # Put the image files in the order of the IDs (if there are any labels loaded)
        sorted_paths = []

        if self.__all_labels is not None:
            for image_id in self.__all_ids:
                path = list(filter(lambda item: item.endswith('/' + image_id), [p for p in image_files]))
                assert len(path) == 1, 'Found no image or multiple images for %r' % image_id
                sorted_paths.append(path[0])
        else:
            sorted_paths = image_files

        self.__total_raw_samples = len(sorted_paths)

        self.__log('Total raw examples is %d' % self.__total_raw_samples)
        self.__log('Parsing dataset...')

        # do preprocessing
        images = self.__apply_preprocessing(sorted_paths)

        # prepare images for training (if there are any labels loaded)
        if self.__all_labels is not None:
            self.__raw_image_files = image_files
            self.__raw_labels = self.__all_labels

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
            path = filter(lambda item: item.endswith('/' + image_id), [p for p in image_files])
            assert len(path) == 1, 'Found no image or multiple images for %r' % image_id
            sorted_paths.append(path[0])

        self.__training_augmentation_images = sorted_paths
        self.__training_augmentation_labels = labels

    def load_pascal_voc_labels_from_directory(self, dir):
        """Loads single per-image bounding boxes from XML files in Pascal VOC format."""

        self.__all_ids = []
        self.__all_labels = []

        file_paths = [os.path.join(dir, name) for name in os.listdir(dir) if
                      os.path.isfile(os.path.join(dir, name)) & name.endswith('.xml')]

        for voc_file in file_paths:
            id, x_min, x_max, y_min, y_max = loaders.read_single_bounding_box_from_pascal_voc(voc_file)

            # re-scale coordinates if images are being resized
            if self.__resize_images:
                x_min = int(x_min * (float(self.__image_width) / self.__image_width_original))
                x_max = int(x_max * (float(self.__image_width) / self.__image_width_original))
                y_min = int(y_min * (float(self.__image_height) / self.__image_height_original))
                y_max = int(y_max * (float(self.__image_height) / self.__image_height_original))

            self.__all_ids.append(id)
            self.__all_labels.append([x_min, x_max, y_min, y_max])

    def __apply_preprocessing(self, images):
        if not len(self.__preprocessing_steps) == 0:
            self.__log('Performing preprocessing steps...')

            if not os.path.isdir(self.__processed_images_dir):
                os.mkdir(self.__processed_images_dir)

            for step in self.__preprocessing_steps:
                if step == 'auto-segmentation':
                    self.__log('Performing auto-segmentation...')

                    self.__log('Initializing bounding box regressor model...')
                    bbr = networks.boundingBoxRegressor(height=self.__image_height, width=self.__image_width)

                    self.__log('Performing bounding box estimation...')
                    bbs = bbr.forward_pass(images)

                    bbr.shut_down()
                    bbr = None

                    images = zip(images, bbs)

                    self.__log('Bounding box estimation finished, performing segmentation...')

                    processed_images = Parallel(n_jobs=self.__num_threads) \
                        (delayed(preprocessing.do_parallel_auto_segmentation)
                         (i[0], i[1], self.__processed_images_dir, self.__image_height, self.__image_width) for i in
                         images)

                    images = processed_images

        return images

    def __parse_dataset(self, train_images, train_labels, test_images, test_labels, image_type='png', train_mf=None,
                        test_mf=None):
        """Takes training and testing images and labels, creates input queues internally to this instance"""
        with self.__graph.as_default():

            # house keeping
            if isinstance(train_images, tf.Tensor):
                self.__total_training_samples = train_images.get_shape().as_list()[0]
            else:
                self.__total_training_samples = len(train_images)

            if self.__total_training_samples is None:
                self.__total_training_samples = int(self.__total_raw_samples * self.__train_test_split)

            # moderation features queues
            if train_mf is not None:
                train_moderation_queue = tf.train.slice_input_producer([train_mf], shuffle=False)
                self.__train_moderation_features = tf.cast(train_moderation_queue[0], tf.float32)

            if test_mf is not None:
                test_moderation_queue = tf.train.slice_input_producer([test_mf], shuffle=False)
                self.__test_moderation_features = tf.cast(test_moderation_queue[0], tf.float32)

            # calculate number of batches to run
            batches_per_epoch = self.__total_training_samples / float(self.__batch_size)
            self.__maximum_training_batches = int(self.__maximum_training_batches * batches_per_epoch)

            self.__log('Batches per epoch: {:f}'.format(batches_per_epoch))
            self.__log('Running to {0} batches'.format(self.__maximum_training_batches))

            if self.__batch_size > self.__total_training_samples:
                self.__log('Less than one batch in training set, exiting now')
                exit()

            # create input queues
            train_input_queue = tf.train.slice_input_producer([train_images, train_labels], shuffle=False)
            test_input_queue = tf.train.slice_input_producer([test_images, test_labels], shuffle=False)

            if self.__problem_type is definitions.ProblemType.SEMANTICSEGMETNATION:
                self.__test_labels = tf.image.decode_png(tf.read_file(test_input_queue[1]), channels=1)
                self.__train_labels = tf.image.decode_png(tf.read_file(train_input_queue[1]),
                                                          channels=1)

                # normalize to 1.0
                self.__train_labels = tf.image.convert_image_dtype(self.__train_labels, dtype=tf.float32)
                self.__test_labels = tf.image.convert_image_dtype(self.__test_labels, dtype=tf.float32)

                # resize if we are using that
                if self.__resize_images:
                    self.__train_labels = tf.image.resize_images(self.__train_labels,
                                                                 [self.__image_height, self.__image_width])
                    self.__test_labels = tf.image.resize_images(self.__test_labels,
                                                                [self.__image_height, self.__image_width])

                    # make into a binary mask
                    self.__test_labels = tf.reduce_mean(self.__test_labels, axis=2)
                    self.__train_labels = tf.reduce_mean(self.__train_labels, axis=2)
            else:
                self.__test_labels = test_input_queue[1]
                self.__train_labels = train_input_queue[1]

            # pre-processing for training and testing images
            if image_type is 'jpg':
                self.__train_images = tf.image.decode_jpeg(tf.read_file(train_input_queue[0]),
                                                           channels=self.__image_depth)
                self.__test_images = tf.image.decode_jpeg(tf.read_file(test_input_queue[0]),
                                                          channels=self.__image_depth)
            else:
                self.__train_images = tf.image.decode_png(tf.read_file(train_input_queue[0]),
                                                          channels=self.__image_depth)
                self.__test_images = tf.image.decode_png(tf.read_file(test_input_queue[0]), channels=self.__image_depth) ## HERE [0]

            # convert images to float and normalize to 1.0
            self.__train_images = tf.image.convert_image_dtype(self.__train_images, dtype=tf.float32)
            self.__test_images = tf.image.convert_image_dtype(self.__test_images, dtype=tf.float32)

            if self.__resize_images is True:
                self.__train_images = tf.image.resize_images(self.__train_images,
                                                             [self.__image_height, self.__image_width])
                self.__test_images = tf.image.resize_images(self.__test_images,
                                                            [self.__image_height, self.__image_width])

            if self.__augmentation_crop is True:

                self.__image_height = int(self.__image_height * self.__crop_amount)
                self.__image_width = int(self.__image_width * self.__crop_amount)

                self.__train_images = tf.random_crop(self.__train_images, [self.__image_height, self.__image_width, 3])
                self.__test_images = tf.image.resize_image_with_crop_or_pad(self.__test_images, self.__image_height,
                                                                            self.__image_width)

            if self.__crop_or_pad_images is True:
                # pad or crop to deal with images of different sizes
                self.__train_images = tf.image.resize_image_with_crop_or_pad(self.__train_images, self.__image_height,
                                                                             self.__image_width)
                self.__test_images = tf.image.resize_image_with_crop_or_pad(self.__test_images, self.__image_height,
                                                                            self.__image_width)

                if self.__problem_type == definitions.ProblemType.SEMANTICSEGMETNATION:
                    self.__train_labels = tf.image.resize_image_with_crop_or_pad(self.__train_labels,
                                                         self.__image_height, self.__image_width)
                    self.__test_labels = tf.image.resize_image_with_crop_or_pad(self.__test_labels,
                                                        self.__image_height, self.__image_width)

            if self.__augmentation_flip_horizontal is True:
                # apply flip horizontal augmentation
                self.__train_images = tf.image.random_flip_left_right(self.__train_images)

            if self.__augmentation_flip_vertical is True:
                # apply flip vertical augmentation
                self.__train_images = tf.image.random_flip_up_down(self.__train_images)

            if self.__augmentation_contrast is True:
                # apply random contrast and brightness augmentation
                self.__train_images = tf.image.random_brightness(self.__train_images, max_delta=63)
                self.__train_images = tf.image.random_contrast(self.__train_images, lower=0.2, upper=1.8)

            # mean-center all inputs
            self.__train_images = tf.image.per_image_standardization(self.__train_images)
            self.__test_images = tf.image.per_image_standardization(self.__test_images)

            # define the shape of the image tensors so it matches the shape of the images
            self.__train_images.set_shape([self.__image_height, self.__image_width, self.__image_depth])
            self.__test_images.set_shape([self.__image_height, self.__image_width, self.__image_depth])


    def __parse_images(self, images, image_type='png'):
        """Takes some images as input, creates producer of processed images internally to this instance"""
        with self.__graph.as_default():
            input_queue = tf.train.string_input_producer(images, shuffle=False)

            reader = tf.WholeFileReader()
            key, file = reader.read(input_queue)

            # pre-processing for all images

            if image_type is 'jpg':
                input_images = tf.image.decode_jpeg(file, channels=self.__image_depth)
            else:
                input_images = tf.image.decode_png(file, channels=self.__image_depth)

            # convert images to float and normalize to 1.0
            input_images = tf.image.convert_image_dtype(input_images, dtype=tf.float32)

            if self.__resize_images is True:
                input_images = tf.image.resize_images(input_images, [self.__image_height, self.__image_width])

            if self.__augmentation_crop is True:
                self.__image_height = int(self.__image_height * self.__crop_amount)
                self.__image_width = int(self.__image_width * self.__crop_amount)
                input_images = tf.image.resize_image_with_crop_or_pad(input_images, self.__image_height,
                                                                      self.__image_width)

            if self.__crop_or_pad_images is True:
                # pad or crop to deal with images of different sizes
                input_images = tf.image.resize_image_with_crop_or_pad(input_images, self.__image_height,
                                                                      self.__image_width)

            # mean-center all inputs
            input_images = tf.image.per_image_standardization(input_images)

            # define the shape of the image tensors so it matches the shape of the images
            input_images.set_shape([self.__image_height, self.__image_width, self.__image_depth])

            self.__all_images = input_images
