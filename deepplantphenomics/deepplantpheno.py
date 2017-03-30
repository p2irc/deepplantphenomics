from . import layers
from . import loaders
from . import preprocessing
from . import definitions
from . import networks
import tensorflow as tf
import numpy as np
from joblib import Parallel, delayed
import os
import datetime
import time
import warnings


class DPPModel(object):
    # Operation settings
    __problem_type = definitions.ProblemType.CLASSIFICATION
    __has_trained = False
    __save_checkpoints = None

    # Input options
    __total_classes = 0
    __total_raw_samples = 0
    __total_training_samples = 0

    __image_width = None
    __image_height = None
    __image_width_original = None
    __image_height_original = None
    __image_depth = None

    __crop_or_pad_images = False
    __resize_images = False

    __preprocessing_steps = []
    __processed_images_dir = './DPP-Processed'

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

    __all_moderation_features = None
    __has_moderation = False
    __moderation_features_size = None
    __train_moderation_features = None
    __test_moderation_features = None

    # Network internal representation
    __session = None
    __graph = None
    __layers = []
    __global_epoch = 0

    __num_layers_norm = 0
    __num_layers_conv = 0
    __num_layers_pool = 0
    __num_layers_fc = 0
    __num_layers_dropout = 0

    # Network options
    __batch_size = None
    __train_test_split = None
    __maximum_training_batches = None
    __reg_coeff = None
    __optimizer = 'Adam'
    __weight_initializer = 'normal'

    __learning_rate = None
    __lr_decay_factor = None
    __lr_decay_epochs = None

    __num_regression_outputs = 1

    # Wrapper options
    __debug = None
    __load_from_saved = None
    __tb_dir = None
    __queue_capacity = 2000
    __report_rate = None

    # Multithreading
    __num_threads = 1
    __coord = None
    __threads = None

    def __init__(self, debug=False, load_from_saved=False, save_checkpoints=True, initialize=True, tensorboard_dir=None, report_rate=100):
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

        # Add the run level to the tensorboard path
        if self.__tb_dir is not None:
            self.__tb_dir = "{0}/{1}".format(self.__tb_dir, datetime.datetime.now().strftime("%d%B%Y%I:%M%p"))

        if initialize:
            self.__log('TensorFlow loaded...')

            self.__graph = tf.Graph()
            self.__session = tf.Session(graph=self.__graph)

    def __log(self, message):
        if self.__debug:
            print('{0}: {1}'.format(datetime.datetime.now().strftime("%I:%M%p"), message))

    def __last_layer(self):
        return self.__layers[-1]

    def __first_layer(self):
        return next(layer for layer in self.__layers if isinstance(layer, layers.convLayer) or isinstance(layer, layers.fullyConnectedLayer))

    def __initialize_queue_runners(self):
        self.__log('Initializing queue runners...')
        self.__coord = tf.train.Coordinator()
        self.__threads = tf.train.start_queue_runners(sess=self.__session, coord=self.__coord)

    def set_number_of_threads(self, num_threads):
        """Set number of threads for input queue runners and preprocessing tasks"""
        self.__num_threads = num_threads

    def set_processed_images_dir(self, dir):
        """Set the directory for storing processed images when pre-processing is used"""
        self.__processed_images_dir = dir

    def set_batch_size(self, size):
        """Set the batch size"""
        self.__batch_size = size

    def set_num_regression_outputs(self, num):
        """Set the number of regression response variables"""
        self.__num_regression_outputs = num

    def set_train_test_split(self, ratio):
        """Set a ratio for the number of samples to use as training set"""
        self.__train_test_split = ratio

    def set_maximum_training_epochs(self, epochs):
        """Set the max number of training epochs"""
        self.__maximum_training_batches = epochs

    def set_learning_rate(self, rate):
        """Set the initial learning rate"""
        self.__learning_rate = rate

    def set_crop_or_pad_images(self, crop_or_pad):
        """Apply padding or cropping images to, which is required if the dataset has images of different sizes"""
        self.__crop_or_pad_images = crop_or_pad

    def set_resize_images(self, resize):
        """Up-sample or down-sample images to specified size"""
        self.__resize_images = resize

    def set_augmentation_flip_horizontal(self, flip):
        """Randomly flip training images horizontally"""
        self.__augmentation_flip_horizontal = flip

    def set_augmentation_flip_vertical(self, flip):
        """Randomly flip training images vertically"""
        self.__augmentation_flip_vertical = flip

    def set_augmentation_crop(self, resize, crop_ratio=0.75):
        """Randomly crop images during training, and crop images to center during testing"""
        self.__augmentation_crop = resize
        self.__crop_amount = crop_ratio

    def set_augmentation_brightness_and_contrast(self, contr):
        """Randomly adjust contrast and/or brightness on training images"""
        self.__augmentation_contrast = contr

    def set_regularization_coefficient(self, lamb):
        """Set lambda for L2 weight decay"""
        self.__reg_coeff = lamb

    def set_learning_rate_decay(self, decay_factor, epochs_per_decay):
        """Set learning rate decay"""
        self.__lr_decay_factor = decay_factor
        self.__lr_decay_epochs = epochs_per_decay

    def set_optimizer(self, optimizer):
        """Set the optimizer to use"""
        self.__optimizer = optimizer

    def set_weight_initializer(self, initializer):
        """Set the initialization scheme used by convolutional and fully connected layers"""
        self.__weight_initializer = initializer

    def set_image_dimensions(self, image_height, image_width, image_depth):
        """Specify the image dimensions for images in the dataset (depth is the number of channels)"""
        self.__image_width = image_width
        self.__image_height = image_height
        self.__image_depth = image_depth

    def set_original_image_dimensions(self, image_height, image_width):
        """
        Specify the original size of the image, before resizing.
        This is only needed in special cases, for instance, if you are resizing input images but using image coordinate
        labels which reference the original size.
        """
        self.__image_width_original = image_width
        self.__image_height_original = image_height

    def add_moderation_features(self, moderation_features):
        """Specify moderation features for examples in the dataset"""
        self.__has_moderation = True
        self.__moderation_features_size = moderation_features.shape[1]
        self.__all_moderation_features = moderation_features

    def add_preprocessor(self, selection):
        """Add a data preprocessing step"""
        self.__preprocessing_steps.append(selection)

    def clear_preprocessors(self):
        """Clear all preprocessing steps"""
        self.__preprocessing_steps = []

    def set_problem_type(self, type):
        """Set the problem type to be solved, either classification or regression"""
        if type == 'classification':
            self.__problem_type = definitions.ProblemType.CLASSIFICATION
        elif type == 'regression':
            self.__problem_type = definitions.ProblemType.REGRESSION
        else:
            warnings.warn('Problem type specified not supported', stacklevel=2)

    def begin_training(self):
        """
        Initialize the network and either run training to the specified max epoch, or load trainable variables.
        The full test accuracy is calculated immediately afterward. Finally, the trainable parameters are saved and
        the session is shut down.
        Before calling this function, the images and labels should be loaded, as well as all relevant hyperparameters.
        """

        self.__log('Beginning training...')

        with self.__graph.as_default():
            # Define batches
            if self.__has_moderation:
                x, y, mod_w = tf.train.shuffle_batch([self.__train_images, self.__train_labels, self.__train_moderation_features],
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

            # If this is a regression problem, unserialize the label
            if self.__problem_type == definitions.ProblemType.REGRESSION:
                y = loaders.label_string_to_tensor(y, self.__batch_size, self.__num_regression_outputs)

            # Run the network operations
            if self.__has_moderation:
                xx = self.forward_pass(x, deterministic=False, moderation_features=mod_w)
            else:
                xx = self.forward_pass(x, deterministic=False)

            if self.__problem_type == definitions.ProblemType.CLASSIFICATION:
                class_predictions = tf.argmax(tf.nn.softmax(xx), 1)

            # Define regularization cost
            if self.__reg_coeff is not None:
                l2_cost = [layer.regularization_coefficient * tf.nn.l2_loss(layer.weights) for layer in self.__layers
                           if isinstance(layer, layers.fullyConnectedLayer) or isinstance(layer, layers.convLayer)]
            else:
                l2_cost = [0.0]

            # Define cost function and set optimizer
            if self.__problem_type == definitions.ProblemType.CLASSIFICATION:
                sf_logits = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=xx, labels=tf.argmax(y, 1))
                cost = tf.reduce_mean(tf.concat([sf_logits, l2_cost], axis=0))
            elif self.__problem_type == definitions.ProblemType.REGRESSION:
                cost = self.__batch_mean_l2_loss(tf.subtract(xx, y))

            if self.__optimizer == 'Adagrad':
                optimizer = tf.train.AdagradOptimizer(self.__learning_rate).minimize(cost)
                self.__log('Using Adagrad optimizer')
            elif self.__optimizer == 'Adadelta':
                optimizer = tf.train.AdadeltaOptimizer(self.__learning_rate).minimize(cost)
                self.__log('Using Adadelta optimizer')
            elif self.__optimizer == 'SGD':
                optimizer = tf.train.GradientDescentOptimizer(self.__learning_rate).minimize(cost)
                self.__log('Using SGD optimizer')
            else:
                optimizer = tf.train.AdamOptimizer(self.__learning_rate).minimize(cost)
                self.__log('Using Adam optimizer')

            if self.__problem_type == definitions.ProblemType.CLASSIFICATION:
                correct_predictions = tf.equal(class_predictions, tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

            # Calculate validation accuracy
            if self.__has_moderation:
                x_test, y_test, mod_w_test = tf.train.shuffle_batch([self.__test_images, self.__test_labels, self.__test_moderation_features],
                                                        batch_size=self.__batch_size,
                                                        num_threads=self.__num_threads,
                                                        capacity=self.__queue_capacity,
                                                        min_after_dequeue=self.__batch_size)
            else:
                x_test, y_test = tf.train.shuffle_batch([self.__test_images, self.__test_labels],
                                                        batch_size=self.__batch_size,
                                                        num_threads=self.__num_threads,
                                                        capacity=self.__queue_capacity,
                                                        min_after_dequeue=self.__batch_size)

            if self.__problem_type == definitions.ProblemType.REGRESSION:
                y_test = loaders.label_string_to_tensor(y_test, self.__batch_size, self.__num_regression_outputs)

            x_test = tf.reshape(x_test, shape=[-1, self.__image_height, self.__image_width, self.__image_depth])

            if self.__has_moderation:
                x_test_predicted = self.forward_pass(x_test, deterministic=True, moderation_features=mod_w_test)
            else:
                x_test_predicted = self.forward_pass(x_test, deterministic=True)

            if self.__problem_type == definitions.ProblemType.CLASSIFICATION:
                test_class_predictions = tf.argmax(tf.nn.softmax(x_test_predicted), 1)
                test_correct_predictions = tf.equal(test_class_predictions, tf.argmax(y_test, 1))
                test_accuracy = tf.reduce_mean(tf.cast(test_correct_predictions, tf.float32))
            elif self.__problem_type == definitions.ProblemType.REGRESSION:
                test_batch_loss = tf.subtract(x_test_predicted, y_test)
                test_cost = self.__batch_mean_l2_loss(test_batch_loss)

            full_test_op = self.compute_full_test_accuracy()

            # Epoch summaries for Tensorboard
            if self.__tb_dir is not None:
                # Summaries for any problem type
                tf.summary.scalar('train/loss', cost, collections=['custom_summaries'])
                tf.summary.scalar('train/learning_rate', self.__learning_rate, collections=['custom_summaries'])
                tf.summary.scalar('train/l2_loss', tf.reduce_mean(l2_cost), collections=['custom_summaries'])
                filter_summary = self.__get_weights_as_image(self.__first_layer().weights)
                tf.summary.image('filters/first', filter_summary, collections=['custom_summaries'])

                # Summaries for classification problems
                if self.__problem_type == definitions.ProblemType.CLASSIFICATION:
                    tf.summary.scalar('train/accuracy', accuracy, collections=['custom_summaries'])
                    tf.summary.scalar('test/accuracy', test_accuracy, collections=['custom_summaries'])
                    tf.summary.histogram('train/class_predictions', class_predictions, collections=['custom_summaries'])
                    tf.summary.histogram('test/class_predictions', test_class_predictions, collections=['custom_summaries'])

                # Summaries for regression
                if self.__problem_type == definitions.ProblemType.REGRESSION:
                    tf.summary.scalar('test/loss', test_cost, collections=['custom_summaries'])

                    if test_batch_loss.get_shape().as_list()[1] == 1:
                        tf.summary.histogram('test/batch_loss', test_batch_loss, collections=['custom_summaries'])

                # Summaries for each layer
                for layer in self.__layers:
                    if hasattr(layer, 'name'):
                        tf.summary.histogram('weights/'+layer.name, layer.weights, collections=['custom_summaries'])
                        tf.summary.histogram('biases/'+layer.name, layer.biases, collections=['custom_summaries'])
                        tf.summary.histogram('activations/'+layer.name, layer.activations, collections=['custom_summaries'])

                merged = tf.summary.merge_all(key='custom_summaries')
                train_writer = tf.summary.FileWriter(self.__tb_dir, self.__session.graph)

            # Either load the network parameters from a checkpoint file or start training
            if self.__load_from_saved is not False:
                self.load_state()

                self.__initialize_queue_runners()

                self.__log('Computing total test accuracy/regression loss...')

                if self.__problem_type == definitions.ProblemType.CLASSIFICATION:
                    mean = self.__session.run(full_test_op)

                    self.__log('Average test accuracy: {:.5f}'.format(mean))
                elif self.__problem_type == definitions.ProblemType.REGRESSION:
                    mean, std, abs_mean, abs_std, min, max, hist = self.__session.run(full_test_op)

                    self.__log('Mean loss: {}'.format(mean))
                    self.__log('Loss standard deviation: {}'.format(std))
                    self.__log('Mean absolute loss: {}'.format(abs_mean))
                    self.__log('Absolute loss standard deviation: {}'.format(abs_std))
                    self.__log('Min error: {}'.format(min))
                    self.__log('Max error: {}'.format(max))
                    self.__log('Histogram of L2 losses:')
                    self.__log(hist)

                self.shut_down()
            else:
                self.__log('Initializing parameters...')
                init_op = tf.global_variables_initializer()
                self.__session.run(init_op)

                self.__initialize_queue_runners()

                for i in range(self.__maximum_training_batches):
                    start_time = time.time()
                    self.__global_epoch = i

                    self.__session.run(optimizer)

                    if self.__global_epoch > 0 and self.__global_epoch % self.__report_rate == 0:
                        elapsed = time.time() - start_time
                        self.__set_learning_rate()

                        if self.__tb_dir is not None:
                            summary = self.__session.run(merged)
                            train_writer.add_summary(summary, i)

                        if self.__problem_type == definitions.ProblemType.CLASSIFICATION:
                            loss, epoch_accuracy, epoch_test_accuracy = self.__session.run([cost, accuracy, test_accuracy])

                            samples_per_sec = self.__batch_size / elapsed

                            self.__log(
                                'Results for batch {} (epoch {}) - Loss: {:.5f}, Training Accuracy: {:.4f}, samples/sec: {:.2f}'
                                .format(i,
                                        i / (self.__total_training_samples / self.__batch_size),
                                        loss,
                                        epoch_accuracy,
                                        samples_per_sec))
                        elif self.__problem_type == definitions.ProblemType.REGRESSION:
                            loss, epoch_test_loss = self.__session.run([cost, test_cost])

                            samples_per_sec = self.__batch_size / elapsed

                            self.__log(
                                'Results for batch {} (epoch {}) - Regression Loss: {:.5f}, samples/sec: {:.2f}'
                                    .format(i,
                                            i / (self.__total_training_samples / self.__batch_size),
                                            loss,
                                            samples_per_sec))

                        if self.__save_checkpoints and self.__global_epoch % (self.__report_rate * 100) == 0:
                            self.save_state()
                    else:
                        loss = self.__session.run([cost])

                    if loss == 0.0:
                        self.__log('Stopping due to zero loss')
                        break

                    if i == self.__maximum_training_batches-1:
                        self.__log('Stopping due to maximum epochs')

                self.save_state()

                self.__log('Computing total test accuracy/regression loss...')

                if self.__problem_type == definitions.ProblemType.CLASSIFICATION:
                    mean = self.__session.run(full_test_op)

                    self.__log('Average test accuracy: {:.5f}'.format(mean))
                elif self.__problem_type == definitions.ProblemType.REGRESSION:
                    mean, std, abs_mean, abs_std, min, max, hist = self.__session.run(full_test_op)

                    self.__log('Mean loss: {}'.format(mean))
                    self.__log('Loss standard deviation: {}'.format(std))
                    self.__log('Mean absolute loss: {}'.format(abs_mean))
                    self.__log('Absolute loss standard deviation: {}'.format(abs_std))
                    self.__log('Min error: {}'.format(min))
                    self.__log('Max error: {}'.format(max))
                    self.__log('Histogram of L2 losses:')
                    self.__log(hist)

                self.shut_down()

    def compute_full_test_accuracy(self):
        """Returns statistics of the test losses depending on the type of task"""

        with self.__graph.as_default():
            num_test = self.__total_raw_samples - self.__total_training_samples
            num_batches = int(num_test/self.__batch_size)+1

            if num_batches == 0:
                warnings.warn('Less than a batch of testing data')
                exit()

            sum = 0.0
            all_losses = np.empty(shape=(4, 1))

            for i in range(num_batches):
                if self.__has_moderation:
                    x_test, y_test, mod_w_test = tf.train.batch([self.__test_images, self.__test_labels, self.__test_moderation_features],
                                                            batch_size=self.__batch_size,
                                                            num_threads=self.__num_threads)
                else:
                    x_test, y_test = tf.train.batch([self.__test_images, self.__test_labels],
                                                            batch_size=self.__batch_size,
                                                            num_threads=self.__num_threads)

                x_test = tf.reshape(x_test, shape=[-1, self.__image_height, self.__image_width, self.__image_depth])

                if self.__has_moderation:
                    x_test_predicted = self.forward_pass(x_test, deterministic=True, moderation_features=mod_w_test)
                else:
                    x_test_predicted = self.forward_pass(x_test, deterministic=True)

                if self.__problem_type == definitions.ProblemType.CLASSIFICATION:
                    test_class_predictions = tf.argmax(tf.nn.softmax(x_test_predicted), 1)

                    test_correct_predictions = tf.equal(test_class_predictions, tf.argmax(y_test, 1))
                    test_acc = tf.reduce_mean(tf.cast(test_correct_predictions, tf.float32))

                    sum = sum + test_acc
                elif self.__problem_type == definitions.ProblemType.REGRESSION:
                    y_test = loaders.label_string_to_tensor(y_test, self.__batch_size, self.__num_regression_outputs)
                    losses = tf.subtract(x_test_predicted, y_test)
                    all_losses = tf.concat([all_losses, losses], axis=0)

            if self.__problem_type == definitions.ProblemType.CLASSIFICATION:
                # For classification problems (assumed to be multi-class), we want accuracy and confusion matrix
                mean = (sum / num_batches)

                return mean
            elif self.__problem_type == definitions.ProblemType.REGRESSION:
                # For regression problems we want relative and abs mean, std of L2 norms, plus a histogram of errors
                abs_mean, abs_var = tf.nn.moments(tf.abs(all_losses), axes=[0])
                abs_std = tf.sqrt(abs_var)

                mean, var = tf.nn.moments(all_losses, axes=[0])
                std = tf.sqrt(var)

                max = tf.reduce_max(all_losses)
                min = tf.reduce_min(all_losses)
                hist = tf.histogram_fixed_width(all_losses, [min, max], nbins=100)

                return mean, std, abs_mean, abs_std, min, max, hist

    def shut_down(self):
        """Stop all queues and end session. The model cannot be used anymore after a shut down is completed."""
        self.__log('Shutdown requested, ending session...')

        self.__coord.request_stop()
        self.__coord.join(self.__threads)

        self.__session.close()

    def __get_weights_as_image(self, kernel):
        """Filter visualization, adapted with permission from https://gist.github.com/kukuruza/03731dc494603ceab0c5"""
        with self.__graph.as_default():
            pad = 1
            grid_X = 4
            grid_Y = (kernel.get_shape().as_list()[-1] / 4)

            # pad X and Y
            x1 = tf.pad(kernel, tf.constant([[pad, 0], [pad, 0], [0, 0], [0, 0]]))

            # X and Y dimensions, w.r.t. padding
            Y = kernel.get_shape()[0] + pad
            X = kernel.get_shape()[1] + pad

            # pack into image with proper dimensions for tf.image_summary
            x2 = tf.transpose(x1, (3, 0, 1, 2))
            x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, 3]))
            x4 = tf.transpose(x3, (0, 2, 1, 3))
            x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, 3]))
            x6 = tf.transpose(x5, (2, 1, 3, 0))
            x7 = tf.transpose(x6, (3, 0, 1, 2))

            # scale to [0, 1]
            x_min = tf.reduce_min(x7)
            x_max = tf.reduce_max(x7)
            x8 = (x7 - x_min) / (x_max - x_min)

        return x8

    def save_state(self):
        """Save all trainable variables as a checkpoint in the current working path"""
        self.__log('Saving parameters...')

        with self.__graph.as_default():
            saver = tf.train.Saver(tf.trainable_variables())
            saver.save(self.__session, 'tfhSaved')

        self.__has_trained = True

    def load_state(self):
        """
        Load all trainable variables from a checkpoint file specified from the load_from_saved parameter in the
        class constructor.
        """
        if self.__load_from_saved is not False:
            self.__log('Loading from checkpoint file...')

            with self.__graph.as_default():
                saver = tf.train.Saver(tf.trainable_variables())
                saver.restore(self.__session, tf.train.latest_checkpoint(self.__load_from_saved))

            self.__has_trained = True
        else:
            warnings.warn('Tried to load state with no file given. Make sure load_from_saved is set in constructor.')

    def __set_learning_rate(self):
        if self.__lr_decay_factor is not None:
            self.__learning_rate = tf.train.exponential_decay(self.__learning_rate,
                                            self.__global_epoch,
                                            self.__lr_decay_epochs,
                                            self.__lr_decay_factor,
                                            staircase=True)

            tf.summary.scalar('learning_rate', self.__learning_rate)

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
            else:
                warnings.warn('Problem type is not recognized')

            num_batches = len(x) / self.__batch_size
            remainder = len(x) % self.__batch_size

            if remainder != 0:
                num_batches += 1
                remainder = self.__batch_size - remainder

            self.load_images_from_list(x)

            x_test = tf.train.batch([self.__all_images], batch_size=self.__batch_size, num_threads=self.__num_threads)

            x_test = tf.reshape(x_test, shape=[-1, self.__image_height, self.__image_width, self.__image_depth])

            x_pred = self.forward_pass(x_test, deterministic=True)

            self.load_state()

            self.__initialize_queue_runners()

            for i in range(num_batches):
                xx = self.__session.run(x_pred)
                total_outputs = np.append(total_outputs, xx, axis=0)

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
            agg = tf.map_fn(lambda ex: tf.sqrt(tf.reduce_sum(ex ** 2)), x)
            mean = tf.reduce_mean(agg)

        return mean

    def add_input_layer(self):
        """Add an input layer to the network"""
        self.__log('Adding the input layer...')

        apply_crop = (self.__augmentation_crop and self.__all_images is None and self.__train_images is None)

        if apply_crop:
            size = [self.__batch_size, int(self.__image_height * self.__crop_amount),
                    int(self.__image_width * self.__crop_amount), self.__image_depth]
        else:
            size = [self.__batch_size, self.__image_height, self.__image_width, self.__image_depth]

        with self.__graph.as_default():
            layer = layers.inputLayer(size)

        self.__layers.append(layer)

    def add_moderation_layer(self):
        """Add a moderation layer to the network"""
        self.__log('Adding moderation layer...')

        reshape = isinstance(self.__last_layer(), layers.convLayer) or isinstance(self.__last_layer(), layers.poolingLayer)
        feat_size = self.__moderation_features_size

        with self.__graph.as_default():
            layer = layers.moderationLayer(self.__last_layer().output_size, feat_size, reshape, self.__batch_size)

        self.__layers.append(layer)

    def add_convolutional_layer(self, filter_dimension, stride_length, activation_function, regularization_coefficient=None):
        """
        Add a convolutional layer to the model.

        :param filter_dimension: array of dimensions in the format [x_size, y_size, depth, num_filters]
        :param stride_length: convolution stride length
        :param activation_function: the activation function to apply to the activation map
        :param regularization_coefficient: optionally, an L2 decay coefficient for this layer (overrides the coefficient
         set by set_regularization_coefficient)
        """
        self.__num_layers_conv += 1
        layer_name = 'conv%d' % self.__num_layers_conv
        self.__log('Adding convolutional layer %s...' % layer_name)

        if regularization_coefficient is None and self.__reg_coeff is not None:
            regularization_coefficient = self.__reg_coeff
        elif regularization_coefficient is None and self.__reg_coeff is None:
            regularization_coefficient = 0.0

        with self.__graph.as_default():
            layer = layers.convLayer(layer_name,
                                     self.__last_layer().output_size,
                                     filter_dimension,
                                     stride_length,
                                     activation_function,
                                     self.__weight_initializer,
                                     regularization_coefficient)

        self.__log('Filter dimensions: {0} Outputs: {1}'.format(filter_dimension, layer.output_size))

        self.__layers.append(layer)

    def add_pooling_layer(self, kernel_size, stride_length, pooling_type='max'):
        """
        Add a pooling layer to the model.

        :param kernel_size: an integer representing the width and height dimensions of the pooling operation
        :param stride_length: convolution stride length
        :param pooling_type: optional, the type of pooling operation
        """
        self.__num_layers_pool += 1
        layer_name = 'pool%d' % self.__num_layers_pool
        self.__log('Adding pooling layer %s...' % layer_name)

        with self.__graph.as_default():
            layer = layers.poolingLayer(self.__last_layer().output_size, kernel_size, stride_length, pooling_type)

        self.__log('Outputs: %s' % layer.output_size)

        self.__layers.append(layer)

    def add_normalization_layer(self):
        """Add a local response normalization layer to the model"""
        self.__num_layers_norm += 1
        layer_name = 'norm%d' % self.__num_layers_pool
        self.__log('Adding pooling layer %s...' % layer_name)

        with self.__graph.as_default():
            layer = layers.normLayer(self.__last_layer().output_size)

        self.__layers.append(layer)

    def add_dropout_layer(self, p):
        """
        Add a DropOut layer to the model.

        :param p: the keep-probability parameter for the DropOut operation
        """
        self.__num_layers_dropout += 1
        layer_name = 'drop%d' % self.__num_layers_dropout
        self.__log('Adding dropout layer %s...' % layer_name)

        with self.__graph.as_default():
            layer = layers.dropoutLayer(self.__last_layer().output_size, p)

        self.__layers.append(layer)

    def add_fully_connected_layer(self, output_size, activation_function, regularization_coefficient=None):
        """
        Add a fully connected layer to the model.

        :param output_size: the number of units in the layer
        :param activation_function: optionally, the activation function to use
        :param regularization_coefficient: optionally, an L2 decay coefficient for this layer (overrides the coefficient
         set by set_regularization_coefficient)
        """
        self.__num_layers_fc += 1
        layer_name = 'fc%d' % self.__num_layers_fc
        self.__log('Adding fully connected layer %s...' % layer_name)

        reshape = isinstance(self.__last_layer(), layers.convLayer) or isinstance(self.__last_layer(), layers.poolingLayer)

        if regularization_coefficient is None and self.__reg_coeff is not None:
            regularization_coefficient = self.__reg_coeff
        if regularization_coefficient is None and self.__reg_coeff is None:
            regularization_coefficient = 0.0

        with self.__graph.as_default():
            layer = layers.fullyConnectedLayer(layer_name,
                                               self.__last_layer().output_size,
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
        self.__log('Adding output layer...')

        reshape = isinstance(self.__last_layer(), layers.convLayer) or isinstance(self.__last_layer(), layers.poolingLayer)

        if regularization_coefficient is None and self.__reg_coeff is not None:
            regularization_coefficient = self.__reg_coeff
        if regularization_coefficient is None and self.__reg_coeff is None:
            regularization_coefficient = 0.0

        if output_size is None:
            if self.__problem_type == definitions.ProblemType.CLASSIFICATION:
                num_out = self.__total_classes
            elif self.__problem_type == definitions.ProblemType.REGRESSION:
                num_out = self.__num_regression_outputs
            else:
                warnings.warn('Problem type is not recognized')
        else:
            num_out = output_size

        with self.__graph.as_default():
            layer = layers.fullyConnectedLayer('output',
                                               self.__last_layer().output_size,
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

        image_files = [os.path.join(dirname, name) for name in os.listdir(dirname) if
                       os.path.isfile(os.path.join(dirname, name)) & name.endswith('.png')]

        labels = loaders.read_csv_labels(labels_file, column_number)

        self.__total_raw_samples = len(image_files)
        self.__total_classes = len(set(labels))

        self.__log('Total raw examples is %d' % self.__total_raw_samples)
        self.__log('Total classes is %d' % self.__total_classes)
        self.__log('Parsing dataset...')

        with self.__graph.as_default():
            if self.__has_moderation:
                train_images, train_labels, test_images, test_labels, train_mf, test_mf = \
                    loaders.split_raw_data(image_files, labels, self.__train_test_split, moderation_features=self.__all_moderation_features)
                self.__parse_dataset(train_images, train_labels, test_images, test_labels, train_mf=train_mf, test_mf=test_mf)
            else:
                train_images, train_labels, test_images, test_labels = loaders.split_raw_data(image_files, labels, self.__train_test_split)
                self.__parse_dataset(train_images, train_labels, test_images, test_labels)

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
        self.__log('Parsing dataset...')

        # split data
        with self.__graph.as_default():
            if self.__has_moderation:
                train_images, train_labels, test_images, test_labels, train_mf, test_mf = \
                    loaders.split_raw_data(image_files, labels, self.__train_test_split, moderation_features=self.__all_moderation_features)
                self.__parse_dataset(train_images, train_labels, test_images, test_labels, train_mf=train_mf, test_mf=test_mf)
            else:
                train_images, train_labels, test_images, test_labels = loaders.split_raw_data(image_files, labels, self.__train_test_split)
                self.__parse_dataset(train_images, train_labels, test_images, test_labels)

    def load_ippn_tray_dataset_from_directory(self, dirname):
        """
        Loads the RGB tray images and plant bounding box labels from the International Plant Phenotyping Network
        dataset.
        """

        images = [os.path.join(dirname, name) for name in os.listdir(dirname) if
                       os.path.isfile(os.path.join(dirname, name)) & name.endswith('_rgb.png')]

        label_files = [os.path.join(dirname, name) for name in os.listdir(dirname) if
                       os.path.isfile(os.path.join(dirname, name)) & name.endswith('_bbox.csv')]

        labels = [loaders.read_csv_labels(label_file) for label_file in label_files]

        self.__all_labels = []

        for label in labels:
            self.__all_labels.append([loaders.box_coordinates_to_pascal_voc_coordinates(l) for l in label])

        self.__total_raw_samples = len(images)

        self.__log('Total raw examples is %d' % self.__total_raw_samples)
        self.__log('Parsing dataset...')

        # do preprocessing
        images = self.__apply_preprocessing(images)

        # prepare images for training (if there are any labels loaded)
        if self.__all_labels is not None:
            with self.__graph.as_default():
                # split data
                train_images, train_labels, test_images, test_labels = loaders.split_raw_data(images, self.__all_labels, self.__train_test_split)

                # create batches of input data and labels for training
                self.__parse_dataset(train_images, train_labels, test_images, test_labels)

    def load_ippn_leaf_count_dataset_from_directory(self, dirname):
        """Loads the RGB images and species labels from the International Plant Phenotyping Network dataset."""

        labels, ids = loaders.read_csv_labels_and_ids(os.path.join(dirname, 'Leaf_counts.csv'), 1, 0)

        # labels must be lists
        labels = [[label] for label in labels]

        image_files = [os.path.join(dirname, id + '_rgb.png') for id in ids]

        self.__total_raw_samples = len(image_files)

        self.__log('Total raw examples is %d' % self.__total_raw_samples)
        self.__log('Parsing dataset...')

        with self.__graph.as_default():
            train_images, train_labels, test_images, test_labels = loaders.split_raw_data(image_files, labels, self.__train_test_split)
            self.__parse_dataset(train_images, train_labels, test_images, test_labels)

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

        # split data
        with self.__graph.as_default():
            if self.__has_moderation:
                train_images, train_labels, test_images, test_labels, train_mf, test_mf = \
                    loaders.split_raw_data(image_files, labels, self.__train_test_split, moderation_features=self.__all_moderation_features)
                self.__parse_dataset(train_images, train_labels, test_images, test_labels, train_mf=train_mf, test_mf=test_mf)
            else:
                train_images, train_labels, test_images, test_labels = loaders.split_raw_data(image_files, labels, self.__train_test_split)
                self.__parse_dataset(train_images, train_labels, test_images, test_labels)

    def load_cifar10_dataset_from_directory(self, dirname):
        """
        Loads the images and labels from a directory containing the CIFAR-10 image classification dataset as
        downloaded by nvidia DIGITS.
        """

        train_dir = os.path.join(dirname, 'train')
        test_dir = os.path.join(dirname, 'test')
        self.__total_classes = 10
        self.__queue_capacity = 60000

        train_labels, train_images = loaders.read_csv_labels_and_ids(os.path.join(train_dir, 'train.txt'), 1, 0, character=' ')

        # transform into numerical one-hot labels
        train_labels = [int(label) for label in train_labels]
        train_labels = tf.one_hot(train_labels, self.__total_classes)

        test_labels, test_images = loaders.read_csv_labels_and_ids(os.path.join(test_dir, 'test.txt'), 1, 0, character=' ')

        # transform into numerical one-hot labels
        test_labels = [int(label) for label in test_labels]
        test_labels = tf.one_hot(test_labels, self.__total_classes)

        self.__total_raw_samples = len(train_images) + len(test_images)

        self.__log('Total raw examples is %d' % self.__total_raw_samples)
        self.__log('Total classes is %d' % self.__total_classes)
        self.__log('Parsing dataset...')

        # create batches of input data and labels for training
        with self.__graph.as_default():
            self.__parse_dataset(train_images, train_labels, test_images, test_labels)

    def load_dataset_from_directory_with_auto_labels(self, dirname):
        """Loads the png images in the given directory, using subdirectories to separate classes."""

        # Load all file names and labels into arrays
        subdirs = filter(lambda item: os.path.isdir(item) & (item != '.DS_Store'),
                         [os.path.join(dirname, f) for f in os.listdir(dirname)])

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

        # split data
        with self.__graph.as_default():
            if self.__has_moderation:
                train_images, train_labels, test_images, test_labels, train_mf, test_mf = \
                    loaders.split_raw_data(image_files, labels, self.__train_test_split, moderation_features=self.__all_moderation_features)
                self.__parse_dataset(train_images, train_labels, test_images, test_labels, train_mf=train_mf, test_mf=test_mf)
            else:
                train_images, train_labels, test_images, test_labels = loaders.split_raw_data(image_files, labels, self.__train_test_split)
                self.__parse_dataset(train_images, train_labels, test_images, test_labels)

    def load_lemnatec_images_from_directory(self, dirname):
        """
        Loads the RGB (VIS) images from a Lemnatec plant scanner image dataset. Unless you only want to do
        preprocessing, regression or classification labels MUST be loaded first.
        """

        # Load all snapshot subdirectories
        subdirs = filter(lambda item: os.path.isdir(item) & (item != '.DS_Store'),
                         [os.path.join(dirname, f) for f in os.listdir(dirname)])

        image_files = []

        # Load the VIS images in each subdirectory
        for sd in subdirs:
            image_paths = [os.path.join(sd, name) for name in os.listdir(sd) if
                           os.path.isfile(os.path.join(sd, name)) & name.startswith('VIS_SV_')]

            image_files = image_files + image_paths

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

            with self.__graph.as_default():
                if self.__has_moderation:
                    train_images, train_labels, test_images, test_labels, train_mf, test_mf = \
                        loaders.split_raw_data(images, labels, self.__train_test_split, moderation_features=self.__all_moderation_features)
                    self.__parse_dataset(train_images, train_labels, test_images, test_labels, train_mf=train_mf, test_mf=test_mf)
                else:
                    train_images, train_labels, test_images, test_labels = loaders.split_raw_data(images, labels, self.__train_test_split)
                    self.__parse_dataset(train_images, train_labels, test_images, test_labels)

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
            # split data
            with self.__graph.as_default():
                labels = self.__all_labels

                if self.__has_moderation:
                    train_images, train_labels, test_images, test_labels, train_mf, test_mf = \
                        loaders.split_raw_data(images, labels, self.__train_test_split, moderation_features=self.__all_moderation_features)
                    self.__parse_dataset(train_images, train_labels, test_images, test_labels, train_mf=train_mf, test_mf=test_mf)
                else:
                    train_images, train_labels, test_images, test_labels = loaders.split_raw_data(images, labels, self.__train_test_split)
                    self.__parse_dataset(train_images, train_labels, test_images, test_labels)
        else:
            with self.__graph.as_default():
                self.__parse_images(images)

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

            with self.__graph.as_default():
                if self.__has_moderation:
                    train_images, train_labels, test_images, test_labels, train_mf, test_mf = \
                        loaders.split_raw_data(images, labels, self.__train_test_split,
                                               moderation_features=self.__all_moderation_features)
                    self.__parse_dataset(train_images, train_labels, test_images, test_labels, train_mf=train_mf,
                                         test_mf=test_mf)
                else:
                    train_images, train_labels, test_images, test_labels = loaders.split_raw_data(images, labels, self.__train_test_split)
                    self.__parse_dataset(train_images, train_labels, test_images, test_labels)

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

                    processed_images = Parallel(n_jobs=self.__num_threads)\
                        (delayed(preprocessing.do_parallel_auto_segmentation)
                         (i[0], i[1], self.__processed_images_dir, self.__image_height, self.__image_width) for i in images)

                    images = processed_images

        return images

    def __parse_dataset(self, train_images, train_labels, test_images, test_labels, image_type='png', train_mf=None, test_mf=None):
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

            self.__test_labels = test_input_queue[1]
            self.__train_labels = train_input_queue[1]

            # pre-processing for training and testing images

            if image_type is 'jpg':
                self.__train_images = tf.image.decode_jpeg(tf.read_file(train_input_queue[0]), channels=self.__image_depth)
                self.__test_images = tf.image.decode_jpeg(tf.read_file(test_input_queue[0]), channels=self.__image_depth)
            else:
                self.__train_images = tf.image.decode_png(tf.read_file(train_input_queue[0]), channels=self.__image_depth)
                self.__test_images = tf.image.decode_png(tf.read_file(test_input_queue[0]), channels=self.__image_depth)

            # convert images to float and normalize to 1.0
            self.__train_images = tf.image.convert_image_dtype(self.__train_images, dtype=tf.float32)
            self.__test_images = tf.image.convert_image_dtype(self.__test_images, dtype=tf.float32)

            if self.__resize_images is True:
                self.__train_images = tf.image.resize_images(self.__train_images, [self.__image_height, self.__image_width])
                self.__test_images = tf.image.resize_images(self.__test_images, [self.__image_height, self.__image_width])

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
                input_images = tf.image.resize_image_with_crop_or_pad(input_images, self.__image_height, self.__image_width)

            if self.__crop_or_pad_images is True:
                # pad or crop to deal with images of different sizes
                input_images = tf.image.resize_image_with_crop_or_pad(input_images, self.__image_height, self.__image_width)

            # mean-center all inputs
            input_images = tf.image.per_image_standardization(input_images)

            # define the shape of the image tensors so it matches the shape of the images
            input_images.set_shape([self.__image_height, self.__image_width, self.__image_depth])

            self.__all_images = input_images