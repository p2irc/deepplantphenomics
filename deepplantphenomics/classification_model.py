from . import layers, loaders, definitions, DPPModel
import numpy as np
import tensorflow as tf
import os
import datetime
import time
import warnings
import copy
from tqdm import tqdm


class ClassificationModel(DPPModel):
    _problem_type = definitions.ProblemType.CLASSIFICATION
    _loss_fn = 'softmax cross entropy'
    _supported_loss_fns = ['softmax cross entropy']
    _supported_augmentations = [definitions.AugmentationType.FLIP_HOR,
                                definitions.AugmentationType.FLIP_VER,
                                definitions.AugmentationType.CROP,
                                definitions.AugmentationType.CONTRAST_BRIGHT,
                                definitions.AugmentationType.ROTATE]

    def __init__(self, debug=False, load_from_saved=False, save_checkpoints=True, initialize=True, tensorboard_dir=None,
                 report_rate=100, save_dir=None):
        super().__init__(debug, load_from_saved, save_checkpoints, initialize, tensorboard_dir, report_rate, save_dir)

        # State variables specific to classification for constructing the graph and passing to Tensorboard
        self.__class_predictions = None
        self.__val_class_predictions = None

    def _graph_tensorboard_summary(self, l2_cost, gradients, variables, global_grad_norm):
        super()._graph_tensorboard_common_summary(l2_cost, gradients, variables, global_grad_norm)

        # Summaries specific to classification problems
        tf.summary.scalar('train/accuracy', self._graph_ops['accuracy'], collections=['custom_summaries'])
        tf.summary.histogram('train/class_predictions', self.__class_predictions, collections=['custom_summaries'])
        if self._validation:
            tf.summary.scalar('validation/accuracy', self._graph_ops['val_accuracy'],
                              collections=['custom_summaries'])
            tf.summary.histogram('validation/class_predictions', self.__val_class_predictions,
                                 collections=['custom_summaries'])

        self._graph_ops['merged'] = tf.summary.merge_all(key='custom_summaries')

    def _assemble_graph(self):
        with self._graph.as_default():
            self._log('Assembling graph...')
            self._log('Graph: Parsing dataset...')
            self._graph_parse_data()  # Always done on CPU

            # Define batches
            if self._has_moderation:
                x, y, mod_w = tf.train.shuffle_batch(
                    [self._train_images, self._train_labels, self._train_moderation_features],
                    batch_size=self._batch_size,
                    num_threads=self._num_threads,
                    capacity=self._queue_capacity,
                    min_after_dequeue=self._batch_size)
            else:
                x, y = tf.train.shuffle_batch([self._train_images, self._train_labels],
                                              batch_size=self._batch_size,
                                              num_threads=self._num_threads,
                                              capacity=self._queue_capacity,
                                              min_after_dequeue=self._batch_size)

            # Reshape input to the expected image dimensions
            x = tf.reshape(x, shape=[-1, self._image_height, self._image_width, self._image_depth])

            # If we are using patching, we extract a random patch from the image here
            if self._with_patching:
                x, offsets = self._graph_extract_patch(x)

            # Run the training on possibly multiple GPUs
            device_gradients = []
            device_variables = []
            for n, d in enumerate(self._get_device_list()):  # Build a graph on either a CPU or all of the GPUs
                with tf.device(d), tf.name_scope('tower_' + str(n)):
                    self._log('Graph: Creating layer parameters...')
                    self._add_layers_to_graph()

                    # Run the network operations
                    if self._has_moderation:
                        xx = self.forward_pass(x, deterministic=False, moderation_features=mod_w)
                    else:
                        xx = self.forward_pass(x, deterministic=False)

                    # Define regularization cost
                    self._log('Graph: Calculating loss and gradients...')
                    if self._reg_coeff is not None:
                        l2_cost = tf.squeeze(tf.reduce_sum(
                            [layer.regularization_coefficient * tf.nn.l2_loss(layer.weights) for layer in self._layers
                             if isinstance(layer, layers.fullyConnectedLayer)]))
                    else:
                        l2_cost = 0.0

                    # Define cost function based on which one was selected via set_loss_function
                    if self._loss_fn == 'softmax cross entropy':
                        sf_logits = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=xx, labels=tf.argmax(y, 1))
                    self._graph_ops['cost'] = tf.add(tf.reduce_mean(tf.concat([sf_logits], axis=0)), l2_cost)

                    # For classification problems, we will compute the training accuracy as well; this is also used
                    # for Tensorboard
                    self.__class_predictions = tf.argmax(tf.nn.softmax(xx), 1)
                    correct_predictions = tf.equal(self.__class_predictions, tf.argmax(y, 1))
                    self._graph_ops['accuracy'] = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

                    # Set the optimizer and get the gradients from it
                    gradients, variables, global_grad_norm = self._graph_get_gradients(self._graph_ops['cost'])
                    device_gradients.append(gradients)
                    device_variables.append(variables)

            average_gradients = self._graph_average_gradients(device_gradients)
            opt_variables = device_variables[0]
            self._graph_ops['optimizer'] = self._graph_apply_gradients(average_gradients, opt_variables)

            # Calculate test and validation accuracy
            if self._has_moderation:
                if self._testing:
                    x_test, self._graph_ops['y_test'], mod_w_test = tf.train.batch(
                        [self._test_images, self._test_labels, self._test_moderation_features],
                        batch_size=self._batch_size,
                        num_threads=self._num_threads,
                        capacity=self._queue_capacity)
                if self._validation:
                    x_val, self._graph_ops['y_val'], mod_w_val = tf.train.batch(
                        [self._val_images, self._val_labels, self._val_moderation_features],
                        batch_size=self._batch_size,
                        num_threads=self._num_threads,
                        capacity=self._queue_capacity)
            else:
                if self._testing:
                    x_test, self._graph_ops['y_test'] = tf.train.batch([self._test_images, self._test_labels],
                                                                       batch_size=self._batch_size,
                                                                       num_threads=self._num_threads,
                                                                       capacity=self._queue_capacity)
                if self._validation:
                    x_val, self._graph_ops['y_val'] = tf.train.batch([self._val_images, self._val_labels],
                                                                     batch_size=self._batch_size,
                                                                     num_threads=self._num_threads,
                                                                     capacity=self._queue_capacity)
            if self._testing:
                x_test = tf.reshape(x_test,
                                    shape=[-1, self._image_height, self._image_width, self._image_depth])
            if self._validation:
                x_val = tf.reshape(x_val,
                                   shape=[-1, self._image_height, self._image_width, self._image_depth])

            # If using patching, we need to properly pull similar patches from the test and validation images
            if self._with_patching:
                if self._testing:
                    x_test, _ = self._graph_extract_patch(x_test, offsets)
                if self._validation:
                    x_val, _ = self._graph_extract_patch(x_val, offsets)

            # Run the testing and validation, whose graph should only be on 1 device
            if self._has_moderation:
                if self._testing:
                    self._graph_ops['x_test_predicted'] = self.forward_pass(x_test, deterministic=True,
                                                                            moderation_features=mod_w_test)
                if self._validation:
                    self._graph_ops['x_val_predicted'] = self.forward_pass(x_val, deterministic=True,
                                                                           moderation_features=mod_w_val)
            else:
                if self._testing:
                    self._graph_ops['x_test_predicted'] = self.forward_pass(x_test, deterministic=True)
                if self._validation:
                    self._graph_ops['x_val_predicted'] = self.forward_pass(x_val, deterministic=True)

            # Compute the loss and accuracy for testing and validation
            if self._testing:
                test_class_predictions = tf.argmax(tf.nn.softmax(self._graph_ops['x_test_predicted']), 1)
                test_correct_predictions = tf.equal(test_class_predictions,
                                                    tf.argmax(self._graph_ops['y_test'], 1))
                self._graph_ops['test_losses'] = test_correct_predictions
                self._graph_ops['test_accuracy'] = tf.reduce_mean(tf.cast(test_correct_predictions, tf.float32))
            if self._validation:
                self.__val_class_predictions = tf.argmax(tf.nn.softmax(self._graph_ops['x_val_predicted']), 1)
                val_correct_predictions = tf.equal(self.__val_class_predictions,
                                                   tf.argmax(self._graph_ops['y_val'], 1))
                self._graph_ops['val_losses'] = val_correct_predictions
                self._graph_ops['val_accuracy'] = tf.reduce_mean(tf.cast(val_correct_predictions, tf.float32))

            # Epoch summaries for Tensorboard
            if self._tb_dir is not None:
                self._graph_tensorboard_summary(l2_cost, average_gradients, opt_variables, global_grad_norm)

    def _training_batch_results(self, batch_num, start_time, tqdm_range, train_writer=None):
        elapsed = time.time() - start_time

        if train_writer is not None:
            summary = self._session.run(self._graph_ops['merged'])
            train_writer.add_summary(summary, batch_num)

        if self._validation:
            loss, epoch_accuracy, epoch_val_accuracy = self._session.run([self._graph_ops['cost'],
                                                                          self._graph_ops['accuracy'],
                                                                          self._graph_ops['val_accuracy']])
            samples_per_sec = self._batch_size / elapsed

            desc_str = "{}: Results for batch {} (epoch {:.1f}) " + \
                       "- Loss: {:.5f}, Training Accuracy: {:.4f}, Validation Accuracy: {:.4f}, samples/sec: {:.2f}"
            tqdm_range.set_description(
                desc_str.format(datetime.datetime.now().strftime("%I:%M%p"),
                                batch_num,
                                batch_num / (self._total_training_samples / self._batch_size),
                                loss,
                                epoch_accuracy,
                                epoch_val_accuracy,
                                samples_per_sec))
        else:
            loss, epoch_accuracy = self._session.run([self._graph_ops['cost'],
                                                      self._graph_ops['accuracy']])
            samples_per_sec = self._batch_size / elapsed

            desc_str = "{}: Results for batch {} (epoch {:.1f}) " + \
                       "- Loss: {:.5f}, Training Accuracy: {:.4f}, samples/sec: {:.2f}"
            tqdm_range.set_description(
                desc_str.format(datetime.datetime.now().strftime("%I:%M%p"),
                                batch_num,
                                batch_num / (self._total_training_samples / self._batch_size),
                                loss,
                                epoch_accuracy,
                                samples_per_sec))

    def compute_full_test_accuracy(self):
        self._log('Computing total test accuracy/regression loss...')

        with self._graph.as_default():
            num_batches = int(np.ceil(self._total_testing_samples / self._batch_size))

            if num_batches == 0:
                warnings.warn('Less than a batch of testing data')
                exit()

            # Initialize storage for the retrieved test variables
            loss_sum = 0.0

            # Main test loop
            for _ in tqdm(range(num_batches)):
                batch_mean = self._session.run([self._graph_ops['test_losses']])
                loss_sum = loss_sum + np.mean(batch_mean)

            # For classification problems (assumed to be multi-class), we want accuracy and confusion matrix (not
            # implemented)
            mean = (loss_sum / num_batches)
            self._log('Average test accuracy: {:.5f}'.format(mean))
            return 1.0 - mean.astype(np.float32)

    def forward_pass_with_file_inputs(self, x):
        with self._graph.as_default():
            total_outputs = np.empty([1, self._last_layer().output_size])

            num_batches = len(x) // self._batch_size
            remainder = len(x) % self._batch_size

            if remainder != 0:
                num_batches += 1
                remainder = self._batch_size - remainder

            # self.load_images_from_list(x) no longer calls following 2 lines so we needed to force them here
            images = x
            self._parse_images(images)

            x_test = tf.train.batch([self._all_images], batch_size=self._batch_size, num_threads=self._num_threads)
            x_test = tf.reshape(x_test, shape=[-1, self._image_height, self._image_width, self._image_depth])

            if self._load_from_saved:
                self.load_state()
            self._initialize_queue_runners()
            # Run model on them
            x_pred = self.forward_pass(x_test, deterministic=True)

            for i in range(int(num_batches)):
                xx = self._session.run(x_pred)
                for img in np.array_split(xx, self._batch_size):
                    total_outputs = np.append(total_outputs, img, axis=0)

            # delete weird first row
            total_outputs = np.delete(total_outputs, 0, 0)

            # delete any outputs which are overruns from the last batch
            if remainder != 0:
                for i in range(remainder):
                    total_outputs = np.delete(total_outputs, -1, 0)

        return total_outputs

    def forward_pass_with_interpreted_outputs(self, x):
        # Perform forward pass of the network to get raw outputs and apply a softmax
        xx = self.forward_pass_with_file_inputs(x)
        interpreted_outputs = np.exp(xx) / np.sum(np.exp(xx), axis=1, keepdims=True)
        return interpreted_outputs

    def add_output_layer(self, regularization_coefficient=None, output_size=None):
        if len(self._layers) < 1:
            raise RuntimeError("An output layer cannot be the first layer added to the model. " +
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

        self._log('Adding output layer...')

        reshape = self._last_layer_outputs_volume()

        if regularization_coefficient is None and self._reg_coeff is not None:
            regularization_coefficient = self._reg_coeff
        if regularization_coefficient is None and self._reg_coeff is None:
            regularization_coefficient = 0.0

        if output_size is None:
            num_out = self._total_classes
        else:
            num_out = output_size

        with self._graph.as_default():
            layer = layers.fullyConnectedLayer('output',
                                               copy.deepcopy(self._last_layer().output_size),
                                               num_out,
                                               reshape,
                                               self._batch_size,
                                               None,
                                               self._weight_initializer,
                                               regularization_coefficient)

        self._log('Inputs: {0} Outputs: {1}'.format(layer.input_size, layer.output_size))
        self._layers.append(layer)

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
            current_labels[self._total_classes, :] = 1
            labels = np.hstack([labels, current_labels]) if labels.size else current_labels
            self._total_classes += 1

        labels = tf.transpose(labels)

        self._total_raw_samples = len(image_files)

        self._log('Total raw examples is %d' % self._total_raw_samples)
        self._log('Total classes is %d' % self._total_classes)
        self._log('Parsing dataset...')

        self._raw_image_files = image_files
        self._raw_labels = labels

    def load_ippn_dataset_from_directory(self, dirname, column='strain'):
        """Loads the RGB images and species labels from the International Plant Phenotyping Network dataset."""

        labels = []
        ids = []
        if column == 'treatment':
            labels, ids = loaders.read_csv_labels_and_ids(os.path.join(dirname, 'Metadata.csv'), 2, 0)
        elif column == 'strain':
            labels, ids = loaders.read_csv_labels_and_ids(os.path.join(dirname, 'Metadata.csv'), 1, 0)
        elif column == 'DAG':
            labels, ids = loaders.read_csv_labels_and_ids(os.path.join(dirname, 'Metadata.csv'), 3, 0)
        else:
            warnings.warn('Unknown column in IPPN dataset')
            exit()

        image_files = [os.path.join(dirname, im_id + '_rgb.png') for im_id in ids]

        self._total_raw_samples = len(image_files)

        self._total_classes = len(set(labels))

        # transform into numerical one-hot labels
        with self._graph.as_default():
            labels = loaders.string_labels_to_sequential(labels)
            labels = tf.one_hot(labels, self._total_classes)

        self._log('Total classes is %d' % self._total_classes)
        self._log('Total raw examples is %d' % self._total_raw_samples)

        self._raw_image_files = image_files
        self._raw_labels = labels
