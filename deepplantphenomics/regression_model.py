from . import layers, loaders, definitions, DPPModel
import numpy as np
import tensorflow as tf
import os
import warnings
import copy
from tqdm import tqdm


class RegressionModel(DPPModel):
    _problem_type = definitions.ProblemType.REGRESSION
    _loss_fn = 'l2'
    _supported_loss_fns = ['l2', 'l1', 'smooth l1', 'log loss']
    _supported_augmentations = [definitions.AugmentationType.FLIP_HOR,
                                definitions.AugmentationType.FLIP_VER,
                                definitions.AugmentationType.CROP,
                                definitions.AugmentationType.CONTRAST_BRIGHT,
                                definitions.AugmentationType.ROTATE]
    _num_regression_outputs = 1

    # State variables specific to regression for constructing the graph and passing to Tensorboard
    _regression_loss = None

    def __init__(self, debug=False, load_from_saved=False, save_checkpoints=True, initialize=True, tensorboard_dir=None,
                 report_rate=100, save_dir=None):
        super().__init__(debug, load_from_saved, save_checkpoints, initialize, tensorboard_dir, report_rate, save_dir)

    def set_num_regression_outputs(self, num):
        """Set the number of regression response variables"""
        if not isinstance(num, int):
            raise TypeError("num must be an int")
        if num <= 0:
            raise ValueError("num must be positive")

        self._num_regression_outputs = num

    def _graph_tensorboard_summary(self, l2_cost, gradients, variables, global_grad_norm):
        super()._graph_tensorboard_summary(l2_cost, gradients, variables, global_grad_norm)

        # Summaries specific to regression problems
        if self._num_regression_outputs == 1:
            tf.summary.scalar('train/regression_loss', self._regression_loss, collections=['custom_summaries'])
            if self._validation:
                tf.summary.scalar('validation/loss', self._graph_ops['val_cost'],
                                  collections=['custom_summaries'])
                tf.summary.histogram('validation/batch_losses', self._graph_ops['val_losses'],
                                     collections=['custom_summaries'])

    def _assemble_graph(self):
        with self._graph.as_default():

            self._log('Parsing dataset...')
            self._graph_parse_data()

            self._log('Creating layer parameters...')
            self._add_layers_to_graph()

            self._log('Assembling graph...')

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

            # This is a regression problem, so we should deserialize the label
            y = loaders.label_string_to_tensor(y, self._batch_size, self._num_regression_outputs)

            # If we are using patching, we extract a random patch from the image here
            if self._with_patching:
                x, offsets = self._graph_extract_patch(x)

            # Run the network operations
            if self._has_moderation:
                xx = self.forward_pass(x, deterministic=False, moderation_features=mod_w)
            else:
                xx = self.forward_pass(x, deterministic=False)

            # Define regularization cost
            if self._reg_coeff is not None:
                l2_cost = tf.squeeze(tf.reduce_sum(
                    [layer.regularization_coefficient * tf.nn.l2_loss(layer.weights) for layer in self._layers
                     if isinstance(layer, layers.fullyConnectedLayer)]))
            else:
                l2_cost = 0.0

            # Define cost function based on which one was selected via set_loss_function
            if self._loss_fn == 'l2':
                self._regression_loss = self.__batch_mean_l2_loss(tf.subtract(xx, y))
            elif self._loss_fn == 'l1':
                self._regression_loss = self.__batch_mean_l1_loss(tf.subtract(xx, y))
            elif self._loss_fn == 'smooth l1':
                self._regression_loss = self.__batch_mean_smooth_l1_loss(tf.subtract(xx, y))
            elif self._loss_fn == 'log loss':
                self._regression_loss = self.__batch_mean_log_loss(tf.subtract(xx, y))
            self._graph_ops['cost'] = tf.add(self._regression_loss, l2_cost)

            # Set the optimizer and get the gradients from it
            gradients, variables, global_grad_norm = self._graph_add_optimizer()

            # Calculate test accuracy
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
                x_test = tf.reshape(x_test, shape=[-1, self._image_height, self._image_width, self._image_depth])
            if self._validation:
                x_val = tf.reshape(x_val, shape=[-1, self._image_height, self._image_width, self._image_depth])

            if self._testing:
                self._graph_ops['y_test'] = loaders.label_string_to_tensor(self._graph_ops['y_test'],
                                                                           self._batch_size,
                                                                           self._num_regression_outputs)
            if self._validation:
                self._graph_ops['y_val'] = loaders.label_string_to_tensor(self._graph_ops['y_val'],
                                                                          self._batch_size,
                                                                          self._num_regression_outputs)

            # If using patching, we need to properly pull similar patches from the test and validation images
            if self._with_patching:
                if self._testing:
                    x_test, _ = self._graph_extract_patch(x_test, offsets)
                if self._validation:
                    x_val, _ = self._graph_extract_patch(x_val, offsets)

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

            # compute the loss and accuracy based on problem type
            if self._testing:
                if self._num_regression_outputs == 1:
                    self._graph_ops['test_losses'] = tf.squeeze(tf.stack(
                        tf.subtract(self._graph_ops['x_test_predicted'], self._graph_ops['y_test'])))
                else:
                    self._graph_ops['test_losses'] = self.__l2_norm(
                        tf.subtract(self._graph_ops['x_test_predicted'], self._graph_ops['y_test']))
            if self._validation:
                if self._num_regression_outputs == 1:
                    self._graph_ops['val_losses'] = tf.squeeze(
                        tf.stack(tf.subtract(self._graph_ops['x_val_predicted'], self._graph_ops['y_val'])))
                else:
                    self._graph_ops['val_losses'] = self.__l2_norm(
                        tf.subtract(self._graph_ops['x_val_predicted'], self._graph_ops['y_val']))
                self._graph_ops['val_cost'] = tf.reduce_mean(tf.abs(self._graph_ops['val_losses']))

            # Epoch summaries for Tensorboard
            self._graph_tensorboard_summary(l2_cost, gradients, variables, global_grad_norm)

    def compute_full_test_accuracy(self):
        self._log('Computing total test accuracy/regression loss...')

        with self._graph.as_default():
            num_batches = int(np.ceil(self._total_testing_samples / self._batch_size))

            if num_batches == 0:
                warnings.warn('Less than a batch of testing data')
                exit()

            all_losses = np.empty(shape=1)
            all_y = np.empty(shape=1)
            all_predictions = np.empty(shape=1)

            # Main test loop
            for _ in tqdm(range(num_batches)):
                r_losses, r_y, r_predicted = self._session.run([self._graph_ops['test_losses'],
                                                                self._graph_ops['y_test'],
                                                                self._graph_ops['x_test_predicted']])
                all_losses = np.concatenate((all_losses, r_losses), axis=0)
                all_y = np.concatenate((all_y, np.squeeze(r_y)), axis=0)
                all_predictions = np.concatenate((all_predictions, np.squeeze(r_predicted)), axis=0)

            all_losses = np.delete(all_losses, 0)
            all_y = np.delete(all_y, 0)
            all_predictions = np.delete(all_predictions, 0)

            # Delete the extra entries (e.g. batch_size is 4 and 1 sample left, it will loop and have 3 repeats that
            # we want to get rid of)
            extra = self._batch_size - (self._total_testing_samples % self._batch_size)
            if extra != self._batch_size:
                mask_extra = np.ones(self._batch_size * num_batches, dtype=bool)
                mask_extra[range(self._batch_size * num_batches - extra, self._batch_size * num_batches)] = False
                all_losses = all_losses[mask_extra, ...]
                all_y = all_y[mask_extra, ...]
                all_predictions = all_predictions[mask_extra, ...]

            # For regression problems we want relative and abs mean, std of L2 norms, plus a histogram of errors
            abs_mean = np.mean(np.abs(all_losses))
            abs_var = np.var(np.abs(all_losses))
            abs_std = np.sqrt(abs_var)

            mean = np.mean(all_losses)
            var = np.var(all_losses)
            mse = np.mean(np.square(all_losses))
            std = np.sqrt(var)
            loss_max = np.amax(all_losses)
            loss_min = np.amin(all_losses)

            hist, _ = np.histogram(all_losses, bins=100)

            self._log('Mean loss: {}'.format(mean))
            self._log('Loss standard deviation: {}'.format(std))
            self._log('Mean absolute loss: {}'.format(abs_mean))
            self._log('Absolute loss standard deviation: {}'.format(abs_std))
            self._log('Min error: {}'.format(loss_min))
            self._log('Max error: {}'.format(loss_max))
            self._log('MSE: {}'.format(mse))

            all_y_mean = np.mean(all_y)
            total_error = np.sum(np.square(all_y - all_y_mean))
            unexplained_error = np.sum(np.square(all_losses))
            # division by zero can happen when using small test sets
            if total_error == 0:
                r2 = -np.inf
            else:
                r2 = 1. - (unexplained_error / total_error)

            self._log('R^2: {}'.format(r2))
            self._log('All test labels:')
            self._log(all_y)

            self._log('All predictions:')
            self._log(all_predictions)

            self._log('Histogram of {} losses:'.format(self._loss_fn))
            self._log(hist)

            return abs_mean.astype(np.float32)

    def forward_pass_with_file_inputs(self, x):
        with self._graph.as_default():
            total_outputs = np.empty([1, self._num_regression_outputs])

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
        # Nothing special required for regression
        interpreted_outputs = self.forward_pass_with_file_inputs(x)
        return interpreted_outputs

    def __batch_mean_l2_loss(self, x):
        """Given a batch of vectors, calculates the mean per-vector L2 norm"""
        with self._graph.as_default():
            agg = self.__l2_norm(x)
            mean = tf.reduce_mean(agg)

        return mean

    def __l2_norm(self, x):
        """Returns the L2 norm of a tensor"""
        with self._graph.as_default():
            y = tf.map_fn(lambda ex: tf.norm(ex, ord=2), x)

        return y

    def __batch_mean_l1_loss(self, x):
        """Given a batch of vectors, calculates the mean per-vector L1 norm"""
        with self._graph.as_default():
            agg = self.__l1_norm(x)
            mean = tf.reduce_mean(agg)

        return mean

    def __l1_norm(self, x):
        """Returns the L1 norm of a tensor"""
        with self._graph.as_default():
            y = tf.map_fn(lambda ex: tf.norm(ex, ord=1), x)

        return y

    def __batch_mean_smooth_l1_loss(self, x):
        """Given a batch of vectors, calculates the mean per-vector smooth L1 norm"""
        with self._graph.as_default():
            agg = self.__smooth_l1_norm(x)
            mean = tf.reduce_mean(agg)

        return mean

    def __smooth_l1_norm(self, x):
        """Returns the smooth L1 norm of a tensor"""
        huber_delta = 1  # may want to make this a tunable hyper parameter in future
        with self._graph.as_default():
            x = tf.abs(x)
            y = tf.map_fn(lambda ex: tf.where(ex < huber_delta,
                                              0.5*ex**2,
                                              huber_delta*(ex-0.5*huber_delta)), x)

        return y

    def __batch_mean_log_loss(self, x):
        """Given a batch of vectors, calculates the mean per-vector log loss"""
        with self._graph.as_default():
            x = tf.abs(x)
            x = tf.clip_by_value(x, 0, 0.9999999)
            agg = -tf.log(1-x)
            mean = tf.reduce_mean(agg)

        return mean

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
            num_out = self._num_regression_outputs
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

        labels = [[label] for label in labels]

        self._log('Total raw examples is %d' % self._total_raw_samples)

        self._raw_image_files = image_files
        self._raw_labels = labels
