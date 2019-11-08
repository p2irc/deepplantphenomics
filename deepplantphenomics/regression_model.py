from . import layers, loaders, definitions, DPPModel
import numpy as np
import tensorflow.compat.v1 as tf
import os
import warnings
import copy
from tqdm import tqdm


class RegressionModel(DPPModel):
    _supported_loss_fns = ['l2', 'l1', 'smooth l1']
    _supported_augmentations = [definitions.AugmentationType.FLIP_HOR,
                                definitions.AugmentationType.FLIP_VER,
                                definitions.AugmentationType.CROP,
                                definitions.AugmentationType.CONTRAST_BRIGHT,
                                definitions.AugmentationType.ROTATE]

    def __init__(self, debug=False, load_from_saved=False, save_checkpoints=True, initialize=True, tensorboard_dir=None,
                 report_rate=100, save_dir=None):
        super().__init__(debug, load_from_saved, save_checkpoints, initialize, tensorboard_dir, report_rate, save_dir)
        self._loss_fn = 'l2'
        self._num_regression_outputs = 1

        # State variables specific to regression for constructing the graph and passing to Tensorboard
        self._regression_loss = None

    def set_num_regression_outputs(self, num):
        """Set the number of regression response variables"""
        if not isinstance(num, int):
            raise TypeError("num must be an int")
        if num <= 0:
            raise ValueError("num must be positive")

        self._num_regression_outputs = num

    def _graph_tensorboard_summary(self, l2_cost, gradients, variables, global_grad_norm):
        super()._graph_tensorboard_common_summary(l2_cost, gradients, variables, global_grad_norm)

        # Summaries specific to regression problems
        if self._num_regression_outputs == 1:
            tf.summary.scalar('train/regression_loss', self._regression_loss, collections=['custom_summaries'])
            if self._validation:
                tf.summary.scalar('validation/loss', self._graph_ops['val_cost'],
                                  collections=['custom_summaries'])
                tf.summary.histogram('validation/batch_losses', self._graph_ops['val_losses'],
                                     collections=['custom_summaries'])

        self._graph_ops['merged'] = tf.summary.merge_all(key='custom_summaries')

    def _assemble_graph(self):
        with self._graph.as_default():
            self._log('Assembling graph...')

            self._log('Graph: Parsing dataset...')
            with tf.device('device:cpu:0'):  # Only do preprocessing on the CPU to limit data transfer between devices
                # Generate training, testing, and validation datasets
                self._graph_parse_data()

                # For regression, we need to also deserialize the labels before batching the datasets
                def _deserialize_label(im, lab):
                    lab = tf.cond(tf.equal(tf.rank(lab), 0),
                                  lambda: tf.reshape(lab, [1]),
                                  lambda: lab)
                    sparse_lab = tf.string_split(lab, sep=' ')
                    lab_values = tf.strings.to_number(sparse_lab.values)
                    lab = tf.reshape(lab_values, [self._num_regression_outputs])
                    return im, lab

                # Batch the datasets and create iterators for them
                self._train_dataset = self._train_dataset.map(_deserialize_label, num_parallel_calls=self._num_threads)
                train_iter = self._batch_and_iterate(self._train_dataset, shuffle=True)
                if self._testing:
                    self._test_dataset = self._test_dataset.map(_deserialize_label,
                                                                num_parallel_calls=self._num_threads)
                    test_iter = self._batch_and_iterate(self._test_dataset)
                if self._validation:
                    self._val_dataset = self._val_dataset.map(_deserialize_label, num_parallel_calls=self._num_threads)
                    val_iter = self._batch_and_iterate(self._val_dataset)

                if self._has_moderation:
                    train_mod_iter = self._batch_and_iterate(self._train_moderation_features)
                    if self._testing:
                        test_mod_iter = self._batch_and_iterate(self._test_moderation_features)
                    if self._validation:
                        val_mod_iter = self._batch_and_iterate(self._val_moderation_features)

                # # If we are using patching, we extract a random patch from the image here
                # if self._with_patching:
                #     x, offsets = self._graph_extract_patch(x)

            # Create an optimizer object for all of the devices
            optimizer = self._graph_make_optimizer()

            # Set up the graph layers
            self._log('Graph: Creating layer parameters...')
            self._add_layers_to_graph()

            # Do the forward pass and training output calcs on possibly multiple GPUs
            device_costs = []
            device_gradients = []
            device_variables = []
            for n, d in enumerate(self._get_device_list()):  # Build a graph on either the CPU or all of the GPUs
                with tf.device(d), tf.name_scope('tower_' + str(n)):
                    x, y = train_iter.get_next()

                    # Run the network operations
                    if self._has_moderation:
                        mod_w = train_mod_iter.get_next()
                        xx = self.forward_pass(x, deterministic=False, moderation_features=mod_w)
                    else:
                        xx = self.forward_pass(x, deterministic=False)

                    # Define regularization cost
                    self._log('Graph: Calculating loss and gradients...')
                    l2_cost = self._graph_layer_loss()

                    # Define the cost function
                    pred_loss = self._graph_problem_loss(xx, y)
                    gpu_cost = tf.reduce_mean(pred_loss) + l2_cost
                    cost_sum = tf.reduce_sum(pred_loss)
                    device_costs.append(cost_sum)

                    # Set the optimizer and get the gradients from it
                    gradients, variables, global_grad_norm = self._graph_get_gradients(gpu_cost, optimizer)
                    device_gradients.append(gradients)
                    device_variables.append(variables)

            # Average the gradients from each GPU and apply them
            average_gradients = self._graph_average_gradients(device_gradients)
            opt_variables = device_variables[0]
            self._graph_ops['optimizer'] = self._graph_apply_gradients(average_gradients, opt_variables, optimizer)

            # Average the costs and accuracies from each GPU
            self._regression_loss = tf.reduce_sum(device_costs) / self._batch_size
            self._graph_ops['cost'] = self._regression_loss + l2_cost

            # Calculate test and validation accuracy (on a single device at Tensorflow's discretion)
            # # If using patching, we need to properly pull similar patches from the test and validation images
            # if self._with_patching:
            #     if self._testing:
            #         x_test, _ = self._graph_extract_patch(x_test, offsets)
            #     if self._validation:
            #         x_val, _ = self._graph_extract_patch(x_val, offsets)
            if self._testing:
                x_test, self._graph_ops['y_test'] = test_iter.get_next()

                if self._has_moderation:
                    mod_w_test = test_mod_iter.get_next()
                    self._graph_ops['x_test_predicted'] = self.forward_pass(x_test, deterministic=True,
                                                                            moderation_features=mod_w_test)
                else:
                    self._graph_ops['x_test_predicted'] = self.forward_pass(x_test, deterministic=True)

                if self._num_regression_outputs == 1:
                    # For 1 output, taking a norm does nothing, so skip it; the loss is just the difference
                    self._graph_ops['test_losses'] = tf.squeeze(
                        self._graph_ops['x_test_predicted'] - self._graph_ops['y_test'], axis=1)
                else:
                    self._graph_ops['test_losses'] = self._graph_problem_loss(self._graph_ops['x_test_predicted'],
                                                                              self._graph_ops['y_test'])

            if self._validation:
                x_val, self._graph_ops['y_val'] = val_iter.get_next()

                if self._has_moderation:
                    mod_w_val = val_mod_iter.get_next()
                    self._graph_ops['x_val_predicted'] = self.forward_pass(x_val, deterministic=True,
                                                                           moderation_features=mod_w_val)
                else:
                    self._graph_ops['x_val_predicted'] = self.forward_pass(x_val, deterministic=True)

                if self._num_regression_outputs == 1:
                    # For 1 output, taking a norm does nothing, so skip it; the loss is just the difference
                    self._graph_ops['val_losses'] = tf.squeeze(
                        self._graph_ops['x_val_predicted'] - self._graph_ops['y_val'], axis=1)
                else:
                    self._graph_ops['val_losses'] = self._graph_problem_loss(self._graph_ops['x_val_predicted'],
                                                                             self._graph_ops['y_val'])
                self._graph_ops['val_cost'] = tf.reduce_mean(tf.abs(self._graph_ops['val_losses']))

            # Epoch summaries for Tensorboard
            if self._tb_dir is not None:
                self._graph_tensorboard_summary(l2_cost, gradients, variables, global_grad_norm)

    def _graph_problem_loss(self, pred, lab):
        val_diffs = pred - lab
        if self._loss_fn == 'l2':
            return self.__l2_loss(val_diffs)
        elif self._loss_fn == 'l1':
            return self.__l1_loss(val_diffs)
        elif self._loss_fn == 'smooth l1':
            return self.__smooth_l1_loss(val_diffs)

        raise RuntimeError("Could not calculate problem loss for a loss function of " + self._loss_fn)

    def __l2_loss(self, x):
        """
        Calculates the L2 loss of prediction difference Tensors for each item in a batch
        :param x: A Tensor with prediction differences for each item in a batch
        :return: A Tensor with the scalar L2 loss for each item
        """
        y = tf.map_fn(lambda ex: tf.reduce_sum(ex ** 2), x)
        return y

    def __l1_loss(self, x):
        """
        Calculates the L1 loss of prediction difference Tensors for each item in a batch
        :param x: A Tensor with prediction differences for each item in a batch
        :return: A Tensor with the scalar L1 loss for each item
        """
        y = tf.map_fn(lambda ex: tf.reduce_sum(tf.abs(ex)), x)
        return y

    def __smooth_l1_loss(self, x, huber_delta=1):
        """
        Calculates the smooth-L1 loss of prediction difference Tensors for each item in a batch. This amounts to
        evaluating the Huber loss of each individual value and taking the sum.
        :param x: A Tensor with prediction differences for each item in a batch
        :param huber_delta: A parameter for calculating the Huber loss; roughly corresponds to the value where the Huber
        loss transitions from quadratic growth to linear growth
        :return: A Tensor with the scalar smooth-L1 loss for each item
        """
        x = tf.abs(x)
        y = tf.map_fn(lambda ex: tf.reduce_sum(tf.where(ex < huber_delta,
                                                        0.5 * ex ** 2,
                                                        huber_delta * (ex - 0.5 * huber_delta))), x)
        return y

    def compute_full_test_accuracy(self):
        self._log('Computing total test accuracy/regression loss...')

        with self._graph.as_default():
            num_batches = int(np.ceil(self._total_testing_samples / self._batch_size))

            if num_batches == 0:
                warnings.warn('Less than a batch of testing data')
                exit()

            all_losses = []
            all_y = []
            all_predictions = []

            # Main test loop
            for _ in tqdm(range(num_batches)):
                r_losses, r_y, r_predicted = self._session.run([self._graph_ops['test_losses'],
                                                                self._graph_ops['y_test'],
                                                                self._graph_ops['x_test_predicted']])
                all_losses.append(r_losses)
                all_y.append(r_y)
                all_predictions.append(r_predicted)

            all_losses = np.concatenate(all_losses, axis=0)
            all_y = np.concatenate(all_y, axis=0)
            all_predictions = np.concatenate(all_predictions, axis=0)

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

    def forward_pass_with_file_inputs(self, images):
        with self._graph.as_default():
            num_batches = len(images) // self._batch_size
            if len(images) % self._batch_size != 0:
                num_batches += 1

            self._parse_images(images)
            im_data = self._all_images.batch(self._batch_size).prefetch(1)
            x_test = im_data.make_one_shot_iterator().get_next()

            if self._load_from_saved:
                self.load_state()

            # Run model on them
            x_pred = self.forward_pass(x_test, deterministic=True)

            total_outputs = []
            for i in range(int(num_batches)):
                xx = self._session.run(x_pred)
                for img in np.array_split(xx, xx.shape[0]):
                    total_outputs.append(img)

            total_outputs = np.concatenate(total_outputs, axis=0)

        return total_outputs

    def forward_pass_with_interpreted_outputs(self, x):
        # Nothing special required for regression
        interpreted_outputs = self.forward_pass_with_file_inputs(x)
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
            num_out = self._num_regression_outputs
        else:
            num_out = output_size

        with self._graph.as_default():
            layer = layers.fullyConnectedLayer('output', copy.deepcopy(self._last_layer().output_size), num_out,
                                               reshape, None, self._weight_initializer, regularization_coefficient)

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
