from . import layers, loaders, definitions, DPPModel
import numpy as np
import tensorflow.compat.v1 as tf
import os
import datetime
import time
import warnings
import copy
from tqdm import tqdm


class ClassificationModel(DPPModel):
    _supported_loss_fns = ['softmax cross entropy']
    _supported_augmentations = [definitions.AugmentationType.FLIP_HOR,
                                definitions.AugmentationType.FLIP_VER,
                                definitions.AugmentationType.CROP,
                                definitions.AugmentationType.CONTRAST_BRIGHT,
                                definitions.AugmentationType.ROTATE]

    def __init__(self, debug=False, load_from_saved=False, save_checkpoints=True, initialize=True, tensorboard_dir=None,
                 report_rate=100, save_dir=None):
        super().__init__(debug, load_from_saved, save_checkpoints, initialize, tensorboard_dir, report_rate, save_dir)
        self._loss_fn = 'softmax cross entropy'

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
            with tf.device("/device:cpu:0"):  # Only do preprocessing on the CPU to limit data transfer between devices
                # Generate training, testing, and validation datasets
                self._graph_parse_data()

                # Batch the datasets and create iterators for them
                train_iter = self._batch_and_iterate(self._train_dataset, shuffle=True)
                if self._testing:
                    test_iter = self._batch_and_iterate(self._test_dataset)
                if self._validation:
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
            device_accuracies = []
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

                    # Define the cost function, then get the cost for this device's sub-batch and any parts of the cost
                    # needed to get the overall batch's cost later
                    pred_loss = self._graph_problem_loss(xx, y)
                    gpu_cost = tf.reduce_mean(tf.concat([pred_loss], axis=0)) + l2_cost
                    cost_sum = tf.reduce_sum(tf.concat([pred_loss], axis=0))
                    device_costs.append(cost_sum)

                    # For classification, we need the training accuracy as well so we can report it in Tensorboard
                    self.__class_predictions, correct_predictions = self._graph_compare_predictions(xx, y)
                    accuracy_sum = tf.reduce_sum(tf.cast(correct_predictions, tf.float32))
                    device_accuracies.append(accuracy_sum)

                    # Set the optimizer and get the gradients from it
                    gradients, variables, global_grad_norm = self._graph_get_gradients(gpu_cost, optimizer)
                    device_gradients.append(gradients)
                    device_variables.append(variables)

            # Average the gradients from each GPU and apply them
            average_gradients = self._graph_average_gradients(device_gradients)
            opt_variables = device_variables[0]
            self._graph_ops['optimizer'] = self._graph_apply_gradients(average_gradients, opt_variables, optimizer)

            # Average the costs and accuracies from each GPU
            self._graph_ops['cost'] = tf.reduce_sum(device_costs) / self._batch_size + l2_cost
            self._graph_ops['accuracy'] = tf.reduce_sum(device_accuracies) / self._batch_size

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

                _, self._graph_ops['test_losses'] = self._graph_compare_predictions(self._graph_ops['x_test_predicted'],
                                                                                    self._graph_ops['y_test'])
                self._graph_ops['test_accuracy'] = tf.reduce_mean(tf.cast(self._graph_ops['test_losses'], tf.float32))

            if self._validation:
                x_val, self._graph_ops['y_val'] = val_iter.get_next()

                if self._has_moderation:
                    mod_w_val = val_mod_iter.get_next()
                    self._graph_ops['x_val_predicted'] = self.forward_pass(x_val, deterministic=True,
                                                                           moderation_features=mod_w_val)
                else:
                    self._graph_ops['x_val_predicted'] = self.forward_pass(x_val, deterministic=True)

                _, self._graph_ops['val_losses'] = self._graph_compare_predictions(self._graph_ops['x_val_predicted'],
                                                                                   self._graph_ops['y_val'])
                self._graph_ops['val_accuracy'] = tf.reduce_mean(tf.cast(self._graph_ops['val_losses'], tf.float32))

            # Epoch summaries for Tensorboard
            if self._tb_dir is not None:
                self._graph_tensorboard_summary(l2_cost, average_gradients, opt_variables, global_grad_norm)

    def _graph_problem_loss(self, pred, lab):
        if self._loss_fn == 'softmax cross entropy':
            lab_idx = tf.argmax(lab, axis=1)
            return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=lab_idx)

        raise RuntimeError("Could not calculate problem loss for a loss function of " + self._loss_fn)

    def _graph_compare_predictions(self, pred, lab):
        """
        Compares the prediction and label classification for each item in a batch, returning
        :param pred: Model class predictions for the batch; no softmax should be applied to it yet
        :param lab: Labels for the correct class, with the same shape as pred
        :return: 2 Tensors: one with the simplified class predictions (i.e. as a single number), and one with integer
        flags (i.e. 1's and 0's) for whether predictions are correct
        """
        pred_idx = tf.argmax(tf.nn.softmax(pred), axis=1)
        lab_idx = tf.argmax(lab, axis=1)
        is_correct = tf.equal(pred_idx, lab_idx)
        return pred_idx, is_correct

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
            layer = layers.fullyConnectedLayer('output', copy.deepcopy(self._last_layer().output_size), num_out,
                                               reshape, None, self._weight_initializer, regularization_coefficient)

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
