from . import layers, definitions, deepplantpheno
import numpy as np
import tensorflow as tf
import datetime
import time
import os
import warnings
from tqdm import tqdm
import pickle
from PIL import Image


class CountCeptionModel(deepplantpheno.DPPModel):
    _problem_type = definitions.ProblemType.OBJECT_COUNTING
    _loss_fn = 'l1'
    _supported_loss_fns = ['l1']
    _supported_augmentations = []

    def __init__(self, debug=False, load_from_saved=False, save_checkpoints=True, initialize=True, tensorboard_dir=None,
                 report_rate=100, save_dir=None):
        super().__init__(debug, load_from_saved, save_checkpoints, initialize, tensorboard_dir, report_rate, save_dir)

    def _graph_tensorboard_summary(self, l2_cost, gradients, variables, global_grad_norm):
        super()._graph_tensorboard_common_summary(l2_cost, gradients, variables, global_grad_norm)

        # Summaries specific to classification problems
        tf.summary.scalar('train/accuracy', self._graph_ops['accuracy'], collections=['custom_summaries'])
        if self._validation:
            tf.summary.scalar('validation/loss', self._graph_ops['val_losses'],
                              collections=['custom_summaries'])
            tf.summary.scalar('validation/accuracy', self._graph_ops['val_accuracy'],
                              collections=['custom_summaries'])

        self._graph_ops['merged'] = tf.summary.merge_all(key='custom_summaries')

    def _assemble_graph(self):
        with self._graph.as_default():
            self._log('Assembling graph...')

            self._log('Graph: Parsing dataset...')
            # Only do preprocessing on the CPU to limit data transfer between devices
            with tf.device('/device:cpu:0'):
                self._graph_parse_data()

                x, y = tf.train.shuffle_batch([self._train_images, self._train_labels],
                                              batch_size=self._batch_size,
                                              num_threads=self._num_threads,
                                              capacity=self._queue_capacity,
                                              min_after_dequeue=self._batch_size)

                # Reshape input to the expected image dimensions
                x = tf.reshape(x, shape=[-1, self._image_height, self._image_width, self._image_depth])

                # Split the current training batch into sub-batches if we are constructing more than 1 training tower
                x_sub_batches = tf.split(x, self._num_gpus, axis=0)
                y_sub_batches = tf.split(y, self._num_gpus, axis=0)

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
                    # Run the network operations
                    xx = self.forward_pass(x_sub_batches[n], deterministic=False)

                    # Define regularization cost
                    self._log('Graph: Calculating loss and gradients...')
                    if self._reg_coeff is not None:
                        l2_cost = tf.squeeze(tf.reduce_sum(
                            [layer.regularization_coefficient * tf.nn.l2_loss(layer.weights) for layer in self._layers
                             if isinstance(layer, layers.fullyConnectedLayer)]))
                    else:
                        l2_cost = 0.0

                    # Define cost function
                    if self._loss_fn == 'l1':
                        val_diff = tf.abs(tf.subtract(xx, y_sub_batches[n]))
                        gt = tf.reduce_sum(y, axis=[1, 2, 3]) / (32 ** 2.0)
                        pr = tf.reduce_sum(xx, axis=[1, 2, 3]) / (32 ** 2.0)
                        acc_diff = tf.abs(gt - pr)
                    gpu_cost = tf.squeeze(tf.reduce_mean(val_diff) + l2_cost)
                    device_costs.append(tf.reduce_sum(val_diff))
                    device_accuracies.append(tf.reduce_sum(acc_diff))

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

            # Calculate test and validation accuracy (on a single device at Tensorflow's discretion)
            if self._testing:
                x_test, self._graph_ops['y_test'] = tf.train.batch([self._test_images, self._test_labels],
                                                                   batch_size=self._batch_size,
                                                                   num_threads=self._num_threads,
                                                                   capacity=self._queue_capacity)
                x_test = tf.reshape(x_test, shape=[-1, self._image_height, self._image_width, self._image_depth])
            if self._validation:
                x_val, self._graph_ops['y_val'] = tf.train.batch([self._val_images, self._val_labels],
                                                                 batch_size=self._batch_size,
                                                                 num_threads=self._num_threads,
                                                                 capacity=self._queue_capacity)
                x_val = tf.reshape(x_val, shape=[-1, self._image_height, self._image_width, self._image_depth])

            # Run the testing and validation, whose graph should only be on 1 device
            if self._testing:
                self._graph_ops['x_test_predicted'] = self.forward_pass(x_test, deterministic=True)
            if self._validation:
                self._graph_ops['x_val_predicted'] = self.forward_pass(x_val, deterministic=True)

            # Compute the loss and accuracy for testing and validation
            if self._testing:
                if self._loss_fn == 'l1':
                    self._graph_ops['test_losses'] = tf.reduce_mean(tf.abs(tf.subtract(
                        self._graph_ops['y_test'], self._graph_ops['x_test_predicted'])))
                    gt_test = tf.reduce_sum(self._graph_ops['y_test'], axis=[1, 2, 3]) / (32 ** 2.0)
                    pr_test = tf.reduce_sum(self._graph_ops['x_test_predicted'], axis=[1, 2, 3]) / (32 ** 2.0)
                    self._graph_ops['gt_test'] = gt_test
                    self._graph_ops['pr_test'] = pr_test
                    self._graph_ops['test_accuracy'] = tf.reduce_mean(tf.abs(gt_test - pr_test))
            if self._validation:
                if self._loss_fn == 'l1':
                    self._graph_ops['val_losses'] = tf.reduce_mean(tf.abs(tf.subtract(
                        self._graph_ops['y_val'], self._graph_ops['x_val_predicted'])))
                    gt_val = tf.reduce_sum(self._graph_ops['y_val'], axis=[1, 2, 3]) / (32 ** 2.0)
                    pr_val = tf.reduce_sum(self._graph_ops['x_val_predicted'], axis=[1, 2, 3]) / (32 ** 2.0)
                    self._graph_ops['val_accuracy'] = tf.reduce_mean(tf.abs(gt_val - pr_val))

            # Epoch summaries for Tensorboard
            if self._tb_dir is not None:
                self._graph_tensorboard_summary(l2_cost, gradients, variables, global_grad_norm)

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
        self._log('Computing total test accuracy...')

        with self._graph.as_default():
            num_batches = int(np.ceil(self._total_testing_samples / self._batch_size))

            if num_batches == 0:
                warnings.warn('Less than a batch of testing data')
                exit()

            # Initialize storage for the retrieved test variables
            loss_sum = 0.0
            abs_diff_sum = 0.0

            # Main test loop
            for _ in tqdm(range(num_batches)):
                batch_loss, batch_abs_diff, batch_gt, batch_pr = self._session.run(
                    [self._graph_ops['test_losses'], self._graph_ops['test_accuracy'],
                     self._graph_ops['gt_test'], self._graph_ops['pr_test']])
                loss_sum = loss_sum + batch_loss
                abs_diff_sum = abs_diff_sum + batch_abs_diff

                # Print prediction results for each image as we go
                for idx, gt in enumerate(batch_gt):
                    pr = batch_pr[idx]
                    abs_diff = abs(pr - gt)
                    rel_diff = abs_diff / gt
                    self._log("idx={}, real_count={}, prediction={:.3f}, abs_diff={:.3f}, relative_diff={:.3f}"
                              .format(idx, gt, pr, abs_diff, rel_diff))

            # For counting problems with countception, we want the averaged loss and difference across the images
            loss_mean = (loss_sum / num_batches)
            abs_diff_mean = (abs_diff_sum / num_batches)
            self._log('Average test loss: {:.3f}'.format(loss_mean))
            self._log('Average test absolute difference: {:.3f}'.format(abs_diff_mean))
            return 1.0 - loss_mean.astype(np.float32), abs_diff_mean

    def forward_pass_with_file_inputs(self, x):
        with self._graph.as_default():

            self._parse_images(x)

            dataset_batch = tf.data.Dataset.from_tensor_slices(self._all_images) \
                .batch(self._batch_size, drop_remainder=True) \
                .prefetch(self._batch_size * 2)

            iterator = dataset_batch.make_one_shot_iterator()
            image_data = iterator.get_next()

            if self._load_from_saved is not False:
                self.load_state()

            # queue is not used, but this is still called to keep compatible with the shut_down() function
            self._initialize_queue_runners()

            # Run model on them
            x_pred = self.forward_pass(image_data, deterministic=True)

            total_outputs = []
            try:
                while True:
                    x_pred_value = self._session.run(x_pred)
                    for pr in x_pred_value:
                        total_outputs.append(np.squeeze(pr))

            except tf.errors.OutOfRangeError:
                pass

        return total_outputs

    def forward_pass_with_interpreted_outputs(self, x):
        xx = self.forward_pass_with_file_inputs(x)

        # Get the predicted count
        patch_size = 32
        interpreted_outputs = [y / (patch_size ** 2.0) for y in np.sum(xx, axis=(1, 2))]
        return interpreted_outputs

    def add_output_layer(self, regularization_coefficient=None, output_size=None):
        # There is no need to do this in the countception model
        pass

    def _parse_images(self, images):
        """
        Parse and put input images into self._all_images.
        This is usually called in forward_pass_with_file_inputs(), when a pre-trained network is used for prediction.
        """
        image_data_list = []
        for img in images:
            image_data_raw = np.array(Image.open(img).getdata())
            image_data = image_data_raw.reshape((self._image_height, self._image_width, self._image_depth))
            image_data_list.append(image_data)

        self._all_images = np.asarray(image_data_list).astype(np.float32)

    def _parse_dataset(self, train_images, train_labels, train_mf,
                       test_images, test_labels, test_mf,
                       val_images, val_labels, val_mf):
        # Countception uses arrays from pickle files for its image and label inputs, so enough differences exist to
        # override this whole function
        with self._graph.as_default():
            # Get the number of samples the normal way right off the bat, since other methods of getting the counts
            # depend on Tensors
            self._total_training_samples = int(self._total_raw_samples)
            if self._testing:
                self._total_testing_samples = int(self._total_raw_samples * self._test_split)
                self._total_training_samples = self._total_training_samples - self._total_testing_samples
            if self._validation:
                self._total_validation_samples = int(self._total_raw_samples * self._validation_split)
                self._total_training_samples = self._total_training_samples - self._total_validation_samples

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

            # Create input queues
            train_input_queue = tf.train.slice_input_producer([train_images, train_labels], shuffle=False)
            if self._testing:
                test_input_queue = tf.train.slice_input_producer([test_images, test_labels], shuffle=False)
            if self._validation:
                val_input_queue = tf.train.slice_input_producer([val_images, val_labels], shuffle=False)

            # Apply pre-processing for training and testing images. Images need just a dtype conversion and labels need
            # nothing whatsoever
            self._train_images = tf.image.convert_image_dtype(train_input_queue[0], dtype=tf.float32)
            if self._testing:
                self._test_images = tf.image.convert_image_dtype(test_input_queue[0], dtype=tf.float32)
            if self._validation:
                self._val_images = tf.image.convert_image_dtype(val_input_queue[0], dtype=tf.float32)

            self._train_labels = train_input_queue[1]
            if self._testing:
                self._test_labels = test_input_queue[1]
            if self._validation:
                self._val_labels = val_input_queue[1]

            # Image standardization doesn't apply, so we skip it

            # Manually set the shape of the image tensors so it matches the shape of the images
            self._parse_force_set_shape()

    def load_countception_dataset_from_pkl_file(self, pkl_file_name):
        """
        Loads the dataset(image data and ground truth count map data) from a pickle file into an internal
        representation.
        For more information about data format in the pickle file, please refer to the paper
        https://arxiv.org/abs/1703.08710
        :param pkl_file_name: the path of the pickle file containing the dataset
        """
        if not isinstance(pkl_file_name, str):
            raise TypeError("pkl_file_name must be a str")
        if not os.path.isfile(pkl_file_name):
            raise ValueError("'" + pkl_file_name + "' does not exist")
        if not pkl_file_name.endswith('.pkl'):
            raise ValueError("'" + pkl_file_name + "' is not a pickle file")

        try:
            # try to load dataset from the given pickle file
            dataset = pickle.load(open(pkl_file_name, "rb"))
            dataset_x = np.asarray([d[0] for d in dataset]).astype(np.float32)
            dataset_y = np.transpose(np.asarray([d[1] for d in dataset]), [0, 2, 3, 1]).astype(np.float32)
        except Exception:
            raise TypeError("'" + pkl_file_name + "' does not contain data in required format.")

        self._total_raw_samples = len(dataset_x)

        self._log('Total raw examples is %d' % self._total_raw_samples)
        self._log('Loading dataset...')

        self._raw_image_files = dataset_x
        self._raw_labels = dataset_y

        self._split_labels = False
