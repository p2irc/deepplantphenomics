from . import DPPModel
import numpy as np
import tensorflow.compat.v1 as tf
import datetime
import time
import os
import warnings
from tqdm import tqdm
import pickle


class CountCeptionModel(DPPModel):
    _supported_loss_fns = ['l1']
    _supported_augmentations = []
    _supports_standardization = False

    def __init__(self, debug=False, load_from_saved=False, save_checkpoints=True, initialize=True, tensorboard_dir=None,
                 report_rate=100, save_dir=None):
        super().__init__(debug, load_from_saved, save_checkpoints, initialize, tensorboard_dir, report_rate, save_dir)
        self._loss_fn = 'l1'

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
            with tf.device('/device:cpu:0'):  # Only do preprocessing on the CPU to limit data transfer between devices
                self._graph_parse_data()

                # Batch the datasets and create iterators for them
                train_iter = self._batch_and_iterate(self._train_dataset, shuffle=True)
                if self._testing:
                    test_iter = self._batch_and_iterate(self._test_dataset)
                if self._validation:
                    val_iter = self._batch_and_iterate(self._val_dataset)

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
                    xx = self.forward_pass(x, deterministic=False)

                    # Define regularization cost
                    self._log('Graph: Calculating loss and gradients...')
                    l2_cost = self._graph_layer_loss()

                    # Define cost function
                    pred_loss = self._graph_problem_loss(xx, y)
                    gpu_cost = tf.reduce_mean(pred_loss) + l2_cost
                    cost_sum = tf.reduce_sum(pred_loss)
                    device_costs.append(cost_sum)

                    # Get the accuracy of the predictions as well
                    _, _, acc_diff = self._graph_count_accuracy(xx, y)
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
                x_test, self._graph_ops['y_test'] = test_iter.get_next()

                self._graph_ops['x_test_predicted'] = self.forward_pass(x_test, deterministic=True)

                test_loss = self._graph_problem_loss(self._graph_ops['x_test_predicted'], self._graph_ops['y_test'])
                self._graph_ops['test_losses'] = tf.reduce_mean(test_loss)

                self._graph_ops['pr_test'], self._graph_ops['gt_test'], test_diff = self._graph_count_accuracy(
                    self._graph_ops['x_test_predicted'], self._graph_ops['y_test'])
                self._graph_ops['test_accuracy'] = tf.reduce_mean(test_diff)

            if self._validation:
                x_val, self._graph_ops['y_val'] = val_iter.get_next()

                self._graph_ops['x_val_predicted'] = self.forward_pass(x_val, deterministic=True)

                val_loss = self._graph_problem_loss(self._graph_ops['x_val_predicted'], self._graph_ops['y_val'])
                self._graph_ops['val_losses'] = tf.reduce_mean(val_loss)

                _, _, val_diff = self._graph_count_accuracy(self._graph_ops['x_val_predicted'],
                                                            self._graph_ops['y_val'])
                self._graph_ops['val_accuracy'] = tf.reduce_mean(val_diff)

            # Epoch summaries for Tensorboard
            if self._tb_dir is not None:
                self._graph_tensorboard_summary(l2_cost, gradients, variables, global_grad_norm)

    def _graph_problem_loss(self, pred, lab):
        if self._loss_fn == 'l1':
            return self.__l1_loss(pred - lab)

        raise RuntimeError("Could not calculate problem loss for a loss function of " + self._loss_fn)

    def __l1_loss(self, x):
        """
        Calculates the L1 loss of prediction difference Tensors for each item in a batch
        :param x: A Tensor with prediction differences for each item in a batch
        :return: A Tensor with the scalar L1 loss for each item
        """
        y = tf.map_fn(lambda ex: tf.reduce_sum(tf.abs(ex)), x)
        return y

    def _graph_count_accuracy(self, pred, lab):
        """
        Calculates the total count from the predictions and labels for each item in a batch
        :param pred: Model predictions for the count heatmap
        :param lab: Labels for the correct heatmaps and counts, with the same size as pred
        :return: The total count for the predictions and labels and their absolute difference
        """
        pred_count = tf.reduce_sum(pred, axis=[1, 2, 3]) / (32 ** 2.0)
        true_count = tf.reduce_sum(lab, axis=[1, 2, 3]) / (32 ** 2.0)
        count_diff = tf.abs(pred_count - true_count)
        return pred_count, true_count, count_diff

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
                for idx, (gt, pr) in enumerate(zip(batch_gt, batch_pr)):
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

            dataset_batch = self._all_images.batch(self._batch_size).prefetch(self._batch_size * 2)
            iterator = dataset_batch.make_one_shot_iterator()
            image_data = iterator.get_next()

            if self._load_from_saved:
                self.load_state()

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
        # There is no need to do this in the Countception model
        pass

    def _parse_read_images(self, images, channels=1, image_type=tf.float32):
        # With Countception, we can have either strings from an inference forward pass, or straight arrays from a
        # pickle file during training.
        if images.dtype == tf.string:
            images = super()._parse_read_images(images, channels)
        else:
            images = tf.image.convert_image_dtype(images, dtype=image_type)
        return images

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
