from . import layers, definitions, deepplantpheno
import numpy as np
import tensorflow as tf
import os
import datetime
import time
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

        super()._graph_tensorboard_summary(l2_cost, gradients, variables, global_grad_norm)

        # Summaries specific to classification problems
        tf.summary.scalar('train/loss', self._graph_ops['cost'], collections=['custom_summaries'])
        tf.summary.scalar('train/accuracy', self._graph_ops['accuracy'], collections=['custom_summaries'])
        if self._validation:
            tf.summary.scalar('validation/loss', self._graph_ops['val_losses'],
                              collections=['custom_summaries'])
            tf.summary.scalar('validation/accuracy', self._graph_ops['val_accuracy'],
                              collections=['custom_summaries'])

    def _assemble_graph(self):

        self._log('Parsing dataset...')
        self._graph_parse_data()

        self._log('Creating layer parameters...')
        self._add_layers_to_graph()

        self._log('Assembling graph...')

        x, y = tf.train.shuffle_batch([self._train_images, self._train_labels],
                                      batch_size=self._batch_size,
                                      num_threads=self._num_threads,
                                      capacity=self._queue_capacity,
                                      min_after_dequeue=self._batch_size)

        # Reshape input to the expected image dimensions
        x = tf.reshape(x, shape=[-1, self._image_height, self._image_width, self._image_depth])

        # Run the network operations
        xx = self.forward_pass(x, deterministic=False)

        # Define regularization cost
        if self._reg_coeff is not None:
            l2_cost = tf.squeeze(tf.reduce_sum(
                [layer.regularization_coefficient * tf.nn.l2_loss(layer.weights) for layer in self._layers
                 if isinstance(layer, layers.fullyConnectedLayer)]))
        else:
            l2_cost = 0.0

        # Define cost function
        if self._loss_fn == 'l1':
            l1_loss = tf.reduce_mean(tf.abs(tf.subtract(xx, y)))
            gt = tf.reduce_sum(y, axis=[1, 2, 3]) / (32 ** 2.0)
            pr = tf.reduce_sum(xx, axis=[1, 2, 3]) / (32 ** 2.0)
            accuracy = tf.reduce_mean(tf.abs(gt - pr))
            self._graph_ops['cost'] = tf.squeeze(tf.add(l1_loss, l2_cost))
            self._graph_ops['accuracy'] = accuracy

        # Set the optimizer and get the gradients from it
        gradients, variables, global_grad_norm = self._graph_add_optimizer()

        # Calculate validation and test accuracy
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

        if self._testing:
            self._graph_ops['x_test_predicted'] = self.forward_pass(x_test, deterministic=True)
        if self._validation:
            self._graph_ops['x_val_predicted'] = self.forward_pass(x_val, deterministic=True)

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
        self._graph_tensorboard_summary(l2_cost, gradients, variables, global_grad_norm)

    def begin_training(self, return_test_loss=False):

        with self._graph.as_default():

            self._assemble_graph()
            print('assembled the graph')

            # Either load the network parameters from a checkpoint file or start training
            if self._load_from_saved is not False:
                self._has_trained = True
                self.load_state()
                self._initialize_queue_runners()
                self.compute_full_test_accuracy()
                self.shut_down()

            else:

                if self._tb_dir is not None:
                    train_writer = tf.summary.FileWriter(self._tb_dir, self._session.graph)

                self._log('Initializing parameters...')
                self._session.run(tf.global_variables_initializer())

                self._initialize_queue_runners()

                self._log('Beginning training...')

                self._set_learning_rate()

                # Needed for batch norm
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                self._graph_ops['optimizer'] = tf.group([self._graph_ops['optimizer'], update_ops])

                # for i in range(self._maximum_training_batches):
                tqdm_range = tqdm(range(self._maximum_training_batches))
                for i in tqdm_range:
                    start_time = time.time()

                    self._global_epoch = i
                    self._session.run(self._graph_ops['optimizer'])
                    if self._global_epoch > 0 and self._global_epoch % self._report_rate == 0:
                        elapsed = time.time() - start_time

                        if self._tb_dir is not None:
                            summary = self._session.run(self._graph_ops['merged'])
                            train_writer.add_summary(summary, i)
                        if self._validation:
                            loss, epoch_accuracy, epoch_val_accuracy = self._session.run(
                                [self._graph_ops['cost'],
                                 self._graph_ops['accuracy'],
                                 self._graph_ops['val_accuracy']])

                            samples_per_sec = self._batch_size / elapsed

                            desc_str = "{}: Results for batch {} (epoch {:.1f}) " + \
                                       "- Loss: {:.5f}, Training Accuracy: {:.4f}, samples/sec: {:.2f}"
                            tqdm_range.set_description(
                                desc_str.format(datetime.datetime.now().strftime("%I:%M%p"),
                                                i,
                                                i / (self._total_training_samples / self._batch_size),
                                                loss,
                                                epoch_accuracy,
                                                samples_per_sec))
                            self._log('Batch {}, train_loss {:.3f}, train_accu {:.3f}, val_accu {:.3f}'
                                      .format(i, loss, epoch_accuracy, epoch_val_accuracy))

                        else:
                            loss, epoch_accuracy = self._session.run(
                                [self._graph_ops['cost'],
                                 self._graph_ops['accuracy']])

                            samples_per_sec = self._batch_size / elapsed

                            desc_str = "{}: Results for batch {} (epoch {:.1f}) " + \
                                       "- Loss: {:.5f}, Training Accuracy: {:.4f}, samples/sec: {:.2f}"
                            tqdm_range.set_description(
                                desc_str.format(datetime.datetime.now().strftime("%I:%M%p"),
                                                i,
                                                i / (self._total_training_samples / self._batch_size),
                                                loss,
                                                epoch_accuracy,
                                                samples_per_sec))
                            self._log('Batch {}, train_loss {:.3f}, train_accu {:.3f}'.format(i, loss, epoch_accuracy))

                        if self._save_checkpoints and self._global_epoch % (self._report_rate * 100) == 0:
                            self.save_state(self._save_dir)
                    else:
                        loss = self._session.run([self._graph_ops['cost']])

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

    def compute_full_test_accuracy(self):
        self._log('Computing total test accuracy...')

        with self._graph.as_default():
            num_batches = int(np.ceil(self._total_testing_samples / self._batch_size))

            if num_batches == 0:
                warnings.warn('Less than a batch of testing data')
                exit()

            # Initialize storage for the retreived test variables
            loss_sum = 0.0
            abs_diff_sum = 0.0

            # Main test loop
            for _ in tqdm(range(num_batches)):
                batch_loss, batch_abs_diff, batch_gt, batch_pr = self._session.run(
                    [self._graph_ops['test_losses'], self._graph_ops['test_accuracy']])
                loss_sum = loss_sum + batch_loss
                abs_diff_sum = abs_diff_sum + batch_abs_diff

            # For classification problems (assumed to be multi-class), we want accuracy and confusion matrix (not
            # implemented)
            loss_mean = (loss_sum / num_batches)
            abs_diff_mean = (abs_diff_sum / num_batches)
            self._log('Average test loss: {:.3f}'.format(loss_mean))
            self._log('Average test absolute difference: {:.3f}'.format(abs_diff_mean))
            return 1.0-loss_mean.astype(np.float32), abs_diff_mean

    def forward_pass_with_file_inputs(self, x):

        with self._graph.as_default():

            total_outputs = []

            num_batches = len(x) // self._batch_size
            remainder = len(x) % self._batch_size

            if remainder != 0:
                num_batches += 1
                remainder = self._batch_size - remainder

            self._parse_images(x)

            x_test = tf.train.batch([self._all_images], batch_size=self._batch_size, num_threads=self._num_threads)
            x_test = tf.reshape(x_test, shape=[-1, self._image_height, self._image_width, self._image_depth])

            if self._load_from_saved is not False:
                self.load_state()
            self._initialize_queue_runners()

            # Run model on them
            x_pred = self.forward_pass(x_test, deterministic=True)

            for i in range(int(num_batches)):
                xx = self._session.run(x_pred)
                for img in np.array_split(xx, self._batch_size):
                    total_outputs.append(np.squeeze(img))

            # delete any outputs which are overruns from the last batch
            if remainder != 0:
                for i in range(remainder):
                    total_outputs = np.delete(total_outputs, -1, 0)

        return total_outputs

    def forward_pass_with_interpreted_outputs(self, x):

        xx = self.forward_pass_with_file_inputs(x)

        # Get the predicted count
        patch_size = 32
        interpreted_outputs = [y / (patch_size ** 2.0) for y in np.sum(xx, axis=(1,2))]
        return interpreted_outputs

    def add_output_layer(self, regularization_coefficient=None, output_size=None):
        """
        Add an output layer to the network (no need to do this in the count ception model)

        :param regularization_coefficient: optionally, an L2 decay coefficient for this layer (overrides the coefficient
         set by set_regularization_coefficient)
        :param output_size: optionally, override the output size of this layer. Typically not needed, but required for
        use cases such as creating the output layer before loading data.
        """
        pass


    def _parse_images(self, images):
        """
        Parse and put input images into self._all_images.
        This is usually called in forward_pass_with_file_inputs(), when trained network is used for prediction.
        """
        input_queue = tf.train.slice_input_producer([images], shuffle=False)
        # '*255' because tf.io.decode_image() returns values between 0 and 1
        # In the pickle file for training, image data values are between 0 and 255
        images = tf.io.decode_image(tf.read_file(input_queue[0]), channels=self._image_depth, dtype=tf.float32) * 255
        images.set_shape([self._image_height, self._image_width, self._image_depth])
        self._all_images = images


    def _parse_dataset(self, train_images, train_labels, train_mf,
                       test_images, test_labels, test_mf,
                       val_images, val_labels, val_mf,
                       image_type='png'):
        """Takes training and testing images and labels, creates input queues internally to this instance"""
        with self._graph.as_default():

            # Get the number of samples the normal way
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

            self._train_labels = train_input_queue[1]
            if self._testing:
                self._test_labels = test_input_queue[1]
            if self._validation:
                self._val_labels = val_input_queue[1]

            # Apply pre-processing for training and testing images
            self._train_images = tf.image.convert_image_dtype(train_input_queue[0], dtype=tf.float32)
            if self._testing:
                self._test_images = tf.image.convert_image_dtype(test_input_queue[0], dtype=tf.float32)
            if self._validation:
                self._val_images = tf.image.convert_image_dtype(val_input_queue[0], dtype=tf.float32)

            # define the shape of the image tensors so it matches the shape of the images
            self._train_images.set_shape([self._image_height, self._image_width, self._image_depth])
            if self._testing:
                self._test_images.set_shape([self._image_height, self._image_width, self._image_depth])
            if self._validation:
                self._val_images.set_shape([self._image_height, self._image_width, self._image_depth])

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



