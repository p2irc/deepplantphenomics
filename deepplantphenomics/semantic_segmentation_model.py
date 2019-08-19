from . import layers, definitions, DPPModel
import numpy as np
import tensorflow as tf
import os
import warnings
import copy
import math
from tqdm import tqdm


class SemanticSegmentationModel(DPPModel):
    _problem_type = definitions.ProblemType.SEMANTIC_SEGMETNATION
    _loss_fn = 'sigmoid cross entropy'
    _supported_loss_fns = ['sigmoid cross entropy']
    _valid_augmentations = [definitions.AugmentationType.CONTRAST_BRIGHT]

    # State variables specific to semantic segmentation for constructing the graph and passing to Tensorboard
    _graph_forward_pass = None

    def __init__(self, debug=False, load_from_saved=False, save_checkpoints=True, initialize=True, tensorboard_dir=None,
                 report_rate=100, save_dir=None):
        super().__init__(debug, load_from_saved, save_checkpoints, initialize, tensorboard_dir, report_rate, save_dir)

    def _graph_tensorboard_summary(self, l2_cost, gradients, variables, global_grad_norm):
        super()._graph_tensorboard_summary(l2_cost, gradients, variables, global_grad_norm)

        # Summaries specific to semantic segmentation
        # We send in the last layer's output size (i.e. the final image dimensions) to get_weights_as_image
        # because xx and x_test_predicted have dynamic dims [?,?,?,?], so we need actual numbers passed in
        train_images_summary = self.__get_weights_as_image(
            tf.transpose(tf.expand_dims(self._graph_forward_pass, -1), (1, 2, 3, 0)),
            self._layers[-1].output_size)
        tf.summary.image('masks/train', train_images_summary, collections=['custom_summaries'])
        if self._validation:
            tf.summary.scalar('validation/loss', self._graph_ops['val_cost'],
                              collections=['custom_summaries'])
            val_images_summary = self.__get_weights_as_image(
                tf.transpose(tf.expand_dims(self._graph_ops['x_val_predicted'], -1), (1, 2, 3, 0)),
                self._layers[-1].output_size)
            tf.summary.image('masks/validation', val_images_summary, collections=['custom_summaries'])

    def __assemble_graph(self):
        with self._graph.as_default():

            self.__log('Parsing dataset...')
            self._graph_parse_data()

            self.__log('Creating layer parameters...')
            self.__add_layers_to_graph()

            self.__log('Assembling graph...')

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

            # Reshape input and labels to the expected image dimensions
            x = tf.reshape(x, shape=[-1, self._image_height, self._image_width, self._image_depth])
            y = tf.reshape(y, shape=[-1, self._image_height, self._image_width, 1])

            # If we are using patching, we extract a random patch from the image here
            if self._with_patching:
                x, offsets = self._graph_extract_patch(x)

            # Run the network operations
            if self._has_moderation:
                xx = self.forward_pass(x, deterministic=False, moderation_features=mod_w)
            else:
                xx = self.forward_pass(x, deterministic=False)
            self._graph_forward_pass = xx  # Needed to output raw forward pass output to Tensorboard

            # Define regularization cost
            if self._reg_coeff is not None:
                l2_cost = tf.squeeze(tf.reduce_sum(
                    [layer.regularization_coefficient * tf.nn.l2_loss(layer.weights) for layer in self._layers
                     if isinstance(layer, layers.fullyConnectedLayer)]))
            else:
                l2_cost = 0.0

            # Define cost function  based on which one was selected via set_loss_function
            if self._loss_fn == 'sigmoid cross entropy':
                pixel_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=xx, labels=y))
            self._graph_ops['cost'] = tf.squeeze(tf.add(pixel_loss, l2_cost))

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
                self._graph_ops['y_test'] = tf.reshape(self._graph_ops['y_test'],
                                                       shape=[-1, self._image_height, self._image_width, 1])
            if self._validation:
                x_val = tf.reshape(x_val, shape=[-1, self._image_height, self._image_width, self._image_depth])
                self._graph_ops['y_val'] = tf.reshape(self._graph_ops['y_val'],
                                                      shape=[-1, self._image_height, self._image_width, 1])

            # If using patching, we need to properly pull similar patches from the test and validation images (and
            # labels)
            if self._with_patching:
                if self._testing:
                    x_test, _ = self._graph_extract_patch(x_test, offsets)
                    self._graph_ops['y_test'], _ = self._graph_extract_patch(self._graph_ops['y_test'], offsets)
                if self._validation:
                    x_val, _ = self._graph_extract_patch(x_val, offsets)
                    self._graph_ops['y_val'], _ = self._graph_extract_patch(self._graph_ops['y_val'], offsets)

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
                self._graph_ops['test_losses'] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self._graph_ops['x_test_predicted'],
                    labels=self._graph_ops['y_test']),
                    axis=2)
                self._graph_ops['test_losses'] = tf.reshape(tf.reduce_mean(self._graph_ops['test_losses'],
                                                                           axis=1),
                                                            [self._batch_size])
            if self._validation:
                self._graph_ops['val_losses'] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self._graph_ops['x_val_predicted'], labels=self._graph_ops['y_val']),
                    axis=2)
                self._graph_ops['val_losses'] = tf.transpose(
                    tf.reduce_mean(self._graph_ops['val_losses'], axis=1))
                self._graph_ops['val_cost'] = tf.reduce_mean(self._graph_ops['val_losses'])

            # Epoch summaries for Tensorboard
            self._graph_tensorboard_summary(l2_cost, gradients, variables, global_grad_norm)

    def compute_full_test_accuracy(self):
        self.__log('Computing total test accuracy/regression loss...')

        with self._graph.as_default():
            num_batches = int(np.ceil(self._total_testing_samples / self._batch_size))

            if num_batches == 0:
                warnings.warn('Less than a batch of testing data')
                exit()

            # Initialize storage for the retreived test variables
            all_losses = np.empty(shape=1)

            # Main test loop
            for _ in tqdm(range(num_batches)):
                r_losses = self._session.run(self._graph_ops['test_losses'])
                all_losses = np.concatenate((all_losses, r_losses), axis=0)

            all_losses = np.delete(all_losses, 0)

            # Delete the extra entries (e.g. batch_size is 4 and 1 sample left, it will loop and have 3 repeats that
            # we want to get rid of)
            extra = self._batch_size - (self._total_testing_samples % self._batch_size)
            if extra != self._batch_size:
                mask_extra = np.ones(self._batch_size * num_batches, dtype=bool)
                mask_extra[range(self._batch_size * num_batches - extra, self._batch_size * num_batches)] = False
                all_losses = all_losses[mask_extra, ...]

            # For semantic segmentation problems we want relative and abs mean, std of L2 norms, plus a histogram of
            # errors
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

            self.__log('Mean loss: {}'.format(mean))
            self.__log('Loss standard deviation: {}'.format(std))
            self.__log('Mean absolute loss: {}'.format(abs_mean))
            self.__log('Absolute loss standard deviation: {}'.format(abs_std))
            self.__log('Min error: {}'.format(loss_min))
            self.__log('Max error: {}'.format(loss_max))
            self.__log('MSE: {}'.format(mse))

            self.__log('Histogram of {} losses:'.format(self._loss_fn))
            self.__log(hist)

            return abs_mean.astype(np.float32)

    def forward_pass_with_file_inputs(self, x):
        with self._graph.as_default():
            if self._with_patching:
                # we want the largest multiple of of patch height/width that is smaller than the original
                # image height/width, for the final image dimensions
                patch_height = self._patch_height
                patch_width = self._patch_width
                final_height = (self._image_height // patch_height) * patch_height
                final_width = (self._image_width // patch_width) * patch_width
                # find image differences to determine recentering crop coords, we divide by 2 so that the leftover
                # is equal on all sides of image
                offset_height = (self._image_height - final_height) // 2
                offset_width = (self._image_width - final_width) // 2
                # pre-allocate output dimensions
                total_outputs = np.empty([1, final_height, final_width])
            else:
                total_outputs = np.empty([1, self._image_height, self._image_width])

            num_batches = len(x) // self._batch_size
            remainder = len(x) % self._batch_size

            if remainder != 0:
                num_batches += 1
                remainder = self._batch_size - remainder

            # self.load_images_from_list(x) no longer calls following 2 lines so we needed to force them here
            images = x
            self.__parse_images(images)

            x_test = tf.train.batch([self._all_images], batch_size=self._batch_size, num_threads=self._num_threads)
            x_test = tf.reshape(x_test, shape=[-1, self._image_height, self._image_width, self._image_depth])
            if self._with_patching:
                x_test = tf.image.crop_to_bounding_box(x_test, offset_height, offset_width, final_height, final_width)
                # Split the images up into the multiple slices of size patch_height x patch_width
                ksizes = [1, patch_height, patch_width, 1]
                strides = [1, patch_height, patch_width, 1]
                rates = [1, 1, 1, 1]
                x_test = tf.extract_image_patches(x_test, ksizes, strides, rates, "VALID")
                x_test = tf.reshape(x_test, shape=[-1, patch_height, patch_width, self._image_depth])

            if self._load_from_saved:
                self.load_state()
            self.__initialize_queue_runners()
            # Run model on them
            x_pred = self.forward_pass(x_test, deterministic=True)

            if self._with_patching:
                num_patch_rows = final_height // patch_height
                num_patch_cols = final_width // patch_width
                for i in range(num_batches):
                    xx = self._session.run(x_pred)

                    # generalized image stitching
                    for img in np.array_split(xx, self._batch_size):  # for each img in current batch
                        # we are going to build a list of rows of imgs called img_rows, where each element
                        # of img_rows is a row of img's concatenated together horizontally (axis=1), then we will
                        # iterate through img_rows concatenating the rows vertically (axis=0) to build
                        # the full img

                        img_rows = []
                        # for each row
                        for j in range(num_patch_rows):
                            curr_row = img[j*num_patch_cols]  # start new row with first img
                            # iterate through the rest of the row, concatenating img's together
                            for k in range(1, num_patch_cols):
                                # horizontal cat
                                curr_row = np.concatenate((curr_row, img[k+(j*num_patch_cols)]), axis=1)
                            img_rows.append(curr_row)  # add row of img's to the list

                        # start full img with the first full row of imgs
                        full_img = img_rows[0]
                        # iterate through rest of rows, concatenating rows together
                        for row_num in range(1, num_patch_rows):
                            # vertical cat
                            full_img = np.concatenate((full_img, img_rows[row_num]), axis=0)

                        # need to match total_outputs dimensions, so we add a dimension to the shape to match
                        full_img = np.array([full_img])  # shape transformation: (x,y) --> (1,x,y)
                        total_outputs = np.append(total_outputs, full_img, axis=0)  # add the final img to the list
            else:
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
        with self._graph.as_default():
            # check for patching needs
            if self._with_patching:
                # we want the largest multiple of of patch height/width that is smaller than the original
                # image height/width, for the final image dimensions
                patch_height = self._patch_height
                patch_width = self._patch_width
                final_height = (self._image_height // patch_height) * patch_height
                final_width = (self._image_width // patch_width) * patch_width
                # find image differences to determine recentering crop coords, we divide by 2 so that the leftover
                # is equal on all sides of image
                offset_height = (self._image_height - final_height) // 2
                offset_width = (self._image_width - final_width) // 2
                # pre-allocate output dimensions
                total_outputs = np.empty([1, final_height, final_width])
            else:
                total_outputs = np.empty([1, self._image_height, self._image_width])

            num_batches = len(x) // self._batch_size
            remainder = len(x) % self._batch_size
            if remainder != 0:
                num_batches += 1
                remainder = self._batch_size - remainder

            # self.load_images_from_list(x) no longer calls following 2 lines so we needed to force them here
            images = x
            self.__parse_images(images)

            # set up and then initialize the queue
            x_test = tf.train.batch(
                [self._all_images], batch_size=self._batch_size, num_threads=self._num_threads)
            x_test = tf.reshape(x_test, shape=[-1, self._image_height, self._image_width, self._image_depth])
            # if using patching we have to determine different image dimensions
            if self._with_patching:
                x_test = tf.image.crop_to_bounding_box(
                    x_test, offset_height, offset_width, final_height, final_width)
                # Split the images up into the multiple slices of size patch_height x patch_width
                ksizes = [1, patch_height, patch_width, 1]
                strides = [1, patch_height, patch_width, 1]
                rates = [1, 1, 1, 1]
                x_test = tf.extract_image_patches(x_test, ksizes, strides, rates, "VALID")
                x_test = tf.reshape(x_test, shape=[-1, patch_height, patch_width, self._image_depth])

            if self._load_from_saved:
                self.load_state()
            self.__initialize_queue_runners()
            x_pred = self.forward_pass(x_test, deterministic=True)

            # check if we need to perform patching
            if self._with_patching:
                num_patch_rows = final_height // patch_height
                num_patch_cols = final_width // patch_width
                for i in range(num_batches):
                    xx = self._session.run(x_pred)
                    # generalized image stitching
                    for img in np.array_split(xx, self._batch_size):  # for each img in current batch
                        # we are going to build a list of rows of imgs called img_rows, where each element
                        # of img_rows is a row of img's concatenated together horizontally (axis=1), then we will
                        # iterate through img_rows concatenating the rows vertically (axis=0) to build
                        # the full img
                        img_rows = []
                        for j in range(num_patch_rows):  # for each row
                            curr_row = img[j*num_patch_cols]  # start new row with first img
                            # iterate through the rest of the row, concatenating img's together
                            for k in range(1, num_patch_cols):
                                # vertical cat
                                curr_row = np.concatenate((curr_row, img[k+(j*num_patch_cols)]), axis=1)
                            img_rows.append(curr_row)  # add row of img's to the list
                        # start full img with the first full row of imgs
                        full_img = img_rows[0]
                        # iterate through rest of rows, concatenating rows together
                        for row_num in range(1, num_patch_rows):
                            # vertical cat
                            full_img = np.concatenate((full_img, img_rows[row_num]), axis=0)
                        # need to match total_outputs dimensions, so we add a dimension to the shape to match
                        full_img = np.array([full_img])  # shape transformation: (x,y) --> (1,x,y)
                        # this appending may be causing a border, might need to rewrite and specifically index
                        total_outputs = np.append(total_outputs, full_img, axis=0)  # add the final img to the list
            else:
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

        # normalize and then threshold
        interpreted_outputs = np.zeros(total_outputs.shape, dtype=np.uint8)
        for i, img in enumerate(total_outputs):
            # normalize
            x_min = np.min(img)
            x_max = np.max(img)
            mask = (img - x_min) / (x_max - x_min)
            # threshold
            mask[mask >= 0.5] = 255
            mask[mask < 0.5] = 0
            # store
            interpreted_outputs[i, :, :] = mask

        return interpreted_outputs

    def add_output_layer(self, regularization_coefficient=None, output_size=None):
        if len(self._layers) < 1:
            raise RuntimeError("An output layer cannot be the first layer added to the model. " +
                               "Add an input layer with DPPModel.add_input_layer() first.")
        if regularization_coefficient is not None:
            warnings.warn("Object detection doesn't use regularization_coefficient in its output layer")
        if output_size is not None:
            raise RuntimeError("output_size should be None for semantic segmentation")

        self.__log('Adding output layer...')

        filter_dimension = [1, 1, copy.deepcopy(self.__last_layer().output_size[3]), 1]

        with self._graph.as_default():
            layer = layers.convLayer('output',
                                     copy.deepcopy(self.__last_layer().output_size),
                                     filter_dimension,
                                     1,
                                     None,
                                     self._weight_initializer)

        self.__log('Inputs: {0} Outputs: {1}'.format(layer.input_size, layer.output_size))
        self._layers.append(layer)

    def load_dataset_from_directory_with_segmentation_masks(self, dirname, seg_dirname):
        """
        Loads the png images in the given directory into an internal representation, using binary segmentation
        masks from another file with the same filename as ground truth.

        :param dirname: the path of the directory containing the images
        :param seg_dirname: the path of the directory containing ground-truth binary segmentation masks
        """

        image_files = [os.path.join(dirname, name) for name in os.listdir(dirname) if
                       os.path.isfile(os.path.join(dirname, name)) & name.endswith('.png')]

        seg_files = [os.path.join(seg_dirname, name) for name in os.listdir(seg_dirname) if
                     os.path.isfile(os.path.join(seg_dirname, name)) & name.endswith('.png')]

        self._total_raw_samples = len(image_files)

        self.__log('Total raw examples is %d' % self._total_raw_samples)

        self._raw_image_files = image_files
        self._raw_labels = seg_files
        self._split_labels = False  # Band-aid fix

    def __parse_dataset(self, train_images, train_labels, train_mf,
                        test_images, test_labels, test_mf,
                        val_images, val_labels, val_mf,
                        image_type='png'):
        """Takes training and testing images and labels, creates input queues internally to this instance"""
        with self._graph.as_default():

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

            # Logging verbosity
            self.__log('Total training samples is {0}'.format(self._total_training_samples))
            self.__log('Total validation samples is {0}'.format(self._total_validation_samples))
            self.__log('Total testing samples is {0}'.format(self._total_testing_samples))

            # Create moderation features queues
            if train_mf is not None:
                train_moderation_queue = tf.train.slice_input_producer([train_mf], shuffle=False)
                self._train_moderation_features = tf.cast(train_moderation_queue[0], tf.float32)

            if test_mf is not None:
                test_moderation_queue = tf.train.slice_input_producer([test_mf], shuffle=False)
                self._test_moderation_features = tf.cast(test_moderation_queue[0], tf.float32)

            if val_mf is not None:
                val_moderation_queue = tf.train.slice_input_producer([val_mf], shuffle=False)
                self._val_moderation_features = tf.cast(val_moderation_queue[0], tf.float32)

            # Calculate number of batches to run
            batches_per_epoch = self._total_training_samples / float(self._batch_size)
            self._maximum_training_batches = int(self._maximum_training_batches * batches_per_epoch)

            if self._batch_size > self._total_training_samples:
                self.__log('Less than one batch in training set, exiting now')
                exit()
            self.__log('Batches per epoch: {:f}'.format(batches_per_epoch))
            self.__log('Running to {0} batches'.format(self._maximum_training_batches))

            # Create input queues
            train_input_queue = tf.train.slice_input_producer([train_images, train_labels], shuffle=False)
            if self._testing:
                test_input_queue = tf.train.slice_input_producer([test_images, test_labels], shuffle=False)
            if self._validation:
                val_input_queue = tf.train.slice_input_producer([val_images, val_labels], shuffle=False)

            # Apply pre-processing to the image labels (which are images for semantic segmentation)
            self._train_labels = tf.image.decode_png(tf.read_file(train_input_queue[1]), channels=1)
            # normalize to 1.0
            self._train_labels = tf.image.convert_image_dtype(self._train_labels, dtype=tf.float32)
            # resize if we are using that
            if self._resize_images:
                self._train_labels = tf.image.resize_images(self._train_labels,
                                                            [self._image_height, self._image_width])
                # make into a binary mask
                self._train_labels = tf.reduce_mean(self._train_labels, axis=2)

            # if using testing and/or validation, do all the above for testing as well
            if self._testing:
                self._test_labels = tf.image.decode_png(tf.read_file(test_input_queue[1]), channels=1)
                self._test_labels = tf.image.convert_image_dtype(self._test_labels, dtype=tf.float32)
                if self._resize_images:
                    self._test_labels = tf.image.resize_images(self._test_labels,
                                                               [self._image_height, self._image_width])
                    self._test_labels = tf.reduce_mean(self._test_labels, axis=2)
            if self._validation:
                self._val_labels = tf.image.decode_png(tf.read_file(val_input_queue[1]),
                                                       channels=1)
                self._val_labels = tf.image.convert_image_dtype(self._val_labels, dtype=tf.float32)
                if self._resize_images:
                    self._val_labels = tf.image.resize_images(self._val_labels,
                                                              [self._image_height, self._image_width])
                    self._val_labels = tf.reduce_mean(self._val_labels, axis=2)

            # Apply pre-processing for training and testing images
            if image_type is 'jpg':
                self._train_images = tf.image.decode_jpeg(tf.read_file(train_input_queue[0]),
                                                          channels=self._image_depth)
                if self._testing:
                    self._test_images = tf.image.decode_jpeg(tf.read_file(test_input_queue[0]),
                                                             channels=self._image_depth)
                if self._validation:
                    self._val_images = tf.image.decode_jpeg(tf.read_file(val_input_queue[0]),
                                                            channels=self._image_depth)
            else:
                self._train_images = tf.image.decode_png(tf.read_file(train_input_queue[0]),
                                                         channels=self._image_depth)
                if self._testing:
                    self._test_images = tf.image.decode_png(tf.read_file(test_input_queue[0]),
                                                            channels=self._image_depth)
                if self._validation:
                    self._val_images = tf.image.decode_png(tf.read_file(val_input_queue[0]),
                                                           channels=self._image_depth)

            # Convert images to float and normalize to 1.0
            self._train_images = tf.image.convert_image_dtype(self._train_images, dtype=tf.float32)
            if self._testing:
                self._test_images = tf.image.convert_image_dtype(self._test_images, dtype=tf.float32)
            if self._validation:
                self._val_images = tf.image.convert_image_dtype(self._val_images, dtype=tf.float32)

            if self._resize_images:
                self._train_images = tf.image.resize_images(self._train_images,
                                                            [self._image_height, self._image_width])
                if self._testing:
                    self._test_images = tf.image.resize_images(self._test_images,
                                                               [self._image_height, self._image_width])
                if self._validation:
                    self._val_images = tf.image.resize_images(self._val_images,
                                                              [self._image_height, self._image_width])

            # Apply the various augmentations to the images
            if self._augmentation_crop:
                # Apply random crops to images
                self._image_height = int(self._image_height * self._crop_amount)
                self._image_width = int(self._image_width * self._crop_amount)

                self._train_images = tf.random_crop(self._train_images, [self._image_height, self._image_width, 3])
                if self._testing:
                    self._test_images = tf.image.resize_image_with_crop_or_pad(self._test_images, self._image_height,
                                                                               self._image_width)
                if self._validation:
                    self._val_images = tf.image.resize_image_with_crop_or_pad(self._val_images, self._image_height,
                                                                              self._image_width)

            if self._crop_or_pad_images:
                # Apply padding or cropping to deal with images of different sizes. For semantic segmentation, the
                # corresponding mask would also need to be cropped/padded
                self._train_images = tf.image.resize_image_with_crop_or_pad(self._train_images,
                                                                            self._image_height,
                                                                            self._image_width)
                self._train_labels = tf.image.resize_image_with_crop_or_pad(self._train_labels,
                                                                            self._image_height,
                                                                            self._image_width)
                if self._testing:
                    self._test_images = tf.image.resize_image_with_crop_or_pad(self._test_images,
                                                                               self._image_height,
                                                                               self._image_width)
                    self._test_labels = tf.image.resize_image_with_crop_or_pad(self._test_labels,
                                                                               self._image_height,
                                                                               self._image_width)
                if self._validation:
                    self._val_images = tf.image.resize_image_with_crop_or_pad(self._val_images,
                                                                              self._image_height,
                                                                              self._image_width)
                    self._val_labels = tf.image.resize_image_with_crop_or_pad(self._val_labels,
                                                                              self._image_height,
                                                                              self._image_width)

            if self._augmentation_flip_horizontal:
                # Apply random horizontal flips
                self._train_images = tf.image.random_flip_left_right(self._train_images)

            if self._augmentation_flip_vertical:
                # Apply random vertical flips
                self._train_images = tf.image.random_flip_up_down(self._train_images)

            if self._augmentation_contrast:
                # Apply random contrast and brightness adjustments
                self._train_images = tf.image.random_brightness(self._train_images, max_delta=63)
                self._train_images = tf.image.random_contrast(self._train_images, lower=0.2, upper=1.8)

            if self._augmentation_rotate:
                # Apply random rotations, then optionally crop out black borders and resize
                angle = tf.random_uniform([], maxval=2*math.pi)
                self._train_images = tf.contrib.image.rotate(self._train_images, angle, interpolation='BILINEAR')
                if self._rotate_crop_borders:
                    # Cropping is done using the smallest fraction possible for the image's aspect ratio to maintain a
                    # consistent scale across the images
                    small_crop_fraction = self.__smallest_crop_fraction()
                    self._train_images = tf.image.central_crop(self._train_images, small_crop_fraction)
                    self._train_images = tf.image.resize_images(self._train_images,
                                                                [self._image_height, self._image_width])

            # mean-center all inputs
            self._train_images = tf.image.per_image_standardization(self._train_images)
            if self._testing:
                self._test_images = tf.image.per_image_standardization(self._test_images)
            if self._validation:
                self._val_images = tf.image.per_image_standardization(self._val_images)

            # define the shape of the image tensors so it matches the shape of the images
            self._train_images.set_shape([self._image_height, self._image_width, self._image_depth])
            if self._testing:
                self._test_images.set_shape([self._image_height, self._image_width, self._image_depth])
            if self._validation:
                self._val_images.set_shape([self._image_height, self._image_width, self._image_depth])
