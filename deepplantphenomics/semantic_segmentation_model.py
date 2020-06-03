from . import loaders, layers, definitions, DPPModel
import numpy as np
import tensorflow.compat.v1 as tf
import os
import warnings
import copy
import itertools
import shutil
from math import ceil
from tqdm import tqdm, trange
from PIL import Image


class SemanticSegmentationModel(DPPModel):
    _supported_loss_fns = ['sigmoid cross entropy', 'softmax cross entropy']
    _supported_augmentations = [definitions.AugmentationType.CONTRAST_BRIGHT]

    def __init__(self, debug=False, load_from_saved=False, save_checkpoints=True, initialize=True, tensorboard_dir=None,
                 report_rate=100, save_dir=None):
        super().__init__(debug, load_from_saved, save_checkpoints, initialize, tensorboard_dir, report_rate, save_dir)
        self._loss_fn = 'sigmoid cross entropy'
        self._num_seg_class = 2

        # State variables specific to semantic segmentation for constructing the graph and passing to Tensorboard
        self._graph_forward_pass = None

    def set_num_segmentation_classes(self, num_class):
        """
        Sets the number of classes to segment images into
        :param num_class: The number of segmentation classes
        """
        if not isinstance(num_class, int):
            raise TypeError("num must be an int")
        if num_class < 2:
            raise ValueError("Semantic segmentation requires at least 2 different classes")

        self._num_seg_class = num_class
        if num_class == 2:
            self._loss_fn = 'sigmoid cross entropy'
        else:
            self._loss_fn = 'softmax cross entropy'

    def _graph_tensorboard_summary(self, l2_cost, gradients, variables, global_grad_norm):
        super()._graph_tensorboard_common_summary(l2_cost, gradients, variables, global_grad_norm)

        # Summaries specific to semantic segmentation
        # We send in the last layer's output size (i.e. the final image dimensions) to get_weights_as_image
        # because xx and x_test_predicted have dynamic dims [?,?,?,?], so we need actual numbers passed in

        tf.summary.image('masks/train', self._graph_forward_pass, collections=['custom_summaries'])

        tf.summary.image('masks/target', self._graph_target, collections=['custom_summaries'])

        tf.summary.image('input_image', self._graph_input, collections=['custom_summaries'])

        if self._validation:
            tf.summary.scalar('validation/loss', self._graph_ops['val_cost'], collections=['custom_summaries'])

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
                    self._graph_forward_pass = xx  # Needed to output raw forward pass output to Tensorboard
                    self._graph_input = x
                    self._graph_target = y

                    # Define regularization cost
                    self._log('Graph: Calculating loss and gradients...')
                    l2_cost = self._graph_layer_loss()

                    # Define cost function based on which one was selected via set_loss_function
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
            self._graph_ops['cost'] = tf.reduce_sum(device_costs) / self._batch_size + l2_cost

            # Calculate test  and validation accuracy (on a single device at Tensorflow's discretion)
            # # If using patching, we need to properly pull similar patches from the test and validation images (and
            # # labels)
            # if self._with_patching:
            #     if self._testing:
            #         x_test, _ = self._graph_extract_patch(x_test, offsets)
            #         self._graph_ops['y_test'], _ = self._graph_extract_patch(self._graph_ops['y_test'], offsets)
            #     if self._validation:
            #         x_val, _ = self._graph_extract_patch(x_val, offsets)
            #         self._graph_ops['y_val'], _ = self._graph_extract_patch(self._graph_ops['y_val'], offsets)

            if self._testing:
                x_test, self._graph_ops['y_test'] = test_iter.get_next()

                if self._has_moderation:
                    mod_w_test = test_mod_iter.get_next()
                    self._graph_ops['x_test_predicted'] = self.forward_pass(x_test, deterministic=True,
                                                                            moderation_features=mod_w_test)
                else:
                    self._graph_ops['x_test_predicted'] = self.forward_pass(x_test, deterministic=True)

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

                self._graph_ops['val_losses'] = self._graph_problem_loss(self._graph_ops['x_val_predicted'],
                                                                         self._graph_ops['y_val'])

                self._graph_ops['val_cost'] = tf.reduce_mean(tf.abs(self._graph_ops['val_losses']))

            # Epoch summaries for Tensorboard
            if self._tb_dir is not None:
                self._graph_tensorboard_summary(l2_cost, gradients, variables, global_grad_norm)

    def _graph_problem_loss(self, pred, lab):
        if self._loss_fn == 'sigmoid cross entropy':
            if self._num_seg_class != 2:
                raise RuntimeError("Sigmoid cross entropy only applies to binary semantic segmentation (i.e. with 2 "
                                   "classes)")
            sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=lab)
            return tf.squeeze(tf.reduce_mean(sigmoid_loss, axis=[1, 2]), axis=1)
        elif self._loss_fn == 'softmax cross entropy':
            if self._num_seg_class <= 2:
                raise RuntimeError("Softmax cross entropy only applies to multi-class semantic segmentation (i.e. with "
                                   "3+ classes classes)")
            lab = tf.cast(tf.squeeze(lab, axis=3), tf.int32)
            pixel_softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=lab)
            return tf.reduce_mean(pixel_softmax_loss, axis=[1, 2])

        raise RuntimeError("Could not calculate problem loss for a loss function of " + self._loss_fn)

    def compute_full_test_accuracy(self):
        self._log('Computing total test accuracy/regression loss...')

        with self._graph.as_default():
            num_batches = int(np.ceil(self._total_testing_samples / self._batch_size))

            if num_batches == 0:
                warnings.warn('Less than a batch of testing data')
                exit()

            # Initialize storage for the retrieved test variables
            all_losses = []

            # Main test loop
            for _ in tqdm(range(num_batches)):
                r_losses = self._session.run(self._graph_ops['test_losses'])
                all_losses.append(r_losses)

            all_losses = np.concatenate(all_losses, axis=0)

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

            self._log('Mean loss: {}'.format(mean))
            self._log('Loss standard deviation: {}'.format(std))
            self._log('Mean absolute loss: {}'.format(abs_mean))
            self._log('Absolute loss standard deviation: {}'.format(abs_std))
            self._log('Min error: {}'.format(loss_min))
            self._log('Max error: {}'.format(loss_max))
            self._log('MSE: {}'.format(mse))

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

            if self._with_patching:
                # Processing and returning whole images is more important than preventing erroneous results from
                # padding, so we the image size with the required padding to accommodate the patch size
                patch_height = self._patch_height
                patch_width = self._patch_width
                num_patch_rows = ceil(self._image_height / patch_height)
                num_patch_cols = ceil(self._image_width / patch_width)
                final_height = num_patch_rows * patch_height
                final_width = num_patch_cols * patch_width

                # Apply any padding to the images if, then extract the patches. Padding is only added to the bottom and
                # right sides.
                x_test = tf.image.pad_to_bounding_box(x_test, 0, 0, final_height, final_width)
                sizes = [1, patch_height, patch_width, 1]
                strides = [1, patch_height, patch_width, 1]  # Same as sizes in order to tightly tile patches
                rates = [1, 1, 1, 1]
                x_test = tf.image.extract_image_patches(x_test, sizes=sizes, strides=strides, rates=rates,
                                                        padding="VALID")
                x_test = tf.reshape(x_test, shape=[-1, patch_height, patch_width, self._image_depth])

            # Run model on them
            x_pred = self.forward_pass(x_test, deterministic=True)

            total_outputs = []
            if self._with_patching:
                n_patches = num_patch_rows * num_patch_cols
                for i in range(num_batches):
                    xx = self._session.run(x_pred)

                    for img_patches in np.array_split(xx, xx.shape[0] / n_patches):
                        # Stitch individual rows together, than stitch the rows into a full image
                        full_img = []
                        for row_of_patches in np.array_split(img_patches, num_patch_rows):
                            row_patches = [row_of_patches[i] for i in range(num_patch_cols)]
                            full_img.append(np.concatenate(row_patches, axis=1))
                        full_img = np.concatenate(full_img, axis=0)

                        # Trim off any padding that was added
                        full_img = full_img[0:self._image_height, 0:self._image_width, :]

                        # Keep the final image, but with an extra dimension to concatenate the images together
                        total_outputs.append(np.expand_dims(full_img, axis=0))
            else:
                for i in range(num_batches):
                    xx = self._session.run(x_pred)
                    for img_patches in np.array_split(xx, xx.shape[0]):
                        total_outputs.append(img_patches)

            total_outputs = np.concatenate(total_outputs, axis=0)

        return total_outputs

    def forward_pass_with_interpreted_outputs(self, x):
        total_outputs = self.forward_pass_with_file_inputs(x)

        if self._num_seg_class == 2:
            # Get a binary mask for each image by normalizing and thresholding them
            interpreted_outputs = np.zeros(total_outputs.shape, dtype=np.uint8)
            for i, img in enumerate(total_outputs):
                x_min = np.min(img)
                x_max = np.max(img)
                mask = (img - x_min) / (x_max - x_min)
                mask[mask >= 0.5] = 255
                mask[mask < 0.5] = 0
                interpreted_outputs[i, :, :] = mask

            return interpreted_outputs
        else:
            # Apply a softmax to the outputs and find the appropriate per-pixel class from the highest probability
            total_outputs = np.exp(total_outputs) / np.sum(np.exp(total_outputs), axis=3, keepdims=True)
            total_outputs = np.argmax(total_outputs, axis=3)
            return total_outputs

    def add_output_layer(self, regularization_coefficient=None, output_size=None):
        if len(self._layers) < 1:
            raise RuntimeError("An output layer cannot be the first layer added to the model. " +
                               "Add an input layer with DPPModel.add_input_layer() first.")
        if regularization_coefficient is not None:
            warnings.warn("Semantic segmentation doesn't use regularization_coefficient in its output layer")
        if output_size is not None:
            raise RuntimeError("output_size should be None for semantic segmentation")

        self._log('Adding output layer...')

        if self._num_seg_class == 2:
            filter_dimension = [1, 1, copy.deepcopy(self._last_layer().output_size[3]), 1]
        else:
            filter_dimension = [1, 1, copy.deepcopy(self._last_layer().output_size[3]), self._num_seg_class]

        with self._graph.as_default():
            layer = layers.convLayer('output',
                                     copy.deepcopy(self._last_layer().output_size),
                                     filter_dimension,
                                     1,
                                     None,
                                     self._weight_initializer)

        self._log('Inputs: {0} Outputs: {1}'.format(layer.input_size, layer.output_size))
        self._layers.append(layer)

    def load_dataset_from_directory_with_segmentation_masks(self, dirname, seg_dirname):
        """
        Loads the png images in the given directory into an internal representation, using binary segmentation
        masks from another file with the same filename as ground truth.

        :param dirname: the path of the directory containing the images
        :param seg_dirname: the path of the directory containing ground-truth binary segmentation masks
        """
        self._raw_image_files = loaders.get_dir_images(dirname)
        self._raw_labels = loaders.get_dir_images(seg_dirname)
        if self._with_patching:
            self._raw_image_files, self._raw_labels = self.__autopatch_segmentation_dataset()

        self._total_raw_samples = len(self._raw_image_files)
        self._log('Total raw examples is %d' % self._total_raw_samples)

        self._split_labels = False  # Band-aid fix

    def __autopatch_segmentation_dataset(self, patch_dir=None):
        """
        Generates a dataset of image patches from a loaded dataset of larger images and returns the new images and
        labels. This will check for existing patches first and load them if found unless data overwriting is turned on.
        :param patch_dir: The directory to place patched images into, or where to read previous patches from
        :return The patched dataset as lists of the image and segmentation mask filenames
        """
        if not patch_dir:
            patch_dir = os.path.curdir
        patch_dir = os.path.join(patch_dir, 'train_patch', '')
        im_dir = os.path.join(patch_dir, 'im_patch', '')
        seg_dir = os.path.join(patch_dir, 'mask_patch', '')

        if os.path.exists(patch_dir) and not self._gen_data_overwrite:
            # If there already is a patched dataset, just load it
            self._log("Loading preexisting patched data from " + patch_dir)
            image_files = loaders.get_dir_images(im_dir)
            seg_files = loaders.get_dir_images(seg_dir)
            return image_files, seg_files

        self._log("Patching dataset: Patches will be in " + patch_dir)
        if os.path.exists(patch_dir):
            self._log("Overwriting preexisting patched data...")
            shutil.rmtree(patch_dir)
        os.mkdir(patch_dir)
        os.mkdir(im_dir)
        os.mkdir(seg_dir)

        # We need to construct patches from the previously loaded dataset. We'll take as many of them as we can fit
        # from the centre of the image.
        patch_num = 0
        n_image = len(self._raw_image_files)
        image_files = []
        seg_files = []
        for n, im_file, seg_file in zip(trange(n_image), self._raw_image_files, self._raw_labels):
            im = np.array(Image.open(im_file))
            seg = np.array(Image.open(seg_file))

            patch_start, patch_end = self._autopatch_get_patch_coords(im)
            num_patch = len(patch_start)

            for i, tl_coord, br_coord in zip(itertools.count(patch_num), patch_start, patch_end):
                im_patch = Image.fromarray(self._autopatch_extract_patch(im, tl_coord, br_coord))
                seg_patch = Image.fromarray(self._autopatch_extract_patch(seg, tl_coord, br_coord))
                im_name = os.path.join(im_dir, 'im_{:0>6d}.png'.format(i))
                seg_name = os.path.join(seg_dir, 'seg_{:0>6d}.png'.format(i))
                im_patch.save(im_name)
                seg_patch.save(seg_name)
                image_files.append(im_name)
                seg_files.append(seg_name)

            patch_num += num_patch

        return image_files, seg_files

    def _autopatch_get_patch_coords(self, im):
        """
        Gets the starting (top-left) and ending (bottom-right) coordinates for splitting an image into patches. Patches
        are taken starting from the top and left edges of the image and continue, padding the bottom and right images
        with black if they go over the edge.
        :param im: A numpy array with an image to split into patches
        :return: Lists of tuples with the starting (top-left) and ending (bottom-right) coordinates for patches
        """
        im_height, im_width, _ = im.shape
        num_patch_h = ceil(im_height / self._patch_height)
        num_patch_w = ceil(im_width / self._patch_width)

        patch_start = [(y * self._patch_height, x * self._patch_width)
                       for y in range(num_patch_h) for x in range(num_patch_w)]
        patch_end = [(y + self._patch_height, x + self._patch_width) for (y, x) in patch_start]

        return patch_start, patch_end

    def _autopatch_extract_patch(self, im, tl_coord, br_coord):
        """
        Extracts a patch from an image, padding it with black if it extends over the edge
        :param im: An ndarray for the image to extract the patch from
        :param tl_coord: A tuple for the top-left (y, x) corner of the patch
        :param br_coord: A tuple for the bottom-right (y, x) corner of the patch
        :return: An ndarray of the extracted patch suitable for saving as a PNG
        """
        y0, x0 = tl_coord
        y1, x1 = br_coord
        patch_x = x1 - x0
        patch_y = y1 - y0
        if im.ndim == 2:
            im = np.expand_dims(im, axis=-1)  # Give 2D images an explicit 1-channel dimension
        im_height, im_width, im_depth = im.shape

        fill_x = x1 - im_width if x1 > im_width else 0
        fill_y = y1 - im_height if y1 > im_height else 0
        if x1 > im_width:
            x1 -= fill_x
        if y1 > im_height:
            y1 -= fill_y

        im_patch = np.full((patch_y, patch_x, im_depth), 0, dtype=np.uint8)
        im_patch[0:patch_y - fill_y, 0:patch_x - fill_x, :] = im[y0:y1, x0:x1, :].astype(np.uint8)

        if im_depth == 1:
            return im_patch.squeeze(axis=2)  # Remove the 1-channel dimension; some image libraries don't like it
        return im_patch

    def _parse_apply_preprocessing(self, images, labels):
        # Apply pre-processing to the image labels too (which are images for semantic segmentation). If there are
        # multiples classes encoded as 0, 1, 2, ..., we want to maintain the read-in uint8 type and do a simple cast
        # to float32 instead of a full image type conversion to prevent value scaling.
        images = self._parse_read_images(images, channels=self._image_depth)
        if self._num_seg_class > 2:
            labels = self._parse_read_images(labels, channels=1, image_type=tf.uint8)
            labels = tf.cast(labels, tf.float32)
        else:
            labels = self._parse_read_images(labels, channels=1)
        return images, labels

    def _parse_resize_images(self, images, labels, height, width):
        images = tf.image.resize_images(images, [height, width])
        labels = tf.image.resize_images(labels, [height, width])
        return images, labels

    def _parse_crop_or_pad(self, images, labels, height, width):
        images = tf.image.resize_image_with_crop_or_pad(images, height, width)
        labels = tf.image.resize_image_with_crop_or_pad(labels, height, width)
        return images, labels

    def _parse_force_set_shape(self, images, labels, height, width, depth):
        images.set_shape([height, width, depth])
        labels.set_shape([height, width, 1])
        return images, labels
