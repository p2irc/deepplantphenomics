from deepplantphenomics import loaders, layers, SemanticSegmentationModel
import tensorflow.compat.v1 as tf
import numpy as np
import os
import warnings
import numbers
import itertools
import shutil
from tqdm import tqdm, trange
from PIL import Image
import cv2
import copy


class HeatmapObjectCountingModel(SemanticSegmentationModel):
    _supported_loss_fns = ['l2', 'l1', 'smooth l1']
    _multiplier = 100.

    def __init__(self, debug=False, load_from_saved=False, save_checkpoints=True, initialize=True, tensorboard_dir=None,
                 report_rate=100, save_dir=None):
        super().__init__(debug, load_from_saved, save_checkpoints, initialize, tensorboard_dir, report_rate, save_dir)
        self._loss_fn = 'l2'

        # This is needed for reading in heatmap labels expressed as object locations, since we want to convert points
        # to gaussians when reading them in and constructing ground truth heatmaps
        self._density_sigma = 1

        # This in needed to ensure that dataset parsing in the graph is done correctly whether or not the heatmap labels
        # come from an external image or are generated
        self.__label_from_image_file = False

    def set_density_map_sigma(self, sigma):
        """
        Sets the standard deviation to use for gaussian points when generating ground truth heatmaps from object
        locations.
        :param sigma: The standard deviation to use for gaussians in generated heatmaps
        """
        if not isinstance(sigma, numbers.Real):
            raise TypeError("sigma must be a real number")

        self._density_sigma = sigma

    def _graph_problem_loss(self, pred, lab):
        heatmap_diffs = pred - lab
        if self._loss_fn == 'l2':
            return self.__l2_loss(heatmap_diffs)
        elif self._loss_fn == 'l1':
            return self.__l1_loss(heatmap_diffs)
        elif self._loss_fn == 'smooth l1':
            return self.__smooth_l1_loss(heatmap_diffs)

        raise RuntimeError("Could not calculate problem loss for a loss function of " + self._loss_fn)

    def __l2_loss(self, x):
        """
        Calculates the L2 loss of prediction difference Tensors for each item in a batch
        :param x: A Tensor with prediction differences for each item in a batch
        :return: A Tensor with the scalar L2 loss for each item
        """
        y = tf.reduce_mean(tf.square(x), axis=[1,2,3])
        return y

    def __l1_loss(self, x):
        """
        Calculates the L1 loss of prediction difference Tensors for each item in a batch
        :param x: A Tensor with prediction differences for each item in a batch
        :return: A Tensor with the scalar L1 loss for each item
        """
        y = tf.reduce_mean(tf.abs(x), axis=[1,2,3])
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

            # Initialize storage for the retrieved test variables
            all_losses = []
            all_y = []
            all_predictions = []

            # Main test loop
            for _ in tqdm(range(num_batches)):
                r_losses, r_y, r_predictions = self._session.run([self._graph_ops['test_losses'],
                                                                  self._graph_ops['y_test'],
                                                                  self._graph_ops['x_test_predicted']])
                all_losses.append(r_losses)
                all_y.append(r_y)
                all_predictions.append(r_predictions)

            all_losses = np.concatenate(all_losses, axis=0)
            all_y = np.concatenate(all_y, axis=0)
            all_predictions = np.concatenate(all_predictions, axis=0)

            # For heatmap object counting losses, like with semantic segmentation, we want relative and abs mean, std
            # of L2 norms, plus a histogram of errors
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

            self._log('Heatmap Losses:')
            self._log('Mean loss: {}'.format(mean))
            self._log('Loss standard deviation: {}'.format(std))
            self._log('Mean absolute loss: {}'.format(abs_mean))
            self._log('Absolute loss standard deviation: {}'.format(abs_std))
            self._log('Min error: {}'.format(loss_min))
            self._log('Max error: {}'.format(loss_max))
            self._log('MSE: {}'.format(mse))

            self._log('Histogram of {} losses:'.format(self._loss_fn))
            self._log(hist)

            # Specifically for heatmap object counting, we also want to determine an accuracy in terms of how the sums
            # over the predicted and ground truth heatmaps compare to each other
            heatmap_differences = [self.__heatmap_difference(all_predictions[i, ...], all_y[i, ...])
                                   for i in range(all_y.shape[0])]
            heatmap_differences = np.array(heatmap_differences)
            overall_difference = np.mean(heatmap_differences)
            self._log('Heatmap Differences: {}'.format(heatmap_differences))
            self._log('Mean Heatmap Difference: {}'.format(overall_difference))

            return overall_difference

    def __heatmap_difference(self, predict_heatmap, label_heatmap):
        """
        Calculates the difference in an image's predicted heatmap compared to its ground truth based on sums over their
        pixel values (i.e. their object count)
        :param predict_heatmap: The model's predicted heatmap as an ndarray
        :param label_heatmap: The image's corresponding label heatmap as an ndarray
        :return: The accuracy of the heatmap prediction
        """
        predicted_count = np.sum(predict_heatmap / self._multiplier)
        label_count = np.sum(label_heatmap / self._multiplier)
        return np.abs(predicted_count - label_count)

    def forward_pass_with_interpreted_outputs(self, x):
        total_outputs = super().forward_pass_with_file_inputs(x)

        # Interpreted output for heatmap counting is the sum over the heatmap pixel values, which should be the number
        # of objects
        return np.array(map(lambda i: np.sum[total_outputs[i, ...]], range(total_outputs.shape[0])))

    def load_dataset_from_directory_with_segmentation_masks(self, dirname, seg_dirname):
        """
        Loads a dataset of png images in the given directory into an internal representation, using custom heatmaps from
        another file with the same filename as the corresponding ground truth. This differs from semantic segmentation
        in that heatmaps are grayscale images while segmentation masks are binary images.
        :param dirname: The path to the directory with the image files
        :param seg_dirname: The path to the directory with the heatmap image files
        """
        # This functionality is nice but a better name would be ... better. It doesn't have a better name because we
        # want to inherit from SemanticSegmentationModel but get a loader that requires reinterpretation to be useful.
        # The ultimate solution is to refactor loader methods into mixin classes, but that will come later.
        super().load_dataset_from_directory_with_segmentation_masks(dirname, seg_dirname)
        self.__label_from_image_file = True

    def load_heatmap_dataset_with_csv_from_directory(self, dirname, label_file, ext='jpg'):
        """
        Loads in a dataset for heatmap object counting. This dataset should consist of a directory of image files to
        train on and a csv file that maps image names to multiple x and y labels (formatted like x1,y1,x2,y2,...)
        :param dirname: The path to the directory with the image files and label file
        :param label_file: The path to the csv file with heatmap point labels
        """
        # self._raw_image_files = loaders.get_dir_images(dirname)

        filename = os.path.join(dirname, label_file)
        labels, ids = loaders.read_csv_multi_labels_and_ids(filename, 0)

        self._raw_image_files = [os.path.join(dirname, id) + '.' + ext for id in ids]

        if any([len(im_labels) % 2 == 1 for im_labels in labels]):
            raise ValueError("Unpaired coordinate found in points labels from " + label_file)

        labels = loaders.csv_points_to_tuples(labels)

        if self._with_patching:
            self._raw_image_files, labels = self.__autopatch_heatmap_dataset(labels)

        heatmaps = self.__labels_to_heatmaps(labels)

        self._total_raw_samples = len(self._raw_image_files)
        self._log('Total raw examples is %d' % self._total_raw_samples)

        self._raw_labels = heatmaps
        self._split_labels = False  # Band-aid fix

    def load_heatmap_dataset_with_json_files_from_directory(self, dirname):
        """
        Loads in a dataset for heatmap object counting. This dataset should consist of a directory of image files to
        train on and JSON files that store the labels as x and y lists
        :param dirname: The path to the directory with the image files and label file
        """
        self._raw_image_files, labels = loaders.read_dataset_from_directory_with_json_labels(dirname)

        if self._with_patching:
            self._raw_image_files, labels = self.__autopatch_heatmap_dataset(labels)

        heatmaps = self.__labels_to_heatmaps(labels)

        self._total_raw_samples = len(self._raw_image_files)
        self._log('Total raw examples is %d' % self._total_raw_samples)

        self._raw_labels = heatmaps
        self._split_labels = False  # Band-aid fix

    def __labels_to_heatmaps(self, labels):
        """
        Converts point labels to heatmap labels and stores them as binary files. This will check for existing heatmaps
        first and load them if found unless data overwriting is turned on.
        :param labels: A list of lists of tuples with the point labels for each image
        :return: A list of file names for the generated heatmaps
        """
        out_dir = os.path.join(os.path.curdir, 'generated_heatmaps')
        if os.path.exists(out_dir) and not self._gen_data_overwrite:
            # If we've already generated heatmaps, just load their filenames
            im_names = [os.path.splitext(os.path.basename(f))[0] for f in self._raw_image_files]
            heatmap_files = ["{}.npy".format(os.path.join(out_dir, f)) for f in im_names]
            return heatmap_files

        if os.path.exists(out_dir):
            self._log("Overwriting preexisting heatmaps...")
            shutil.rmtree(out_dir)
        os.mkdir(out_dir)

        heatmaps = []
        for filename, coords in zip(self._raw_image_files, labels):
            if len(coords) > 0:
                heatmap = self.__points_to_density_map(coords)
            else:
                # There are no points, so the heatmap is blank
                heatmap = np.full([self._image_height, self._image_width, 1], 0, dtype=np.float32)

            heatmap_file = self.__save_heatmap_as_binary(heatmap, os.path.splitext(os.path.basename(filename))[0],
                                                         out_dir=out_dir)
            heatmaps.append(heatmap_file)

        return heatmaps

    def __save_heatmap_as_binary(self, heatmap, filename, out_dir=None):
        """
        Saves a floating-point heatmap array as a binary .npy file for later use in training
        :param heatmap: A float ndarray with a heatmap
        :param filename: The filename to save the heatmap array with, excluding the extension
        :return: The file path for later use in reloading the array
        """
        if out_dir is None:
            out_dir = os.path.curdir
        out_name = os.path.join(out_dir, '{}.npy'.format(filename))

        np.save(out_name, heatmap)
        return out_name

    def __points_to_density_map(self, points):
        """
        Convert point labels for a heatmap into a grayscale image with a gaussian placed at heatmap points
        :param points: A list of (x,y) tuples for object locations in an image
        :return: An ndarray of the heatmap image
        """

        output_img = np.zeros([self._image_height, self._image_width], dtype=np.float32)

        diameter = int(self._density_sigma * 6)
        radius = diameter / 2

        h = self._image_height
        w = self._image_width

        gauss = cv2.getGaussianKernel(diameter, self._density_sigma)
        gauss2d = gauss * gauss.T
        gauss2d = (gauss2d / np.sum(gauss2d)) * self._multiplier

        for (x, y) in points:
            gx1 = 0
            gx2 = diameter

            gy1 = 0
            gy2 = diameter

            if x - radius < 0:
                x1 = 0
                gx1 = int(radius - x)
            else:
                x1 = int(x - radius)

            if x + radius > w:
                x2 = int(w)
                gx2 = int(radius + (w - x))
            else:
                x2 = int(x + radius)

            if y - radius < 0:
                y1 = 0
                gy1 = int(radius - y)
            else:
                y1 = int(y - radius)

            if y + radius > h:
                y2 = h
                gy2 = int(radius + (h - y))
            else:
                y2 = int(y + radius)

            output_img[y1:y2, x1:x2] += gauss2d[gy1:gy2, gx1:gx2]

        return np.expand_dims(output_img, -1)

    def __autopatch_heatmap_dataset(self, labels, patch_dir=None):
        """
        Generates a dataset of image patches from a loaded dataset of larger images and returns the new images and
        labels. This will check for existing patches first and load them if found unless data overwriting is turned on.
        :param labels: A nested list of point tuple labels for the original images (i.e. [[(x,y), (x,y), ...], ...]
        :param patch_dir: The directory to place patched images into, or where to read previous patches from
        :return: The patched dataset as a list of image filenames and a nested list of their corresponding point labels
        """
        if not patch_dir:
            patch_dir = os.path.curdir
        patch_dir = os.path.join(patch_dir, 'train_patch', '')
        im_dir = patch_dir
        point_file = os.path.join(patch_dir, 'patch_point_labels.csv')

        if os.path.exists(patch_dir) and not self._gen_data_overwrite:
            # If there already is a patched dataset, just load it
            self._log("Loading preexisting patched data from " + patch_dir)
            #image_files = loaders.get_dir_images(im_dir)
            new_labels, ids = loaders.read_csv_multi_labels_and_ids(point_file, 0)

            image_files = [os.path.join(patch_dir, id) + '.png' for id in ids]

            new_labels = loaders.csv_points_to_tuples(new_labels)
            return image_files, new_labels

        self._log("Patching dataset: Patches will be in " + patch_dir)
        if os.path.exists(patch_dir):
            self._log("Overwriting preexisting patched data...")
            shutil.rmtree(patch_dir)
        os.mkdir(patch_dir)

        # We need to construct patches from the previously loaded dataset. We'll take as many of them as we can fit
        # from the centre of the image, though at the risk of excluding any points that get cut off at the edges.
        patch_num = 0
        n_image = len(self._raw_image_files)
        image_files = []
        new_labels = []
        out_labels = []
        label_str = []
        for n, im_file, im_labels in zip(trange(n_image), self._raw_image_files, labels):
            im = np.array(Image.open(im_file))

            def place_points_in_patches(tl_corner, br_corner, points):
                # The slow, O(mn) way
                for (py0, px0), (py1, px1) in zip(tl_corner, br_corner):
                    points_in_patch = [(x - px0, y - py0) for (x, y) in points if py0 <= y < py1 and px0 <= x < px1]
                    serial_points = [c for p in points_in_patch for c in p]  # Convert (x,y) tuples to flat x,y list
                    new_labels.append(points_in_patch)
                    out_labels.append(serial_points)

            patch_start, patch_end = self._autopatch_get_patch_coords(im)
            num_patch = len(patch_start)
            place_points_in_patches(patch_start, patch_end, im_labels)

            for i, tl_coord, br_coord in zip(itertools.count(patch_num), patch_start, patch_end):
                im_patch = Image.fromarray(self._autopatch_extract_patch(im, tl_coord, br_coord))
                im_name = os.path.join(im_dir, 'im_{:0>6d}.png'.format(i))
                im_patch.save(im_name)
                image_files.append(im_name)

                label_str.append('im_{:0>6d},'.format(i) + ','.join([str(x) for x in out_labels[i]]))

            patch_num += num_patch

        with open(point_file, 'w') as f:
            for line in label_str:
                f.write(line + '\n')

        return image_files, new_labels

    def _parse_load_heatmap_binary(self, filename):
        return np.load(filename)

    def _parse_apply_preprocessing(self, images, labels):
        if not self.__label_from_image_file:
            # If we generated the heatmaps from points in a CSV or JSON file, then we want to treat the labels like
            # other labels, with the wrinkle that loading them requires wrapping a binary loader with tf.py_func
            images = self._parse_read_images(images, channels=self._image_depth)
            labels = tf.numpy_function(self._parse_load_heatmap_binary, [labels], tf.float32)
            return images, labels
        else:
            # If we instead read in the heatmaps as images, then we want to use the version in
            # SemanticSegmentationModel, which treats the labels like regular images.
            return super()._parse_apply_preprocessing(images, labels)

    def _parse_resize_images(self, images, labels, height, width):
        # See _parse_apply_preprocessing for an explanation of whats going on here
        if not self.__label_from_image_file:
            # Skip over the version in SemanticSegmentationModel to use the one in DPPModel
            return super(SemanticSegmentationModel, self)._parse_resize_images(images, labels, height, width)
        else:
            return super()._parse_resize_images(images, labels, height, width)

    def _parse_crop_or_pad(self, images, labels, height, width):
        # See _parse_apply_preprocessing for an explanation of whats going on here
        if not self.__label_from_image_file:
            # Skip over the version in SemanticSegmentationModel to use the one in DPPModel
            return super(SemanticSegmentationModel, self)._parse_crop_or_pad(images, labels, height, width)
        else:
            return super()._parse_crop_or_pad(images, labels, height, width)

    def _parse_force_set_shape(self, images, labels, height, width, depth):
        # See _parse_apply_preprocessing for an explanation of whats going on here
        if not self.__label_from_image_file:
            # Skip over the version in SemanticSegmentationModel to use the one in DPPModel
            return super(SemanticSegmentationModel, self)._parse_force_set_shape(images, labels, height, width, depth)
        else:
            return super()._parse_force_set_shape(images, labels, height, width, depth)

    def add_output_layer(self, regularization_coefficient=None, output_size=None):
        if len(self._layers) < 1:
            raise RuntimeError("An output layer cannot be the first layer added to the model. " +
                               "Add an input layer with DPPModel.add_input_layer() first.")
        if regularization_coefficient is not None:
            warnings.warn("Heatmap counter doesn't use regularization_coefficient in its output layer")
        if output_size is not None:
            raise RuntimeError("output_size should be None for heatmap counting")

        self._log('Adding output layer...')

        filter_dimension = [1, 1, copy.deepcopy(self._last_layer().output_size[3]), 1]

        with self._graph.as_default():
            layer = layers.convLayer('output',
                                     copy.deepcopy(self._last_layer().output_size),
                                     filter_dimension,
                                     1,
                                     None,
                                     self._weight_initializer,
                                     use_bias=False)

        self._log('Inputs: {0} Outputs: {1}'.format(layer.input_size, layer.output_size))
        self._layers.append(layer)
