from deepplantphenomics import loaders, SemanticSegmentationModel
import tensorflow as tf
import numpy as np
import os
import warnings
import numbers
import itertools
from tqdm import tqdm, trange
from scipy.ndimage import gaussian_filter
from PIL import Image


class HeatmapObjectCountingModel(SemanticSegmentationModel):
    _supported_loss_fns = ['l2', 'l1', 'smooth l1']

    def __init__(self, debug=False, load_from_saved=False, save_checkpoints=True, initialize=True, tensorboard_dir=None,
                 report_rate=100, save_dir=None):
        super().__init__(debug, load_from_saved, save_checkpoints, initialize, tensorboard_dir, report_rate, save_dir)
        self._loss_fn = 'l2'

        # This is needed for reading in heatmap labels expressed as object locations, since we want to convert points
        # to gaussians when reading them in and constructing ground truth heatmaps
        self._density_sigma = 5

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
        predicted_count = np.sum(predict_heatmap)
        label_count = np.sum(label_heatmap)
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

    def load_heatmap_dataset_with_csv_from_directory(self, dirname, label_file):
        """
        Loads in a dataset for heatmap object counting. This dataset should consist of a directory of image files to
        train on and a csv file that maps image names to multiple x and y labels (formatted like x1,y1,x2,y2,...)
        :param dirname: The path to the directory with the image files and label file
        :param label_file: The path to the csv file with heatmap point labels
        """
        self._raw_image_files = loaders.get_dir_images(dirname)

        filename = os.path.join(dirname, label_file)
        labels, ids = loaders.read_csv_multi_labels_and_ids(filename, 0)
        labels = [list(map(int, x)) for x in labels]

        if self._with_patching:
            self._raw_image_files, labels = self.__autopatch_heatmap_dataset(labels)

        # The labels are [x1,y1,x2,y2,...] points, which we need to turn into (x,y) tuples and use to generate the
        # ground truth heatmap
        heatmaps = []
        for coords in labels:
            if coords:
                if len(coords) % 2 == 1:
                    # There is an odd number of coordinates, which is problematic
                    raise ValueError("Unpaired coordinate found in points labels from " + label_file)

                points = zip(coords[0::2], coords[1::2])
                heatmaps.append(self.__points_to_density_map(points))
            else:
                # There are no objects, so the heatmap is blank
                heatmaps.append(np.full([self._image_height, self._image_width, 1], 0, dtype=np.float32))

        self._total_raw_samples = len(self._raw_image_files)
        self._log('Total raw examples is %d' % self._total_raw_samples)

        heatmaps = np.stack(heatmaps)
        self._raw_labels = heatmaps
        self._split_labels = False  # Band-aid fix

    def __points_to_density_map(self, points):
        """
        Convert point labels for a heatmap into a grayscale image with a gaussian placed at heatmap points
        :param points: A list of (x,y) tuples for object locations in an image
        :return: An ndarray of the heatmap image
        """
        # The simple way to do this is to place single pixel 1's on a blank image at each point and then apply a
        # gaussian filter to that image. The result is our density map.
        den_map = np.zeros([self._image_height, self._image_width, 1], dtype=np.float32)
        for (x, y) in points:
            den_map[y, x, 0] = 1
        den_map = gaussian_filter(den_map, self._density_sigma, mode='constant')

        return den_map

    def __autopatch_heatmap_dataset(self, labels, patch_dir=None):
        """
        Generates a dataset of image patches from a loaded dataset of larger images, or simply sets the dataset to a
        set of patches made previously
        :param labels: A nested list of point labels for the original images
        :param patch_dir: The directory to place patched images into, or where to read previous patches from
        :return: The patched dataset as a list of image filenames and a nested list of their corresponding point labels
        """
        if not patch_dir:
            patch_dir = os.path.curdir
        patch_dir = os.path.join(patch_dir, 'train_patch', '')
        im_dir = patch_dir
        point_file = os.path.join(patch_dir, 'patch_point_labels.csv')

        if os.path.exists(patch_dir):
            # If there already is a patched dataset, just load it
            self._log("Loading preexisting patched data from " + patch_dir)
            image_files = loaders.get_dir_images(im_dir)
            new_labels, _ = loaders.read_csv_multi_labels_and_ids(point_file, 0)
            new_labels = [list(map(int, x)) for x in new_labels]
            return image_files, new_labels

        self._log("Patching dataset: Patches will be in " + patch_dir)
        os.mkdir(patch_dir)

        # We need to construct patches from the previously loaded dataset. We'll take as many of them as we can fit
        # from the centre of the image, though at the risk of excluding any points that get cut off at the edges.
        patch_num = 0
        n_image = len(self._raw_image_files)
        image_files = []
        new_labels = []
        label_str = []
        for n, im_file, im_labels in zip(trange(n_image), self._raw_image_files, labels):
            im = np.array(Image.open(im_file))

            def place_points_in_patches(tl_coord, br_coord, serial_points):
                # The slow, O(mn) way
                points = list(zip(serial_points[0::2], serial_points[1::2]))
                for (py0, px0), (py1, px1) in zip(tl_coord, br_coord):
                    points_in_patch = [(x - px0, y - py0) for (x, y) in points if py0 <= y < py1 and px0 <= x < px1]
                    points_in_patch = [c for p in points_in_patch for c in p]  # Convert (x,y) tuples to flat x,y list
                    new_labels.append(points_in_patch)

            patch_start, patch_end = self._autopatch_get_patch_coords(im)
            num_patch = len(patch_start)
            place_points_in_patches(patch_start, patch_end, im_labels)

            for i, (y0, x0), (y1, x1), patch_labels in zip(itertools.count(patch_num),
                                                           patch_start, patch_end, new_labels):
                im_patch = Image.fromarray(im[y0:y1, x0:x1].astype(np.uint8))
                im_name = os.path.join(im_dir, 'im_{:0>6d}.png'.format(i))
                im_patch.save(im_name)
                image_files.append(im_name)

                label_str.append('im_{:0>6d},'.format(i) + ','.join([str(x) for x in patch_labels]))

            patch_num += num_patch

        with open(point_file, 'w') as f:
            for line in label_str:
                f.write(line + '\n')

        return image_files, new_labels

    def _parse_apply_preprocessing(self, images, labels):
        # This is tricky. If we read in the heatmaps as images, then we want to use the version in
        # SemanticSegmentationModel, which treats the labels like regular images. If we instead generated the heatmaps
        # from points in a CSV file, then we want to treat the labels like other labels, which the version in DPPModel
        # does. super() will let us choose which one by making it look through this or Semantic...Model's MRO.
        if not self.__label_from_image_file:
            # Skip over the version in SemanticSegmentationModel to use the one in DPPModel
            return super(SemanticSegmentationModel, self)._parse_apply_preprocessing(images, labels)
        else:
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
