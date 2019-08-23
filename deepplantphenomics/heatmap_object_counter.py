from deepplantphenomics import definitions, loaders, SemanticSegmentationModel
import numpy as np
import os
import warnings
import numbers
from tqdm import tqdm
from scipy.ndimage import gaussian_filter


class HeatmapObjectCountingModel(SemanticSegmentationModel):
    _problem_type = definitions.ProblemType.HEATMAP_COUNTING

    def __init__(self, debug=False, load_from_saved=False, save_checkpoints=True, initialize=True, tensorboard_dir=None,
                 report_rate=100, save_dir=None):
        super().__init__(debug, load_from_saved, save_checkpoints, initialize, tensorboard_dir, report_rate, save_dir)

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

    def compute_full_test_accuracy(self):
        self._log('Computing total test accuracy/regression loss...')

        with self._graph.as_default():
            num_batches = int(np.ceil(self._total_testing_samples / self._batch_size))

            if num_batches == 0:
                warnings.warn('Less than a batch of testing data')
                exit()

            # Initialize storage for the retrieved test variables
            all_losses = np.empty(shape=1)
            all_y = np.empty(shape=[self._batch_size, self._image_height, self._image_width])
            all_predictions = np.empty(shape=[self._batch_size, self._image_height, self._image_width])

            # Main test loop
            for _ in tqdm(range(num_batches)):
                r_losses, r_y, r_predictions = self._session.run([self._graph_ops['test_losses'],
                                                                  self._graph_ops['y_test'],
                                                                  self._graph_ops['x_test_predicted']])
                all_losses = np.concatenate((all_losses, r_losses), axis=0)
                all_y = np.concatenate((all_y, r_y), axis=0)
                all_predictions = np.concatenate((all_predictions, r_predictions), axis=0)

            all_losses = np.delete(all_losses, 0)
            all_y = np.delete(all_y, 0, axis=0)
            all_predictions = np.delete(all_predictions, 0, axis=0)

            # Delete the extra entries (e.g. batch_size is 4 and 1 sample left, it will loop and have 3 repeats that
            # we want to get rid of)
            extra = self._batch_size - (self._total_testing_samples % self._batch_size)
            if extra != self._batch_size:
                mask_extra = np.ones(self._batch_size * num_batches, dtype=bool)
                mask_extra[range(self._batch_size * num_batches - extra, self._batch_size * num_batches)] = False
                all_losses = all_losses[mask_extra, ...]
                all_y = all_y[mask_extra, ...]
                all_predictions = all_predictions[mask_extra, ...]

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
            heatmap_accuracies = np.array(map(
                lambda i: self.__heatmap_accuracy(all_predictions[i, ...], all_y[i, ...]),
                range(all_y.shape[0])))
            overall_accuracy = np.mean(heatmap_accuracies)
            self._log('Heatmap Accuracies: {}'.format(heatmap_accuracies))
            self._log('Heatmap Accuracies: {}'.format(overall_accuracy))

            return overall_accuracy

    def __heatmap_accuracy(self, predict_heatmap, label_heatmap):
        """
        Calculates the accuracy of an image's predicted heatmap compared to its ground truth based on sums over their
        pixel values
        :param predict_heatmap: The model's predicted heatmap as an ndarray
        :param label_heatmap: The image's corresponding label heatmap as an ndarray
        :return: The accuracy of the heatmap prediction
        """
        return np.abs(np.sum(predict_heatmap) - np.sum(label_heatmap))

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

    def load_heatmap_dataset_from_csv(self, dirname, label_file):
        """
        Loads in a dataset for heatmap object counting. This dataset should consist of a directory of image files to
        train on and a csv file that maps image names to multiple x and y labels (formatted like x1,y1,x2,y2,...)
        :param dirname: The path to the directory with the image files and label file
        :param label_file: The path to the csv file with heatmap point labels
        """
        labels, ids = loaders.read_csv_multi_labels_and_ids(label_file, 0)

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
                heatmaps.append(np.full([self._image_height, self._image_width], 0))

        image_files = [os.path.join(dirname, filename) for filename in ids]
        self._total_raw_samples = len(image_files)
        self._log('Total raw examples is %d' % self._total_raw_samples)

        self._raw_image_files = image_files
        self._raw_labels = labels
        self._split_labels = False  # Band-aid fix

    def __points_to_density_map(self, points):
        """
        Convert point labels for a heatmap into a grayscale image with a gaussian placed at heatmap points
        :param points: A list of (x,y) tuples for object locations in an image
        :return: An ndarray of the heatmap image
        """
        # The simple way to do this is to place single pixel 1's on a blank image at each point and then apply a
        # gaussian filter to that image. The result is our density map.
        den_map = np.empty([self._image_height, self._image_width])
        for p in points:
            den_map[p[0], p[1]] = 1
        den_map = gaussian_filter(den_map, self._density_sigma, mode='constant')

        return den_map

    def _parse_apply_preprocessing(self, test_input_queue, train_input_queue, val_input_queue):
        # This is tricky. If we read in the heatmaps as images, then we want to use the version in
        # SemanticSegmentationModel, which treats the labels like regular images. If we instead generated the heatmaps
        # from points in a CSV file, then we want to treat the labels like other labels, which the version in DPPModel
        # does. super() will let us choose which one by making it look through this or Semantic...Model's MRO.
        if self.__label_from_image_file:
            super(SemanticSegmentationModel, self)._parse_apply_preprocessing(
                test_input_queue, train_input_queue, val_input_queue)
        else:
            super()._parse_apply_preprocessing(test_input_queue, train_input_queue, val_input_queue)
