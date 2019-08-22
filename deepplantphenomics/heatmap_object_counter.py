from deepplantphenomics import definitions, SemanticSegmentationModel


class HeatmapObjectCountingModel(SemanticSegmentationModel):
    _problem_type = definitions.ProblemType.HEATMAP_COUNTING

    def __init__(self, debug=False, load_from_saved=False, save_checkpoints=True, initialize=True, tensorboard_dir=None,
                 report_rate=100, save_dir=None):
        super().__init__(debug, load_from_saved, save_checkpoints, initialize, tensorboard_dir, report_rate, save_dir)

    def _assemble_graph(self):
        # TODO
        # Might be the same as for semantic segmentation
        pass

    def compute_full_test_accuracy(self):
        # TODO
        pass

    def __heatmap_accuracy(self, predict_heatmap, label_heatmap):
        """
        Calculates the accuracy of an image's predicted heatmap compared to the heatmap label
        :param predict_heatmap: The model's predicted heatmap
        :param label_heatmap: The image's corresponding label heatmap
        :return: The accuracy of the heatmap prediction
        """
        return 0

    def forward_pass_with_file_inputs(self, x):
        # TODO
        pass

    def forward_pass_with_interpreted_outputs(self, x):
        # TODO
        pass

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

    def load_heatmap_dataset_from_csv(self, dirname, label_file):
        """
        Loads in a dataset for heatmap object counting. This dataset should consist of a directory of image files to
        train on and a csv file that maps image names to x and y labels
        :param dirname: The path to the directory with the image files and label file
        :param label_file: The path to the csv file with heatmap point labels
        """
        pass

    def __points_to_density_map(self, points):
        """
        Convert point labels for a heatmap into a grayscale image with gaussians placed at heatmap points
        :param points: A list of (x,y) tuples for object locations in an image
        :return: An ndarray of the heatmap image
        """
        return []
