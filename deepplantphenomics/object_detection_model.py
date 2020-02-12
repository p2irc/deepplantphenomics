from . import layers, loaders, definitions, DPPModel
import numpy as np
import tensorflow.compat.v1 as tf
import os
import json
import warnings
import copy
import itertools
import shutil
from math import ceil
from collections.abc import Sequence
from scipy.special import expit
from PIL import Image
from tqdm import tqdm


class ObjectDetectionModel(DPPModel):
    _supported_loss_fns = ['yolo']
    _supported_augmentations = [definitions.AugmentationType.CONTRAST_BRIGHT]

    def __init__(self, debug=False, load_from_saved=False, save_checkpoints=True, initialize=True, tensorboard_dir=None,
                 report_rate=100, save_dir=None):
        super().__init__(debug, load_from_saved, save_checkpoints, initialize, tensorboard_dir, report_rate, save_dir)
        self._loss_fn = 'yolo'

        # A flag to tell the object detection loaders whether to automatically convert JSON labels to YOLO format. This
        # exists because the dataset loader `load_yolo_dataset_from_directory` doesn't want that to happen
        self._json_no_convert = False

        # State variables specific to object detection for constructing the graph and passing to Tensorboard
        self._yolo_loss = None

        # Yolo-specific parameters, non-default values defined by set_yolo_parameters
        self._grid_w = 7
        self._grid_h = 7
        self._LABELS = ['plant']
        self._NUM_CLASSES = 1
        self._RAW_ANCHORS = [(159, 157), (103, 133), (91, 89), (64, 65), (142, 101)]
        self._ANCHORS = None  # Scaled version, but grid and image sizes are needed so default is deferred
        self._NUM_BOXES = 5
        self._THRESH_SIG = 0.6
        self._THRESH_OVERLAP = 0.3
        self._THRESH_CORRECT = 0.5

    def set_image_dimensions(self, image_height, image_width, image_depth):
        super().set_image_dimensions(image_height, image_width, image_depth)

        # Generate image-scaled anchors for YOLO object detection once the image dimensions are set
        if self._RAW_ANCHORS:
            scale_w = self._grid_w / self._image_width
            scale_h = self._grid_h / self._image_height
            self._ANCHORS = [(anchor[0] * scale_w, anchor[1] * scale_h) for anchor in self._RAW_ANCHORS]

    def set_yolo_parameters(self, grid_size=None, labels=None, anchors=None):
        """
        Set YOLO parameters for the grid size, class labels, and anchor/prior sizes
        :param grid_size: 2-element list/tuple with the width and height of the YOLO grid. Default = [7,7]
        :param labels: List of class labels for detection. Default = ['plant']
        :param anchors: List of 2-element anchor/prior widths and heights.
        Default = [[159, 157], [103, 133], [91, 89], [64, 65], [142, 101]]
        """
        if not self._image_width or not self._image_height:
            raise RuntimeError("Image dimensions need to be chosen before setting YOLO parameters")

        # Do type checks and fill in list parameters with arguments or defaults, because mutable function defaults are
        # dangerous
        if grid_size:
            if not isinstance(grid_size, Sequence) or len(grid_size) != 2 \
                    or not all([isinstance(x, int) for x in grid_size]):
                raise TypeError("grid_size should be a 2-element integer list")
            self._grid_w, self._grid_h = grid_size
        else:
            self._grid_w, self._grid_h = [7, 7]

        if labels:
            if not isinstance(labels, Sequence) or isinstance(labels, str) \
                    or not all([isinstance(lab, str) for lab in labels]):
                raise TypeError("labels should be a string list")
            self._LABELS = labels
            self._NUM_CLASSES = len(labels)
        else:
            self._LABELS = ['plant']
            self._NUM_CLASSES = 1

        if anchors:
            if not isinstance(anchors, Sequence):
                raise TypeError("anchors should be a list/tuple of integer lists/tuples")
            if not all([(isinstance(a, Sequence) and len(a) == 2
                         and isinstance(a[0], int) and isinstance(a[1], int)) for a in anchors]):
                raise TypeError("anchors should contain 2-element lists/tuples")
            self._RAW_ANCHORS = anchors
        else:
            self._RAW_ANCHORS = [(159, 157), (103, 133), (91, 89), (64, 65), (142, 101)]

        # Fill in non-mutable parameters
        self._NUM_BOXES = len(self._RAW_ANCHORS)

        # Scale anchors to the grid size
        scale_w = self._grid_w / self._image_width
        scale_h = self._grid_h / self._image_height
        self._ANCHORS = [(anchor[0] * scale_w, anchor[1] * scale_h) for anchor in self._RAW_ANCHORS]

    def set_yolo_thresholds(self, thresh_sig=0.6, thresh_overlap=0.3, thresh_correct=0.5):
        """Set YOLO IoU thresholds for bounding box significance (during output filtering), overlap (during non-maximal
        suppression), and correctness (for mAP calculation)"""
        self._THRESH_SIG = thresh_sig
        self._THRESH_OVERLAP = thresh_overlap
        self._THRESH_CORRECT = thresh_correct

    def _yolo_compute_iou(self, pred_box, true_box):
        """Helper function to compute the intersection over union of pred_box and true_box
        pred_box and true_box represent multiple boxes with coords being x,y,w,h (0-indexed 0-3)"""
        # numerator
        # get coords of intersection rectangle, then compute intersection area
        x1 = tf.maximum(pred_box[..., 0] - 0.5 * pred_box[..., 2],
                        true_box[..., 0:1] - 0.5 * true_box[..., 2:3])
        y1 = tf.maximum(pred_box[..., 1] - 0.5 * pred_box[..., 3],
                        true_box[..., 1:2] - 0.5 * true_box[..., 3:4])
        x2 = tf.minimum(pred_box[..., 0] + 0.5 * pred_box[..., 2],
                        true_box[..., 0:1] + 0.5 * true_box[..., 2:3])
        y2 = tf.minimum(pred_box[..., 1] + 0.5 * pred_box[..., 3],
                        true_box[..., 1:2] + 0.5 * true_box[..., 3:4])
        intersection_area = tf.multiply(tf.maximum(0., x2 - x1), tf.maximum(0., y2 - y1))

        # denominator
        # compute area of pred and truth, compute union area
        pred_area = tf.multiply(pred_box[..., 2], pred_box[..., 3])
        true_area = tf.multiply(true_box[..., 2:3], true_box[..., 3:4])
        union_area = tf.subtract(tf.add(pred_area, true_area), intersection_area)

        # compute iou
        iou = tf.divide(intersection_area, union_area)
        return iou

    def _yolo_loss_function(self, y_true, y_pred):
        """
        Loss function based on YOLO
        See the paper for details: https://pjreddie.com/media/files/papers/yolo.pdf

        :param y_true: Tensor with ground truth bounding boxes for each grid square in each image. Labels have 6
        elements: [object/no-object, class, x, y, w, h]
        :param y_pred: Tensor with predicted bounding boxes for each grid square in each image. Predictions consist of
        one box and confidence [x, y, w, h, conf] for each anchor plus 1 element for specifying the class (only one atm)
        :return Scalar Tensor with the Yolo loss for the bounding box predictions. Technically, this is the sum of the
        Yolo loss for each image in the passed-in batch.
        """

        prior_boxes = tf.convert_to_tensor(self._ANCHORS)

        # object/no-object masks #
        # create masks for grid cells with objects and with no objects
        obj_mask = tf.cast(y_true[..., 0], dtype=bool)
        no_obj_mask = tf.logical_not(obj_mask)
        obj_pred = tf.boolean_mask(y_pred, obj_mask)
        obj_true = tf.boolean_mask(y_true, obj_mask)
        no_obj_pred = tf.boolean_mask(y_pred, no_obj_mask)

        # bbox coordinate loss #
        # build a tensor of the predicted bounding boxes and confidences, classes will be stored separately
        # [x1,y1,w1,h1,conf1,x2,y2,w2,h2,conf2,x3,y3,w3,h3,conf3,...]
        pred_classes = obj_pred[..., self._NUM_BOXES * 5:]
        # we take the x,y,w,h,conf's that are altogether (dim is 1xB*5) and turn into Bx5, where B is num_boxes
        obj_pred = tf.reshape(obj_pred[..., 0:self._NUM_BOXES * 5], [-1, self._NUM_BOXES, 5])
        no_obj_pred = tf.reshape(no_obj_pred[..., 0:self._NUM_BOXES * 5], [-1, self._NUM_BOXES, 5])
        t_x, t_y, t_w, t_h = obj_pred[..., 0], obj_pred[..., 1], obj_pred[..., 2], obj_pred[..., 3]
        t_o = obj_pred[..., 4]
        pred_x = tf.sigmoid(t_x) + 0.00001  # concerned about underflow (might not actually be necessary)
        pred_y = tf.sigmoid(t_y) + 0.00001
        pred_w = (tf.exp(t_w) + 0.00001) * prior_boxes[:, 0]
        pred_h = (tf.exp(t_h) + 0.00001) * prior_boxes[:, 1]
        pred_conf = tf.sigmoid(t_o) + 0.00001
        predicted_boxes = tf.stack([pred_x, pred_y, pred_w, pred_h, pred_conf], axis=2)

        # find responsible boxes by computing iou's and select the best one
        ious = self._yolo_compute_iou(
            predicted_boxes, obj_true[..., 1 + self._NUM_CLASSES:1 + self._NUM_CLASSES + 4])
        greatest_iou_indices = tf.argmax(ious, 1)
        argmax_one_hot = tf.one_hot(indices=greatest_iou_indices, depth=5)
        resp_box_mask = tf.cast(argmax_one_hot, dtype=bool)
        responsible_boxes = tf.boolean_mask(predicted_boxes, resp_box_mask)

        # compute loss on responsible boxes
        loss_xy = tf.square(tf.subtract(responsible_boxes[..., 0:2],
                                        obj_true[..., 1 + self._NUM_CLASSES:1 + self._NUM_CLASSES + 2]))
        loss_wh = tf.square(tf.subtract(tf.sqrt(responsible_boxes[..., 2:4]),
                                        tf.sqrt(obj_true[..., 1 + self._NUM_CLASSES + 2:1 + self._NUM_CLASSES + 4])))
        coord_loss = tf.reduce_sum(tf.add(loss_xy, loss_wh))

        # confidence loss #
        # grids that do contain an object, 1 * iou means we simply take the difference between the
        # iou's and the predicted confidence
        obj_num_grids = tf.shape(predicted_boxes)[0]  # [num_boxes, 5, 5]
        loss_obj = tf.cast((1 / obj_num_grids), dtype='float32') * tf.reduce_sum(
            tf.square(tf.subtract(ious, predicted_boxes[..., 4])))

        # grids that do not contain an object, 0 * iou means we simply take the predicted confidences of the
        # grids that do not have an object and square and sum (because they should be 0)
        no_obj_confs = tf.sigmoid(no_obj_pred[..., 4])
        no_obj_num_grids = tf.shape(no_obj_confs)[0]  # [number_of_grids_without_an_object, 5]
        loss_no_obj = tf.cast(1 / no_obj_num_grids, dtype='float32') * tf.reduce_sum(tf.square(no_obj_confs))
        # incase obj_pred or no_obj_confs is empty (e.g. no objects in the image) we need to make sure we dont
        # get nan's in our losses...
        loss_obj = tf.cond(tf.count_nonzero(y_true[..., 4]) > 0, lambda: loss_obj, lambda: 0.)
        loss_no_obj = tf.cond(tf.count_nonzero(y_true[..., 4]) < self._grid_w * self._grid_h,
                              lambda: loss_no_obj, lambda: 0.)
        conf_loss = tf.add(loss_obj, loss_no_obj)

        # classification loss #
        # currently only one class, plant, will need to be made more general for multi-class in the future
        class_probs_pred = tf.nn.softmax(pred_classes)
        class_diffs = tf.subtract(obj_true[..., 1:1 + self._NUM_CLASSES], class_probs_pred)
        class_loss = tf.reduce_sum(tf.square(class_diffs))

        total_loss = coord_loss + conf_loss + class_loss
        return total_loss

    def _graph_tensorboard_summary(self, l2_cost, gradients, variables, global_grad_norm):
        super()._graph_tensorboard_common_summary(l2_cost, gradients, variables, global_grad_norm)

        # Summaries specific to object detection
        tf.summary.scalar('train/yolo_loss', self._yolo_loss, collections=['custom_summaries'])
        if self._validation:
            tf.summary.scalar('validation/loss', self._graph_ops['val_losses'],
                              collections=['custom_summaries'])

        self._graph_ops['merged'] = tf.summary.merge_all(key='custom_summaries')

    def _assemble_graph(self):
        with self._graph.as_default():
            self._log('Assembling graph...')

            self._log('Graph: Parsing dataset...')
            with tf.device('device:cpu:0'):  # Only do preprocessing on the CPU to limit data transfer between devices
                self._graph_parse_data()

                # For object detection, we need to also deserialize the labels before batching the datasets
                def _deserialize_label(im, lab):
                    lab = tf.cond(tf.equal(tf.rank(lab), 0),
                                  lambda: tf.reshape(lab, [1]),
                                  lambda: lab)
                    sparse_lab = tf.string_split(lab, sep=' ')
                    lab_values = tf.strings.to_number(sparse_lab.values)
                    lab = tf.reshape(lab_values, [self._grid_w * self._grid_h, 5 + self._NUM_CLASSES])
                    return im, lab

                # Batch the datasets and create iterators for them
                self._train_dataset = self._train_dataset.map(_deserialize_label, num_parallel_calls=self._num_threads)
                train_iter = self._batch_and_iterate(self._train_dataset, shuffle=True)
                if self._testing:
                    self._test_dataset = self._test_dataset.map(_deserialize_label,
                                                                num_parallel_calls=self._num_threads)
                    test_iter = self._batch_and_iterate(self._test_dataset)
                if self._validation:
                    self._val_dataset = self._val_dataset.map(_deserialize_label, num_parallel_calls=self._num_threads)
                    val_iter = self._batch_and_iterate(self._val_dataset)

                if self._has_moderation:
                    train_mod_iter = self._batch_and_iterate(self._train_moderation_features)
                    if self._testing:
                        test_mod_iter = self._batch_and_iterate(self._test_moderation_features)
                    if self._validation:
                        val_mod_iter = self._batch_and_iterate(self._val_moderation_features)

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

                    # Define regularization cost
                    self._log('Graph: Calculating loss and gradients...')
                    l2_cost = self._graph_layer_loss()

                    # Define the cost function
                    xx = tf.reshape(xx, [-1, self._grid_w * self._grid_h,
                                         self._NUM_BOXES * 5 + self._NUM_CLASSES])
                    num_image_loss = tf.cast(tf.shape(xx)[0], tf.float32)

                    pred_loss = self._graph_problem_loss(xx, y)
                    gpu_cost = tf.squeeze(pred_loss / num_image_loss + l2_cost)
                    device_costs.append(pred_loss)

                    # Set the optimizer and get the gradients from it
                    gradients, variables, global_grad_norm = self._graph_get_gradients(gpu_cost, optimizer)
                    device_gradients.append(gradients)
                    device_variables.append(variables)

            # Average the gradients from each GPU and apply them
            average_gradients = self._graph_average_gradients(device_gradients)
            opt_variables = device_variables[0]
            self._graph_ops['optimizer'] = self._graph_apply_gradients(average_gradients, opt_variables, optimizer)

            # Average the costs and accuracies from each GPU
            self._yolo_loss = 0
            self._graph_ops['cost'] = tf.reduce_sum(device_costs) / self._batch_size + l2_cost

            # Calculate test and validation accuracy (on a single device at Tensorflow's discretion)
            if self._testing:
                x_test, self._graph_ops['y_test'] = test_iter.get_next()
                n_images = tf.cast(tf.shape(x_test)[0], tf.float32)

                if self._has_moderation:
                    mod_w_test = test_mod_iter.get_next()
                    self._graph_ops['x_test_predicted'] = self.forward_pass(x_test, deterministic=True,
                                                                            moderation_features=mod_w_test)
                else:
                    self._graph_ops['x_test_predicted'] = self.forward_pass(x_test, deterministic=True)
                self._graph_ops['x_test_predicted'] = tf.reshape(self._graph_ops['x_test_predicted'],
                                                                 [-1,
                                                                  self._grid_w * self._grid_h,
                                                                  self._NUM_BOXES * 5 + self._NUM_CLASSES])

                self._graph_ops['test_losses'] = self._graph_problem_loss(self._graph_ops['x_test_predicted'],
                                                                          self._graph_ops['y_test']) / n_images

            if self._validation:
                x_val, self._graph_ops['y_val'] = val_iter.get_next()
                n_images = tf.cast(tf.shape(x_val)[0], tf.float32)

                if self._has_moderation:
                    mod_w_val = val_mod_iter.get_next()
                    self._graph_ops['x_val_predicted'] = self.forward_pass(x_val, deterministic=True,
                                                                           moderation_features=mod_w_val)
                else:
                    self._graph_ops['x_val_predicted'] = self.forward_pass(x_val, deterministic=True)
                self._graph_ops['x_val_predicted'] = tf.reshape(self._graph_ops['x_val_predicted'],
                                                                [-1,
                                                                 self._grid_w * self._grid_h,
                                                                 self._NUM_BOXES * 5 + self._NUM_CLASSES])

                self._graph_ops['val_losses'] = self._graph_problem_loss(self._graph_ops['x_val_predicted'],
                                                                         self._graph_ops['y_val']) / n_images

            # Epoch summaries for Tensorboard
            if self._tb_dir is not None:
                self._graph_tensorboard_summary(l2_cost, gradients, variables, global_grad_norm)

    def _graph_problem_loss(self, pred, lab):
        if self._loss_fn == 'yolo':
            return self._yolo_loss_function(lab, pred)

        raise RuntimeError("Could not calculate problem loss for a loss function of " + self._loss_fn)

    def compute_full_test_accuracy(self):
        self._log('Computing total test accuracy/regression loss...')

        with self._graph.as_default():
            num_test = self._total_raw_samples - self._total_training_samples
            num_batches = int(np.ceil(num_test / self._batch_size))

            if num_batches == 0:
                warnings.warn('Less than a batch of testing data')
                exit()

            # Initialize storage for the retrieved test variables
            all_y = []
            all_predictions = []

            # Main test loop
            for _ in tqdm(range(num_batches)):
                r_y, r_predicted = self._session.run([self._graph_ops['y_test'],
                                                      self._graph_ops['x_test_predicted']])
                all_y.append(r_y)
                all_predictions.append(r_predicted)

            all_y = np.concatenate(all_y, axis=0)
            all_predictions = np.concatenate(all_predictions, axis=0)

            # Make the images heterogeneous, storing their separate grids in a list
            test_labels = [all_y[i, ...] for i in range(all_y.shape[0])]
            test_preds = [all_predictions[i, ...] for i in range(all_predictions.shape[0])]
            n_images = len(test_labels)

            # Convert coordinates, then filter out the positive ground truth labels and significant predictions
            for i in range(n_images):
                conv_label, conv_pred = self.__yolo_coord_convert(test_labels[i], test_preds[i])
                truth_mask = conv_label[..., 0] == 1
                if not np.any(truth_mask):
                    conv_label = None
                else:
                    conv_label = conv_label[truth_mask, :]
                conv_pred = self.__yolo_filter_predictions(conv_pred)
                test_labels[i] = conv_label
                test_preds[i] = conv_pred

            # Get and log the map
            yolo_map = self.__yolo_map(test_labels, test_preds)
            self._log('Yolo mAP: {}'.format(yolo_map))
            return yolo_map.astype(np.float32)

    def __yolo_coord_convert(self, labels=None, preds=None):
        """
        Converts Yolo labeled and predicted bounding boxes from xywh coords to x1y1x2y2 coords. Also accounts for
        required sigmoid and exponential conversions in the predictions (including the confidences)

        :param labels: ndarray with Yolo ground-truth bounding boxes (size ?x(NUM_CLASSES+5))
        :param preds: ndarray with Yolo predicted bounding boxes (size ?x(NUM_BOXES*5))
        :return: `labels` and `preds` with the bounding box coords changed from xywh to x1y1x2y2 and predicted box
        confidences converted to percents
        """

        def xywh_to_xyxy(x, y, w, h):
            x_centre = np.arange(self._grid_w * self._grid_h) % self._grid_w
            y_centre = np.arange(self._grid_w * self._grid_h) // self._grid_w
            scale_x = self._image_width / self._grid_w
            scale_y = self._image_height / self._grid_h

            x = (x + x_centre) * scale_x
            y = (y + y_centre) * scale_y
            w = w * scale_x
            h = h * scale_y

            x1 = x - w/2
            x2 = x + w/2
            y1 = y - h/2
            y2 = y + h/2
            return x1, y1, x2, y2

        if labels is not None:
            # Labels are already sensible numbers, so convert them first
            lab_coord_idx = np.arange(labels.shape[-1]-4, labels.shape[-1])
            lab_class, lab_x, lab_y, lab_w, lab_h = np.split(labels, lab_coord_idx, axis=-1)
            lab_x1, lab_y1, lab_x2, lab_y2 = xywh_to_xyxy(np.squeeze(lab_x),  # Squeezing to aid broadcasting in helper
                                                          np.squeeze(lab_y),
                                                          np.squeeze(lab_w),
                                                          np.squeeze(lab_h))
            labels = np.concatenate([lab_class,
                                     lab_x1[:, np.newaxis],  # Dummy dimensions to enable concatenation
                                     lab_y1[:, np.newaxis],
                                     lab_x2[:, np.newaxis],
                                     lab_y2[:, np.newaxis]], axis=-1)

        if preds is not None:
            # Extract the class predictions and reorganize the predicted boxes
            class_preds = preds[..., self._NUM_BOXES * 5:]
            preds = np.reshape(preds[..., 0:self._NUM_BOXES * 5], preds.shape[:-1] + (self._NUM_BOXES, 5))

            # Predictions are not sensible numbers, so apply sigmoids and exponentials first and then convert them
            anchors = np.array(self._ANCHORS)
            pred_x = expit(preds[..., 0])
            pred_y = expit(preds[..., 1])
            pred_w = np.exp(preds[..., 2]) * anchors[:, 0]
            pred_h = np.exp(preds[..., 3]) * anchors[:, 1]
            pred_conf = expit(preds[..., 4])
            pred_x1, pred_y1, pred_x2, pred_y2 = xywh_to_xyxy(pred_x.T,  # Transposes to aid broadcasting in helper
                                                              pred_y.T,
                                                              pred_w.T,
                                                              pred_h.T)
            preds[..., :] = np.stack([pred_x1.T,  # Transposes to restore original shape
                                      pred_y1.T,
                                      pred_x2.T,
                                      pred_y2.T,
                                      pred_conf], axis=-1)

            # Reattach the class predictions
            preds = np.reshape(preds, preds.shape[:-2] + (self._NUM_BOXES * 5,))
            preds = np.concatenate([preds, class_preds], axis=-1)

        return labels, preds

    def __yolo_filter_predictions(self, preds):
        """
        Filters the predicted bounding boxes by eliminating insignificant and overlapping predictions

        :param preds: ndarray with predicted bounding boxes for one image in each grid square. Predictions
        are a list of, for each box, [x1, y1, x2, y2, conf] followed by a list of class predictions
        :return: `preds` with only the significant and maximal confidence predictions remaining
        """
        # Extract the class predictions and separate the predicted boxes
        grid_count = preds.shape[0]
        class_preds = preds[..., self._NUM_BOXES * 5:]
        preds = np.reshape(preds[..., 0:self._NUM_BOXES * 5], preds.shape[:-1] + (self._NUM_BOXES, 5))

        # In each grid square, the highest confidence box is the one responsible for prediction
        max_conf_idx = np.argmax(preds[..., 4], axis=-1)
        responsible_boxes = [preds[i, max_conf_idx[i], :] for i in range(grid_count)]
        preds = np.stack(responsible_boxes, axis=0)

        # Eliminate insignificant predicted boxes
        sig_mask = preds[:, 4] > self._THRESH_SIG
        if not np.any(sig_mask):
            return None
        class_preds = class_preds[sig_mask, :]
        preds = preds[sig_mask, :]

        # Apply non-maximal suppression (i.e. eliminate boxes that overlap with a more confidant box)
        maximal_idx = []
        sig_grid_count = preds.shape[0]
        conf_order = np.argsort(preds[:, 4])
        pair_iou = np.array([self.__compute_iou(preds[i, 0:4], preds[j, 0:4])
                             for i in range(sig_grid_count) for j in range(sig_grid_count)])
        pair_iou = pair_iou.reshape(sig_grid_count, sig_grid_count)
        while len(conf_order) > 0:
            # Take the most confidant box, then cull the list down to boxes that don't overlap with it
            cur_grid = conf_order[-1]
            maximal_idx.append(cur_grid)
            non_overlap = pair_iou[cur_grid, conf_order] < self._THRESH_OVERLAP
            if np.any(non_overlap):
                conf_order = conf_order[non_overlap]
                # Make sure that the most confidant box itself isn't still in the list (usually by having a self-IOU of
                # 0.999999... when the overlap threshold is 1)
                conf_order = conf_order[conf_order != cur_grid]
            else:
                break

        # Stick things back together. maximal_idx is not sorted, but box and class predictions should still match up
        # and the original grid order shouldn't matter for mAP calculations
        class_preds = class_preds[maximal_idx, :]
        preds = preds[maximal_idx, :]
        preds = np.concatenate([preds, class_preds], axis=-1)

        return preds

    def __yolo_map(self, labels, preds):
        """
        Calculates the mean average precision of Yolo object and class predictions

        :param labels: List of ndarrays with ground truth bounding box labels for each image. Labels are a 6-value
        list: [object-ness, class, x1, y1, x2, y2]
        :param preds: List of ndarrays with significant predicted bounding boxes in each image. Predictions are a list
        of box parameters [x1, y1, x2, y2, conf] followed by a list of class predictions
        :return: The mean average precision (mAP) of the predictions
        """
        # Go over each prediction in each image and determine if it's a true or false positive
        detections = []
        for im_lab, im_pred in zip(labels, preds):
            # No predictions means no positives
            if im_pred is None:
                continue
            n_pred = im_pred.shape[0]

            # No labels means all false positives
            if im_lab is None:
                for i in range(n_pred):
                    detections.append((im_pred[i, 4], 0))
                continue
            n_lab = im_lab.shape[0]

            # Add a 7th value to the labels so we can tell which ones get matched up with true positives
            im_lab = np.concatenate([im_lab, np.zeros((n_lab, 1))], axis=-1)

            # Calculate the IoUs of all the prediction and label pairings, then record each detection as a true or
            # false positive with the prediction confidence
            pair_ious = np.array([self.__compute_iou(im_pred[i, 0:4], im_lab[j, 2:6])
                                  for i in range(n_pred) for j in range(n_lab)])
            pair_ious = np.reshape(pair_ious, (n_pred, n_lab))
            for i in range(n_pred):
                j = np.argmax(pair_ious[i, :])
                if pair_ious[i, j] >= self._THRESH_CORRECT and not im_lab[j, 6]:
                    detections.append((im_pred[i, 4], 1))
                    im_lab[j, 6] = 1
                else:
                    detections.append((im_pred[i, 4], 0))

        # If there are no valid predictions at all, the mAP is 0
        if not detections:
            return np.float32(0)  # We need to play nice with compute_full_test_accuracy()'s return

        # With multiple classes, we would also have class tags in the detection tuples so the below code could generate
        # and iterate over class-separated detection lists, giving multiple AP values and one true mean AP. We aren't
        # doing that right now because of our one-class plant detector assumption

        # Determine the precision-recall curve from the cumulative detected true and false positives (in order of
        # descending confidence)
        detections = np.array(sorted(detections, key=lambda d: d[0], reverse=True))
        n_truths = sum([x.shape[0] if (x is not None) else 0
                        for x in labels])
        n_positives = detections.shape[0]
        true_positives = np.cumsum(detections[:, 1])
        precision = true_positives / np.arange(1, n_positives+1)
        recall = true_positives / n_truths

        # Calculate the area under the precision-recall curve (== AP)
        for i in range(precision.size - 1, 0, -1):  # Make precision values the maximum precision at further recalls
            precision[i - 1] = np.max((precision[i], precision[i-1]))
        ap = np.sum(precision[1:] * (recall[1:] - recall[0:-1]))

        return ap

    def forward_pass_with_file_inputs(self, images):
        with self._graph.as_default():
            num_batches = len(images) // self._batch_size
            if len(images) % self._batch_size != 0:
                num_batches += 1

            self._parse_images(images)
            im_data = self._all_images.batch(self._batch_size).prefetch(1)
            x_test = im_data.make_one_shot_iterator().get_next()

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

            if self._load_from_saved:
                self.load_state()

            # Run model on them
            x_pred = self.forward_pass(x_test, deterministic=True)

            if self._with_patching:
                xx_output_size = [-1, num_patch_rows * num_patch_cols,
                                  self._grid_w * self._grid_h, 5 * self._NUM_BOXES + self._NUM_CLASSES]
            else:
                xx_output_size = [-1,
                                  self._grid_w * self._grid_h, 5 * self._NUM_BOXES + self._NUM_CLASSES]

            total_outputs = []
            for i in range(int(num_batches)):
                xx = self._session.run(x_pred)
                xx = np.reshape(xx, xx_output_size)
                for img in np.array_split(xx, self._batch_size):
                    total_outputs.append(img)

            # Delete the weird first row and any outputs which are overruns from the last batch
            total_outputs = np.concatenate(total_outputs, axis=0)

        return total_outputs

    def forward_pass_with_interpreted_outputs(self, x):
        total_outputs = self.forward_pass_with_file_inputs(x)
        n_images = total_outputs.shape[0]

        if self._with_patching:
            num_patches_vert = self._image_height // self._patch_height
            num_patches_horiz = self._image_width // self._patch_width
            num_patches = num_patches_horiz * num_patches_vert

            im_preds = []
            for i in range(n_images):
                patch_preds = []
                for j in range(num_patches):
                    _, conv_preds = self.__yolo_coord_convert(None, total_outputs[i, j, ...])
                    filtered_preds = self.__yolo_filter_predictions(conv_preds)
                    patch_preds.append(filtered_preds)
                im_preds.append(patch_preds)
        else:
            im_preds = []
            for i in range(n_images):
                _, conv_preds = self.__yolo_coord_convert(None, total_outputs[i, ...])
                filtered_preds = self.__yolo_filter_predictions(conv_preds)
                im_preds.append(filtered_preds)

        return im_preds

    def __compute_iou(self, box1, box2):
        """
        Need to somehow merge with the iou helper function in the yolo cost function.

        :param box1: x1, y1, x2, y2
        :param box2: x1, y1, x2, y2
        :return: Intersection Over Union of box1 and box2
        """
        x1 = np.maximum(box1[0], box2[0])
        y1 = np.maximum(box1[1], box2[1])
        x2 = np.minimum(box1[2], box2[2])
        y2 = np.minimum(box1[3], box2[3])

        intersection_area = np.maximum(0., x2 - x1) * np.maximum(0., y2 - y1)
        union_area = \
            ((box1[2] - box1[0]) * (box1[3] - box1[1])) + \
            ((box2[2] - box2[0]) * (box2[3] - box2[1])) - \
            intersection_area

        return intersection_area / union_area

    def add_output_layer(self, regularization_coefficient=None, output_size=None):
        if len(self._layers) < 1:
            raise RuntimeError("An output layer cannot be the first layer added to the model. " +
                               "Add an input layer with DPPModel.add_input_layer() first.")
        if regularization_coefficient is not None:
            warnings.warn("Object detection doesn't use regularization_coefficient in its output layer")
        if output_size is not None:
            if output_size is not None:
                raise RuntimeError("output_size should be None for object detection")

        self._log('Adding output layer...')

        filter_dimension = [1, 1,
                            copy.deepcopy(self._last_layer().output_size[3]),
                            (5 * self._NUM_BOXES + self._NUM_CLASSES)]

        with self._graph.as_default():
            layer = layers.convLayer('output',
                                     copy.deepcopy(self._last_layer().output_size),
                                     filter_dimension,
                                     1,
                                     None,
                                     self._weight_initializer)

        self._log('Inputs: {0} Outputs: {1}'.format(layer.input_size, layer.output_size))
        self._layers.append(layer)

    def load_ippn_tray_dataset_from_directory(self, dirname):
        """
        Loads the RGB tray images and plant bounding box labels from the International Plant Phenotyping Network
        dataset.
        """
        self._resize_bbox_coords = True

        images = [os.path.join(dirname, name) for name in sorted(os.listdir(dirname)) if
                  os.path.isfile(os.path.join(dirname, name)) & name.endswith('_rgb.png')]

        label_files = [os.path.join(dirname, name) for name in sorted(os.listdir(dirname)) if
                       os.path.isfile(os.path.join(dirname, name)) & name.endswith('_bbox.csv')]

        # currently reads columns, need to read rows instead!!!
        labels = [loaders.read_csv_rows(label_file) for label_file in label_files]

        self._all_labels = []
        for label in labels:
            curr_label = []
            for nums in label:
                # yolo wants x,y,w,h for coords
                curr_label.extend(loaders.box_coordinates_to_xywh_coordinates(nums))
            self._all_labels.append(curr_label)

        self._total_raw_samples = len(images)

        # need to add object-ness flag and one-hot encodings for class
        # it will be 1 or 0 for object-ness, one-hot for the class, then 4 bbox coords (x,y,w,h)
        # e.g. [1,0,0,...,1,...,0,223,364,58,62] but since there is only one class for the ippn dataset we get
        # [1,1,x,y,w,h]

        # for scaling bbox coords
        # scaling image down to the grid size
        scale_ratio_w = self._grid_w / self._image_width_original
        scale_ratio_h = self._grid_h / self._image_height_original

        labels_with_one_hot = []
        for curr_img_coords in self._all_labels:
            curr_img_labels = []
            num_boxes = len(curr_img_coords) // 4
            for i in range(num_boxes):
                # start the current box label with the object-ness flag and class label (there is only one class
                # for ippn)
                curr_box = [1, 1]
                # add scaled bbox coords
                j = i * 4
                # x and y offsets from grid position
                x_grid = curr_img_coords[j] * scale_ratio_w
                y_grid = curr_img_coords[j + 1] * scale_ratio_h
                x_grid_offset, x_grid_loc = np.modf(x_grid)
                y_grid_offset, y_grid_loc = np.modf(y_grid)
                # w and h ratios from anchor box
                w_ratio = curr_img_coords[j + 2] / self._ANCHORS[0]
                h_ratio = curr_img_coords[j + 3] / self._ANCHORS[1]
                curr_box.append(x_grid_offset)
                curr_box.append(y_grid_offset)
                curr_box.append(w_ratio)
                curr_box.append(h_ratio)
                curr_img_labels.extend(curr_box)
            labels_with_one_hot.append(curr_img_labels)
        self._raw_labels = labels_with_one_hot

        self._log('Total raw examples is %d' % self._total_raw_samples)
        self._log('Parsing dataset...')

        self._raw_image_files = images
        self._raw_labels = self._all_labels

    def load_yolo_dataset_from_directory(self, data_dir, label_file, image_dir):
        """
        Loads in labels and images for object detection tasks, converting the labels to YOLO format and automatically
        patching the images if necessary.

        :param data_dir: String, The directory where the labels and images are stored in
        :param label_file: String, The filename for the JSON file with the labels
        :param image_dir: String, The directory with the images
        """
        label_path = os.path.join(data_dir, label_file)
        image_path = os.path.join(data_dir, image_dir, '')

        self._json_no_convert = True
        self.load_json_labels_from_file(label_path)
        images_list = loaders.get_dir_images(image_path)
        self.load_images_from_list(images_list)
        self._json_no_convert = False

        # Perform automatic image patching if necessary
        if self._with_patching:
            self._raw_image_files, self._all_labels = self.__autopatch_object_detection_dataset(patch_dir=data_dir)
            self._total_raw_samples = len(self._raw_image_files)
            self._log('Total raw patch examples is now %d' % self._total_raw_samples)

        self._all_labels = self.__convert_labels_to_yolo_format()
        self._raw_labels = self._all_labels

    def __autopatch_object_detection_dataset(self, patch_dir=None):
        """
        Generates a dataset of image patches from a loaded dataset of larger images and returns the new images and
        labels. This will check for existing patches first and load them if found unless data overwriting is turned on.
        :param patch_dir: The directory to place patched images into, or where to read previous patches from
        :return: The patched dataset as a list of image filenames and a nested list of their corresponding point labels
        """
        if not patch_dir:
            patch_dir = os.path.curdir
        patch_dir = os.path.join(patch_dir, 'tmp_train', '')
        img_dir = os.path.join(patch_dir, 'image_patches', '')
        json_file = os.path.join(patch_dir, 'train_patches.json')

        if os.path.exists(patch_dir) and not self._gen_data_overwrite:
            # If there already is a patched dataset, just load it
            self._log("Loading preexisting patched data from " + patch_dir)
            self._json_no_convert = True
            self.load_json_labels_from_file(json_file)
            img_list = loaders.get_dir_images(img_dir)
            self.load_images_from_list(img_list)
            self._json_no_convert = False
            return self._raw_image_files, self._all_labels

        self._log("Patching dataset: Patches will be in " + patch_dir)
        if os.path.exists(patch_dir):
            self._log("Overwriting preexisting patched data...")
            shutil.rmtree(patch_dir)
        os.makedirs(patch_dir)
        os.makedirs(img_dir)

        # We need to construct a patched dataset, but we'll be picking them out with various methods
        img_dict = {}
        new_raw_image_files = []
        new_raw_labels = []

        def add_patch_to_dataset(patch, file_boxes, raw_boxes, patch_idx):
            patch_name = os.path.join(img_dir + "{:0>6d}.png".format(patch_idx))
            patch_img = Image.fromarray(patch.astype(np.uint8))
            patch_img.save(patch_name)

            img_dict["{:0>6d}".format(patch_idx)] = {"height": self._patch_height,
                                                     "width": self._patch_width,
                                                     "file_name": "{:0>6d}.png".format(patch_idx),
                                                     "plants": file_boxes}
            new_raw_image_files.append(patch_name)
            new_raw_labels.append(raw_boxes)

        def xywh_to_tblr_coords(cx, cy, width, height):
            top = cy - height // 2
            bottom = top + height
            left = cx - width // 2
            right = left + width
            return [top, bottom, left, right]

        def xyxy_to_xywh_coords(x1, x2, y1, y2):
            bw = x2 - x1
            bh = y2 - y1
            x = x1 + bw // 2
            y = y1 + bh // 2
            return [x, y, bw, bh]

        def image_to_patch_xy(xi, yi, xp, yp, p_width, p_height):
            dx = xi - xp
            dy = yi - yp
            cx_shift = p_width // 2 + dx
            cy_shift = p_height // 2 + dy
            return cx_shift, cy_shift

        def get_random_patch(orig_img, p_width, p_height):
            px_centre, py_centre = p_width // 2, p_height // 2
            min_width, max_width = px_centre, orig_img.shape[1] - px_centre
            min_height, max_height = py_centre, orig_img.shape[0] - py_centre

            rand_x = np.random.randint(min_width, max_width + 1)
            rand_y = np.random.randint(min_height, max_height + 1)
            top, bot, left, right = xywh_to_tblr_coords(rand_x, rand_y, p_width, p_height)
            patch = orig_img[top:bot, left:right]

            return patch, [top, bot, left, right]

        def get_boxes_in_patch(p_tblr, boxes):
            p_top, p_bot, p_left, p_right = p_tblr
            p_width, p_height = (p_right - p_left), (p_bot - p_top)
            patch_boxes = []
            for orig_box in boxes:
                orig_x, orig_y, orig_w, orig_h = xyxy_to_xywh_coords(*orig_box)
                if p_left <= orig_x <= p_right and p_top <= orig_y <= p_bot:
                    cx, cy = p_left + p_width // 2, p_top + p_height // 2
                    patch_x, patch_y = image_to_patch_xy(orig_x, orig_y, cx, cy, p_width, p_height)
                    patch_y_min, patch_y_max, patch_x_min, patch_x_max = xywh_to_tblr_coords(patch_x, patch_y,
                                                                                             orig_w, orig_h)
                    patch_boxes.append([patch_x_min, patch_x_max, patch_y_min, patch_y_max])

            return patch_boxes

        num_orig_images = len(self._raw_image_files)
        img_name_idx = 0

        # First set of patches: attempt to get patches such that every YOLO grid cell will see a plant at some point
        # and learn to recognize them during training. The patches should be a small distance from the edges of the
        # image, so plants in the patches should be about 1 patch-length away from the edges to allow shifting them
        # into the appropriate grid cell.
        for img_num, img_name, img_boxes in zip(range(num_orig_images), self._raw_image_files, self._all_labels):
            img = np.array(Image.open(img_name))

            for i, j in itertools.product(range(self._grid_h), range(self._grid_w)):
                found_one = False
                random_indices = list(range(len(img_boxes)))
                while random_indices and not found_one:
                    rand_idx = np.random.randint(0, len(random_indices))
                    rand_plant_idx = random_indices.pop(rand_idx)
                    box_x, box_y, box_w, box_h = xyxy_to_xywh_coords(*img_boxes[rand_plant_idx])
                    if (self._patch_width + 5) < box_x < (img.shape[1] - (self._patch_width + 5)) \
                            and (self._patch_height + 5) < box_y < (img.shape[0] - (self._patch_height + 5)):
                        # This plant box meets our criteria, so get the center of the patch that places it in grid cell
                        # (i, j)
                        delta_x = j - self._grid_w // 2
                        delta_y = i - self._grid_h // 2
                        new_x = int(box_x - (delta_x * (self._patch_width / self._grid_w)))
                        new_y = int(box_y - (delta_y * (self._patch_height / self._grid_h)))
                        top_row, bot_row, left_col, right_col = xywh_to_tblr_coords(
                            new_x, new_y, self._patch_width, self._patch_height)
                        img_patch = img[top_row:bot_row, left_col:right_col]

                        new_raw_boxes = get_boxes_in_patch([top_row, bot_row, left_col, right_col], img_boxes)
                        new_boxes = []
                        for box in new_raw_boxes:
                            new_boxes.append({"all_points_x": box[0:2], "all_points_y": box[2:4]})

                        # Save patch to disk and store labels
                        add_patch_to_dataset(img_patch, new_raw_boxes, new_boxes, img_name_idx)
                        img_name_idx += 1
                        found_one = True
                if not found_one:
                    # If this happens, then none of the plants can meet our criteria and no patches like this can be
                    # made for this image
                    break

            self._log(str(img_num + 1) + '/' + str(len(self._all_labels)))
        self._log('Completed baseline patches. Total images so far: ' + str(img_name_idx))

        # Second set of patches: pick patches at random with some plants in them and randomly augment them with
        # rotations, flips, and brightness adjustments
        self._log('Creating augmentation patches...')
        for i in range(self._grid_h * self._grid_w):
            for img_name, img_boxes in zip(self._raw_image_files, self._all_labels):
                img = np.array(Image.open(img_name))

                # Randomly grab a patch of the image and make sure it has at least one plant in it
                img_patch = None
                new_boxes = []
                while not new_boxes:
                    img_patch, img_tblr = get_random_patch(img, self._patch_width, self._patch_height)
                    new_boxes = get_boxes_in_patch(img_tblr, img_boxes)

                # Randomly choose one of three augmentations to apply
                aug = np.random.randint(1, 4)  # 1 == rotation, 2 == brightness, 3 == flip
                if aug == 1:  # rotation
                    k = np.random.randint(1, 4)
                    rot_img_patch = np.rot90(img_patch, k)
                    theta = np.radians(90 * k)
                    x0 = self._patch_width // 2
                    y0 = self._patch_height // 2
                    rot_boxes = []
                    raw_rot_boxes = []
                    for box in new_boxes:
                        rot_x_min = x0 + (box[0] - x0) * np.cos(theta) + (box[2] - y0) * np.sin(theta)
                        rot_y_min = y0 - (box[0] - x0) * np.sin(theta) + (box[2] - y0) * np.cos(theta)
                        w = box[1] - box[0]
                        h = box[3] - box[2]
                        if k == 1:  # w and h flip, x_min y_min become x_min y_max
                            w, h = h, w
                            rot_y_min -= h
                        elif k == 2:  # w and h stay same, x_min y_min become x_max y_max
                            rot_x_min -= w
                            rot_y_min -= h
                        else:  # w and h flip, x_min y_min become x_max y_min
                            w, h = h, w
                            rot_x_min -= w
                        rot_x_max = rot_x_min + w
                        rot_y_max = rot_y_min + h

                        rot_boxes.append({"all_points_x": [rot_x_min, rot_x_max],
                                          "all_points_y": [rot_y_min, rot_y_max]})
                        raw_rot_boxes.append([rot_x_min, rot_x_max, rot_y_min, rot_y_max])

                    # Save patch to disk and store labels
                    add_patch_to_dataset(rot_img_patch, rot_boxes, raw_rot_boxes, img_name_idx)
                    img_name_idx += 1
                elif aug == 2:  # brightness
                    value = np.random.randint(40, 76)  # just a 'nice amount' of brightness change
                    k = np.random.random()
                    if k < 0.5:  # brighter
                        bright_img_patch = np.where((255 - img_patch) < value, 255, img_patch + value)
                    else:  # dimmer
                        bright_img_patch = np.where(img_patch < value, 0, img_patch - value)

                    bright_boxes = []
                    raw_bright_boxes = []
                    for box in new_boxes:
                        bright_boxes.append({"all_points_x": [box[0], box[1]],
                                             "all_points_y": [box[2], box[3]]})
                        raw_bright_boxes.append([box[0], box[1], box[2], box[3]])

                    # Save patch to disk and store labels
                    add_patch_to_dataset(bright_img_patch, bright_boxes, raw_bright_boxes, img_name_idx)
                    img_name_idx += 1
                else:  # flip (k == 3)
                    flip_boxes = []
                    raw_flip_boxes = []
                    k = np.random.random()
                    if k < 0.5:
                        flip_img_patch = np.fliplr(img_patch)
                        for box in new_boxes:
                            w = box[1] - box[0]
                            x_min = self._patch_width - box[1]
                            x_max = x_min + w
                            y_min = box[2]
                            y_max = box[3]

                            flip_boxes.append({"all_points_x": [x_min, x_max],
                                               "all_points_y": [y_min, y_max]})
                            raw_flip_boxes.append([x_min, x_max, y_min, y_max])
                    else:
                        flip_img_patch = np.flipud(img_patch)
                        for box in new_boxes:
                            h = box[3] - box[2]
                            x_min = box[0]
                            x_max = box[1]
                            y_min = self._patch_height - box[3]
                            y_max = y_min + h

                            flip_boxes.append({"all_points_x": [x_min, x_max],
                                               "all_points_y": [y_min, y_max]})
                            raw_flip_boxes.append([x_min, x_max, y_min, y_max])

                    # Save patch to disk and store labels
                    add_patch_to_dataset(flip_img_patch, flip_boxes, raw_flip_boxes, img_name_idx)
                    img_name_idx += 1
            self._log(str(i + 1) + '/' + str(self._grid_w * self._grid_h))
        self._log('Completed augmentation patches. Total images so far: ' + str(img_name_idx))

        # Third set of patches: pick patches completely at random so as to double the number of patches in our dataset
        self._log('Generating random patches...')
        rand_patches_per_img = img_name_idx // len(self._raw_image_files)
        for img_num, img_name, img_boxes in zip(range(num_orig_images), self._raw_image_files, self._all_labels):
            img = np.array(Image.open(img_name))

            for _ in range(rand_patches_per_img):
                img_patch, img_tblr = get_random_patch(img, self._patch_width, self._patch_height)
                raw_new_boxes = get_boxes_in_patch(img_tblr, img_boxes)
                new_boxes = []
                for box in raw_new_boxes:
                    new_boxes.append({"all_points_x": box[0:2], "all_points_y": box[2:4]})

                # Save patch to disk and store labels
                add_patch_to_dataset(img_patch, new_boxes, raw_new_boxes, img_name_idx)
                img_name_idx += 1

            self._log(str(img_num + 1) + '/' + str(num_orig_images))

        # Save all of the patch labels as a JSON file before returning the patch filenames and labels
        with open(json_file + 'train_patches.json', 'w', encoding='utf-8') as outfile:
            json.dump(img_dict, outfile)

        return new_raw_image_files, new_raw_labels

    def load_pascal_voc_labels_from_directory(self, data_dir):
        super().load_pascal_voc_labels_from_directory(data_dir)

        # need to add object-ness flag and one-hot encodings for class
        # it will be 1 or 0 for object-ness, one-hot for the class, then 4 bbox coords (x,y,w,h)
        # e.g. [1,0,0,...,1,...,0,223,364,58,62]
        # for scaling bbox coords
        # scaling image down to the grid size
        scale_ratio_w = self._grid_w / self._image_width
        scale_ratio_h = self._grid_h / self._image_height

        labels_with_one_hot = []
        for curr_img_coords in self._all_labels:
            curr_img_grid_locs = []  # for duplicates; current hacky fix
            curr_img_labels = np.zeros((self._grid_w * self._grid_h) * (1 + self._NUM_CLASSES + 4))

            # only one object per image so no need to loop here
            # add scaled bbox coords
            # x and y offsets from grid position
            w = curr_img_coords[1] - curr_img_coords[0]
            h = curr_img_coords[3] - curr_img_coords[2]
            x_center = (w / 2) + curr_img_coords[0]
            y_center = (h / 2) + curr_img_coords[2]
            x_grid = x_center * scale_ratio_w
            y_grid = y_center * scale_ratio_h
            x_grid_offset, x_grid_loc = np.modf(x_grid)
            y_grid_offset, y_grid_loc = np.modf(y_grid)

            # for duplicate object in grid checking
            if (x_grid_loc, y_grid_loc) in curr_img_grid_locs:
                continue
            else:
                curr_img_grid_locs.append((x_grid_loc, y_grid_loc))

            # w and h values on grid scale
            w_grid = w * scale_ratio_w
            h_grid = h * scale_ratio_h

            # compute grid-cell location
            # grid is defined as left-right, down, left-right, down... so in a 3x3 grid the middle left cell
            # would be 4 (or 3 when 0-indexing)
            grid_loc = (y_grid_loc * self._grid_w) + x_grid_loc

            # 1 for obj then 1 since only once class <- needs to be made more general for multiple classes
            # should be [1,0,...,1,...,0,x,y,w,h] where 0,...,1,...,0 represents the one-hot encoding of classes
            # maybe define a new list inside the loop, append a 1, then extend a one-hot list, then append
            # x,y,w,h then use the in this next line below
            # cur_box = []... vec_size = len(currbox)....
            vec_size = (1 + self._NUM_CLASSES + 4)
            curr_img_labels[int(grid_loc)*vec_size:(int(grid_loc)+1)*vec_size] = \
                [1, 1, x_grid_offset, y_grid_offset, w_grid, h_grid]
            # using extend because I had trouble with converting a list of lists to a tensor, so making it one list
            # of all the numbers and then reshaping later when we pull y off the train shuffle batch has been the
            # current hacky fix
            labels_with_one_hot.append(curr_img_labels)

        self._all_labels = labels_with_one_hot

    def load_json_labels_from_file(self, filename):
        super().load_json_labels_from_file(filename)

        if not self._json_no_convert:
            self._all_labels = self.__convert_labels_to_yolo_format()

    def __convert_labels_to_yolo_format(self):
        """Takes labels loaded from the custom json format and turns them into formatted arrays that the network and
        yolo loss function are expecting to work with
        :return: The converted labels
        """
        # for scaling bbox coords
        # scaling image down to the grid size
        scale_ratio_w = self._grid_w / self._image_width
        scale_ratio_h = self._grid_h / self._image_height

        labels_with_one_hot = []
        for curr_img_coords in self._all_labels:
            curr_img_grid_locs = []  # for duplicates; current hacky fix
            curr_img_labels = np.zeros((self._grid_w * self._grid_h) * (1 + self._NUM_CLASSES + 4))
            num_boxes = len(curr_img_coords)
            for i in range(num_boxes):
                # add scaled bbox coords
                # x and y offsets from grid position
                w = curr_img_coords[i][1] - curr_img_coords[i][0]
                h = curr_img_coords[i][3] - curr_img_coords[i][2]
                x_center = (w / 2) + curr_img_coords[i][0]
                y_center = (h / 2) + curr_img_coords[i][2]
                x_grid = x_center * scale_ratio_w
                y_grid = y_center * scale_ratio_h
                x_grid_offset, x_grid_loc = np.modf(x_grid)
                y_grid_offset, y_grid_loc = np.modf(y_grid)

                # for duplicate object in grid checking
                if (x_grid_loc, y_grid_loc) in curr_img_grid_locs:
                    continue
                else:
                    curr_img_grid_locs.append((x_grid_loc, y_grid_loc))

                # w and h values on grid scale
                w_grid = w * scale_ratio_w
                h_grid = h * scale_ratio_h

                # compute grid-cell location
                # grid is defined as left-right, down, left-right, down... so in a 3x3 grid the middle left cell
                # would be 4 (or 3 when 0-indexing)
                grid_loc = ((y_grid_loc * self._grid_w) + x_grid_loc) % (self._grid_h * self._grid_w)
                # the % (self._grid_h*self._grid_w) is to handle the rare case we are right on the edge and
                # we want the last 0-indexed grid position (off by 1 error, get 49 for 7x7 grid when should have 48)

                # 1 for obj then 1 since only one class <- needs to be made more general for multiple classes #
                # should be [1,0,...,1,...,0,x,y,w,h] where 0,...,1,...,0 represents the one-hot encoding of classes
                curr_box = [1, 1, x_grid_offset, y_grid_offset, w_grid, h_grid]

                vec_size = (1 + self._NUM_CLASSES + 4)
                curr_img_labels[int(grid_loc) * vec_size:(int(grid_loc) + 1) * vec_size] = curr_box
                # using extend because I had trouble with converting a list of lists to a tensor, so making it one list
                # of all the numbers and then reshaping later when we pull y off the train shuffle batch has been the
                # current hacky fix
            labels_with_one_hot.append(curr_img_labels)

        return labels_with_one_hot
