from . import layers, loaders, definitions, DPPModel
import numpy as np
import tensorflow as tf
import os
import json
import warnings
import copy
from collections.abc import Sequence
from scipy.special import expit
from PIL import Image
from tqdm import tqdm


class ObjectDetectionModel(DPPModel):
    _problem_type = definitions.ProblemType.OBJECT_DETECTION
    _loss_fn = 'yolo'
    _supported_loss_fns = ['yolo']
    _valid_augmentations = [definitions.AugmentationType.CONTRAST_BRIGHT]

    # State variables specific to object detection for constructing the graph and passing to Tensorboard
    _yolo_loss = None

    # Yolo-specific parameters, non-default values defined by set_yolo_parameters
    _grid_w = 7
    _grid_h = 7
    _LABELS = ['plant']
    _NUM_CLASSES = 1
    _RAW_ANCHORS = [(159, 157), (103, 133), (91, 89), (64, 65), (142, 101)]
    _ANCHORS = None  # Scaled version, but grid and image sizes are needed so default is deferred
    _NUM_BOXES = 5
    _THRESH_SIG = 0.6
    _THRESH_OVERLAP = 0.3
    _THRESH_CORRECT = 0.5

    def __init__(self, debug=False, load_from_saved=False, save_checkpoints=True, initialize=True, tensorboard_dir=None,
                 report_rate=100, save_dir=None):
        super().__init__(debug, load_from_saved, save_checkpoints, initialize, tensorboard_dir, report_rate, save_dir)

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
        :return Scalar Tensor with the Yolo loss for the bounding box predictions
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

        # this is how the paper does it, the above 6 lines is experimental
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
        super()._graph_tensorboard_summary(l2_cost, gradients, variables, global_grad_norm)

        # Summaries specific to object detection
        tf.summary.scalar('train/yolo_loss', self._yolo_loss, collections=['custom_summaries'])
        if self._validation:
            tf.summary.scalar('validation/loss', self._graph_ops['val_losses'],
                              collections=['custom_summaries'])

    def _assemble_graph(self):
        with self._graph.as_default():

            self._log('Parsing dataset...')
            self._graph_parse_data()

            self._log('Creating layer parameters...')
            self._add_layers_to_graph()

            self._log('Assembling graph...')

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

            # Reshape input to the expected image dimensions
            x = tf.reshape(x, shape=[-1, self._image_height, self._image_width, self._image_depth])

            # Deserialize the label
            y = loaders.label_string_to_tensor(y, self._batch_size)
            vec_size = 1 + self._NUM_CLASSES + 4
            y = tf.reshape(y, [self._batch_size, self._grid_w * self._grid_h, vec_size])

            # Run the network operations
            if self._has_moderation:
                xx = self.forward_pass(x, deterministic=False, moderation_features=mod_w)
            else:
                xx = self.forward_pass(x, deterministic=False)

            # Define regularization cost
            if self._reg_coeff is not None:
                l2_cost = tf.squeeze(tf.reduce_sum(
                    [layer.regularization_coefficient * tf.nn.l2_loss(layer.weights) for layer in self._layers
                     if isinstance(layer, layers.fullyConnectedLayer)]))
            else:
                l2_cost = 0.0

            # Define cost function  based on which one was selected via set_loss_function
            if self._loss_fn == 'yolo':
                self._yolo_loss = self._yolo_loss_function(
                    y, tf.reshape(xx, [self._batch_size,
                                       self._grid_w * self._grid_h,
                                       self._NUM_BOXES * 5 + self._NUM_CLASSES]))
            self._graph_ops['cost'] = tf.squeeze(tf.add(self._yolo_loss, l2_cost))

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

            vec_size = 1 + self._NUM_CLASSES + 4
            if self._testing:
                x_test = tf.reshape(x_test, shape=[-1, self._image_height, self._image_width, self._image_depth])
                self._graph_ops['y_test'] = loaders.label_string_to_tensor(self._graph_ops['y_test'],
                                                                           self._batch_size)
                self._graph_ops['y_test'] = tf.reshape(self._graph_ops['y_test'],
                                                       shape=[self._batch_size,
                                                              self._grid_w * self._grid_h,
                                                              vec_size])
            if self._validation:
                x_val = tf.reshape(x_val, shape=[-1, self._image_height, self._image_width, self._image_depth])
                self._graph_ops['y_val'] = loaders.label_string_to_tensor(self._graph_ops['y_val'],
                                                                          self._batch_size)
                self._graph_ops['y_val'] = tf.reshape(self._graph_ops['y_val'],
                                                      shape=[self._batch_size,
                                                             self._grid_w * self._grid_h,
                                                             vec_size])

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

            # For object detection, the network outputs need to be reshaped to match y_test and y_val
            if self._testing:
                self._graph_ops['x_test_predicted'] = tf.reshape(self._graph_ops['x_test_predicted'],
                                                                 [self._batch_size,
                                                                  self._grid_w * self._grid_h,
                                                                  self._NUM_BOXES * 5 + self._NUM_CLASSES])
            if self._validation:
                self._graph_ops['x_val_predicted'] = tf.reshape(self._graph_ops['x_val_predicted'],
                                                                [self._batch_size,
                                                                 self._grid_w * self._grid_h,
                                                                 self._NUM_BOXES * 5 + self._NUM_CLASSES])

            # compute the loss and accuracy based on problem type
            if self._testing:
                if self._loss_fn == 'yolo':
                    self._graph_ops['test_losses'] = self._yolo_loss_function(self._graph_ops['y_test'],
                                                                              self._graph_ops['x_test_predicted'])
            if self._validation:
                if self._loss_fn == 'yolo':
                    self._graph_ops['val_losses'] = self._yolo_loss_function(self._graph_ops['y_val'],
                                                                             self._graph_ops['x_val_predicted'])

            # Epoch summaries for Tensorboard
            self._graph_tensorboard_summary(l2_cost, gradients, variables, global_grad_norm)

    def compute_full_test_accuracy(self):
        self._log('Computing total test accuracy/regression loss...')

        with self._graph.as_default():
            num_test = self._total_raw_samples - self._total_training_samples
            num_batches = int(np.ceil(num_test / self._batch_size))

            if num_batches == 0:
                warnings.warn('Less than a batch of testing data')
                exit()

            # Initialize storage for the retreived test variables. Object detection needs some special care given to
            # its variable's shape
            all_y = np.empty(shape=(1,
                                    self._grid_w * self._grid_h,
                                    1 + self._NUM_CLASSES + 4))
            all_predictions = np.empty(shape=(1,
                                              self._grid_w * self._grid_h,
                                              5 * self._NUM_BOXES + self._NUM_CLASSES))

            # Main test loop
            for _ in tqdm(range(num_batches)):
                r_y, r_predicted = self._session.run([self._graph_ops['y_test'],
                                                      self._graph_ops['x_test_predicted']])
                all_y = np.concatenate((all_y, r_y), axis=0)
                all_predictions = np.concatenate((all_predictions, r_predicted), axis=0)

            # Delete the weird first entries in losses, y values, and predictions (because creating empty arrays
            # isn't a thing)
            if self._problem_type == definitions.ProblemType.OBJECT_DETECTION:
                # These are multi-dimensional for object detection, so first entry = first slice
                all_y = np.delete(all_y, 0, axis=0)
                all_predictions = np.delete(all_predictions, 0, axis=0)

            # Delete the extra entries (e.g. batch_size is 4 and 1 sample left, it will loop and have 3 repeats that
            # we want to get rid of)
            extra = self._batch_size - (self._total_testing_samples % self._batch_size)
            if extra != self._batch_size:
                mask_extra = np.ones(self._batch_size * num_batches, dtype=bool)
                mask_extra[range(self._batch_size * num_batches - extra, self._batch_size * num_batches)] = False
                all_y = all_y[mask_extra, ...]
                all_predictions = all_predictions[mask_extra, ...]

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

    def __yolo_coord_convert(self, labels, preds):
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

        # Extract the class predictions and reorganize the predicted boxes
        class_preds = preds[..., self._NUM_BOXES * 5:]
        preds = np.reshape(preds[..., 0:self._NUM_BOXES * 5], preds.shape[:-1] + (self._NUM_BOXES, 5))

        # Predictions are not, so apply sigmoids and exponentials first and then convert them
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
            return 0

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

    def forward_pass_with_file_inputs(self, x):
        with self._graph.as_default():
            if self._with_patching:
                # we want the largest multiple of patch height/width that is smaller than the original
                # image height/width, for the final image dimensions
                patch_height = self._patch_height
                patch_width = self._patch_width
                final_height = (self._image_height // patch_height) * patch_height
                final_width = (self._image_width // patch_width) * patch_width
                num_patches_vert = self._image_height // patch_height
                num_patches_horiz = self._image_width // patch_width
                # find image differences to determine recentering crop coords, we divide by 2 so that the leftover
                # is equal on all sides of image
                offset_height = (self._image_height - final_height) // 2
                offset_width = (self._image_width - final_width) // 2
                # pre-allocate output dimensions
                total_outputs = np.empty([1,
                                          num_patches_horiz * num_patches_vert,
                                          self._grid_w * self._grid_h * (5 * self._NUM_BOXES + self._NUM_CLASSES)])
            else:
                total_outputs = np.empty(
                    [1, self._grid_w * self._grid_h * (5 * self._NUM_BOXES + self._NUM_CLASSES)])

            num_batches = len(x) // self._batch_size
            remainder = len(x) % self._batch_size

            if remainder != 0:
                num_batches += 1
                remainder = self._batch_size - remainder

            # self.load_images_from_list(x) no longer calls following 2 lines so we needed to force them here
            images = x
            self._parse_images(images)

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
            self._initialize_queue_runners()
            # Run model on them
            x_pred = self.forward_pass(x_test, deterministic=True)

            if self._with_patching:
                for i in range(int(num_batches)):
                    xx = self._session.run(x_pred)
                    xx = np.reshape(xx, [self._batch_size, num_patches_vert * num_patches_horiz, -1])
                    for img in np.array_split(xx, self._batch_size):
                        total_outputs = np.append(total_outputs, img, axis=0)
            else:
                for i in range(int(num_batches)):
                    xx = self._session.run(x_pred)
                    xx = np.reshape(xx, [self._batch_size, -1])
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
                # we want the largest multiple of patch height/width that is smaller than the original
                # image height/width, for the final image dimensions
                patch_height = self._patch_height
                patch_width = self._patch_width
                final_height = (self._image_height // patch_height) * patch_height
                final_width = (self._image_width // patch_width) * patch_width
                num_patches_vert = self._image_height // patch_height
                num_patches_horiz = self._image_width // patch_width
                # find image differences to determine recentering crop coords, we divide by 2 so that the leftover
                # is equal on all sides of image
                offset_height = (self._image_height - final_height) // 2
                offset_width = (self._image_width - final_width) // 2
                # pre-allocate output dimensions
                total_outputs = np.empty([1,
                                          num_patches_horiz * num_patches_vert,
                                          self._grid_w * self._grid_h * (5 * self._NUM_BOXES + self._NUM_CLASSES)])
            else:
                total_outputs = np.empty(
                    [1, self._grid_w * self._grid_h * (5 * self._NUM_BOXES + self._NUM_CLASSES)])
            num_batches = len(x) // self._batch_size
            remainder = len(x) % self._batch_size

            if remainder != 0:
                num_batches += 1
                remainder = self._batch_size - remainder

            # self.load_images_from_list(x) no longer calls following 2 lines so we needed to force them here
            images = x
            self._parse_images(images)

            x_test = tf.train.batch([self._all_images], batch_size=self._batch_size,
                                    num_threads=self._num_threads)
            x_test = tf.reshape(x_test, shape=[-1, self._image_height, self._image_width, self._image_depth])
            if self._with_patching:
                x_test = tf.image.crop_to_bounding_box(x_test, offset_height, offset_width, final_height,
                                                       final_width)
                # Split the images up into the multiple slices of size patch_height x patch_width
                ksizes = [1, patch_height, patch_width, 1]
                strides = [1, patch_height, patch_width, 1]
                rates = [1, 1, 1, 1]
                x_test = tf.extract_image_patches(x_test, ksizes, strides, rates, "VALID")
                x_test = tf.reshape(x_test, shape=[-1, patch_height, patch_width, self._image_depth])

            if self._load_from_saved:
                self.load_state()
            self._initialize_queue_runners()
            # Run model on them
            x_pred = self.forward_pass(x_test, deterministic=True)

            if self._with_patching:
                for i in range(int(num_batches)):
                    xx = self._session.run(x_pred)
                    xx = np.reshape(xx, [self._batch_size, num_patches_vert * num_patches_horiz, -1])
                    for img in np.array_split(xx, self._batch_size):
                        total_outputs = np.append(total_outputs, img, axis=0)
            else:
                for i in range(int(num_batches)):
                    xx = self._session.run(x_pred)
                    xx = np.reshape(xx, [self._batch_size, -1])
                    for img in np.array_split(xx, self._batch_size):
                        total_outputs = np.append(total_outputs, img, axis=0)

            # delete weird first row
            total_outputs = np.delete(total_outputs, 0, 0)
            # delete any outputs which are overruns from the last batch
            if remainder != 0:
                for i in range(remainder):
                    total_outputs = np.delete(total_outputs, -1, 0)

        # Perform yolo needs
        # this is currently for patching, need a way to be more general or maybe just need to write both ways out
        # fully
        total_pred_boxes = []
        if self._with_patching:
            num_patches = num_patches_vert * num_patches_horiz
            for img_data in total_outputs:
                ###################################################################################################
                # img_data is [x,y,w,h,conf,x,y,w,h,conf,x,y,......, classes]
                # currently 5 boxes and 1 class are fixed amounts, hence we pull 5 box confs and we use multiples
                # of 26 because 5 (boxes) * 5 (x,y,w,h,conf) + 1 (class) = 26
                # this may likely need to be made more general in future
                ###################################################################################################
                for i in range(num_patches):
                    for j in range(self._grid_w * self._grid_h):
                        # We first find the responsible box by finding the one with the highest confidence
                        box_conf1 = expit(img_data[i, j * 26 + 4])
                        box_conf2 = expit(img_data[i, j * 26 + 9])
                        box_conf3 = expit(img_data[i, j * 26 + 14])
                        box_conf4 = expit(img_data[i, j * 26 + 19])
                        box_conf5 = expit(img_data[i, j * 26 + 24])
                        box_confs = [box_conf1, box_conf2, box_conf3, box_conf4, box_conf5]
                        max_conf_idx = int(np.argmax(box_confs))
                        # Then we check if the responsible box is above the threshold for detecting an object
                        if box_confs[max_conf_idx] > 0.6:
                            # This box has detected an object, so we extract and convert its coords
                            pred_box = img_data[i, j*26+5*max_conf_idx:j*26+5*max_conf_idx+4]

                            # centers from which x and y offsets are applied to, these are in 'grid coords'
                            c_x = j % self._grid_w
                            c_y = j // self._grid_w
                            # x and y go from 'grid coords' to 'patch coords' to 'full img coords'
                            x = (expit(pred_box[0]) + c_x) * (patch_width / self._grid_w) \
                                + (i % num_patches_horiz) * patch_width
                            y = (expit(pred_box[1]) + c_y) * (patch_height / self._grid_h) \
                                + (i // num_patches_horiz) * patch_height
                            # get the anchor box based on the highest conf (responsible box)
                            prior_w = self._ANCHORS[max_conf_idx][0]
                            prior_h = self._ANCHORS[max_conf_idx][1]
                            # w and h go from 'grid coords' to 'full img coords'
                            w = (np.exp(pred_box[2]) * prior_w) * (self._image_width / self._grid_w)
                            h = (np.exp(pred_box[3]) * prior_h) * (self._image_height / self._grid_h)
                            # turn into points
                            x1y1 = (int(x - w / 2), int(y - h / 2))
                            x2y2 = (int(x + w / 2), int(y + h / 2))
                            total_pred_boxes.append([x1y1[0], x1y1[1], x2y2[0], x2y2[1], box_confs[max_conf_idx]])

                # Non - maximal suppression (Probably make into a general function)
                all_boxes = np.array(total_pred_boxes)
                idxs = np.argsort(all_boxes[:, 4])  # sorts them smallest to largest by confidence
                final_boxes_idxs = []
                while len(idxs) > 0:  # sometimes we may delete multiple boxes so we use a while instead of for
                    last = len(idxs) - 1  # since sorted in reverse order, the last one has the highest conf
                    i = idxs[last]
                    final_boxes_idxs.append(i)  # add it to the maximal list, then check for duplicates to delete
                    suppress = [last]  # this is the list of idxs of boxes to stop checking (they will deleted)
                    for pos in range(0, last):  # search for duplicates
                        j = idxs[pos]
                        iou = self.__compute_iou(all_boxes[i], all_boxes[j])
                        if iou > 0.3:  # maybe should make this a tunable parameter
                            suppress.append(pos)
                    idxs = np.delete(idxs, suppress)  # remove the box that was added and its duplicates

            # [[x1,y1,x2,y2,conf],[x1,y1,x2,y2,conf],...]
            interpreted_outputs = np.array(all_boxes[final_boxes_idxs, :])
            return interpreted_outputs
        else:
            # no patching
            print(total_outputs.shape)
            for img_data in total_outputs:
                ###################################################################################################
                # img_data is [x,y,w,h,conf,x,y,w,h,conf,x,y,......, classes]
                # currently 5 boxes and 1 class are fixed amounts, hence we pull 5 box confs and we use multiples
                # of 26 because 5 (boxes) * 5 (x,y,w,h,conf) + 1 (class) = 26
                # this may likely need to be made more general in future
                ###################################################################################################
                for i in range(self._grid_w * self._grid_h):
                    # x,y,w,h,conf,x,y,w,h,cong,x,y,...... classes
                    box_conf1 = expit(img_data[i * 26 + 4])
                    box_conf2 = expit(img_data[i * 26 + 9])
                    box_conf3 = expit(img_data[i * 26 + 14])
                    box_conf4 = expit(img_data[i * 26 + 19])
                    box_conf5 = expit(img_data[i * 26 + 24])
                    box_confs = [box_conf1, box_conf2, box_conf3, box_conf4, box_conf5]
                    max_conf_idx = int(np.argmax(box_confs))

                    if box_confs[max_conf_idx] > 0.6:
                        pred_box = img_data[i * 26 + 5 * max_conf_idx: i * 26 + 5 * max_conf_idx + 4]

                        # centers from which x and y offsets are applied to, these are in 'grid coords'
                        c_x = i % self._grid_w
                        c_y = i // self._grid_w
                        # x and y go from 'grid coords' to 'full img coords'
                        x = (expit(pred_box[0]) + c_x) * (self._image_width / self._grid_w)
                        y = (expit(pred_box[1]) + c_y) * (self._image_height / self._grid_h)
                        # get the anchor box based on the highest conf (responsible box)
                        prior_w = self._ANCHORS[max_conf_idx][0]
                        prior_h = self._ANCHORS[max_conf_idx][1]
                        # w and h go from 'grid coords' to 'full img coords'
                        w = (np.exp(pred_box[2]) * prior_w) * (self._image_width / self._grid_w)
                        h = (np.exp(pred_box[3]) * prior_h) * (self._image_height / self._grid_h)
                        x1y1 = (int(x - w / 2), int(y - h / 2))
                        x2y2 = (int(x + w / 2), int(y + h / 2))
                        total_pred_boxes.append([x1y1[0], x1y1[1], x2y2[0], x2y2[1], box_confs[max_conf_idx]])

                # Non - maximal suppression (Probably make into a general function)
                all_boxes = np.array(total_pred_boxes)
                idxs = np.argsort(all_boxes[:, 4])  # sorts them smallest to largest by confidence
                final_boxes_idxs = []
                while len(idxs) > 0:  # sometimes we may delete multiple boxes so we use a while instead of for
                    last = len(idxs) - 1  # since sorted in reverse order, the last one has highest conf
                    i = idxs[last]
                    final_boxes_idxs.append(i)  # add it to the maximal list, then check for duplicates to delete
                    suppress = [last]  # this is the list of idxs of boxes to stop checking (they will deleted)
                    for pos in range(0, last):  # search for duplicates
                        j = idxs[pos]
                        iou = self.__compute_iou(all_boxes[i], all_boxes[j])
                        if iou > 0.3:  # maybe should make this a tunable parameter
                            suppress.append(pos)
                    idxs = np.delete(idxs, suppress)  # remove the box that was added and its duplicates

            # [[x1,y1,x2,y2,conf],[x1,y1,x2,y2,conf],...]
            interpreted_outputs = np.array(all_boxes[final_boxes_idxs, :])
            return interpreted_outputs

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
            if self._problem_type is definitions.ProblemType.OBJECT_DETECTION:
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

    def load_yolo_dataset_from_directory(self, data_dir, label_file=None, image_dir=None):
        """
        Loads in labels and images for object detection tasks, converting the labels to YOLO format and automatically
        patching the images if necessary.

        :param data_dir: String, The directory where the labels and images are stored in
        :param label_file: String, The filename for the JSON file with the labels. Optional if using automatic patching
        which has been done already
        :param image_dir: String, The directory with the images. Optional if using automatic patching which has been
        done already
        """
        load_patched_data = self._with_patching and 'tmp_train' in os.listdir(data_dir)

        # Construct the paths to the labels and images
        if load_patched_data:
            label_path = os.path.join(data_dir, 'tmp_train/json/train_patches.json')
            image_path = os.path.join(data_dir, 'tmp_train/image_patches', '')
        else:
            label_path = os.path.join(data_dir, label_file)
            image_path = os.path.join(data_dir, image_dir, '')

        # Load the labels and images
        if load_patched_data:
            # Hack to make the label reader convert the labels to YOLO format when re-reading image patches
            self._with_patching = False
        self.load_json_labels_from_file(label_path)
        images_list = [image_path + filename for filename in sorted(os.listdir(image_path))
                       if filename.endswith('.png')]
        self.load_images_from_list(images_list)
        if load_patched_data:
            # Remove the hack
            self._with_patching = True

        # Perform automatic image patching if necessary
        if self._with_patching and 'tmp_train' not in os.listdir(data_dir):
            self._raw_image_files, self._all_labels = \
                self.__object_detection_patching_and_augmentation(patch_dir=data_dir)
            self.__convert_labels_to_yolo_format()
            self._raw_labels = self._all_labels
            self._total_raw_samples = len(self._raw_image_files)
            self._log('Total raw patch examples is %d' % self._total_raw_samples)

    def __object_detection_patching_and_augmentation(self, patch_dir=None):
        # make the below a function
        # labels, images = function()
        img_dict = {}
        img_num = 0
        img_name_idx = 1

        if patch_dir:
            patch_dir = os.path.join(patch_dir, 'tmp_train', '')
        else:
            patch_dir = os.path.join(os.path.curdir, 'tmp_train', '')
        if not os.path.exists(patch_dir):
            os.makedirs(patch_dir)
        else:
            raise RuntimeError("Patched images already exist in " + patch_dir + ". Either delete them and run again or "
                               "use them directly (i.e. without patching).")

        img_dir_out = patch_dir + 'image_patches/'
        if not os.path.exists(img_dir_out):
            os.makedirs(img_dir_out)
        json_dir_out = patch_dir + 'json/'
        if not os.path.exists(json_dir_out):
            os.makedirs(json_dir_out)
        new_raw_image_files = []
        new_raw_labels = []

        # first add images such that each grid cell has a plant in it
        # should add num_images*grid many images (e.g. 27(images)*49(7x7grid))
        self._log('Beginning creation of training patches. Images and json are being saved in ' + patch_dir)
        for img_name, img_boxes in zip(self._raw_image_files, self._all_labels):
            img_num += 1
            img = np.array(Image.open(img_name))

            # take patches that have a plant in each grid cell to ensure come training time that each grid cell learns
            # to recognize an object
            for i in range(self._grid_h):
                for j in range(self._grid_w):
                    # choose plant randomly (and far enough from edges)
                    found_one = False
                    failed = False
                    find_count = 0
                    random_indices = list(range(len(img_boxes)))
                    while found_one is False:
                        rand_idx = np.random.randint(0, len(random_indices))
                        rand_plant_idx = random_indices[rand_idx]
                        box_w = img_boxes[rand_plant_idx][1] - img_boxes[rand_plant_idx][0]
                        box_h = img_boxes[rand_plant_idx][3] - img_boxes[rand_plant_idx][2]
                        box_x = img_boxes[rand_plant_idx][0] + box_w / 2
                        box_y = img_boxes[rand_plant_idx][2] + box_h / 2
                        if (self._patch_width + 5) < box_x < (img.shape[1] - (self._patch_width + 5)) \
                                and (self._patch_height + 5) < box_y < (img.shape[0] - (self._patch_height + 5)):
                            found_one = True

                            # adjust center based on target grid location
                            center_x = self._grid_w // 2
                            center_y = self._grid_h // 2
                            delta_x = j - center_x
                            delta_y = i - center_y
                            # note we need to invert the direction of delta_x so as to move the center to where we
                            # want it to be, hence subtraction
                            new_x = int(box_x - (delta_x * (self._patch_width / self._grid_w)))
                            new_y = int(box_y - (delta_y * (self._patch_height / self._grid_h)))

                            top_row = new_y - (self._patch_height // 2)
                            bot_row = top_row + self._patch_height
                            left_col = new_x - (self._patch_width // 2)
                            right_col = left_col + self._patch_width

                            img_patch = img[top_row:bot_row, left_col:right_col]

                            # search for, adjust, and add bbox coords for the json
                            new_boxes = []
                            new_raw_boxes = []
                            for box in img_boxes:
                                # check if box is inside current patch, if so convert the coords and add it to the json
                                box_w = box[1] - box[0]
                                box_h = box[3] - box[2]
                                box_x = box[0] + box_w / 2
                                box_y = box[2] + box_h / 2
                                if (box_x >= left_col) and (box_x <= right_col) and (box_y >= top_row) and (
                                        box_y <= bot_row):
                                    delta_x = box_x - new_x
                                    delta_y = box_y - new_y
                                    new_x_center = self._patch_width // 2 + delta_x
                                    new_y_center = self._patch_height // 2 + delta_y
                                    new_x_min = new_x_center - box_w / 2
                                    new_x_max = new_x_min + box_w
                                    new_y_min = new_y_center - box_h / 2
                                    new_y_max = new_y_min + box_h

                                    new_boxes.append({"all_points_x": [new_x_min, new_x_max],
                                                      "all_points_y": [new_y_min, new_y_max]})
                                    new_raw_boxes.append([new_x_min, new_x_max, new_y_min, new_y_max])

                            # save image to disk
                            # print(top_row, bot_row, left_col, right_col)
                            result = Image.fromarray(img_patch.astype(np.uint8))
                            new_img_name = img_dir_out + "{:0>6d}".format(img_name_idx) + '.png'
                            result.save(new_img_name)

                            new_raw_image_files.append(new_img_name)
                            new_raw_labels.append(new_raw_boxes)

                            img_dict["{:0>6d}".format(img_name_idx)] = {"height": self._patch_height,
                                                                        "width": self._patch_width,
                                                                        "file_name": "{:0>6d}".format(
                                                                            img_name_idx) + '.png',
                                                                        "plants": new_boxes}
                            img_name_idx += 1
                        else:
                            del random_indices[rand_idx]
                        find_count += 1
                        if find_count == len(img_boxes):
                            failed = True
                            break
                    if failed:
                        break

            self._log(str(img_num) + '/' + str(len(self._all_labels)))
        self._log('Completed baseline train patches set. Total images: ' + str(img_name_idx))

        # augmentation images: rotations, brightness, flips
        self._log('Beginning creating of augmentation patches')
        for i in range(self._grid_h * self._grid_w):
            for img_name, img_boxes in zip(self._raw_image_files, self._all_labels):
                img = np.array(Image.open(img_name))
                # randomly grab a patch, make sure it has at least one plant in it
                max_width = img.shape[1] - (self._patch_width // 2)
                min_width = (self._patch_width // 2)
                max_height = img.shape[0] - (self._patch_height // 2)
                min_height = (self._patch_height // 2)
                found_one = False
                while found_one is False:
                    rand_x = np.random.randint(min_width, max_width + 1)
                    rand_y = np.random.randint(min_height, max_height + 1)
                    # determine patch location and slice into mask and img to create patch
                    top_row = rand_y - (self._patch_height // 2)
                    bot_row = top_row + self._patch_height
                    left_col = rand_x - (self._patch_width // 2)
                    right_col = left_col + self._patch_width
                    img_patch = img[top_row:bot_row, left_col:right_col]
                    # objects and corresponding bboxes
                    new_boxes = []
                    for box in img_boxes:
                        cent_x = box[0] + ((box[1] - box[0]) / 2)
                        cent_y = box[2] + ((box[3] - box[2]) / 2)
                        # check if box is inside current patch, if so convert the coords and add it to the json
                        if (cent_x >= left_col) and (cent_x <= right_col) and (cent_y >= top_row) and (
                                cent_y <= bot_row):
                            box_w = box[1] - box[0]
                            box_h = box[3] - box[2]
                            box_x = box[0] + box_w / 2
                            box_y = box[2] + box_h / 2
                            delta_x = box_x - rand_x
                            delta_y = box_y - rand_y
                            new_x_center = self._patch_width // 2 + delta_x
                            new_y_center = self._patch_height // 2 + delta_y
                            new_x_min = new_x_center - box_w / 2
                            new_x_max = new_x_min + box_w
                            new_y_min = new_y_center - box_h / 2
                            new_y_max = new_y_min + box_h
                            new_boxes.append([new_x_min, new_x_max, new_y_min, new_y_max])
                    if len(new_boxes) >= 1:
                        found_one = True

                        # augmentation is a random choice of 3 options
                        # 1 == rotation, 2 == brightness, 3 == flip
                        aug = np.random.randint(1, 4)
                        if aug == 1:
                            # rotation
                            k = np.random.randint(1, 4)
                            rot_img_patch = np.rot90(img_patch, k)
                            theta = np.radians(90 * k)
                            x0 = self._patch_width // 2
                            y0 = self._patch_height // 2
                            rot_boxes = []
                            raw_rot_boxes = []
                            for box in new_boxes:
                                # since only rotating by 90 degrees we could probably hard code in 1's, -1's, and 0's
                                # in the cases instead of using sin and cos
                                rot_x_min = x0 + (box[0] - x0) * np.cos(theta) + (box[2] - y0) * np.sin(theta)
                                rot_y_min = y0 - (box[0] - x0) * np.sin(theta) + (box[2] - y0) * np.cos(theta)
                                w = box[1] - box[0]
                                h = box[3] - box[2]
                                if k == 1:
                                    # w and h flip, x_min y_min become x_min y_max
                                    w, h = h, w
                                    rot_y_min -= h
                                elif k == 2:
                                    # w and h stay same, x_min y_min become x_max y_max
                                    rot_x_min -= w
                                    rot_y_min -= h
                                else:  # k == 3
                                    # w and h flip, x_min y_min become x_max y_min
                                    w, h = h, w
                                    rot_x_min -= w
                                rot_x_max = rot_x_min + w
                                rot_y_max = rot_y_min + h

                                rot_boxes.append({"all_points_x": [rot_x_min, rot_x_max],
                                                  "all_points_y": [rot_y_min, rot_y_max]})
                                raw_rot_boxes.append([rot_x_min, rot_x_max, rot_y_min, rot_y_max])
                            # save image to disk
                            result = Image.fromarray(rot_img_patch.astype(np.uint8))
                            new_img_name = img_dir_out + "{:0>6d}".format(img_name_idx) + '.png'
                            result.save(new_img_name)

                            new_raw_image_files.append(new_img_name)
                            new_raw_labels.append(raw_rot_boxes)

                            img_dict["{:0>6d}".format(img_name_idx)] = {"height": self._patch_height,
                                                                        "width": self._patch_width,
                                                                        "file_name": "{:0>6d}".format(img_name_idx) +
                                                                                     '.png',
                                                                        "plants": rot_boxes}
                            img_name_idx += 1
                        elif aug == 2:
                            # brightness
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

                            # save image to disk
                            result = Image.fromarray(bright_img_patch.astype(np.uint8))
                            new_img_name = img_dir_out + "{:0>6d}".format(img_name_idx) + '.png'
                            result.save(new_img_name)

                            new_raw_image_files.append(new_img_name)
                            new_raw_labels.append(raw_bright_boxes)

                            img_dict["{:0>6d}".format(img_name_idx)] = {"height": self._patch_height,
                                                                        "width": self._patch_width,
                                                                        "file_name": "{:0>6d}".format(img_name_idx) +
                                                                                     '.png',
                                                                        "plants": bright_boxes}
                            img_name_idx += 1

                        else:  # aug == 3
                            # flip
                            k = np.random.random()
                            if k < 0.5:
                                flip_img_patch = np.fliplr(img_patch)
                                flip_boxes = []
                                raw_flip_boxes = []
                                for box in new_boxes:
                                    w = box[1] - box[0]
                                    # h = box[3] - box[2] for reference
                                    x_min = self._patch_width - (box[1])
                                    x_max = x_min + w
                                    y_min = box[2]
                                    y_max = box[3]

                                    flip_boxes.append({"all_points_x": [x_min, x_max],
                                                       "all_points_y": [y_min, y_max]})
                                    raw_flip_boxes.append([x_min, x_max, y_min, y_max])

                                result = Image.fromarray(flip_img_patch.astype(np.uint8))
                                new_img_name = img_dir_out + "{:0>6d}".format(img_name_idx) + '.png'
                                result.save(new_img_name)

                                new_raw_image_files.append(new_img_name)
                                new_raw_labels.append(raw_flip_boxes)

                                img_dict["{:0>6d}".format(img_name_idx)] = {"height": self._patch_height,
                                                                            "width": self._patch_width,
                                                                            "file_name": "{:0>6d}".format(
                                                                                img_name_idx) + '.png',
                                                                            "plants": flip_boxes}
                                img_name_idx += 1
                            else:
                                flip_img_patch = np.flipud(img_patch)
                                flip_boxes = []
                                raw_flip_boxes = []
                                for box in new_boxes:
                                    # w = box[1] - box[0] for reference
                                    h = box[3] - box[2]
                                    x_min = box[0]
                                    x_max = box[1]
                                    y_min = self._patch_height - (box[3])
                                    y_max = y_min + h

                                    flip_boxes.append({"all_points_x": [x_min, x_max],
                                                       "all_points_y": [y_min, y_max]})
                                    raw_flip_boxes.append([x_min, x_max, y_min, y_max])

                                result = Image.fromarray(flip_img_patch.astype(np.uint8))
                                new_img_name = img_dir_out + "{:0>6d}".format(img_name_idx) + '.png'
                                result.save(new_img_name)

                                new_raw_image_files.append(new_img_name)
                                new_raw_labels.append(raw_flip_boxes)

                                img_dict["{:0>6d}".format(img_name_idx)] = {"height": self._patch_height,
                                                                            "width": self._patch_width,
                                                                            "file_name": "{:0>6d}".format(
                                                                                img_name_idx) + '.png',
                                                                            "plants": flip_boxes}
                                img_name_idx += 1
            self._log(str(i + 1) + '/' + str(self._grid_w * self._grid_h))
        self._log('Completed augmentation set. Total images: ' + str(img_name_idx))

        # rest are just random patches
        num_patches = img_name_idx // len(self._raw_image_files)
        self._log('Generating random patches')
        img_num = 0
        random_imgs = 0
        for img_name, img_boxes in zip(self._raw_image_files, self._all_labels):
            img_num += 1
            img = np.array(Image.open(img_name))
            # we will randomly generate centers of the images we are extracting
            #  with size: patch_size x patch_size
            max_width = img.shape[1] - (self._patch_width // 2)
            min_width = (self._patch_width // 2)
            max_height = img.shape[0] - (self._patch_height // 2)
            min_height = (self._patch_height // 2)
            rand_x = np.random.randint(min_width, max_width + 1, num_patches)
            rand_y = np.random.randint(min_height, max_height + 1, num_patches)

            for idx, center in enumerate(zip(rand_x, rand_y)):
                # determine patch location and slice into mask and img to create patch
                top_row = center[1] - (self._patch_height // 2)
                bot_row = top_row + self._patch_height
                left_col = center[0] - (self._patch_width // 2)
                right_col = left_col + self._patch_width
                img_patch = img[top_row:bot_row, left_col:right_col]

                # objects and corresponding bboxes
                new_boxes = []
                raw_new_boxes = []
                for box in img_boxes:
                    cent_x = box[0] + ((box[1] - box[0]) / 2)
                    cent_y = box[2] + ((box[3] - box[2]) / 2)
                    # check if box is inside current patch, if so convert the coords and add it to the json
                    if (cent_x >= left_col) and (cent_x <= right_col) and (cent_y >= top_row) and (cent_y <= bot_row):
                        box_w = box[1] - box[0]
                        box_h = box[3] - box[2]
                        box_x = box[0] + box_w / 2
                        box_y = box[2] + box_h / 2
                        delta_x = box_x - center[0]
                        delta_y = box_y - center[1]
                        new_x_center = self._patch_width // 2 + delta_x
                        new_y_center = self._patch_height // 2 + delta_y
                        new_x_min = new_x_center - box_w / 2
                        new_x_max = new_x_min + box_w
                        new_y_min = new_y_center - box_h / 2
                        new_y_max = new_y_min + box_h

                        new_boxes.append({"all_points_x": [new_x_min, new_x_max],
                                          "all_points_y": [new_y_min, new_y_max]})
                        raw_new_boxes.append([new_x_min, new_x_max, new_y_min, new_y_max])

                # save image to disk
                result = Image.fromarray(img_patch.astype(np.uint8))
                new_img_name = img_dir_out + "{:0>6d}".format(img_name_idx) + '.png'
                result.save(new_img_name)

                new_raw_image_files.append(new_img_name)
                new_raw_labels.append(raw_new_boxes)

                img_dict["{:0>6d}".format(img_name_idx)] = {"height": self._patch_height,
                                                            "width": self._patch_width,
                                                            "file_name": "{:0>6d}".format(img_name_idx) + '.png',
                                                            "plants": new_boxes}
                img_name_idx += 1

            # verbose
            random_imgs += 1
            self._log(str(random_imgs) + '/' + str(len(self._raw_image_files)))

        # save into json
        with open(json_dir_out + 'train_patches.json', 'w') as outfile:
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
            # using extend because I had trouble with converting a list of lists to a tensor using our string
            # queues, so making it one list of all the numbers and then reshaping later when we pull y off the
            # train shuffle batch has been the current hacky fix
            labels_with_one_hot.append(curr_img_labels)

        self._all_labels = labels_with_one_hot

    def load_json_labels_from_file(self, filename):
        super().load_json_labels_from_file(filename)

        if not self._with_patching:
            self.__convert_labels_to_yolo_format()

    def __convert_labels_to_yolo_format(self):
        """Takes the labels that are in the json format and turns them into formatted arrays
        that the network and yolo loss function are expecting to work with"""

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
                # using extend because I had trouble with converting a list of lists to a tensor using our string
                # queues, so making it one list of all the numbers and then reshaping later when we pull y off the
                # train shuffle batch has been the current hacky fix
            labels_with_one_hot.append(curr_img_labels)

        self._all_labels = labels_with_one_hot
