from layers import *
from loaders import *
from preprocessing import *
from definitions import *
import tensorflow as tf
import numpy as np
from joblib import Parallel, delayed
import os
import datetime
import time
import warnings


class DPPModel(object):
    # Operation settings
    __problem_type = ProblemType.CLASSIFICATION

    # Input options
    __total_classes = 0
    __total_raw_samples = 0
    __total_training_samples = 0

    __image_width = None
    __image_height = None
    __image_depth = None
    __image_size = None

    __crop_or_pad_images = False
    __resize_images = False

    __preprocessing_steps = []
    __processed_images_dir = '../DPP-Processed'

    # Augmentation options
    __augmentation_flip = False
    __augmentation_crop = False
    __augmentation_contrast = False

    # Dataset storage
    __all_ids = None
    __train_images = None
    __test_images = None

    __all_labels = None
    __train_labels = None
    __test_labels = None

    # Network internal representation
    __session = None
    __layers = []
    __global_epoch = 0

    __num_layers_norm = 0
    __num_layers_conv = 0
    __num_layers_pool = 0
    __num_layers_fc = 0
    __num_layers_dropout = 0

    # Network options
    __batch_size = None
    __train_test_split = None
    __maximum_training_batches = None
    __reg_coeff = None
    __optimizer = 'Adam'
    __weight_initializer = 'xavier'

    __learning_rate = None
    __lr_decay_factor = None
    __lr_decay_epochs = None

    __dropout_p = 0.75
    __shakeweight_p = 0.75

    # Wrapper options
    __debug = None
    __load_from_saved = None
    __tb_dir = None
    __queue_capacity = 2000
    __report_rate = None

    # Multithreading
    __num_threads = 1
    __coord = None
    __threads = None

    def __init__(self, debug=False, load_from_saved=False, tensorboard_dir=None, report_rate=100):
        self.__debug = debug
        self.__load_from_saved = load_from_saved
        self.__tb_dir = tensorboard_dir
        self.__report_rate = report_rate

        # Add the run level to the tensorboard path
        if self.__tb_dir is not None:
            self.__tb_dir = "{0}/{1}".format(self.__tb_dir, datetime.datetime.now().strftime("%d%B%Y%I:%M%p"))

        self.__log('TensorFlow loaded...')
        self.__session = tf.Session()

    def __log(self, message):
        if self.__debug:
            print '{0}: {1}'.format(datetime.datetime.now().strftime("%I:%M%p"), message)

    def __lastLayer(self):
        return self.__layers[-1]

    def __firstLayer(self):
        return next(layer for layer in self.__layers if isinstance(layer, convLayer) or isinstance(layer, fullyConnectedLayer))

    def __initializeQueueRunners(self):
        self.__log('Initializing queue runners...')
        self.__coord = tf.train.Coordinator()
        self.__threads = tf.train.start_queue_runners(sess=self.__session, coord=self.__coord)

    def setNumberOfThreads(self, num_threads):
        """Set number of threads for input queue runners"""
        self.__num_threads = num_threads

    def setBatchSize(self, size):
        """Setter for batch size"""
        self.__batch_size = size

    def setTrainTestSplit(self, ratio):
        """Setter for a ratio for the number of samples to use as training set"""
        self.__train_test_split = ratio

    def setMaximumTrainingEpochs(self, epochs):
        """Setter for max training epochs"""
        self.__maximum_training_batches = epochs

    def setLearningRate(self, rate):
        """Setter for learning rate"""
        self.__learning_rate = rate

    def setCropOrPadImages(self, crop_or_pad):
        """Setter for padding or cropping images, which is required if the dataset has images of different sizes"""
        self.__crop_or_pad_images = crop_or_pad

    def setResizeImages(self, resize):
        """Setting for up- or down-sampling images to specified size"""
        self.__resize_images = resize

    def setAugmentationFlip(self, flip):
        """Setter for randomly flipping images horizontally augmentation"""
        self.__augmentation_flip = flip

    def setAugmentationCrop(self, resize):
        """Setter for randomly cropping images augmentation"""
        self.__augmentation_crop = resize

    def setAugmentationBrightnessAndContrast(self, contr):
        """Setter for random brightness and contrast augmentation"""
        self.__augmentation_contrast = contr

    def setRegularizationCoefficient(self, lamb):
        """Setter for L2 regularization lambda"""
        self.__reg_coeff = lamb

    def setLearningRateDecay(self, decay_factor, epochs_per_decay):
        """Set learning rate decay"""
        self.__lr_decay_factor = decay_factor
        self.__lr_decay_epochs = epochs_per_decay

    def setOptimizer(self, optimizer):
        """Set the optimizer to use by string"""
        self.__optimizer = optimizer

    def setWeightInitializer(self, initializer):
        """Set the initialization scheme used by convolutional and fully connected layers"""
        self.__weight_initializer = initializer

    def setDropoutProbability(self, p):
        """Set the probability for keeping units in dropout layers"""
        self.__dropout_p = p

    def setImageDimensions(self, image_height, image_width, image_depth):
        """Setter for image dimensions for images in the dataset"""
        self.__image_width = image_width
        self.__image_height = image_height
        self.__image_depth = image_depth
        self.__image_size = image_width*image_height*image_depth

    def addPreprocessor(self, selection):
        """Add a data preprocessing step"""
        self.__preprocessing_steps.append(selection)

    def setProblemType(self, type):
        """Set the problem type to be solved, either classification or regression"""
        if type == 'classification':
            self.__problem_type = ProblemType.CLASSIFICATION
        elif type == 'regression':
            self.__problem_type = ProblemType.REGRESSION
        else:
            warnings.warn('Problem type specified not supported', stacklevel=2)

    def beginTraining(self):
        """Initialize the network and run training to the specified max epoch"""

        self.__log('Beginning training...')

        # Define batches
        x, y = tf.train.shuffle_batch([self.__train_images, self.__train_labels],
                                      batch_size=self.__batch_size,
                                      num_threads=self.__num_threads,
                                      capacity=self.__queue_capacity,
                                      min_after_dequeue=self.__batch_size)

        # Reshape input to the expected image dimensions
        x = tf.reshape(x, shape=[-1, self.__image_height, self.__image_width, self.__image_depth])

        # Run the network operations
        xx = self.forwardPass(x, deterministic=False)
        class_predictions = tf.argmax(tf.nn.softmax(xx), 1)

        # Define regularization cost
        if self.__reg_coeff is not None:
            l2_cost = [layer.regularization_coefficient * tf.nn.l2_loss(layer.weights) for layer in self.__layers
                       if isinstance(layer, fullyConnectedLayer) or isinstance(layer, convLayer)]
        else:
            l2_cost = [0.0]

        # Define cost function and set optimizer
        cost = tf.reduce_mean(tf.concat(0, [tf.nn.sparse_softmax_cross_entropy_with_logits(xx, tf.argmax(y,1)), l2_cost]))

        if self.__optimizer == 'Adagrad':
            optimizer = tf.train.AdagradOptimizer(self.__learning_rate).minimize(cost)
            self.__log('Using Adagrad optimizer')
        elif self.__optimizer == 'Adadelta':
            optimizer = tf.train.AdadeltaOptimizer(self.__learning_rate).minimize(cost)
            self.__log('Using Adadelta optimizer')
        elif self.__optimizer == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(self.__learning_rate).minimize(cost)
            self.__log('Using SGD optimizer')
        else:
            optimizer = tf.train.AdamOptimizer(self.__learning_rate).minimize(cost)
            self.__log('Using Adam optimizer')

        # Calculate classification accuracy
        correct_predictions = tf.equal(class_predictions, tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        # Calculate validation accuracy
        x_test, y_test = tf.train.shuffle_batch([self.__test_images, self.__test_labels],
                                                batch_size=self.__batch_size,
                                                num_threads=self.__num_threads,
                                                capacity=self.__queue_capacity,
                                                min_after_dequeue=self.__batch_size)

        x_test = tf.reshape(x_test, shape=[-1, self.__image_height, self.__image_width, self.__image_depth])

        x_test_predicted = self.forwardPass(x_test, deterministic=True)
        test_class_predictions = tf.argmax(tf.nn.softmax(x_test_predicted), 1)

        test_correct_predictions = tf.equal(test_class_predictions, tf.argmax(y_test, 1))
        test_accuracy = tf.reduce_mean(tf.cast(test_correct_predictions, tf.float32))

        full_test_op = self.computeFullTestAccuracy()

        # Epoch summaries for Tensorboard
        if self.__tb_dir is not None:
            tf.scalar_summary('train/accuracy', accuracy)
            tf.scalar_summary('test/accuracy', test_accuracy)
            tf.scalar_summary('train/loss', cost)
            tf.scalar_summary('train/learning_rate', self.__learning_rate)
            tf.scalar_summary('train/l2_loss', tf.reduce_mean(l2_cost))
            tf.histogram_summary('train/class_predictions', class_predictions)
            tf.histogram_summary('test/class_predictions', test_class_predictions)

            # Plot weights of first conv layer
            filter_summary = self.__getWeightsAsImage(self.__firstLayer().weights)

            tf.image_summary('filters/first', filter_summary)

            # Including summaries for each layer
            for layer in self.__layers:
                if hasattr(layer, 'name'):
                    tf.histogram_summary('weights/'+layer.name, layer.weights)
                    tf.histogram_summary('biases/'+layer.name, layer.biases)
                    tf.histogram_summary('activations/'+layer.name, layer.activations)

            merged = tf.merge_all_summaries()
            train_writer = tf.train.SummaryWriter(self.__tb_dir, self.__session.graph)

        # Either load the network parameters from a checkpoint file or start training
        if self.__load_from_saved:
            self.__log('Loading from checkpoint file...')
            saver = tf.train.Saver()
            saver.restore(self.__session, tf.train.latest_checkpoint('./'))

            self.__initializeQueueRunners()

            self.__log('Computing total test accuracy...')
            tt_error = self.__session.run(full_test_op)
            self.__log('Error: {:.5f}'.format(tt_error))
        else:
            self.__log('Initializing parameters...')
            init_op = tf.initialize_all_variables()
            self.__session.run(init_op)

            self.__initializeQueueRunners()

            for i in range(self.__maximum_training_batches):
                start_time = time.time()
                self.__global_epoch = i
                self.__session.run(optimizer)

                if self.__global_epoch % self.__report_rate == 0:
                    elapsed = time.time() - start_time
                    self.__setLearningRate()

                    if self.__tb_dir is not None:
                        summary = self.__session.run(merged)
                        train_writer.add_summary(summary, i)

                    loss, epoch_accuracy, epoch_test_accuracy = self.__session.run([cost, accuracy, test_accuracy])

                    samples_per_sec = self.__batch_size / elapsed

                    self.__log(
                        'Results for batch {} (epoch {}) - Loss: {:.5f}, Training Accuracy: {:.4f}, samples/sec: {:.2f}'
                        .format(i,
                                i / (self.__total_training_samples / self.__batch_size),
                                loss,
                                epoch_accuracy,
                                samples_per_sec))

                    if self.__global_epoch % (self.__report_rate * 100) == 0:
                        self.saveState()
                else:
                    loss = self.__session.run([cost])

                if loss == 0.0:
                    self.__log('Stopping due to zero loss')
                    break

                if i == self.__maximum_training_batches-1:
                    self.__log('Stopping due to maximum epochs')

            self.saveState()

            self.__log('Computing total test accuracy...')
            tt_error = self.__session.run(full_test_op)
            self.__log('Total test accuracy: {:.5f}'.format(tt_error))

            self.__log('Ending session...')

            self.__coord.request_stop()
            self.__coord.join(self.__threads)

            self.__session.close()

    def computeFullTestAccuracy(self):
        num_test = self.__total_raw_samples - self.__total_training_samples
        num_batches = int(num_test/self.__batch_size)
        sum_correct = 0.0

        for i in range(num_batches):
            x_test, y_test = tf.train.batch([self.__test_images, self.__test_labels],
                                                    batch_size=self.__batch_size,
                                                    num_threads=self.__num_threads)

            x_test = tf.reshape(x_test, shape=[-1, self.__image_height, self.__image_width, self.__image_depth])

            x_test_predicted = self.forwardPass(x_test, deterministic=True)
            test_class_predictions = tf.argmax(tf.nn.softmax(x_test_predicted), 1)

            test_correct_predictions = tf.equal(test_class_predictions, tf.argmax(y_test, 1))
            test_acc = tf.reduce_mean(tf.cast(test_correct_predictions, tf.float32))

            sum_correct = sum_correct + test_acc

        return sum_correct / num_batches

    def __getWeightsAsImage(self, kernel):
        """Filter visualization, adapted with permission from https://gist.github.com/kukuruza/03731dc494603ceab0c5"""

        pad = 1
        grid_X = 4
        grid_Y = (kernel.get_shape().as_list()[-1] / 4)

        # pad X and Y
        x1 = tf.pad(kernel, tf.constant([[pad, 0], [pad, 0], [0, 0], [0, 0]]))

        # X and Y dimensions, w.r.t. padding
        Y = kernel.get_shape()[0] + pad
        X = kernel.get_shape()[1] + pad

        # pack into image with proper dimensions for tf.image_summary
        x2 = tf.transpose(x1, (3, 0, 1, 2))
        x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, 3]))
        x4 = tf.transpose(x3, (0, 2, 1, 3))
        x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, 3]))
        x6 = tf.transpose(x5, (2, 1, 3, 0))
        x7 = tf.transpose(x6, (3, 0, 1, 2))

        # scale to [0, 1]
        x_min = tf.reduce_min(x7)
        x_max = tf.reduce_max(x7)
        x8 = (x7 - x_min) / (x_max - x_min)

        return x8

    def saveState(self):
        self.__log('Saving parameters...')
        saver = tf.train.Saver()
        saver.save(self.__session, 'tfhSaved')

    def __setLearningRate(self):
        if self.__lr_decay_factor is not None:
            self.__learning_rate = tf.train.exponential_decay(self.__learning_rate,
                                            self.__global_epoch,
                                            self.__lr_decay_epochs,
                                            self.__lr_decay_factor,
                                            staircase=True)

            tf.scalar_summary('learning_rate', self.__learning_rate)

    def forwardPass(self, x, deterministic=False):
        for layer in self.__layers:
            x = layer.forwardPass(x, deterministic)

        return x

    def addInputLayer(self):
        self.__log('Adding the input layer...')
        layer = inputLayer([self.__batch_size, self.__image_height, self.__image_width, self.__image_depth])

        self.__layers.append(layer)

    def addConvolutionalLayer(self, filter_dimension, stride_length, activation_function, regularization_coefficient=None):
        self.__num_layers_conv += 1
        layer_name = 'conv%d' % self.__num_layers_conv
        self.__log('Adding convolutional layer %s...' % layer_name)

        if regularization_coefficient is None and self.__reg_coeff is not None:
            regularization_coefficient = self.__reg_coeff
        elif regularization_coefficient is None and self.__reg_coeff is None:
            regularization_coefficient = 0.0

        layer = convLayer(layer_name,
                          self.__lastLayer().output_size,
                          filter_dimension,
                          stride_length,
                          activation_function,
                          self.__weight_initializer,
                          regularization_coefficient)

        self.__log('Filter dimensions: {0} Outputs: {1}'.format(filter_dimension, layer.output_size))

        self.__layers.append(layer)

    def addPoolingLayer(self, kernel_size, stride_length):
        self.__num_layers_pool += 1
        layer_name = 'pool%d' % self.__num_layers_pool
        self.__log('Adding pooling layer %s...' % layer_name)

        layer = poolingLayer(self.__lastLayer().output_size, kernel_size, stride_length)
        self.__log('Outputs: %s' % layer.output_size)

        self.__layers.append(layer)

    def addNormalizationLayer(self):
        self.__num_layers_norm += 1
        layer_name = 'norm%d' % self.__num_layers_pool
        self.__log('Adding pooling layer %s...' % layer_name)

        layer = normLayer(self.__lastLayer().output_size)
        self.__layers.append(layer)

    def addDropoutLayer(self, p=None):
        self.__num_layers_dropout += 1
        layer_name = 'drop%d' % self.__num_layers_dropout
        self.__log('Adding dropout layer %s...' % layer_name)

        if p is None:
            p = self.__dropout_p

        layer = dropoutLayer(self.__lastLayer().output_size, p)
        self.__layers.append(layer)

    def addFullyConnectedLayer(self, output_size, activation_function, shakeweight_p=None,
                               shakeout_p=None, shakeout_c=None, dropconnect_p=None,
                               regularization_coefficient=None):
        self.__num_layers_fc += 1
        layer_name = 'fc%d' % self.__num_layers_fc
        self.__log('Adding fully connected layer %s...' % layer_name)

        reshape = isinstance(self.__lastLayer(), convLayer) or isinstance(self.__lastLayer(), poolingLayer)

        if regularization_coefficient is None and self.__reg_coeff is not None:
            regularization_coefficient = self.__reg_coeff
        if regularization_coefficient is None and self.__reg_coeff is None:
            regularization_coefficient = 0.0

        layer = fullyConnectedLayer(layer_name,
                                    self.__lastLayer().output_size,
                                    output_size,
                                    reshape,
                                    self.__batch_size,
                                    activation_function,
                                    self.__weight_initializer,
                                    regularization_coefficient)

        layer.shakeweight_p = shakeweight_p
        layer.shakeout_p = shakeout_p
        layer.shakeout_c = shakeout_c
        layer.dropconnect_p = dropconnect_p

        self.__log('Inputs: {0} Outputs: {1}'.format(layer.input_size, layer.output_size))

        self.__layers.append(layer)

    def addOutputLayer(self, regularization_coefficient=None):
        self.__log('Adding output layer...')

        reshape = isinstance(self.__lastLayer(), convLayer) or isinstance(self.__lastLayer(), poolingLayer)

        layer = fullyConnectedLayer('output',
                                    self.__lastLayer().output_size,
                                    self.__total_classes,
                                    reshape,
                                    self.__batch_size,
                                    None,
                                    self.__weight_initializer,
                                    regularization_coefficient)

        self.__log('Inputs: {0} Outputs: {1}'.format(layer.input_size, layer.output_size))

        self.__layers.append(layer)

    def loadDatasetFromDirectoryWithCSVLabels(self, dirname, labels_file, column_number=False):
        """Loads the png images in the given directory into an internal representation,
        using the labels provided in a csv file. You can optionally specify a column
        number from the labels file to specify the class label"""

        image_files = [os.path.join(dirname, name) for name in os.listdir(dirname) if
                       os.path.isfile(os.path.join(dirname, name)) & name.endswith('.png')]

        labels = readCSVLabels(labels_file, column_number)

        self.__total_raw_samples = len(image_files)
        self.__total_classes = len(set(labels))

        self.__log('Total raw examples is %d' % self.__total_raw_samples)
        self.__log('Total classes is %d' % self.__total_classes)
        self.__log('Parsing dataset...')

        # split data
        train_images, train_labels, test_images, test_labels = splitRawData(image_files, labels)

        # create batches of input data and labels for training
        self.__parseDataset(train_images, train_labels, test_images, test_labels)

    def loadIPPNDatasetFromDirectory(self, dirname):
        """Loads the RGB images and labels from the IPPN dataset"""

        labels, ids = readCSVLabelsAndIds(os.path.join(dirname, 'Metadata.csv'), 1, 0)

        image_files = [os.path.join(dirname, id + '_rgb.png') for id in ids]

        self.__total_raw_samples = len(image_files)
        self.__total_classes = len(set(labels))

        # transform into numerical one-hot labels
        labels = stringLabelsToSequential(labels)
        labels = tf.one_hot(labels, self.__total_classes)

        self.__log('Total raw examples is %d' % self.__total_raw_samples)
        self.__log('Total classes is %d' % self.__total_classes)
        self.__log('Parsing dataset...')

        # split data
        train_images, train_labels, test_images, test_labels = splitRawData(image_files, labels)

        # create batches of input data and labels for training
        self.__parseDataset(train_images, train_labels, test_images, test_labels)

    def loadINRADatasetFromDirectory(self, dirname):
        """Loads the RGB images and labels from the INRA dataset"""

        labels, ids = readCSVLabelsAndIds(os.path.join(dirname, 'AutomatonImages.csv'), 1, 3, character=';')

        # Remove the header line
        labels.pop(0)
        ids.pop(0)

        image_files = [os.path.join(dirname, id) for id in ids]

        self.__total_raw_samples = len(image_files)
        self.__total_classes = len(set(labels))

        # transform into numerical one-hot labels
        labels = stringLabelsToSequential(labels)
        labels = tf.one_hot(labels, self.__total_classes)

        self.__log('Total raw examples is %d' % self.__total_raw_samples)
        self.__log('Total classes is %d' % self.__total_classes)
        self.__log('Parsing dataset...')

        # split data
        train_images, train_labels, test_images, test_labels = splitRawData(image_files, labels)

        # create batches of input data and labels for training
        self.__parseDataset(train_images, train_labels, test_images, test_labels, image_type='jpg')

    def loadCIFAR10DatasetFromDirectory(self, dirname):
        """Loads a static CIFAR10 data directory"""

        train_dir = os.path.join(dirname, 'train')
        test_dir = os.path.join(dirname, 'test')
        self.__total_classes = 10
        self.__queue_capacity = 60000

        train_labels, train_images = readCSVLabelsAndIds(os.path.join(train_dir, 'train.txt'), 1, 0, character=' ')

        # transform into numerical one-hot labels
        train_labels = [int(label) for label in train_labels]
        train_labels = tf.one_hot(train_labels, self.__total_classes)

        test_labels, test_images = readCSVLabelsAndIds(os.path.join(test_dir, 'test.txt'), 1, 0, character=' ')

        # transform into numerical one-hot labels
        test_labels = [int(label) for label in test_labels]
        test_labels = tf.one_hot(test_labels, self.__total_classes)

        self.__total_raw_samples = len(train_images) + len(test_images)

        self.__log('Total raw examples is %d' % self.__total_raw_samples)
        self.__log('Total classes is %d' % self.__total_classes)
        self.__log('Parsing dataset...')

        # create batches of input data and labels for training
        self.__parseDataset(train_images, train_labels, test_images, test_labels)

    def loadDatasetFromDirectoryWithAutoLabels(self, dirname):
        """Loads the png images in the given directory, using subdirectories to separate classes"""

        # Load all file names and labels into arrays
        subdirs = filter(lambda item: os.path.isdir(item) & (item != '.DS_Store'),
                         [os.path.join(dirname, f) for f in os.listdir(dirname)])

        num_classes = len(subdirs)

        image_files = []
        labels = np.array([])

        for sd in subdirs:
            image_paths = [os.path.join(sd, name) for name in os.listdir(sd) if
                           os.path.isfile(os.path.join(sd, name)) & name.endswith('.png')]
            image_files = image_files + image_paths

            # for one-hot labels
            current_labels = np.zeros((num_classes, len(image_paths)))
            current_labels[self.__total_classes, :] = 1
            labels = np.hstack([labels, current_labels]) if labels.size else current_labels
            self.__total_classes += 1

        labels = tf.transpose(labels)

        self.__total_raw_samples = len(image_files)

        self.__log('Total raw examples is %d' % self.__total_raw_samples)
        self.__log('Total classes is %d' % self.__total_classes)
        self.__log('Parsing dataset...')

        # split data
        train_images, train_labels, test_images, test_labels = splitRawData(image_files, labels, self.__train_test_split)

        # create batches of input data and labels for training
        self.__parseDataset(train_images, train_labels, test_images, test_labels)

    def loadLemnatecDatasetFromDirectory(self, dirname):
        """Loads a Lemnatec plant scanner image dataset. Regression or classification labels MUST be loaded first."""

        # Load all snapshot subdirectories
        subdirs = filter(lambda item: os.path.isdir(item) & (item != '.DS_Store'),
                         [os.path.join(dirname, f) for f in os.listdir(dirname)])

        image_files = []

        # Load the VIS images in each subdirectory
        for sd in subdirs:
            image_paths = [os.path.join(sd, name) for name in os.listdir(sd) if
                           os.path.isfile(os.path.join(sd, name)) & name.startswith('VIS_SV_')]

            image_files = image_files + image_paths

        # Put the image files in the order of the IDs
        sorted_paths = []

        for image_id in self.__all_ids:
            path = filter(lambda item: item.endswith(image_id), [p for p in image_files])
            assert len(path) == 1, 'Found no image or multiple images for %r' % image_id
            sorted_paths.append(path[0])

        self.__total_raw_samples = len(sorted_paths)

        self.__log('Total raw examples is %d' % self.__total_raw_samples)
        self.__log('Parsing dataset...')

        # split data
        train_images, train_labels, test_images, test_labels = splitRawData(sorted_paths, self.__all_labels, self.__train_test_split)

        # create batches of input data and labels for training
        self.__parseDataset(image_files, train_labels, test_images, test_labels)

    def loadMultipleLabelsFromCSV(self, filepath, id_column=0):
        """Load multiple labels from a CSV file, for instance values for regression.
        Parameter id_column is the column number specifying the image file name.
        """

        self.__all_labels, self.__all_ids = readCSVMultiLabelsAndIds(filepath, id_column)

    def __parseDataset(self, train_images, train_labels, test_images, test_labels, image_type='png'):
        # pre-processing

        if not len(self.__preprocessing_steps) == 0:
            self.__log('Performing preprocessing steps...')

            if not os.path.isdir(self.__processed_images_dir):
                os.mkdir(self.__processed_images_dir)

            for step in self.__preprocessing_steps:
                if step == 'auto-segmentation':
                    self.__log('Performing auto-segmentation...')

                    train_images = Parallel(n_jobs=self.__num_threads)(delayed(doParallelAutoSegmentation)(i, self.__processed_images_dir) for i in train_images)
                    test_images = Parallel(n_jobs=self.__num_threads)(delayed(doParallelAutoSegmentation)(i, self.__processed_images_dir) for i in test_images)

        # house keeping
        if isinstance(train_images, tf.Tensor):
            self.__total_training_samples = train_images.get_shape().as_list()[0]
        else:
            self.__total_training_samples = len(train_images)

        if self.__total_training_samples is None:
            self.__total_training_samples = int(self.__total_raw_samples * self.__train_test_split)

        # calculate number of batches to run
        batches_per_epoch = self.__total_training_samples / float(self.__batch_size)
        self.__maximum_training_batches = int(self.__maximum_training_batches * batches_per_epoch)

        self.__log('Batches per epoch: {:f}'.format(batches_per_epoch))
        self.__log('Running to {0} batches'.format(self.__maximum_training_batches))

        if self.__batch_size > self.__total_training_samples:
            self.__log('Less than one batch in training set, exiting now')
            exit()

        # create input queues
        train_input_queue = tf.train.slice_input_producer([train_images, train_labels], shuffle=False)
        test_input_queue = tf.train.slice_input_producer([test_images, test_labels], shuffle=False)

        self.__test_labels = test_input_queue[1]
        self.__train_labels = train_input_queue[1]

        # pre-processing for training and testing images

        if image_type is 'jpg':
            self.__train_images = tf.image.decode_jpeg(tf.read_file(train_input_queue[0]),
                                                       channels=self.__image_depth)
            self.__test_images = tf.image.decode_jpeg(tf.read_file(test_input_queue[0]),
                                                      channels=self.__image_depth)
        else:
            self.__train_images = tf.image.decode_png(tf.read_file(train_input_queue[0]),
                                                      channels=self.__image_depth)
            self.__test_images = tf.image.decode_png(tf.read_file(test_input_queue[0]), channels=self.__image_depth)

        # convert images to float and normalize to 1.0
        self.__train_images = tf.image.convert_image_dtype(self.__train_images, dtype=tf.float32)
        self.__test_images = tf.image.convert_image_dtype(self.__test_images, dtype=tf.float32)

        if self.__augmentation_crop is True:
            self.__image_height = int(self.__image_height * 0.75)
            self.__image_width = int(self.__image_width * 0.75)
            self.__train_images = tf.random_crop(self.__train_images, [self.__image_height, self.__image_width, 3])
            self.__test_images = tf.image.resize_image_with_crop_or_pad(self.__test_images, self.__image_height,
                                                                        self.__image_width)

        if self.__resize_images is True:
            self.__train_images = tf.image.resize_images(self.__train_images,
                                                         [self.__image_height, self.__image_width])
            self.__test_images = tf.image.resize_images(self.__test_images,
                                                        [self.__image_height, self.__image_width])

        if self.__crop_or_pad_images is True:
            # pad or crop to deal with images of different sizes
            self.__train_images = tf.image.resize_image_with_crop_or_pad(self.__train_images, self.__image_height,
                                                                         self.__image_width)
            self.__test_images = tf.image.resize_image_with_crop_or_pad(self.__test_images, self.__image_height,
                                                                        self.__image_width)
        elif self.__augmentation_crop is True:
            self.__train_images = tf.image.resize_image_with_crop_or_pad(self.__train_images, self.__image_height,
                                                                         self.__image_width)

        if self.__augmentation_flip is True:
            # apply flip augmentation
            self.__train_images = tf.image.random_flip_left_right(self.__train_images)

        if self.__augmentation_contrast is True:
            # apply random contrast and brightness augmentation
            self.__train_images = tf.image.random_brightness(self.__train_images, max_delta=63)
            self.__train_images = tf.image.random_contrast(self.__train_images, lower=0.2, upper=1.8)

        # mean-center all inputs
        self.__train_images = tf.image.per_image_whitening(self.__train_images)
        self.__test_images = tf.image.per_image_whitening(self.__test_images)

        # define the shape of the image tensors so it matches the shape of the images
        self.__train_images.set_shape([self.__image_height, self.__image_width, self.__image_depth])
        self.__test_images.set_shape([self.__image_height, self.__image_width, self.__image_depth])