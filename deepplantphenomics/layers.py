import tensorflow as tf
import math


class convLayer(object):
    filter_dimension = None
    __stride_length = None
    __activation_function = None

    weights = None
    biases = None
    activations = None

    input_size = None
    output_size = None
    name = None
    regularization_coefficient = None

    def __init__(self, name, input_size, filter_dimension, stride_length, activation_function, initializer, regularization_coefficient):
        self.name = name
        self.filter_dimension = filter_dimension
        self.__stride_length = stride_length
        self.__activation_function = activation_function
        self.input_size = input_size
        self.output_size = input_size
        self.regularization_coefficient = regularization_coefficient

        padding = 2*(math.floor(filter_dimension[0] / 2))
        self.output_size[1] = int((self.output_size[1] - filter_dimension[0] + padding) / stride_length + 1)
        padding = 2 * (math.floor(filter_dimension[1] / 2))
        self.output_size[2] = int((self.output_size[2] - filter_dimension[1] + padding) / stride_length + 1)
        self.output_size[-1] = filter_dimension[-1]

        if initializer == 'xavier':
            self.weights = tf.get_variable(self.name + '_weights',
                                           shape=self.filter_dimension,
                                           initializer=tf.contrib.layers.xavier_initializer_conv2d())
        else:
            self.weights = tf.get_variable(self.name + '_weights',
                                           shape=self.filter_dimension,
                                           initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                           dtype=tf.float32)

        self.biases = tf.get_variable(self.name + '_bias',
                                      [self.output_size[-1]],
                                      initializer=tf.constant_initializer(0.1),
                                      dtype=tf.float32)

    def forward_pass(self, x, deterministic):
        # For convention, just use a symmetrical stride with same padding
        activations = tf.nn.conv2d(x, self.weights,
                                   strides=[1, self.__stride_length, self.__stride_length, 1],
                                   padding='SAME')

        activations = tf.nn.bias_add(activations, self.biases)

        # Apply a non-linearity specified by the user
        if self.__activation_function == 'relu':
            activations = tf.nn.relu(activations)
        elif self.__activation_function == 'tanh':
            activations = tf.tanh(activations)

        self.activations = activations

        return activations


class poolingLayer(object):
    __kernel_size = None
    __stride_length = None

    input_size = None
    output_size = None
    pooling_type= None

    def __init__(self, input_size, kernel_size, stride_length, pooling_type='max'):
        self.__kernel_size = kernel_size
        self.__stride_length = stride_length
        self.input_size = input_size
        self.pooling_type = pooling_type

        # The pooling operation will reduce the width and height dimensions
        self.output_size = self.input_size
        filter_size_even = (kernel_size % 2 == 0)

        if filter_size_even:
            self.output_size[1] = int(math.floor((self.output_size[1] - kernel_size) / stride_length + 1))
            self.output_size[2] = int(math.floor((self.output_size[2] - kernel_size) / stride_length + 1))
        else:
            self.output_size[1] = int(math.floor((self.output_size[1]-kernel_size)/stride_length + 1) + 1)
            self.output_size[2] = int(math.floor((self.output_size[2]-kernel_size)/stride_length + 1) + 1)

    def forward_pass(self, x, deterministic):
        if self.pooling_type == 'max':
            return tf.nn.max_pool(x,
                                  ksize=[1, self.__kernel_size, self.__kernel_size, 1],
                                  strides=[1, self.__stride_length, self.__stride_length, 1],
                                  padding='SAME')
        elif self.pooling_type == 'avg':
            return tf.nn.avg_pool(x,
                                  ksize=[1, self.__kernel_size, self.__kernel_size, 1],
                                  strides=[1, self.__stride_length, self.__stride_length, 1],
                                  padding='SAME')


class fullyConnectedLayer(object):
    weights = None
    biases = None
    activations = None
    __activation_function = None
    __reshape = None
    regularization_coefficient = None

    input_size = None
    output_size = None
    name = None

    def __init__(self, name, input_size, output_size, reshape, batch_size, activation_function, initializer, regularization_coefficient):
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.__reshape = reshape
        self.__batch_size = batch_size
        self.__activation_function = activation_function
        self.regularization_coefficient = regularization_coefficient

        # compute the vectorized size for weights if we will need to reshape it
        if reshape:
            vec_size = input_size[1]*input_size[2]*input_size[3]
        else:
            vec_size = input_size

        if initializer == 'xavier':
            self.weights = tf.get_variable(self.name + '_weights', shape=[vec_size, output_size],
                                           initializer=tf.contrib.layers.xavier_initializer())
        else:
            self.weights = tf.get_variable(self.name + '_weights',
                                           shape=[vec_size, output_size],
                                           initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0/self.output_size)),
                                           dtype=tf.float32)

        self.biases = tf.get_variable(self.name + '_bias',
                                      [self.output_size],
                                      initializer=tf.constant_initializer(0.1),
                                      dtype=tf.float32)

    def forward_pass(self, x, deterministic):
        # Reshape into a column vector if necessary
        if self.__reshape is True:
            x = tf.reshape(x, [self.__batch_size, -1])

        activations = tf.matmul(x, self.weights)
        activations = tf.add(activations, self.biases)

        # Apply a non-linearity specified by the user
        if self.__activation_function == 'relu':
            activations = tf.nn.relu(activations)
        elif self.__activation_function == 'tanh':
            activations = tf.tanh(activations)

        self.activations = activations

        return activations


class inputLayer(object):
    """An object representing the input layer so it can give information about input size to the next layer"""
    input_size = None
    output_size = None

    def __init__(self, input_size):
        self.input_size = input_size
        self.output_size = input_size

    def forward_pass(self, x, deterministic):
        return x


class normLayer(object):
    """Layer which performs local response normalization"""
    input_size = None
    output_size = None

    def __init__(self, input_size):
        self.input_size = input_size
        self.output_size = input_size

    def forward_pass(self, x, deterministic):
        x = tf.nn.lrn(x)
        return x


class dropoutLayer(object):
    """Layer which performs dropout"""
    input_size = None
    output_size = None
    p = None

    def __init__(self, input_size, p):
        self.input_size = input_size
        self.output_size = input_size
        self.p = p

    def forward_pass(self, x, deterministic):
        if deterministic:
            return x
        else:
            return tf.nn.dropout(x, self.p)


class moderationLayer(object):
    """Layer for fusing moderating data into the input vector"""
    input_size = None
    output_size = None
    __reshape = None
    __batch_size = None

    def __init__(self, input_size, feature_size, reshape, batch_size):
        self.input_size = input_size
        self.__reshape = reshape
        self.__batch_size = batch_size

        # compute the vectorized size for weights if we will need to reshape it
        if reshape:
            vec_size = input_size[1] * input_size[2] * input_size[3]
        else:
            vec_size = input_size

        self.output_size = vec_size + feature_size

    def forward_pass(self, x, deterministic, features):
        # Reshape into a column vector if necessary
        if self.__reshape is True:
            x = tf.reshape(x, [self.__batch_size, -1])

        # Append the moderating features onto the vector
        x = tf.concat([x, features], axis=1)

        return x