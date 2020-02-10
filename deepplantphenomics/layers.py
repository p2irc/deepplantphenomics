import tensorflow.compat.v1 as tf
import tensorflow.contrib
import math
import copy


class convLayer(object):
    def __init__(self, name, input_size, filter_dimension, stride_length,
                 activation_function, initializer, padding=None, batch_norm=False, use_bias=False, epsilon=1e-5, decay=0.9):
        self.name = name
        self.filter_dimension = filter_dimension
        self.__stride_length = stride_length
        self.__activation_function = activation_function
        self.__initializer = initializer
        self.use_bias = use_bias
        self.input_size = input_size
        self.output_size = copy.deepcopy(input_size)
        self.batch_norm_layer = None

        if padding is None:
            padding_row = math.floor(filter_dimension[0] / 2)
            padding_col = math.floor(filter_dimension[1] / 2)
        else:
            padding_row = padding
            padding_col = padding

        self.padding = [[0, 0], [padding_row, padding_row], [padding_col, padding_col], [0, 0]]
        self.output_size[1] = int((self.output_size[1] - filter_dimension[0] + 2 * padding_row) / stride_length + 1)
        self.output_size[2] = int((self.output_size[2] - filter_dimension[1] + 2 * padding_col) / stride_length + 1)
        self.output_size[-1] = filter_dimension[-1]

        if batch_norm:
            self.batch_norm_layer = batchNormLayer(name=self.name + '_batch_norm', input_size=self.output_size,
                                                   epsilon=epsilon, decay=decay)

    def add_to_graph(self):
        if self.__initializer == 'xavier':
            self.weights = tf.get_variable(self.name + '_weights',
                                           shape=self.filter_dimension,
                                           initializer=tensorflow.contrib.layers.xavier_initializer_conv2d())
        else:
            self.weights = tf.get_variable(self.name + '_weights',
                                           shape=self.filter_dimension,
                                           initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                           dtype=tf.float32)

        if self.use_bias:
            self.biases = tf.get_variable(self.name + '_bias',
                                          [self.filter_dimension[-1]],
                                          initializer=tf.constant_initializer(0.1),
                                          dtype=tf.float32)

        if self.batch_norm_layer is not None:
            self.batch_norm_layer.add_to_graph()

    def decay_weights(self):
        return tf.assign(self.weights, self.weights * (1. - 1e-5))

    def forward_pass(self, x, deterministic=False):
        activations = tf.nn.conv2d(x, self.weights,
                                   strides=[1, self.__stride_length, self.__stride_length, 1],
                                   padding=self.padding)

        if self.use_bias:
            activations = tf.nn.bias_add(activations, self.biases)

        if self.batch_norm_layer is not None:
            activations = self.batch_norm_layer.forward_pass(activations, deterministic)

        # Apply a non-linearity specified by the user
        if self.__activation_function == 'relu':
            activations = tf.nn.relu(activations)
        elif self.__activation_function == 'tanh':
            activations = tf.tanh(activations)
        elif self.__activation_function == 'lrelu':
            activations = tf.nn.leaky_relu(activations)
        elif self.__activation_function == 'selu':
            activations = tf.nn.selu(activations)

        self.activations = activations

        return activations


class upsampleLayer(object):
    def __init__(self, name, input_size, filter_size, num_filters, upscale_factor,
                 activation_function, batch_multiplier, initializer, use_bias, regularization_coefficient):
        self.name = name
        self.__activation_function = activation_function
        self.__initializer = initializer
        self.input_size = input_size
        self.strides = [1, upscale_factor, upscale_factor, 1]
        self.upscale_factor = upscale_factor
        self.batch_multiplier = batch_multiplier
        self.num_filters = num_filters
        self.regularization_coefficient = regularization_coefficient
        self.use_bias = use_bias

        # if upscale_factor is an int then height and width are scaled the same
        if isinstance(upscale_factor, int):
            self.strides = [1, upscale_factor, upscale_factor, 1]
            h = self.input_size[1] * upscale_factor
            w = self.input_size[2] * upscale_factor
        else:  # otherwise scaled individually
            self.strides = [1, upscale_factor[0], upscale_factor[1], 1]
            h = self.input_size[1] * upscale_factor[0]
            w = self.input_size[2] * upscale_factor[1]

        # upsampling will have the same batch size self.input_size[0]
        self.output_size = [self.input_size[0], h, w, num_filters]

        # the shape needed to initialize weights is based on
        self.weights_shape = [filter_size, filter_size, num_filters, input_size[-1]]

    def add_to_graph(self):
        if self.__initializer == 'xavier':
            self.weights = tf.get_variable(self.name + '_weights',
                                           shape=self.weights_shape,
                                           initializer=tensorflow.contrib.layers.xavier_initializer_conv2d())
        else:
            self.weights = tf.get_variable(self.name + '_weights',
                                           shape=self.weights_shape,
                                           initializer=tf.truncated_normal_initializer(stddev=5e-2),
                                           dtype=tf.float32)

        if self.use_bias:
            self.biases = tf.get_variable(self.name + '_bias',
                                          [self.num_filters],
                                          initializer=tf.constant_initializer(0.1),
                                          dtype=tf.float32)

    def decay_weights(self):
        return tf.assign(self.weights, self.weights * (1. - 1e-5))

    def forward_pass(self, x, deterministic):
        # upsampling will have the same batch size (first dimension of x),
        # and will preserve the number of filters (self.input_size[-1]), (this is NHWC)
        dyn_input_shape = tf.shape(x)
        batch_size = dyn_input_shape[0]
        h = dyn_input_shape[1] * self.upscale_factor
        w = dyn_input_shape[2] * self.upscale_factor
        output_shape = tf.stack([batch_size, h, w, self.num_filters])

        activations = tf.nn.conv2d_transpose(x, self.weights, output_shape=output_shape,
                                             strides=self.strides, padding='SAME')

        if self.use_bias:
            activations = tf.nn.bias_add(activations, self.biases)

        # Apply a non-linearity specified by the user
        if self.__activation_function == 'relu':
            activations = tf.nn.relu(activations)
        elif self.__activation_function == 'tanh':
            activations = tf.tanh(activations)
        elif self.__activation_function == 'lrelu':
            activations = tf.nn.leaky_relu(activations)
        elif self.__activation_function == 'selu':
            activations = tf.nn.selu(activations)

        self.activations = activations

        if activations.shape[-1] == 1:
            return tf.squeeze(activations)
        else:
            return activations


class poolingLayer(object):
    def __init__(self, input_size, kernel_size, stride_length, pooling_type='max'):
        self.__kernel_size = kernel_size
        self.__stride_length = stride_length
        self.input_size = input_size
        self.pooling_type = pooling_type

        # The pooling operation will reduce the width and height dimensions, but since the padding type is always
        # 'SAME', the output size only depends on input size and stride length
        self.output_size = self.input_size
        self.output_size[1] = int(math.ceil(self.output_size[1] / float(stride_length)))
        self.output_size[2] = int(math.ceil(self.output_size[2] / float(stride_length)))

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
    def __init__(self, name, input_size, output_size, reshape, activation_function, initializer,
                 regularization_coefficient):
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.__reshape = reshape
        self.__activation_function = activation_function
        self.__initializer = initializer
        self.regularization_coefficient = regularization_coefficient

        # compute the vectorized size for weights if we will need to reshape it
        if reshape:
            self.__vec_size = self.input_size[1] * self.input_size[2] * self.input_size[3]
        else:
            self.__vec_size = self.input_size

    def add_to_graph(self):
        if self.__initializer == 'xavier':
            self.weights = tf.get_variable(self.name + '_weights', shape=[self.__vec_size, self.output_size],
                                           initializer=tensorflow.contrib.layers.xavier_initializer())
        else:
            self.weights = tf.get_variable(self.name + '_weights',
                                           shape=[self.__vec_size, self.output_size],
                                           initializer=tf.truncated_normal_initializer(
                                               stddev=math.sqrt(2.0/self.output_size)),
                                           dtype=tf.float32)

        self.biases = tf.get_variable(self.name + '_bias',
                                      [self.output_size],
                                      initializer=tf.constant_initializer(0.1),
                                      dtype=tf.float32)

    def forward_pass(self, x, deterministic):
        # Reshape into a column vector if necessary
        if self.__reshape is True:
            x = tf.reshape(x, [-1, self.__vec_size])

        activations = tf.matmul(x, self.weights)
        activations = tf.add(activations, self.biases)

        # Apply a non-linearity specified by the user
        if self.__activation_function == 'relu':
            activations = tf.nn.relu(activations)
        elif self.__activation_function == 'tanh':
            activations = tf.tanh(activations)
        elif self.__activation_function == 'lrelu':
            activations = tf.nn.leaky_relu(activations)
        elif self.__activation_function == 'selu':
            activations = tf.nn.selu(activations)

        self.activations = activations

        return activations


class inputLayer(object):
    """An object representing the input layer so it can give information about input size to the next layer"""
    def __init__(self, input_size):
        self.input_size = input_size
        self.output_size = input_size

    def forward_pass(self, x, deterministic):
        return x


class normLayer(object):
    """Layer which performs local response normalization"""
    def __init__(self, input_size):
        self.input_size = input_size
        self.output_size = input_size

    def forward_pass(self, x, deterministic):
        x = tf.nn.lrn(x)
        return x


class dropoutLayer(object):
    """Layer which performs dropout"""
    def __init__(self, input_size, p):
        self.input_size = input_size
        self.output_size = input_size
        self.drop_rate = 1 - p

    def forward_pass(self, x, deterministic):
        if deterministic:
            return x
        else:
            return tf.nn.dropout(x, rate=self.drop_rate)


class globalAveragePoolingLayer(object):
    """Layer which performs global average pooling"""
    def __init__(self, name, input_size):
        self.name = name
        self.input_size = input_size
        self.output_size = copy.deepcopy(input_size)
        self.output_size[1] = 1
        self.output_size[2] = 1

    def forward_pass(self, x, deterministic):
        return tf.reduce_mean(x, axis=[1, 2])


class moderationLayer(object):
    """Layer for fusing moderating data into the input vector"""
    def __init__(self, input_size, feature_size, reshape, batch_size):
        self.input_size = input_size
        self.__reshape = reshape
        self.__batch_size = batch_size

        # compute the vectorized size for weights if we will need to reshape it
        if reshape:
            self.__vec_size = input_size[1] * input_size[2] * input_size[3]
        else:
            self.__vec_size = input_size

        self.output_size = self.__vec_size + feature_size

    def forward_pass(self, x, deterministic, features):
        # Reshape into a column vector if necessary
        if self.__reshape is True:
            x = tf.reshape(x, [-1, self.__vec_size])

        # Append the moderating features onto the vector
        x = tf.concat([x, features], axis=1)

        return x


class batchNormLayer(object):
    def __init__(self, name, input_size, epsilon=1e-5, decay=0.9):
        self.name = name
        self.input_size = input_size
        self.output_size = input_size
        self.epsilon = epsilon
        self.decay = decay

    def add_to_graph(self):
        shape = self.output_size[-1]

        zeros = tf.constant_initializer(0.0)
        ones = tf.constant_initializer(1.0)

        self.offset = tf.get_variable(self.name+'_offset', shape=shape, initializer=zeros, trainable=True)
        self.scale = tf.get_variable(self.name+'_scale', shape=shape, initializer=ones, trainable=True)

        self.test_mean = tf.get_variable(self.name+'_pop_mean', shape=shape, initializer=zeros, trainable=False)
        self.test_var = tf.get_variable(self.name+'_pop_var', shape=shape, initializer=ones, trainable=False)

    def forward_pass(self, x, deterministic):
        mean, var = tf.nn.moments(x, axes=(0, 1, 2))

        # deterministic = False in training, True in testing
        if deterministic:
            y = tf.nn.batch_normalization(x, self.test_mean, self.test_var, self.offset, self.scale, self.epsilon,
                                          name=self.name + '_batchnorm')
        else:
            train_mean_op = tf.assign(self.test_mean, self.test_mean * self.decay + mean * (1 - self.decay))
            train_var_op = tf.assign(self.test_var, self.test_var * self.decay + var * (1 - self.decay))

            with tf.control_dependencies([train_mean_op, train_var_op]):
                y = tf.nn.batch_normalization(x, mean, var, self.offset, self.scale, self.epsilon,
                                              name=self.name + '_batchnorm')

        return y


class paralConvBlock(object):
    """A block consists of two parallel convolutional layers"""
    def __init__(self, name, input_size, filter_dimension_1, filter_dimension_2):

        self.name = name

        self.conv1 = convLayer(name=self.name + "_conv1",
                               input_size=input_size,
                               filter_dimension=filter_dimension_1,
                               stride_length=1,
                               activation_function='lrelu',
                               initializer='xavier',
                               padding=0,
                               batch_norm=True,
                               epsilon=1e-5,
                               decay=0.9)

        self.conv2 = convLayer(name=self.name + "_conv2",
                               input_size=input_size,
                               filter_dimension=filter_dimension_2,
                               stride_length=1,
                               activation_function='lrelu',
                               initializer='xavier',
                               padding=1,
                               batch_norm=True,
                               epsilon=1e-5,
                               decay=0.9)

        self.output_size = copy.deepcopy(self.conv1.output_size)
        self.output_size[-1] = self.conv1.output_size[-1] + self.conv2.output_size[-1]

    def add_to_graph(self):
        self.conv1.add_to_graph()
        self.conv2.add_to_graph()

    def forward_pass(self, x, deterministic):
        conv1_out = self.conv1.forward_pass(x, deterministic)
        conv2_out = self.conv2.forward_pass(x, deterministic)
        output = tf.concat([conv1_out, conv2_out], axis=3)

        return output


class skipConnection(object):
    """Makes a skip connection. Addition ops are handled by the graph-level forward_pass function."""
    def __init__(self, name, input_size, downsampled):
        self.name = name
        self.input_size = input_size

        if downsampled:
            filters = self.input_size[-1]
            self.layer = convLayer(name=self.name + '_downsample',
                                   input_size=self.input_size,
                                   filter_dimension=[1, 1, filters / 2, filters],
                                   stride_length=2,
                                   activation_function=None,
                                   initializer='xavier')
            self.output_size = self.layer.output_size
        else:
            self.layer = None
            self.output_size = input_size

    def add_to_graph(self):
        if self.layer is not None:
            self.layer.add_to_graph()

    def forward_pass(self, x, deterministic):
        if self.layer is not None:
            return self.layer.forward_pass(x, deterministic)
        else:
            return x


class copyConnection(object):
    """Defines a coconcatenation connection a la u-net. """
    def __init__(self, name, input_size, mode):
        self.name = name
        self.input_size = input_size
        self.mode = mode
        self.output_size = copy.deepcopy(input_size)

        if mode == 'load':
            self.output_size[-1] = self.output_size[-1] * 2
