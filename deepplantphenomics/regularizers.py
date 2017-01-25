import tensorflow as tf


class regularizers(object):
    @staticmethod
    def shakeWeight_Invert(x, weights, p):
        unifRand = tf.random_uniform(weights.get_shape().as_list())
        falseVec = tf.ones_like(unifRand)
        trueVec = tf.mul(tf.ones_like(unifRand), -1)
        mask = tf.select(tf.less_equal(unifRand, p), falseVec, trueVec)
        perturbed = tf.mul(weights, mask)
        activations = tf.matmul(x, perturbed)

        return activations

    @staticmethod
    def shakeWeight(x, weights, p, c=0.):
        """ Implements the ShakeWeight operation simillar to ShakeOut but instead of
            generating a mask of size [batch_size, num_of_outputs], ShakeWeight
            generates a mask of size [batch_size, num_of_inputs, num_of_outputs].

            shakeWeight computes the pre-activations given the input and the layer
            weights and hence, must be followed by an activation function if
            desired.
            Args:
                x (T): A batch of the input.
                weights (T): The weights of the layer.
                biases (T): The biases of the layer.
                p (float): The shakeOut parameter corresponding to dropout.
                c (float): The shakeOut parameter corresponding to L1 regularization.

            Returns:
                T: Pre-activations of the layer
            Examples:
                pre_activ = shakeWeight(x, weights, biases, c, tau)
        """
        inpS = tf.shape(weights)[0]
        outS = tf.shape(weights)[1]
        bS = tf.shape(x)[0]
        tau = tf.sub(1., p)
        c = tf.select(tf.equal(tau, 0.), 0., c)
        # Generate uniform random variables between [0,1]
        #########################################################
        #                       Shake Weights
        # Note the bS, _inpS_, outS
        unifRand = tf.random_uniform([bS, inpS, outS])
        falseVec = tf.zeros_like(unifRand)
        recipP = tf.div(1.0, tf.sub(1., tau))
        trueVec = tf.mul(tf.ones_like(unifRand), recipP)
        R = tf.select(tf.less_equal(unifRand, tau), trueVec, falseVec)

        # calculate r_j*w_ij
        term1 = tf.mul(R, weights)

        # calculate (r_j - 1)*c*sign(w_ij)
        rjMinus1 = tf.sub(R, 1.0)
        crjMinus1 = tf.mul(rjMinus1, c)
        term2 = tf.mul(crjMinus1, tf.sign(weights))

        # final shaked weights and outputs
        shakedOutW = tf.add(term1, term2)

        xe = tf.expand_dims(x, 1)
        shakedOut_pre_activation = tf.squeeze(tf.batch_matmul(xe, shakedOutW))

        return shakedOut_pre_activation

    @staticmethod
    def shakeOut(x, weights, p, c=0.):
        """ Implements the ShakeOut operation as described in the ShakeOut paper,
            except that the author do not discuss the issue of per example mask
            rather than per batch mask.

            shakeOut computes the pre-activations given the input and the layer
            weights and hence, must be followed by an activation function if
            desired.

            Args:
                x (T): A batch of the input.
                weights (T): The weights of the layer.
                biases (T): The biases of the layer.
                p (float): The shakeOut parameter corresponding to dropout.
                c (float): The shakeOut parameter corresponding to L1 regularization.
            Returns:
                T: Pre-activations of the layer
            Examples:
                pre_activ = shakeOut(x, weights, biases, c, tau)
        """
        inpS = tf.shape(weights)[0]
        outS = tf.shape(weights)[1]
        bS = tf.shape(x)[0]
        tau = tf.sub(1., p)
        c = tf.select(tf.equal(tau, 0.), 0., c)
        # Generate uniform random variables between [0,1]
        #########################################################
        #                       Shake Out
        # Note the bS, _1_, outS
        unifRand = tf.random_uniform([bS, 1, outS])
        falseVec = tf.zeros_like(unifRand)
        recipP = tf.div(1.0, tf.sub(1., tau))
        trueVec = tf.mul(tf.ones_like(unifRand), recipP)
        R = tf.select(tf.less_equal(unifRand, tau), trueVec, falseVec)

        # calculate r_j*w_ij
        term1 = tf.mul(R, weights)

        # calculate (r_j - 1)*c*sign(w_ij)
        rjMinus1 = tf.sub(R, 1.0)
        crjMinus1 = tf.mul(rjMinus1, c)
        term2 = tf.mul(crjMinus1, tf.sign(weights))

        # final shaked weights and outputs
        shakedOutW = tf.add(term1, term2)

        xe = tf.expand_dims(x, 1)
        shakedOut_pre_activation = tf.squeeze(tf.batch_matmul(xe, shakedOutW))

        return shakedOut_pre_activation

    @staticmethod
    def dropConnect(x, weights, p):
        """ Implements the DropConnect regularization by generating a mask of size
            [batch_size, num_of_inputs, num_of_outputs], and multiplies the mask
            before the activation function on the weights rather than on the
            activations as is the case in DropOut.

            dropConnect computes the pre-activations given the input and the layer
            weights and hence, must be followed by an activation function if
            desired.
            Args:
                x (T): A batch of the input.
                weights (T): The weights of the layer.
                biases (T): The biases of the layer.
                p (float): The retention probability of a connection.
            Returns:
                T: Pre-activations of the layer
            Examples:
                pre_activ = dropConnect(prev_layer_output, weights, biases, p)
        """
        inpS = tf.shape(weights)[0]
        outS = tf.shape(weights)[1]
        bS = tf.shape(x)[0]

        #########################################################
        #                       DropConnect
        # Generate uniform random variables between [0,1]
        unifRand = tf.random_uniform([bS, inpS, outS])
        falseVec = tf.zeros_like(unifRand)
        trueVec = tf.mul(tf.ones_like(unifRand), tf.div(1.0, p))
        # Transform uniform random variable into Ber(p)
        R = tf.select(tf.less_equal(unifRand, p), trueVec, falseVec)

        # calculate r_ij*w_ij
        dropedConnect = tf.mul(R, weights)
        xe = tf.expand_dims(x, 1)
        dropedConnect_pre_activation = tf.squeeze(tf.batch_matmul(xe, dropedConnect))
        return dropedConnect_pre_activation