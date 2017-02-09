import networks

class tools(object):
    """
    Provides stand-alone phenotyping tools which can be called statically.
    """

    @staticmethod
    def predict_rosette_leaf_count(x):
        """
        Uses a pre-trained network to predict the number of leaves on rosette plants.
        Images are input as a list of filenames.
        """

        net = networks.rosetteLeafRegressor()

        predictions = net.forward_pass(x)

        net.shut_down()

        return predictions