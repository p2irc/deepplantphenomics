import networks

class tools(object):
    """
    Provides stand-alone phenotyping tools which can be called statically.
    """

    @staticmethod
    def predict_rosette_leaf_count(x, batch_size=8):
        """
        Uses a pre-trained network to predict the number of leaves on rosette plants.
        Images are input as a list of filenames.
        """

        net = networks.rosetteLeafRegressor(batch_size=batch_size)

        predictions = net.forward_pass(x)

        net.shut_down()

        return predictions

    @staticmethod
    def classify_arabidopsis_strain(x, batch_size=32):
        """
        Uses a pre-trained network to classify arabidopsis strain
        """

        net = networks.arabidopsisStrainClassifier(batch_size=batch_size)

        predictions = net.forward_pass(x)

        net.shut_down()

        return predictions