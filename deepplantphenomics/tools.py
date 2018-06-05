from . import networks
import numpy as np
import cv2

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

        # round for leaf counts
        predictions = np.round(predictions)

        return predictions


    @staticmethod
    def classify_arabidopsis_strain(x, batch_size=32):
        """
        Uses a pre-trained network to classify arabidopsis strain
        """

        net = networks.arabidopsisStrainClassifier(batch_size=batch_size)
        predictions = net.forward_pass(x)
        net.shut_down()

        # Convert from class probabilities to labels
        indices = np.argmax(predictions, axis=1)
        mapping = {0: 'Col-0', 1: 'ein2', 2: 'pgm', 3: 'adh1', 4: 'ctr'}
        labels = [mapping[index] for index in indices]

        return labels


    @staticmethod
    def segment_vegetation(x, batch_size=8):
        """
        Uses a pre-trained fully convolutional network to perform vegetation segmentation
        """

        net = networks.vegetationSegmentationNetwork(batch_size=batch_size)
        predictions = net.forward_pass(x)
        net.shut_down()

        # round for binary mask
        #predictions = np.round(predictions)
        _, predictions = cv2.threshold(predictions.astype(np.float32),0.5,1.0,cv2.THRESH_BINARY)

        return predictions
