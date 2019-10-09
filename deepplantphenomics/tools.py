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
    def segment_vegetation(x, batch_size=8):
        """
        Uses a pre-trained fully convolutional network to perform vegetation segmentation
        """
        net = networks.vegetationSegmentationNetwork(batch_size=batch_size)
        predictions = net.forward_pass(x)
        net.shut_down()

        # round for binary mask
        predictions = predictions.astype(np.float32)
        predictions = np.squeeze(predictions, axis=3)
        for i, mask in enumerate(predictions):
            _, thresh_mask = cv2.threshold(mask, 0.5, 1.0, cv2.THRESH_BINARY)
            predictions[i, :, :] = thresh_mask
        return predictions

    @staticmethod
    def count_canola_flowers(x, batch_size=1, image_height=300, image_width=300, image_depth=3):

        net = networks.countCeptionCounter(
            batch_size=batch_size, image_height=image_height, image_width=image_width, image_depth=image_depth)
        predictions = net.forward_pass(x)
        net.shut_down()

        # round for counts
        predictions = np.round(predictions)

        return predictions
