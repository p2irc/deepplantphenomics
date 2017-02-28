import numpy as np
import os

class boundingBoxRegressor(object):
    model = None

    img_height = 257
    img_width = 307

    original_img_height = None
    original_img_width = None

    __dir_name = 'bbox-regressor-lemnatec'

    def __init__(self, height, width, batch_size=4):
        """A network which predicts bounding box coordinates via a convolutional neural net"""

        # Set original image dimensions
        self.original_img_height = height
        self.original_img_width = width

        m_path, _ = os.path.split(__file__)
        checkpoint_path = os.path.join(m_path, 'network_states', self.__dir_name)

        import deepplantpheno as dpp

        self.model = dpp.DPPModel(debug=False, load_from_saved=checkpoint_path)

        self.model.clear_preprocessors()

        # Define model hyperparameters
        self.model.set_batch_size(batch_size)
        self.model.set_number_of_threads(1)
        self.model.set_original_image_dimensions(self.original_img_height, self.original_img_width)
        self.model.set_image_dimensions(self.img_height, self.img_width, 3)
        self.model.set_resize_images(True)

        self.model.set_problem_type('regression')

        # Define a model architecture
        self.model.add_input_layer()

        self.model.add_convolutional_layer(filter_dimension=[5, 5, 3, 16], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
        self.model.add_pooling_layer(kernel_size=3, stride_length=2)

        self.model.add_convolutional_layer(filter_dimension=[5, 5, 16, 64], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
        self.model.add_pooling_layer(kernel_size=3, stride_length=2)

        self.model.add_convolutional_layer(filter_dimension=[5, 5, 64, 64], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
        self.model.add_pooling_layer(kernel_size=3, stride_length=2)

        self.model.add_convolutional_layer(filter_dimension=[5, 5, 64, 64], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
        self.model.add_pooling_layer(kernel_size=3, stride_length=2)

        self.model.add_fully_connected_layer(output_size=384, activation_function='relu')

        self.model.add_output_layer(regularization_coefficient=0.0)

    def forward_pass(self, x):
        y = self.model.forward_pass_with_file_inputs(x)

        # rescale coordinates from network input size to original image size
        height_ratio = (self.original_img_height / float(self.img_height))
        width_ratio = (self.original_img_width / float(self.img_width))

        y[:, 0] = y[:, 0] * width_ratio
        y[:, 1] = y[:, 1] * width_ratio
        y[:, 2] = y[:, 2] * height_ratio
        y[:, 3] = y[:, 3] * height_ratio

        return y

    def shut_down(self):
        self.model.shut_down()


class rosetteLeafRegressor(object):
    model = None

    img_height = 128
    img_width = 128

    __dir_name = 'rosette-leaf-regressor'

    def __init__(self, batch_size=8):
        """A network which predicts bounding box coordinates via a convolutional neural net"""

        m_path, _ = os.path.split(__file__)
        checkpoint_path = os.path.join(m_path, 'network_states', self.__dir_name)

        import deepplantpheno as dpp

        self.model = dpp.DPPModel(debug=False, load_from_saved=checkpoint_path)

        self.model.clear_preprocessors()

        # Define model hyperparameters
        self.model.set_batch_size(batch_size)
        self.model.set_number_of_threads(1)
        self.model.set_image_dimensions(self.img_height, self.img_width, 3)
        self.model.set_resize_images(True)

        self.model.set_problem_type('regression')

        self.model.set_augmentation_crop(True)

        # Define a model architecture
        self.model.add_input_layer()

        self.model.add_convolutional_layer(filter_dimension=[5, 5, 3, 16], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
        self.model.add_pooling_layer(kernel_size=3, stride_length=2)

        self.model.add_convolutional_layer(filter_dimension=[5, 5, 16, 64], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
        self.model.add_pooling_layer(kernel_size=3, stride_length=2)

        self.model.add_convolutional_layer(filter_dimension=[5, 5, 64, 64], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
        self.model.add_pooling_layer(kernel_size=3, stride_length=2)

        self.model.add_convolutional_layer(filter_dimension=[5, 5, 64, 64], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
        self.model.add_pooling_layer(kernel_size=3, stride_length=2)

        self.model.add_fully_connected_layer(output_size=2048, activation_function='relu')
        self.model.add_fully_connected_layer(output_size=2048, activation_function='relu')

        self.model.add_output_layer(regularization_coefficient=0.0)

    def forward_pass(self, x):
        y = self.model.forward_pass_with_file_inputs(x)

        # round for leaf counts
        y = np.round(y)

        return y

    def shut_down(self):
        self.model.shut_down()

class arabidopsisStrainClassifier(object):
    model = None

    img_height = 128
    img_width = 128

    __dir_name = 'arabidopsis-strain-classifier'

    def __init__(self, batch_size=32):
        """A network which predicts bounding box coordinates via a convolutional neural net"""

        m_path, _ = os.path.split(__file__)
        checkpoint_path = os.path.join(m_path, 'network_states', self.__dir_name)

        import deepplantpheno as dpp

        self.model = dpp.DPPModel(debug=False, load_from_saved=checkpoint_path)

        # Define model hyperparameters
        self.model.set_batch_size(batch_size)
        self.model.set_number_of_threads(1)
        self.model.set_image_dimensions(self.img_height, self.img_width, 3)
        self.model.set_resize_images(True)

        self.model.set_augmentation_crop(True)

        # Define a model architecture
        self.model.add_input_layer()

        self.model.add_convolutional_layer(filter_dimension=[5, 5, 3, 32], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
        self.model.add_pooling_layer(kernel_size=3, stride_length=2)

        self.model.add_convolutional_layer(filter_dimension=[5, 5, 32, 64], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
        self.model.add_pooling_layer(kernel_size=3, stride_length=2)

        self.model.add_convolutional_layer(filter_dimension=[5, 5, 64, 64], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
        self.model.add_pooling_layer(kernel_size=3, stride_length=2)

        self.model.add_convolutional_layer(filter_dimension=[5, 5, 64, 64], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
        self.model.add_pooling_layer(kernel_size=3, stride_length=2)

        self.model.add_fully_connected_layer(output_size=4096, activation_function='relu')
        self.model.add_dropout_layer(0.5)
        self.model.add_fully_connected_layer(output_size=4096, activation_function='relu')
        self.model.add_dropout_layer(0.5)

        self.model.add_output_layer(regularization_coefficient=0.0, output_size=5)

    def forward_pass(self, x):
        y = self.model.forward_pass_with_file_inputs(x)

        # Convert from class probabilities to labels
        indices = np.argmax(y, axis=1)
        mapping = {0: 'Col-0', 1: 'ein2', 2: 'pgm', 3: 'adh1', 4: 'ctr'}
        labels = [mapping[index] for index in indices]

        return labels

    def shut_down(self):
        self.model.shut_down()