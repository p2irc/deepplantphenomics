import os

class boundingBoxRegressor(object):
    model = None

    img_height = 257
    img_width = 307

    original_img_height = None
    original_img_width = None

    __dir_name = 'bbox-regressor-lemnatec'

    def __init__(self, height, width):
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
        self.model.set_batch_size(4)
        self.model.set_number_of_threads(4)
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

    def __init__(self):
        """A network which predicts bounding box coordinates via a convolutional neural net"""

        m_path, _ = os.path.split(__file__)
        checkpoint_path = os.path.join(m_path, 'network_states', self.__dir_name)

        import deepplantpheno as dpp

        self.model = dpp.DPPModel(debug=False, load_from_saved=checkpoint_path)

        self.model.clear_preprocessors()

        # Define model hyperparameters
        self.model.set_batch_size(8)
        self.model.set_number_of_threads(4)
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

        return y

    def shut_down(self):
        self.model.shut_down()