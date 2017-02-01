class boundingBoxRegressor(object):
    model = None

    img_height = 257
    img_width = 307

    original_img_height = None
    original_img_width = None

    def __init__(self, height, width):
        """A network which predicts bounding box coordinates via a convolutional neural net"""

        # Set original image dimensions
        self.original_img_height = height
        self.original_img_width = width

        #with tf.variable_scope("bbox-regressor") as scope:
        import deepplantpheno as dpp

        self.model = dpp.DPPModel(debug=True, load_from_saved='./network_states/bbox-regressor-lemnatec')

        self.model.clearPreprocessors()

        # Define model hyperparameters
        self.model.setBatchSize(4)
        self.model.setNumberOfThreads(4)
        self.model.setOriginalImageDimensions(self.original_img_height, self.original_img_width)
        self.model.setImageDimensions(self.img_height, self.img_width, 3)
        self.model.setResizeImages(True)
        self.model.setWeightInitializer('normal')

        self.model.setProblemType('regression')

        # Define a model architecture
        self.model.addInputLayer()

        self.model.addConvolutionalLayer(filter_dimension=[5, 5, 3, 64], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
        self.model.addPoolingLayer(kernel_size=3, stride_length=2)

        self.model.addConvolutionalLayer(filter_dimension=[5, 5, 64, 128], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
        self.model.addPoolingLayer(kernel_size=3, stride_length=2)

        self.model.addConvolutionalLayer(filter_dimension=[5, 5, 128, 128], stride_length=1, activation_function='relu', regularization_coefficient=0.0)
        self.model.addPoolingLayer(kernel_size=3, stride_length=2)

        self.model.addFullyConnectedLayer(output_size=384, activation_function='relu')

        self.model.addOutputLayer(regularization_coefficient=0.0)

    def forwardPass(self, x):
        y = self.model.forwardPassWithFileInputs(x)

        height_ratio = (self.original_img_height / float(self.img_height))
        width_ratio = (self.original_img_width / float(self.img_width))

        # TODO
        y = y * height_ratio

        return y

    def shutDown(self):
        self.model.shutDown()