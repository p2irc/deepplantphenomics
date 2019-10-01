After training a model, if you are happy with the performance, you can deploy it for future use. This tutorial uses the leaf counter as an example. If you are not sure how the leaf counter was trained, please see [Training the Leaf Counter](Tutorial-Training-The-Leaf-Counter.md).

## Move the Network State to a Safe Place

By default, DPP will dump out the full network state (the values for all trainable parameters) to a directory called `saved_state` in the working path when it finishes training. Move these state files somewhere where they will not get overwritten. Let's move the files to a directory called `rosette-leaf-regressor`.

## Create a Network Class

The next thing you should do is make a class to abstract the use of the network. A simplified example for the leaf counter is:

```python
class rosetteLeafRegressor(object):
    model = None

    img_height = 128
    img_width = 128

    __dir_name = 'rosette-leaf-regressor'

    def __init__(self, batch_size=8):
        """A network which predicts rosette leaf count via a convolutional neural net"""

        import deepplantphenomics as dpp

        self.model = dpp.RegressionModel(debug=False, load_from_saved=self.__dir_name)

        # Define model hyperparameters
        self.model.set_batch_size(batch_size)
        self.model.set_number_of_threads(1)
        self.model.set_image_dimensions(self.img_height, self.img_width, 3)
        self.model.set_resize_images(True)

        self.model.set_augmentation_crop(True)

        # Define a model architecture
        self.model.add_input_layer()

        self.model.add_convolutional_layer(filter_dimension=[5, 5, 3, 32], stride_length=1, activation_function='tanh')
        self.model.add_pooling_layer(kernel_size=3, stride_length=2)

        self.model.add_convolutional_layer(filter_dimension=[5, 5, 32, 64], stride_length=1, activation_function='tanh')
        self.model.add_pooling_layer(kernel_size=3, stride_length=2)

        self.model.add_convolutional_layer(filter_dimension=[3, 3, 64, 64], stride_length=1, activation_function='tanh')
        self.model.add_pooling_layer(kernel_size=3, stride_length=2)

        self.model.add_convolutional_layer(filter_dimension=[3, 3, 64, 64], stride_length=1, activation_function='tanh')
        self.model.add_pooling_layer(kernel_size=3, stride_length=2)

        self.model.add_output_layer()

    def forward_pass(self, x):
        y = self.model.forward_pass_with_file_inputs(x)

        return y

    def shut_down(self):
        self.model.shut_down()
```

Note that the `__init__()` function builds the full network, exactly the same as the network which we trained. If the architecture is any different, there will be an error.

The important line here is:

```python
self.model = dpp.RegressionModel(debug=False, load_from_saved=self.__dir_name)
```

The parameter `debug=False` suppresses any console output when we use this class. The parameter `load_from_saved=dir_name` loads all of the parameters from the trained network, so the model is exactly as it was when we finished training.

Also note that `self.model.set_number_of_threads(1)` restricts the preprocessing to only one thread - this is important so that the output of the network will be in the exact same order as the images we feed in. We also must specify `self.model.set_augmentation_crop(True)` so that inputs are cropped to center, as this is the input size our trained network expects due to our use of the crop augmentation during training.

After that, we simply have one function called `forward_pass()` for performing inference, and one function called `shut_down()` to destroy the network and release memory when we are done.

## Test it Out

The only thing left to do is to try out this new class we made, using the forward pass function with a list of filenames of images we want to run.

```python
images = ['plant_1.png', 'plant_2.png', 'plant_3.png']

print('Performing leaf estimation...')

net = rosetteLeafRegressor()
leaf_counts = net.forward_pass(images)
net.shut_down()

for k,v in zip(images, leaf_counts):
    print '%s: %d' % (k, v)
    
print('Done')
```

It's worth noting that if you are performing inference on the same data you trained on, the performance is not representative as you are including images that the model has already fit.