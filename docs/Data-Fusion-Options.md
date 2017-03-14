## Data Fusion Options

There may be cases where you want to add more features to the data, besides the features automatically extracted from the images e.g. by convolutional layers. For example, you may have environmental metadata which corresponds to the images but exists in a CSV file.

In these cases, combining the two types of data is a process known as ***data fusion***. In Deep Plant Phenomics, data which does not come from the feature extractor are called ***moderation features***.

This data should exist as an array of array values in the order corresponding to the order of the images. The moderation features can be loaded into the model with 

```
model.add_moderation_features(my_features)
```

Now the features are set to be included in all of the training and testing batches. The only other thing we need to do is decide where to append the moderation features in the network architecture. This is done using a special layer called a ***moderation layer***. For example:

```
model.add_moderation_layer()
```

This simply appends the moderation features into the vector at that point in the network.