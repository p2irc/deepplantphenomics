The performance of the training process is sensitive to different constants called hyperparameters. DPP supports automatic hyperparameter optimization via the grid search method in order to evaluate performance under many different combinations of hyperparameter values.

## Training with Hyperparameter Search

In order to train with hyperparameter search, you must replace the line 

```
model.begin_training()
```

with a modified version containing the ranges of hyperparameters you would like to evaluate:

```
model.begin_training_with_hyperparameter_search(l2_reg_limits=[0.001, 0.005], lr_limits=[0.0001, 0.001], num_steps=4)
```

Here, you can see that we are searching over values for two hyperparameters: the L2 regularization coefficient (`l2_reg_limits`) and the learning rate (`lr_limits`). If you don't want to search over a particular hyperparameter, just set its limits to `None` and make sure you set it manually in your model (for example, with `set_regularization_coefficient()`). The values in brackets indicate the lowest and highest values to try, respectively. The area in between the low and high values is divided into equal parts depending on the number of steps chosen.

The parameter `num_steps=4` means that the system will search over 4 values for each of the two hyperparameters, meaning that in total 12 runs will be executed. Please note that larger values for `num_steps` will increase the amount of runs exponentially, which will increase the run time dramatically.