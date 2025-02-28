# Week 7 Tutorial: Exploring Keras

I've provided a [notebook](07-assorted-keras.ipynb) that provides a few different ways of interacting with Keras. Feel free to pick and choose which of these exercises to do based on what might be useful for your assignment.

## Exercise 1: A simple neural network
Read through the notebook until you hit the Exercise 1 section. Follow the instructions to create a simple neural network, and look at the training curves. There should be some strange looking curves in there. What do you think is happening? 
- What are the `batch_size` and `validation_batch_size` parameters?
- How do you decide on the minibatch size?
- What are the defaults?
- What impact does modifying the batch sizes have on your history curves?

Don't go too crazy trying to optimize this model, just be aware of the behaviour.

## Exercise 2: The functional API
The [Functional API](https://keras.io/guides/functional_api/) allows you to define layers that behave like functions. These can be strung together sequentially by passing the output of one layer as the function parameter of the next, but are also more flexible and can do things like concatenation.

For this exercise, I've created a few layers but it's up to you to add on the two hidden layers (Dense).

After you've done that, try tweaking the model. You can do things like:
- Split the inputs so that some go through the "deep" layers and some go directly to the output
- Define multiple outputs - for example, predict both the house price and classify it as a "good deal" or not
- Adding auxiliary outputs to do stuff with intermediate layers

The final way of defining models in Keras is [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/), which provides the most flexibility, but it a bit more complicated. It's also the PyTorch way of doing things.

## Exercise 3: Saving and restoring models
Ultimately, you'll want to save your "best" model. Two callbacks that are really useful are `EarlyStopping` and `ModelCheckpoint`. For this exercise, create a [ModelCheckpoint](https://keras.io/api/callbacks/model_checkpoint/) callback that saves the model with the best validation loss.

How would you load the model to use it for inference?

## Exercise 4: Hyperparameter tuning
We could spend days tweaking things, or we could be more systematic about it (like the `GridSearchCV` in scikit-learn from week 1). For that matter, we could use scikit-learn tools directly, but there's also a Keras Tuner library that's built for this.

Think about:
- What are some hyperparameters we could tune?
- Which ones are most important?
- How do we decide on the range of values to try?

## Exercise 5: Image augmentation
CNNs are pretty good about being translation invariant, but they still have a hard time with image rotations, flips, rescaling, etc.

For this exercise, run all the cells under the "Image Augmentation" section and observe the difference in validation accuracy at the last epoch compared to the accuracy when augmentation is applied. Next, try the following:
- Go back to the model definition stage and insert the augmentation layers, then retrain
- What happens to the accuracy of the augmented validation set now?
- Can you think of other augmentations that would be useful in this context?