# Assignment 2: Playing card classification
Due March 7, 2025 at 5 pm

You may work in teams of 2 or 3. Click [here](https://classroom.github.com/a/NaaV3FNQ) to create your team on GitHub Classroom. This can be the same team or different from assignment 1.

## Table of Contents <!-- omit in toc -->
- [Overview](#overview)
- [Dataset](#dataset)
- [Deliverables](#deliverables)
    - [Your code](#your-code)
    - [Your report](#your-report)
- [Marking Scheme](#marking-scheme)

## Overview
The purpose of this assignment is to apply your theoretical knowledge of neural networks (particularly convolutional neural networks) to a real application. You will **build and train a model** from "scratch" (using Keras or some other framework, not really from scratch), and then see how much you can **reduce its size** while minimizing performance degradation. In addition to building and tweaking a neural network, this assignment serves as an introduction to:
- Working with streaming data (such as tensorflow's [Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset))
- Preprocessing for image data
- Evaluating classification models

## Dataset
This time, I'm choosing the dataset: [Playing Cards](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification). This is quite a clean dataset with reasonable class balance, and I was able to get around 80% accuracy on the validation set without really trying. There are lots of implementations of classifiers using this dataset, and if you do look at someone else' work for ideas, make sure to **cite your sources** and **understand** what you are implementing.

## Deliverables
Your assignment should consist of the following:
1. Your notebook(s) and/or Python scripts where you did your experiments, with the final training run and evaluation rendered
2. A report describing your experiments and your final model decisions
3. Your final model in [`.keras`](https://keras.io/guides/serialization_and_saving/) format, or similar if you're using a different framework. **Please make sure that it loads and runs properly using the default package versions in Colab**, which I hope are consistent from one user to the next. I've provided a listing of the [Python package versions](colab_versions.txt) on my Colab instance which you can use if you're training on your own hardware.

I would recommend working in parallel with your teammate(s) and commit your changes after each experiment. It's fine if you have multiple working notebooks, just indicate to me which is the final version.

### Your code
Your code should:
- Load the training data and do some basic data exploration, like looking at samples, number of classes, class distribution, etc. The code in `starter.ipynb` provides some ideas for connecting Colab to Google Drive, defining the training dataset, and inspecting a few samples.
- Do any preprocessing you might want to do (at the very least, you'll probably want to rescale the images from unsigned ints in the range 0-255 to floats in the range 0 to 1)
  > Hint: Keras has some preprocessing layers that you can stick on the start of your model much like Scikit-learn's pipelines. For example, here's the [Rescaling](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Rescaling) layer.
- Define, compile, and train a model, keeping in mind the following:
    - The [input](https://keras.io/api/layers/core_layers/input/) layer must match the size of your image data, e.g. `(SIZE, SIZE, 3)` where 3 is the number of colour channels. You do not need to define the batch size.
    - The output layer must have as many neurons as classes you are trying to predict, with the softmax activation function.
    - Everything in between is a design choice that you can tweak!
- Iterate! I would suggest starting with a simple CNN feeding in to a fully connected output layer. My fairly random and not at all optimized model looked like this:
    ```python
    model = tf.keras.Sequential([
        Input((SIZE, SIZE, 3)),
        Rescaling(scale=1./255),
        Conv2D(32, (3, 3), activation="relu"),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPool2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPool2D((2, 2)),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(53, activation="softmax")
    ])
    model.summary()
    ```
    I'm sure you can do better!
- Train two models: one "best performance" version, where you try to get the highest **accuracy**, and one "size optimized" where you try to maintain reasonably good accuracy with the **fewest parameters**.
- Once you've trained your models, save them using `model.save("filename.keras")`. This is a newish format that includes the model weights and architecture all in one so they can be loaded with `tf.keras.models.load_model("filename.keras")`. If you are using a different framework, please include the code to load the model in your report.

### Your report
In a separate document, summarize your experiments, models, observations, reflections, etc. I've provided a template with more details in the starter code (`report.md`), though you aren't limited to the markdown format.

## Marking Scheme
Each of the following components will be marked on a 4-point scale and weighted.

| Component                                               | Weight |
| ------------------------------------------------------- | ------ |
| Report: Model development and experimentation           | 20%    |
| Report: reflections                                     | 20%    |
| Report: abstract and appendices                         | 20%    |
| Model: load and run on Colab                            | 10%    |
| Model: performance (highest accuracy model)             | 10%    |
| Model: performance / parameters ratio (size optimized)  | 10%    |
| Evidence of collaboration (comments, code reviews, etc) | 10%    |

| Score | Description                                                            |
| ----- | ---------------------------------------------------------------------- |
| 4     | Excellent - thoughtful and creative without any errors or omissions    |
| 3     | Pretty good, but with minor errors or omissions                        |
| 2     | Mostly complete, but with major errors or omissions, lacking in detail |
| 1     | A minimal effort was made, incomplete or incorrect                     |
| 0     | No effort was made, or the submission is plagiarized                   |
