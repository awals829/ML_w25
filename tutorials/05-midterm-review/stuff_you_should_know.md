# 1. Questions for Data Exploration and Cleaning
Some of these things get into modelling territory, but they were discussed in the first couple of classes.

## Big Picture
- Why is domain knowledge important?
- What kinds of machine learning tasks are there?
- What are the general steps in a machine learning project?

## Data Exploration
- Why do we need to reserve a test set?
- What is data peeking (or snooping) bias?
- What problems might arise from randomly shuffling the data to select the test set?
- What is stratified sampling and when is it useful?

## Data Cleaning
- What are the 3 methods of dealing with missing features presented in the book?
- What are the pros/cons of each method?
- What methods could we use to convert categorical or text data to numeric information?
- What are the assumptions about ordinal encoding? When does it make sense to use this approach?
- What are the limitations of one-hot encoding?
- Why do we need to make sure the data are on approximately the same scale?
- What are common methods for rescaling data?
- Which method is more affected by outliers?
- How would you decide which one to use?

## Regression Modelling
- What kind of performance measures make sense for regression tasks?
- What is under and over fitting?
- What are some ways to reduce overfitting?
- What is cross-validation?
- When would you choose cross-validation instead of a test/train/validate split?
- What does it mean if the cross-validation error is higher than the training error?
- What are hyperparameters vs parameters?

# 2. Questions for math review
The following should be answerable at a concept level - no need to memorize formulas.

## Linear Algebra
- What are scalars, vectors, and matrices?
- What is the norm of a vector?
- What is the dot product of two vectors?
- How do you multiply two matrices?

## Calculus
- What is a (partial) derivative?
- Why are derivatives important in machine learning?
- What is the chain rule?
- What is the gradient?

## Probability and Statistics
- What is the difference between a discrete and continuous random variable?
- What are the probability mass and density functions?
- What is the expected value?
- What are covariance and correlation?
- What are the limitations of correlation coefficients?
- Why is the normal distribution so common?

# 3. Questions for training models
## Linear Regression
- What does it mean to have a closed-form solution?
- Why is the normal equation not always the best way to solve linear regression?
- How can linear regression be used to fit polynomials?
- What is the design matrix (X)?

## Gradient Descent
- What is the difference between the normal equation and gradient descent?
- What are stochastic, batch, and minibatch gradient descent?
- Why would you choose one over the other?
- Why is data normalization important for gradient descent?
- What is the learning rate or step size?
- How would you choose when to stop training?
- What is the role of the validation set?
- What criteria does a loss function need to meet to be suitable for gradient descent?

# 4. Questions for perceptron/backpropagation
## Perceptrons
- What is a perceptron?
- Under what conditions is a perceptron guaranteed to converge?
- Why was the perceptron unable to solve the XOR problem?
- What is the nonlinear activation function used in a perceptron?
- Why does a multilayer perceptron (MLP) solve the XOR problem?
- What was the key insight that allowed training of MLP-like networks?

## Backpropagation
- What is being calculated in backpropagation?
- What are the forward and backward passes?
- How are backpropagation and gradient descent related?
- What are some computational efficiencies that are commonly used to speed up backpropagation?
- Why are bias terms necessary in a neural network?
- Why are nonlinear activation functions necessary in a neural network?
- What is the difference between hidden and output activation functions?
- Why is ReLU (and its variations) more popular than the sigmoid function for hidden layers?
- What drives the choice of loss and output activation functions?
- What are the parameters in a basic neural network?

# 5. Questions for classification and more NN details
## Cross-entropy loss
- Why is mean squared error (and its various flavours) not a good loss function for classification problems?
- What is the expected value operator?
- What is a Bernoulli distribution?
- What is the information of an event?
- What is entropy?
- What is cross-entropy measuring?
- How is cross-entropy related to the KL divergence?
- What output activation functions are used to predict probabilities?

## More NN details
- Why is weight initialization important?
- What key number should be considered in weight initialization?
- How is the activation function related to weight initialization?
- What is the vanishing/exploding gradient problem?
- How can batch normalization help stabilize training?