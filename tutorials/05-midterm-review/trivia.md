## Midterm review: pub trivia style

1. Give 2 examples of ways to handle missing data.

<details>
<summary>Answer</summary>
Imputing (with mean, nearest neighbour, constant value, etc), dropping the feature entirely, dropping the record from training (not testing!)

</details>

2. Name 1 limitation of the correlation coefficient.

<details>
<summary>Answer</summary>
Possible options include: only describes linear relationships, can only be used with numeric variables, can be heavily influenced by outliers
</details>

3. Why is normalization important for gradient descent?

<details>
<summary>Answer</summary>
Weight change is a small step in the gradient direction, so all gradients should be more or less the same scale (otherwise some weights have no change while others have big jumps).
</details>

4. Would you ever use gradient descent for linear regression instead of the normal equation? Why or why not?

<details>
<summary>Answer</summary>
Yes, because the normal equation requires an expensive and sometimes numerically unstable matrix inversion.
</details>

5. What does regularization do?

<details>
<summary>Answer</summary>
Helps to prevent overfitting by imposing an additional limitation on the weights.
</details>

6. Under what conditions is a perceptron guaranteed to converge?

<details>
<summary>Answer</summary>
When the data are linearly separable.
</details>

7. What is a key difference in modern neural networks compared to multi layer perceptrons (hint: it allows backpropagation to work)?

<details>
<summary>Answer</summary>
Differentiable activation functions (sigmoid, tanh, ReLU, etc) instead of the step function.
</details>

8. What is being calculated in backpropagation?

<details>
<summary>Answer</summary>
The gradient of the loss with respect to the weights and biases of each layer.
</details>

9. In descriptive/intuitive terms, what is entropy measuring?

<details>
<summary>Answer</summary>
Something like the expected amount of "surprise" of a distribution, average level of uncertainty or average amount of information.
</details>

1.  What activation function should you use for the output of a binary classification model?

<details>
<summary>Answer</summary>
Sigmoid
</details>

11. Why is accuracy not always the best measure of classifier performance?

<details>
<summary>Answer</summary>
It can be misleading with imbalanced classes.
</details>

1.  What is the vanishing/exploding gradient problem?

<details>
<summary>Answer</summary>
As the same terms are being multiplied repeatedly, they can get very large or very small if not carefully initialized.
</details>

