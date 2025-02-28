import numpy as np

# define activation functions and their derivatives as constants
F = {
    "linear": lambda x: x,
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "relu": lambda x: x * (x > 0),
    "tanh": lambda x: np.tanh(x),
}

D_F = {
    "linear": lambda x: np.ones_like(x),
    "sigmoid": lambda x: F["sigmoid"](x) * (1 - F["sigmoid"](x)),
    "relu": lambda x: (x > 0).astype(float),
    "tanh": lambda x: 1 - np.tanh(x) ** 2,
}

L = {
    "mse": lambda y, y_est: np.mean((y - y_est) ** 2),
    "cross_entropy": lambda y, p_hat: -np.mean(
        y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat)
    ),
}

D_L = {
    "mse": lambda y, y_est: y_est - y,
    "cross_entropy": lambda y, p_hat: -(y / p_hat + (1 - y) / (1 - p_hat)),
}


class MLPRegressor:
    def __init__(self, n_inputs: int, loss: str = "mse"):
        """
        Initialize a new multi-layer perceptron with the specified number of inputs.
        """
        # List of numpy arrays for weights and biases (1 set per layer)
        self.weights = []
        self.biases = []

        # keep track of feature dimensions and layer inputs
        self.dims = [n_inputs]
        self.inputs = []

        # and strings to keep track of activation functions
        self.activations = []

        # loss function
        self.loss = loss

    def __repr__(self):
        """
        Print out network architecture.
        """
        repr_str = "MLPRegressor\n"
        for i in range(len(self.activations)):
            repr_str += (
                f"  Layer {i+1}: {self.dims[i]} inputs -> "
                f"{self.dims[i+1]} neurons, activation={self.activations[i]}\n"
            )

        repr_str += f"Total trainable parameters: {sum(w.size + b.size for w, b in zip(self.weights, self.biases))}"

        return repr_str

    def add_layer(self, n_neurons: int, activation: str) -> None:
        """
        Add a new layer with n_neurons and the given activation function.
        Initialize corresponding weight matrix to random values.
        """
        std_glorot = np.sqrt(2 / (self.dims[-1] + n_neurons))
        self.weights.append(np.random.randn(self.dims[-1], n_neurons) * std_glorot)
        self.biases.append(np.zeros(n_neurons))
        self.activations.append(activation)

        # self.n_inputs has 1 more element than the other lists
        self.dims.append(n_neurons)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass to compute estimate based on inputs.
        Output is an inputs.shape[0] dimension vector.
        """
        if inputs.shape[1] != self.dims[0]:
            raise ValueError(
                f"Expected input with {self.dims[0]} features, but got {inputs.shape[1]}"
            )

        # blast away previous inputs
        self.inputs = []
        self.inputs.append(inputs)

        # loop through the layers and calculate intermediate values
        for i in range(len(self.activations) - 1):
            f = F[self.activations[i]]
            self.inputs.append(f(self.inputs[i] @ self.weights[i] + self.biases[i]))

        # the final output is just the same thing again
        return F[self.activations[-1]](
            self.inputs[-1] @ self.weights[-1] + self.biases[-1]
        )

    def backward(self, eta: float, y_est: np.ndarray, y: np.ndarray) -> None:
        """
        Performs backpropagation to compute partial derivatives with respect to weights
        and biases, then updates with a step size eta.
        """
        if y.shape[0] != y_est.shape[0]:
            raise ValueError(
                f"Expected y with {y_est.shape[0]} samples, but got {y.shape[0]}"
            )

        d_prev = D_L[self.loss](y, y_est)

        # loop backwards
        for i in reversed(range(len(self.activations))):
            # Get a more convenient handle to the derivative of the activation function for this layer
            d_f = D_F[self.activations[i]]

            # call the input to this layer for this sample xi
            # This should be a matrix of features (batch_size x n)
            xi = self.inputs[i]

            # multiply d_prev by d_f(z)
            z = xi @ self.weights[i] + self.biases[i]
            d_prev = d_f(z) * d_prev

            # accumulate partials - bias is just previous d/dz times 1
            d_b = d_prev.sum(axis=0)
            d_w = xi.T @ d_prev

            # multiply by dz/df to go to the next layer
            d_prev = d_prev @ self.weights[i].T

            # update weights and biases by the average over samples
            self.weights[i] -= eta * d_w / y.shape[0]
            self.biases[i] -= eta * d_b / y.shape[0]

    def train(
        self, X: np.ndarray, y: np.ndarray, eta: float, epochs: int, batch_size: int
    ) -> np.ndarray:
        """
        Perform batch gradient descent to train model
        """

        batches_per_epoch = X.shape[0] // batch_size

        loss = np.zeros(epochs)
        for epoch in range(epochs):
            for _ in range(batches_per_epoch):
                batch = np.random.choice(range(0, X.shape[0]), batch_size)
                y_est_b = self.forward(X[batch, :])
                self.backward(eta, y_est_b, y[batch])

            y_est = self.forward(X)
            loss[epoch] = L[self.loss](y, y_est)

        return loss


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # basic example to make sure things are behaving
    xor_in = np.array(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ]
    )

    # good old XOR
    y = np.array([0, 1, 1, 0])
    y = y[:, np.newaxis]

    mlp = MLPRegressor(xor_in.shape[1])
    mlp.add_layer(2, "tanh")
    mlp.add_layer(1, "tanh")
    print(mlp)

    # simple test: is forward prop working?
    print("Known weights")
    mlp.weights[0] = np.array([[1, 1], [1, 1]])
    mlp.weights[1] = np.array([[-1], [1]])
    mlp.biases[0] = np.array([-3 / 2, -1 / 2])
    mlp.biases[1] = -1 / 2
    print(mlp.forward(xor_in) > 0)  # good enough, would work with threshold

    # Reset and train
    np.random.seed(12345)
    mlp = MLPRegressor(xor_in.shape[1])
    mlp.add_layer(2, "tanh")
    mlp.add_layer(1, "sigmoid")
    loss = mlp.train(xor_in, y, 0.2, 2000, 4)

    print(mlp.forward(xor_in) > 0.5)
    # take a look at the training curve
    plt.plot(loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (binary cross-entropy)")
    plt.show()
