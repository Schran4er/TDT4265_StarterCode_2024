import numpy as np
import utils
import typing

np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784, f"X.shape[1]: {X.shape[1]}, should be 784"

    mean = X.mean()
    std = X.std()
    print(f"mean: {mean}, std: {std}")

    # X_train, _, X_val, _ = utils.load_full_mnist()
    # X_complete = np.concatenate((X_train, X_val), axis=0)
    # mean = X_complete.mean()
    # std = X_complete.std()
    # print(f"total mean: {mean}, total std: {std}")

    X_normalized = (X - mean) / std

    X_normalized_bias = np.insert(X_normalized, X_normalized.shape[1], values=1, axis=1) 
    # todo: is it correct to add the "bias 1" after the normalization or should it be added before it?
    ## --> Macht keinen Unterschied (solange der bias nicht auf 0 normalisiert wird) würde ich sagen, da das dann dynamisch durch die Gewichte ausgeglichen wird?

    return X_normalized_bias


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray, use_L2_reg=False, w=None, l2_reg_lambda=None):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"

    # C_n_w = -1*(np.sum(targets * np.log(outputs)))
    C_n_w = -1*(np.sum(targets * np.log(outputs + 1e-12)))      # add small number to avoid log(0) = -inf
    if use_L2_reg: C_n_w += l2_reg_lambda * np.sum(w**2)
    loss = C_n_w / len(targets)
    return loss


class SoftmaxModel:

    def __init__(
        self,
        # Number of neurons per layer
        neurons_per_layer: typing.List[int],
        use_improved_sigmoid: bool,  # Task 3b hyperparameter
        use_improved_weight_init: bool,  # Task 3a hyperparameter
        use_relu: bool,  # Task 3c hyperparameter
    ):
        np.random.seed(
            1
        )  # Always reset random seed before weight init to get comparable results.
        # Define number of input nodes
        self.I = 785    # None          # todo? was originally None, changed to 785
        self.use_improved_sigmoid = use_improved_sigmoid
        self.use_relu = use_relu
        self.use_improved_weight_init = use_improved_weight_init

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            w = np.zeros(w_shape)
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        # HINT: For performing the backward pass, you can save intermediate activations in variables in the forward pass.
        # such as self.hidden_layer_output = ...

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        def sigmoid_derivative(x):
            return sigmoid(x) * (1 - sigmoid(x))
        def softmax(Z):
            Z = Z - np.max(Z)       # Softmax numerical stability!
            return np.exp(Z) / np.sum(np.exp(Z))    # for input vector Z

        # input -> hidden layer
        Z_J = X.dot(self.ws[0])
        A_J = sigmoid(Z_J)
        self.hidden_layer_output_A = A_J
        self.hidden_layer_output_sigmoid_derivative = sigmoid_derivative(Z_J)

        # hidden -> output layer
        Z_K = A_J.dot(self.ws[1])
        Y_hat = softmax(Z_K)
    
        return Y_hat

    def backward(self, X: np.ndarray, outputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)
        assert (
            targets.shape == outputs.shape
        ), f"Output shape: {outputs.shape}, targets: {targets.shape}"

        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        # self.grads = [None, None]
        self.zero_grad()

        # # hidden -> output layer
        # grad_temp = np.zeros(self.ws[1].shape).T
        # # evaluate over all samples
        # for i in range(X.shape[0]):
        #     err_2 = -(outputs[i] - targets[i]).reshape(-1, 1)
        #     grad_temp += np.matmul(err_2, self.hidden_layer_output_A[i].T.reshape(1, -1))
        # # transpose and divide by number of samples
        # self.grads[1] = grad_temp.T / X.shape[0]
        err_2 = -(outputs - targets)
        self.grads[1] = np.dot(self.hidden_layer_output_A.T, err_2) / X.shape[0]

        # # input -> hidden layer
        # grad_temp = np.zeros(self.ws[0].shape).T
        # # evaluate over all samples
        # for i in range(X.shape[0]):
        #     err_2 = -(outputs[i] - targets[i]).reshape(-1, 1)
        #     err_1 = np.multiply(self.hidden_layer_output_sigmoid_derivative[i].reshape(-1, 1), np.matmul(self.ws[1], err_2))
        #     grad_temp += np.matmul(err_1, X[i].reshape(1, -1))
        # # transpose and divide by number of samples
        # self.grads[0] = grad_temp.T / X.shape[0]
        err_1 = np.multiply(self.hidden_layer_output_sigmoid_derivative, np.dot(err_2, self.ws[1].T))
        self.grads[0] = np.dot(X.T, err_1) / X.shape[0]

        for grad, w in zip(self.grads, self.ws):
            assert (
                grad.shape == w.shape
            ), f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """   
    one_hot_encoded = np.zeros((len(Y), num_classes))
    one_hot_encoded[np.arange(Y.size), Y.T] = 1
    return one_hot_encoded


def gradient_approximation_test(model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
    Numerical approximation for gradients. Should not be edited.
    Details about this test is given in the appendix in the assignment.
    """

    assert isinstance(X, np.ndarray) and isinstance(
        Y, np.ndarray
    ), f"X and Y should be of type np.ndarray!, got {type(X), type(Y)}"

    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**1, (
                    f"Calculated gradient is incorrect. "
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n"
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n"
                    f"If this test fails there could be errors in your cross entropy loss function, "
                    f"forward function or backward function"
                )


def main():
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert (
        Y[0, 3] == 1 and Y.sum() == 1
    ), f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert (
        X_train.shape[1] == 785
    ), f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_relu = True
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init, use_relu
    )

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)


if __name__ == "__main__":
    main()
