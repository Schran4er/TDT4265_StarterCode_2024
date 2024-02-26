import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    outputs = model.forward(X)
    outputs = np.argmax(outputs, axis=1)
    targets = np.argmax(targets, axis=1)
    accuracy = np.mean(outputs == targets)
    return accuracy


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        # TODO: Implement this function (task 3b)
        outputs = self.model.forward(X_batch)   
        self.model.backward(X_batch, outputs, Y_batch)   
        self.model.w -= self.learning_rate * self.model.grad
        loss = cross_entropy_loss(Y_batch, outputs, self.model.use_L2_reg, self.model.w, self.model.l2_reg_lambda)
        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
            accuracy_train (float): Accuracy on train dataset
            accuracy_val (float): Accuracy on the validation dataset
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(self.Y_val, logits)

        accuracy_train = calculate_accuracy(
            self.X_train, self.Y_train, self.model)
        accuracy_val = calculate_accuracy(
            self.X_val, self.Y_val, self.model)
        return loss, accuracy_train, accuracy_val


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # Intialize model
    model = SoftmaxModel(l2_reg_lambda)     # without regularization
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)


    # Train a model with L2 regularization (task 4b)
    model1 = SoftmaxModel(l2_reg_lambda=1, use_L2_reg=True)
    trainer = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg01, val_history_reg01 = trainer.train(num_epochs)


    # You can finish the rest of task 4 below this point.
    # Plotting of softmax weights (Task 4b)                             
    fig, ax = plt.subplots(ncols=10, nrows=2, figsize=(10, 4))
    fig.subplots_adjust(hspace=-.6)
    for i, w in enumerate([model.w, model1.w]):
        for j in range(10):
            weight = w[:-1, j]  # get the weights corresponding to the j-th class, exclude bias!
            ax[i, j].imshow(weight.reshape(28, 28), cmap='gray')
            ax[i, j].axis('off')
    fig.tight_layout()
    fig.savefig("task4b_softmax_weight.png")
    # plt.imsave("task4b_softmax_weight.png", weight, cmap="gray")


    # Plotting of accuracy for difference values of lambdas (task 4c)       
    l2_lambdas = [1, .1, .01, .001]
    models = []
    accuracies = []
    for l2_lambda in l2_lambdas:
        model_ = SoftmaxModel(l2_reg_lambda=l2_lambda, use_L2_reg=True)
        trainer = SoftmaxTrainer(
            model_, learning_rate, batch_size, shuffle_dataset,
            X_train, Y_train, X_val, Y_val,
        )
        _, val_hist = trainer.train(num_epochs)
        models.append(model_)
        accuracies.append(val_hist["accuracy"])
    
    # # Plot accuracy
    plt.figure()
    # plt.ylim([0.89, .93])
    for (l2_lambda, acc) in zip(l2_lambdas, accuracies): 
        utils.plot_loss(acc, f"lambda={l2_lambda}")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("task4c_l2_reg_accuracy.png")
    plt.show()


    # Task 4d - Plotting of the l2 norm for each weight     
    L2_norm_values = [ np.sum(mod.w**2) for mod in models]
    plt.figure()
    plt.plot(l2_lambdas, L2_norm_values, "-x")
    plt.xlabel("lambda (hyperparameter)")
    plt.ylabel("L2 norm of weights")
    plt.savefig("task4d_l2_reg_norms.png")
    plt.show()


if __name__ == "__main__":
    main()
