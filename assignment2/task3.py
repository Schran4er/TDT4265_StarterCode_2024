import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer
import numpy as np


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50 
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False
    use_relu = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # BASELINE: identical to task 2 model
    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)
    train_history_baseline, val_history_baseline = trainer.train(num_epochs)
        

    # Example created for comparing with and without shuffling.
    # For comparison, show all loss/accuracy curves in the same plot
    # YOU CAN DELETE EVERYTHING BELOW!

    # TASK 3a: improved weights:
    use_improved_sigmoid = False
    use_improved_weight_init = True
    use_momentum = False
    use_relu = False

    model_improved_weights = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_improved_weights = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_improved_weights, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_improved_weights, val_history_improved_weights = trainer_improved_weights.train(num_epochs)


    # TASK 3b: improved sigmoid:
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = False
    use_relu = False

    model_improved_sigmoid = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_improved_sigmoid = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_improved_sigmoid, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_improved_sigmoid, val_history_improved_sigmoid = trainer_improved_sigmoid.train(num_epochs)


    # TASK 3c: momentum:
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    use_relu = False

    learning_rate = 0.02

    model_momentum = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_momentum = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_momentum, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_momentum, val_history_momentum = trainer_momentum.train(num_epochs)


    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    plt.ylim([0.0, 0.9])
    utils.plot_loss(train_history_baseline["loss"],"Task 2 Model (baseline)", npoints_to_average=10)
    utils.plot_loss(train_history_improved_weights["loss"],"Task 3a Model (improved weights)", npoints_to_average=10)
    utils.plot_loss(train_history_improved_sigmoid["loss"], "Task 3b Model (improved sigmoid)", npoints_to_average=10)
    utils.plot_loss(train_history_momentum["loss"], "Task 3c Model (momentum)", npoints_to_average=10)
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    
    plt.subplot(1, 2, 2)
    plt.ylim([0.8, 0.99])
    utils.plot_loss(val_history_baseline["accuracy"], "Task 2 Model (baseline)")
    utils.plot_loss(val_history_improved_weights["accuracy"], "Task 3a Model (improved weights)")
    utils.plot_loss(val_history_improved_sigmoid["accuracy"], "Task 3b Model (improved sigmoid)")
    utils.plot_loss(val_history_momentum["accuracy"], "Task 3c Model (momentum)")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3_train_loss.png")
    plt.show()



if __name__ == "__main__":
    main()
