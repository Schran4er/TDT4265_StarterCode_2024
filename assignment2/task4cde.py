import numpy as np
import matplotlib.pyplot as plt
import utils
from task2a import one_hot_encode, pre_process_images, SoftmaxModel, gradient_approximation_test
from task2 import SoftmaxTrainer


def main():
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785, \
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"


    # params for task 4c model
    # num_epochs = 50 
    num_epochs = 5 # task 4e
    learning_rate = 0.02
    batch_size = 32
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)


    # Task 4c: original model from task 3:
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    use_relu = False

    neurons_per_layer = [64, 10]
    
    model_64 = SoftmaxModel(
        neurons_per_layer, 
        use_improved_sigmoid, 
        use_improved_weight_init, 
        use_relu)

    trainer_64 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_64, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_64, val_history_64 = trainer_64.train(num_epochs)

    # Modify your network here
    # Task 4c: 2 hidden layers with same number of neurons
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    use_relu = False

    learning_rate = 0.02
    neurons_per_layer = [59, 59, 10]
    
    model_59 = SoftmaxModel(
        neurons_per_layer, 
        use_improved_sigmoid, 
        use_improved_weight_init, 
        use_relu)

    trainer_59 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_59, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_59, val_history_59 = trainer_59.train(num_epochs)

    # TASK 4e:
    neurons_per_layer = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10]
    model_10_64 = SoftmaxModel(
        neurons_per_layer, 
        use_improved_sigmoid, 
        use_improved_weight_init, 
        use_relu)

    trainer_10_64 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_10_64, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_10_64, val_history_10_64 = trainer_10_64.train(num_epochs)

    # model = SoftmaxModel(neurons_per_layer, use_improved_sigmoid, use_improved_weight_init, use_relu)
    # Gradient approximation check for 100 images
    # X_train = X_train[:100]
    # Y_train = Y_train[:100]
    # for layer_idx, w in enumerate(model.ws):
    #     model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    # gradient_approximation_test(model, X_train, Y_train)


    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    plt.ylim([0.0, 0.9])
    utils.plot_loss(train_history_64["loss"], "Task 3 Model (1 * 64 hidden)", npoints_to_average=10)
    utils.plot_loss(train_history_59["loss"],"Task 4 Model (2 * 59 hidden)", npoints_to_average=10)
    utils.plot_loss(train_history_10_64["loss"],"Task 4 Model (10 * 64 hidden)", npoints_to_average=10)
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    
    plt.subplot(1, 2, 2)
    plt.ylim([0.8, 0.99])
    utils.plot_loss(val_history_64["accuracy"], "Task 3 Model (1 * 64 hidden)")
    utils.plot_loss(val_history_59["accuracy"], "Task 4 Model (2 * 59 hidden)")
    utils.plot_loss(val_history_10_64["accuracy"], "Task 4e Model (10 * 64 hidden)")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task4e_64_59_10-64neurons_5epochs.png")
    plt.show()




if __name__ == "__main__":
    main()
