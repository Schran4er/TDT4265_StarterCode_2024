import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50 
    learning_rate = .1
    batch_size = 32
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)


    # FINAL MODEL TASK3:
    neurons_per_layer = [64, 10]
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    use_relu = False

    learning_rate = 0.02

    model_3 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_3 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_3, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_3, val_history_3 = trainer_3.train(num_epochs)


    # TASK 4a: 32 neurons
    neurons_per_layer = [32, 10]
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    use_relu = False

    learning_rate = 0.02

    model_4_32neurons = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_4_32neurons = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_4_32neurons, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_4_32neurons, val_history_4_32neurons = trainer_4_32neurons.train(num_epochs)


    # TASK 4b: 128 neurons:
    neurons_per_layer = [128, 10]
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    use_relu = False

    learning_rate = 0.02
    model_4_128neurons = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_4_128neurons = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_4_128neurons, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_4_128neurons, val_history_4_128neurons = trainer_4_128neurons.train(num_epochs)



    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    plt.ylim([0.0, 0.9])
    utils.plot_loss(train_history_3["loss"],"Task 3 Model (baseline)", npoints_to_average=10)
    utils.plot_loss(train_history_4_32neurons["loss"], "Task 4a 32 Neurons", npoints_to_average=10)
    utils.plot_loss(train_history_4_128neurons["loss"], "Task 4b 128 Neurons", npoints_to_average=10)
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    
    plt.subplot(1, 2, 2)
    plt.ylim([0.8, 0.99])
    utils.plot_loss(val_history_3["accuracy"], "Task 3 Model (baseline)")
    utils.plot_loss(val_history_4_32neurons["accuracy"], "Task 4a 32 Neurons)")
    utils.plot_loss(val_history_4_128neurons["accuracy"], "Task 4b 128 Neurons)")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task4_ab_32_128_neurons.png")
    plt.show()



if __name__ == '__main__':
    main()