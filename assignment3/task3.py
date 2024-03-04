# task3.py
import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn, optim
from dataloaders import load_cifar10
from trainer import Trainer
from enum import Enum


class ConvolutionType(Enum):
    POOLING = 0,
    STRIDED_CONVOLUTIONS = 1


class ArchitectureType(Enum):
    BASE_ARCHITECTURE = 0,
    ARCHITECTURE_V1 = 1
    ARCHITECTURE_V2 = 2
    ARCHITECTURE_V3 = 3
    FINAL = 4


def create_plots_multiple(trainers: list[Trainer], plot_names: list[str], name):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.ylim([0.0, 5.0])
    plt.title("Cross Entropy Loss")
    for trainer, plot_name in zip(trainers, plot_names):
        utils.plot_loss(
            trainer.train_history["loss"], label=f"{plot_name}: Training loss", npoints_to_average=10
        )
        utils.plot_loss(trainer.validation_history["loss"], label=f"{plot_name} Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.ylim([0.0, 0.99])
    plt.title("Accuracy")
    for trainer, plot_name in zip(trainers, plot_names):
        utils.plot_loss(trainer.validation_history["accuracy"], label=f"{plot_name} Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()



class ExampleModel(nn.Module):
    def __init__(self, image_channels,
                 num_classes,
                 filter_size,
                 number_of_filters,
                 convolution_type: ConvolutionType,
                 batch_normalization: bool,
                 architecture_type: ArchitectureType):
        """
        Is called when model is initialized.
        Args:
            image_channels. Number of color channels in image (3)
            num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        num_filters = number_of_filters  # Set number of filters in first conv layer
        self.num_classes = num_classes
        self.filter_size = filter_size
        if filter_size == 3:
            padding = 1
        else:
            padding = 2

        if convolution_type == ConvolutionType.POOLING:
            stride = 1 
            pooling_stride = 2
        elif convolution_type == ConvolutionType.STRIDED_CONVOLUTIONS:
            stride = 1
            pooling_stride = 4
            padding = 1

        if batch_normalization:
            if architecture_type == ArchitectureType.BASE_ARCHITECTURE:
                self.feature_extractor = nn.Sequential(
                    # layer 1:
                    nn.Conv2d(
                        in_channels=image_channels,
                        out_channels=num_filters,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.BatchNorm2d(num_filters),
                    nn.ReLU(),
                    nn.MaxPool2d(
                        stride=pooling_stride,
                        kernel_size=2
                    ),

                    # layer 2:
                    nn.Conv2d(
                        in_channels=num_filters,
                        out_channels=num_filters*2,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.BatchNorm2d(num_filters*2),
                    nn.ReLU(),
                    nn.MaxPool2d(
                        stride=pooling_stride,
                        kernel_size=2
                    ),

                    # layer 3:
                    nn.Conv2d(
                        in_channels=num_filters*2,
                        out_channels=num_filters*4,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.BatchNorm2d(num_filters*4),
                    nn.ReLU(),
                    nn.MaxPool2d(
                        stride=pooling_stride,
                        kernel_size=2
                    ),

                    # flatten:
                    nn.Flatten()
                )
            
            elif architecture_type == ArchitectureType.ARCHITECTURE_V1:
                self.feature_extractor = nn.Sequential(
                    # layer 1:
                    nn.Conv2d(
                        in_channels=image_channels,
                        out_channels=num_filters,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(
                        stride=pooling_stride,
                        kernel_size=2
                    ),

                    # layer 2:
                    nn.Conv2d(
                        in_channels=num_filters,
                        out_channels=num_filters*2,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(
                        stride=pooling_stride,
                        kernel_size=2
                    ),

                    # layer 3:
                    nn.Conv2d(
                        in_channels=num_filters*2,
                        out_channels=num_filters*4,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(
                        stride=pooling_stride,
                        kernel_size=2
                    ),


                    # layer 4:
                    nn.Conv2d(
                        in_channels=num_filters*4,
                        out_channels=num_filters*8,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(
                        stride=pooling_stride,
                        kernel_size=2
                    )
                )
            
            elif architecture_type == ArchitectureType.ARCHITECTURE_V2:
                self.feature_extractor = nn.Sequential(
                    # layer 1:
                    nn.Conv2d(
                        in_channels=image_channels,
                        out_channels=num_filters,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=num_filters,
                        out_channels=num_filters,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(
                        stride=pooling_stride,
                        kernel_size=2
                    ),

                    # layer 2:
                    nn.Conv2d(
                        in_channels=num_filters,
                        out_channels=num_filters*2,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=num_filters*2,
                        out_channels=num_filters*2,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(
                        stride=pooling_stride,
                        kernel_size=2
                    ),

                    # layer 3:
                    nn.Conv2d(
                        in_channels=num_filters*2,
                        out_channels=num_filters*4,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=num_filters*4,
                        out_channels=num_filters*4,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(
                        stride=pooling_stride,
                        kernel_size=2
                    ),


                    # layer 4:
                    nn.Conv2d(
                        in_channels=num_filters*4,
                        out_channels=num_filters*8,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=num_filters*8,
                        out_channels=num_filters*8,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(
                        stride=pooling_stride,
                        kernel_size=2
                    )
                )
            
            elif architecture_type == ArchitectureType.ARCHITECTURE_V3:
                self.feature_extractor = nn.Sequential(
                    # layer 1:
                    nn.Conv2d(
                        in_channels=image_channels,
                        out_channels=num_filters,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.BatchNorm2d(num_filters),
                    nn.ReLU(),

                    # layer 2:
                    nn.Conv2d(
                        in_channels=num_filters,
                        out_channels=num_filters*2,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.BatchNorm2d(num_filters*2),
                    nn.ReLU(),

                    # layer 3:
                    nn.Conv2d(
                        in_channels=num_filters*2,
                        out_channels=num_filters*4,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_filters*4),


                    # layer 4:
                    nn.Conv2d(
                        in_channels=num_filters*4,
                        out_channels=num_filters*8,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_filters*8)
                )

            elif architecture_type == ArchitectureType.FINAL:
                self.feature_extractor = nn.Sequential(
                    # layer 1:
                    nn.Conv2d(
                        in_channels=image_channels,
                        out_channels=num_filters,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.LeakyReLU(),
                    nn.Conv2d(
                        in_channels=num_filters,
                        out_channels=num_filters,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.LeakyReLU(),
                    nn.MaxPool2d(
                        stride=pooling_stride,
                        kernel_size=2
                    ),
                    nn.BatchNorm2d(num_filters),

                    # layer 2:
                    nn.Conv2d(
                        in_channels=num_filters,
                        out_channels=num_filters*2,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.LeakyReLU(),
                    nn.Conv2d(
                        in_channels=num_filters*2,
                        out_channels=num_filters*2,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.LeakyReLU(),
                    nn.MaxPool2d(
                        stride=pooling_stride,
                        kernel_size=2
                    ),
                    nn.BatchNorm2d(num_filters*2),


                    # layer 3:
                    nn.Conv2d(
                        in_channels=num_filters*2,
                        out_channels=num_filters*4,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.LeakyReLU(),
                    nn.Conv2d(
                        in_channels=num_filters*4,
                        out_channels=num_filters*4,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.LeakyReLU(),
                    nn.MaxPool2d(
                        stride=pooling_stride,
                        kernel_size=2
                    ),
                    nn.BatchNorm2d(num_filters*4),



                    # layer 4:
                    nn.Conv2d(
                        in_channels=num_filters*4,
                        out_channels=num_filters*8,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.LeakyReLU(),
                    nn.Conv2d(
                        in_channels=num_filters*8,
                        out_channels=num_filters*8,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.LeakyReLU(),
                    nn.MaxPool2d(
                        stride=pooling_stride,
                        kernel_size=2
                    ),
                    nn.BatchNorm2d(num_filters*8)
                )
        else:
            if architecture_type == ArchitectureType.BASE_ARCHITECTURE:
                self.feature_extractor = nn.Sequential(
                    # layer 1:
                    nn.Conv2d(
                        in_channels=image_channels,
                        out_channels=num_filters,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(
                        stride=pooling_stride,
                        kernel_size=2
                    ),

                    # layer 2:
                    nn.Conv2d(
                        in_channels=num_filters,
                        out_channels=num_filters*2,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(
                        stride=pooling_stride,
                        kernel_size=2
                    ),

                    # layer 3:
                    nn.Conv2d(
                        in_channels=num_filters*2,
                        out_channels=num_filters*4,
                        kernel_size=filter_size,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(
                        stride=pooling_stride,
                        kernel_size=2
                    ),

                    # flatten:
                    nn.Flatten()
                )   

        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]

        # self.num_output_features = 32 * 32 * 32
        # following the error messages we get and taks 1 g) this should be equal to: height * width * feature_maps
        if convolution_type == ConvolutionType.POOLING:
            if architecture_type == ArchitectureType.BASE_ARCHITECTURE:
                self.num_output_features = 4 * 4 * num_filters*4
            elif architecture_type == ArchitectureType.ARCHITECTURE_V1 or \
                architecture_type == ArchitectureType.ARCHITECTURE_V2 or \
                architecture_type == ArchitectureType.FINAL:
                self.num_output_features = 4 * 4 * num_filters * 2
            elif architecture_type == ArchitectureType.ARCHITECTURE_V3:
                self.num_output_features = 4 * 4 * num_filters * 2 * 256

        elif convolution_type == ConvolutionType.STRIDED_CONVOLUTIONS:
            self.num_output_features = num_filters*4
        
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss

        # layers 4 + 5:
        if batch_normalization:
            if architecture_type == ArchitectureType.BASE_ARCHITECTURE:
                self.classifier = nn.Sequential(
                    nn.Linear(self.num_output_features, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Linear(64, num_classes),
                )      

            elif architecture_type == ArchitectureType.ARCHITECTURE_V1 \
                or architecture_type == ArchitectureType.ARCHITECTURE_V2 \
                or architecture_type == ArchitectureType.ARCHITECTURE_V3 \
                or architecture_type == ArchitectureType.FINAL:

                self.classifier = nn.Sequential(
                    nn.Linear(self.num_output_features, num_classes)
                )                  

        else:
            if architecture_type == ArchitectureType.BASE_ARCHITECTURE:
                self.classifier = nn.Sequential(
                    nn.Linear(self.num_output_features, 64),
                    nn.ReLU(),
                    nn.Linear(64, num_classes),
                )


    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        x = self.feature_extractor(x) # 64, 128, 2, 2

        if self.filter_size == 3:
            x = x.view(-1, self.num_output_features) # 64, 512
        else:
            x = x.view(batch_size, -1)


        x = self.classifier(x) # my 64,2048 expected: 64, 10
        out = x
        expected_shape = (batch_size, self.num_classes) # expected: out: (64,10) got (400,10)
        assert out.shape == (
            batch_size,
            self.num_classes,
        ), f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result!
    utils.set_seed(0)
    print(f"Using device: {utils.get_device()}")
    epochs = 10

    all_trainers = []
    all_plot_names = []
    train_model = [True, True, True, True, True, True, True, True] 

    # TASK 3: Baseline (Model from task 2)
    if train_model[0]:
        DATA_AUGMENTATION = False
        batch_size = 64
        learning_rate = 5e-2
        early_stop_count = 4
        dataloaders = load_cifar10(batch_size, data_augmentation=DATA_AUGMENTATION)
        model = ExampleModel(image_channels=3, 
                             num_classes=10, 
                             filter_size=5, 
                             number_of_filters=32, 
                             convolution_type=ConvolutionType.POOLING,
                             batch_normalization=False,
                             architecture_type=ArchitectureType.BASE_ARCHITECTURE)
        
        trainer_baseline = Trainer(
            batch_size, learning_rate, early_stop_count, epochs, model, dataloaders
        )
        trainer_baseline.train()

        plot_name_baseline = "task3 baseline"
        all_trainers.append(trainer_baseline)
        all_plot_names.append(plot_name_baseline)

    # TASK 3i: 
    # Enable Data Augmentation
    if train_model[1]:
        DATA_AUGMENTATION = True
        batch_size = 64
        learning_rate = 5e-2
        early_stop_count = 4
        dataloaders = load_cifar10(batch_size, data_augmentation=DATA_AUGMENTATION)
        model = ExampleModel(image_channels=3, 
                             num_classes=10, 
                             filter_size=5, 
                             number_of_filters=32,
                             convolution_type=ConvolutionType.POOLING,
                             batch_normalization=False,
                             architecture_type=ArchitectureType.BASE_ARCHITECTURE)
        
        trainer_3_i = Trainer(
            batch_size, learning_rate, early_stop_count, epochs, model, dataloaders
        )
        trainer_3_i.train()

        plot_name_3_i = "3 i"
        all_trainers.append(trainer_3_i)
        all_plot_names.append(plot_name_3_i)


    # TASK 3 ii: 
    # Enable Data Augmentation
    # Smaller filter size
    if train_model[2]:
        DATA_AUGMENTATION = True
        batch_size = 64
        learning_rate = 5e-2
        early_stop_count = 4
        dataloaders = load_cifar10(batch_size, data_augmentation=DATA_AUGMENTATION)
        model = ExampleModel(image_channels=3, 
                             num_classes=10, 
                             filter_size=3, 
                             number_of_filters=32,
                             convolution_type=ConvolutionType.POOLING,
                             batch_normalization=False,
                             architecture_type=ArchitectureType.BASE_ARCHITECTURE)
        
        trainer_3_ii = Trainer(
            batch_size, learning_rate, early_stop_count, epochs, model, dataloaders
        )
        trainer_3_ii.train()

        plot_name_3_ii = "3 ii"
        all_trainers.append(trainer_3_ii)
        all_plot_names.append(plot_name_3_ii)

    # test showed that a smaller filter_size of 3 performed better than the initial filter_size of 5, so the following
    # models are run with a filter size of 3


    # TASK 3 iii:
    # Try with more filters
    if train_model[3]:
        DATA_AUGMENTATION = True
        batch_size = 64
        learning_rate = 5e-2
        early_stop_count = 4
        dataloaders = load_cifar10(batch_size, data_augmentation=DATA_AUGMENTATION)
        model = ExampleModel(image_channels=3, 
                             num_classes=10, 
                             filter_size=3, 
                             number_of_filters=64,
                             convolution_type=ConvolutionType.POOLING,
                             batch_normalization=False,
                             architecture_type=ArchitectureType.BASE_ARCHITECTURE)
        
        trainer_3_iii = Trainer(
            batch_size, learning_rate, early_stop_count, epochs, model, dataloaders
        )
        trainer_3_iii.train()

        plot_name_3_iii = "3 iii"
        all_trainers.append(trainer_3_iii)
        all_plot_names.append(plot_name_3_iii)

    # test showed that doubling the number of filters from 32 to 64 improved the performance
    
    
    # TASK 3 iv:
    # strided convolutions
    if train_model[4]:
        DATA_AUGMENTATION = True
        batch_size = 64
        learning_rate = 5e-2
        early_stop_count = 4
        dataloaders = load_cifar10(batch_size, data_augmentation=DATA_AUGMENTATION)
        model = ExampleModel(image_channels=3, 
                             num_classes=10, 
                             filter_size=3, 
                             number_of_filters=64,
                             convolution_type=ConvolutionType.STRIDED_CONVOLUTIONS,
                             batch_normalization=False,
                             architecture_type=ArchitectureType.BASE_ARCHITECTURE)
        
        trainer_3_iv = Trainer(
            batch_size, learning_rate, early_stop_count, epochs, model, dataloaders
        )
        trainer_3_iv.train()

        plot_name_3_iv = "3 iv"
        all_trainers.append(trainer_3_iv)
        all_plot_names.append(plot_name_3_iv)

    # test showed that strided convolutions performed worse


    # TASK 3 v:
    # introduce batch normalization
    if train_model[5]:
        DATA_AUGMENTATION = True
        batch_size = 64
        learning_rate = 5e-2
        early_stop_count = 4
        dataloaders = load_cifar10(batch_size, data_augmentation=DATA_AUGMENTATION)
        model = ExampleModel(image_channels=3, 
                             num_classes=10, 
                             filter_size=3, 
                             number_of_filters=64,
                             convolution_type=ConvolutionType.POOLING,
                             batch_normalization=True,
                             architecture_type=ArchitectureType.BASE_ARCHITECTURE)
        
        trainer_3_v = Trainer(
            batch_size, learning_rate, early_stop_count, epochs, model, dataloaders
        )
        trainer_3_v.train()

        plot_name_3_v = "3 v"
        all_trainers.append(trainer_3_v)
        all_plot_names.append(plot_name_3_v)

    # the model learns indeed faster with batch normalization
        
    if train_model[6]:
        train_architectures = [True, True, True]

        if train_architectures[0]:
            DATA_AUGMENTATION = True
            batch_size = 64
            learning_rate = 5e-2
            early_stop_count = 4
            dataloaders = load_cifar10(batch_size, data_augmentation=DATA_AUGMENTATION)
            model = ExampleModel(image_channels=3, 
                                num_classes=10, 
                                filter_size=3, 
                                number_of_filters=64,
                                convolution_type=ConvolutionType.POOLING,
                                batch_normalization=True,
                                architecture_type=ArchitectureType.ARCHITECTURE_V1)
            
            trainer_3_vi_a = Trainer(
                batch_size, learning_rate, early_stop_count, epochs, model, dataloaders
            )
            trainer_3_vi_a.train()

            plot_name_3_vi_a = "3 vi a"
            all_trainers.append(trainer_3_vi_a)
            all_plot_names.append(plot_name_3_vi_a)

        if train_architectures[1]:
            DATA_AUGMENTATION = True
            batch_size = 64
            learning_rate = 5e-2
            early_stop_count = 4
            dataloaders = load_cifar10(batch_size, data_augmentation=DATA_AUGMENTATION)
            model = ExampleModel(image_channels=3, 
                                num_classes=10, 
                                filter_size=3, 
                                number_of_filters=64,
                                convolution_type=ConvolutionType.POOLING,
                                batch_normalization=True,
                                architecture_type=ArchitectureType.ARCHITECTURE_V2)
            
            trainer_3_vi_b = Trainer(
                batch_size, learning_rate, early_stop_count, epochs, model, dataloaders
            )
            trainer_3_vi_b.train()

            plot_name_3_vi_b = "3 vi b"
            all_trainers.append(trainer_3_vi_b)
            all_plot_names.append(plot_name_3_vi_b) 

        if train_architectures[2]:
            DATA_AUGMENTATION = True
            batch_size = 64
            learning_rate = 5e-2
            early_stop_count = 4
            dataloaders = load_cifar10(batch_size, data_augmentation=DATA_AUGMENTATION)
            model = ExampleModel(image_channels=3, 
                                num_classes=10, 
                                filter_size=3, 
                                number_of_filters=64,
                                convolution_type=ConvolutionType.POOLING,
                                batch_normalization=True,
                                architecture_type=ArchitectureType.ARCHITECTURE_V3)
            
            trainer_3_vi_c = Trainer(
                batch_size, learning_rate, early_stop_count, epochs, model, dataloaders
            )
            trainer_3_vi_c.train()

            plot_name_3_vi_c = "3 vi c"
            all_trainers.append(trainer_3_vi_c)
            all_plot_names.append(plot_name_3_vi_c) 
    
    # model b has the best accuracy curve

    # to avoid an excessive amount of models to train and compare, the next recommendations described under 
    # "opimizers" and "activation functions" were added at once
    # on top of that, to make the model learn faster, batch normalization was added
    
    # nn.LogSimoid() produced an unhealthy loss curve with spikes, to the next tried activation function is LeakyReLu() since it is more similar to the 
    # originally used ReLu
    if train_model[7]:
        DATA_AUGMENTATION = True
        batch_size = 64
        learning_rate = 0.001
        early_stop_count = 4
        dataloaders = load_cifar10(batch_size, data_augmentation=DATA_AUGMENTATION)
        model = ExampleModel(image_channels=3, 
                            num_classes=10, 
                            filter_size=3, 
                            number_of_filters=64,
                            convolution_type=ConvolutionType.POOLING,
                            batch_normalization=True,
                            architecture_type=ArchitectureType.FINAL)
        
        trainer_3_vii = Trainer(
            batch_size, learning_rate, early_stop_count, epochs, model, dataloaders
        )
        trainer_3_vii.optimizer = optim.Adam(model.parameters(), learning_rate)

        trainer_3_vii.train()

        plot_name_3_vii = "3 vii"
        all_trainers.append(trainer_3_vii)
        all_plot_names.append(plot_name_3_vii) 
    



    plot_name = "task3"
    create_plots_multiple(all_trainers, all_plot_names, plot_name)



if __name__ == "__main__":
    main()
