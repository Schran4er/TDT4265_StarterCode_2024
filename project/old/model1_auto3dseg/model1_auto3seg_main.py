from monai.bundle.config_parser import ConfigParser
from monai.apps.auto3dseg.auto_runner import AutoRunner
from monai.apps.auto3dseg.bundle_gen import BundleGen
import matplotlib.pyplot as plt
import numpy as np
import nrrd
import nibabel as nib
import os



def train():
    # Create input configuration .yaml file.
    input_yaml = "/cluster/work/felixzr/TDT4265_StarterCode_2024/project/model1_auto3dseg/model1_auto3dseg_input.yaml"

    input_config = {
        "name": "model1_auto3dseg", # optional, it is only for your own record
        "task": "segmentation",  # optional, it is only for your own record
        "modality": "CT",  # required
        "datalist": "/cluster/work/felixzr/TDT4265_StarterCode_2024/project/model1_auto3dseg/data.json",  # required
        "dataroot": "/cluster/work/felixzr/TDT4265_StarterCode_2024/data_flat",  # required
    }

    ConfigParser.export_config_file(input_config, input_yaml)

    # TODO: set custom workdir when using custom ensemble runner
    #Set custom training parameters
    max_epochs = 25
    train_param = {
        "num_epochs_per_validation": 1,
        "num_images_per_batch": 2,
        "num_epochs": max_epochs,
        "num_workers": 2,
        "max_workers": 2,
        "num_warmup_epochs": 1,
        "num_images_per_batch": 2,
    }

    work_dir = "/cluster/work/felixzr/TDT4265_StarterCode_2024/project/model1_auto3dseg"
    runner = AutoRunner(work_dir=work_dir, input=input_yaml)
    runner.set_training_params(params=train_param)

    # Set model ensemble method
    # runner.set_ensemble_method(ensemble_method_name="AlgoEnsembleBestByFold")

    # Set custom inference parameters
    pred_params = {
        "mode": "vote",  # use majority vote instead of mean to ensemble the predictions
        "sigmoid": True,  # when to use sigmoid to binarize the prediction and output the label
    }

    runner.set_prediction_params(params=pred_params)
    runner.run()


def main():
    train()


if __name__ == '__main__':
    main()