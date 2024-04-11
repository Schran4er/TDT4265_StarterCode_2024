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
    input_yaml = "/cluster/work/felixzr/TDT4265_StarterCode_2024/pytorch-lightning-template/nnUnet_kitware_tutorial/input_auto3dseg_v2.yaml"

    input_config = {
        "name": "AortaSeg", # optional, it is only for your own record
        "task": "segmentation",  # optional, it is only for your own record
        "modality": "CT",  # required
        "datalist": "/cluster/work/felixzr/TDT4265_StarterCode_2024/pytorch-lightning-template/nnUnet_kitware_tutorial/data_kitnet/data.json",  # required
        "dataroot": "/cluster/work/felixzr/TDT4265_StarterCode_2024/pytorch-lightning-template/nnUnet_kitware_tutorial/data_kitnet/",  # required
        "output_dir": "/cluster/work/felixzr/TDT4265_StarterCode_2024/pytorch-lightning-template/nnUnet_kitware_tutorial/output"
    }

    ConfigParser.export_config_file(input_config,input_yaml)


    #Set custom training parameters
    max_epochs = 1
    train_param = {
        "num_epochs_per_validation": 1,
        "num_images_per_batch": 2,
        "num_epochs": max_epochs,
        "num_workers":2,
        "max_workers":2,
        "num_warmup_epochs": 1,
        "num_images_per_batch": 2,
        # "patch_size_valid":[48,64,64]
    }

    runner = AutoRunner(input=input_yaml)
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


def plot_by_z_slice_idx(z, lbl, pred, name):

    img_slice = lbl[:, :, z] == 0
    label_slice = lbl[:, :, z] == 1
    background_slice = pred[:, :, z, 0] if pred.ndim == 4 else pred[:, :, z] == 0
    foreground_slice = pred[:, :, z, 1] if pred.ndim == 4 else pred[:, :, z] == 1

    plt.subplot(2, 2, 1)
    plt.imshow(img_slice)
    plt.title("background groundtruth")
    cbar = plt.colorbar(shrink=0.8)
    plt.subplot(2, 2, 2)
    plt.imshow(label_slice)
    plt.title("foreground_groundtruth")
    cbar = plt.colorbar(shrink=0.8)
    plt.subplot(2, 2, 3)
    plt.imshow(background_slice)
    plt.title("background prediction")
    cbar = plt.colorbar(shrink=0.8)
    plt.subplot(2, 2, 4)
    plt.imshow(foreground_slice)
    plt.title("foreground prediction")
    cbar = plt.colorbar(shrink=0.8)
    # set the spacing between subplots
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
    path = f"/cluster/work/felixzr/TDT4265_StarterCode_2024/pytorch-lightning-template/nnUnet_kitware_tutorial/plots_nnUnet_kitware_tutorial/plot_{name}.png"
    plt.savefig(path)


def inference():
    # img_nib = nib.load(os.path.join(dataroot_dir, sim_datalist["testing"][0]["image"]))
    # lbl_nib = nib.load(os.path.join(dataroot_dir, sim_datalist["testing"][0]["label"]))
    # img = np.array(img_nib.dataobj)
    # lbl = np.array(lbl_nib.dataobj)


    # nrrd_data, nrrd_header = nrrd.read(os.path.join(dataroot_dir, sim_datalist["testing"][0]["image"]))
    # img = np.array(nrrd_data)
    # nrrd_data, nrrd_header = nrrd.read(os.path.join(dataroot_dir, sim_datalist["testing"][0]["label"]))

    truth_path = "/cluster/work/felixzr/TDT4265_StarterCode_2024/pytorch-lightning-template/nnUnet_kitware_tutorial/data_kitnet/Annotations/Normal_19.nrrd"
    nrrd_data, nrrd_header = nrrd.read(truth_path)
    lbl = np.array(nrrd_data)

    prediction_path = "/cluster/work/felixzr/work_dir/ensemble_output/CTCTA/Normal_19_ensemble.nii.gz"
    # image_name = sim_datalist["testing"][0]["image"].split(".")[0]
    # prediction_nib = nib.load(os.path.join(work_dir, "ensemble_output", image_name + "_ensemble" + ".nii.gz"))
    prediction_nib = nib.load(prediction_path)
    pred = np.array(prediction_nib.dataobj)

    plot_by_z_slice_idx(150, lbl, pred, "test")

    
    





#     from monai.apps.auto3dseg import (
#         AlgoEnsembleBestN,
#         AlgoEnsembleBuilder,
#         import_bundle_algo_history,
#     )
#     work_dir = "/cluster/work/felixzr/TDT4265_StarterCode_2024/work_dir"
#     history = import_bundle_algo_history(work_dir, only_trained=True)
#     builder = AlgoEnsembleBuilder(history, input_yaml) # input yaml
#     builder.set_ensemble_method(AlgoEnsembleBestN(n_best=5))
#     ensembler = builder.get_ensemble()
#     preds = ensembler()
    
#     pass


    # import torch
    # from monai.networks.nets import DenseNet, SegResNet, SwinUNet

    # # Load the trained model
    # model_path = '/cluster/work/felixzr/TDT4265_StarterCode_2024/work_dir/dints_4/model_fold4/best_metric_model.pt'
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # For DINTS
    # model = DenseNet(
    #     spatial_dims=2, in_channels=1, out_channels=2, init_features=32, growth_rate=16, block_config=(6, 12, 24, 16)
    # ).to(device)

    # # For SegResNet
    # # model = SegResNet(
    # #     blocks_down=[2, 2, 2, 2], blocks_up=[1, 1, 1], init_filters=32, in_channels=1, out_channels=2
    # # ).to(device)

    # # For SwinUNet
    # # model = SwinUNet(
    # #     spatial_dims=2, in_channels=1, out_channels=2, img_size=(256, 256), patch_size=4, hidden_size=128, num_heads=4,
    # # ).to(device)

    # model.load_state_dict(torch.load(model_path))
    # model.eval()



def main():
    # train()
    inference()


if __name__ == '__main__':
    main()