import monai
from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    ScaleIntensityd,
    Resized,
    Padd,
    ToTensord,
    SpatialPadd,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob

from monai.data import load_decathlon_datalist



def preproc(train_data, val_data):
    """
    todo
    """

    # set fixed seed for comparability
    monai.utils.set_determinism(seed=0, additional_settings=None)

    mean, std = -848.3641349994796, 1201.188331923214       ## todo! Only for train data!!


    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),        # load image
            EnsureChannelFirstd(keys=["image", "label"]),       # channel dimension is the first dimension of the input data
            Orientationd(keys=["image", "label"], axcodes="LPS"), # adjust orientation of the input data
            ScaleIntensityd(keys="image"), # normalization (intensity scaled to between 0, 1)
            # ScaleIntensityRanged(keys="image", a_min, a_max, b_min, b_max), # scale a_minmax intensity to b_minmax
            # Spacingd(keys=["image", "label"], pixdim=(0.5, 0.5, 0.75), mode=("bilinear", "nearest")),       # resample to new pixel spacing
                                        ### pixdim: interpolation of x,y,z resolution to new spacing in mm !!
            Resized(keys=["image", "label"], spatial_size=(512, 512, 215), mode=("trilinear", "nearest")),      # resize image resolution, since different resolutions in z!
            # Padd(keys=["image", "label"], spatial_size=(512, 512, 224), mode=("constant", "edge")),           # pad image to desired size. Not need if we use Resized?
            # SpatialPadd(keys=["image", "label"], spatial_size=(512, 512, 210), method="symmetric", mode=("constant", "edge")),    # spatial pad
            CropForegroundd(keys=["image", "label"], source_key="image"),
            # ToTensord(keys=["image", "label"]),       # convert to pytorch tensor
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            # Orientationd(keys=["image", "label"], axcodes="RAS"), # unifies the data orientation based on the affine matrix
            ScaleIntensityd(keys="image"), # normalization
            Spacingd(keys=["image", "label"], pixdim=(0.5, 0.5, 0.75), mode=("bilinear", "nearest")),
            Resized(keys=["image", "label"], spatial_size=(512, 512, 215), mode=("trilinear", "nearest")),
            # Padd(keys=["image", "label"], spatial_size=(512, 512, 210), mode=("constant", "edge")),
            # ToTensord(keys=["image", "label"]),
            # SpatialPadd(keys=["image", "label"], spatial_size=(512, 512, 210), method="symmetric", mode=("constant", "edge")),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            ToTensord(keys=["image", "label"]),
        ]
    )

    # set fixed seed for comparability
    train_transforms.set_random_state(seed=0)
    val_transforms.set_random_state(seed=0)

    train_ds = CacheDataset(data=train_data, transform=train_transforms, cache_rate=1.0, num_workers=4)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

    val_ds = CacheDataset(data=val_data, transform=val_transforms, cache_rate=1.0, num_workers=4)
    # val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)