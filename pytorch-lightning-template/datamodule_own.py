import lightning.pytorch as pl
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
import monai
from monai.data import CacheDataset, load_decathlon_datalist
from monai.transforms import (
    CropForegroundd,
    Spacingd,
    NormalizeIntensityd,
    Resized,
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    Orientationd,
    ToTensor,
    Padd,
    SpatialPadd,
    ScaleIntensityRanged,
)


class ASOCADataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=64, num_workers=11, data_root="./data", train_split_ratio=0.8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = data_root
        self.train_split_ratio = train_split_ratio

    # def prepare_data(self):
    #     # Download the dataset if needed (only using rank 1)
    #     datasets.CIFAR100(root=self.data_root, train=True, download=True)
    

    def setup(self, stage=None):
        # Split the dataset into train and validation sets

        train_files = load_decathlon_datalist(self.data_dir, data_list_key="training")
        val_files = load_decathlon_datalist(self.data_dir, data_list_key="validation")
        test_files = load_decathlon_datalist(self.data_dir, data_list_key="test")

        train_dataset = CustomDataset(data=train_files, transform=self.get_transforms("train"))
        val_dataset = CustomDataset(data=val_files, transform=self.get_transforms("val"))
        test_dataset = CustomDataset(data=test_files, transform=self.get_transforms("test"))

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # TODO: split manually or by param? Cross-Validation?
        # indices = torch.randperm(len(train_dataset))
        # val_size = int(len(train_dataset) * self.train_split_ratio)
        # self.train_dataset = Subset(train_dataset, indices[-val_size:])
        # self.val_dataset = Subset(val_dataset, indices[:-val_size])
       
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,pin_memory=True, shuffle=False)
    
    def get_transforms(self,split):
        # mean = [0.4914, 0.4822, 0.4465]
        # std = [0.2023, 0.1994, 0.2010]
        
        shared_transforms = ([
            # # Add your transformations here
            # LoadImaged(keys=["sample", "label"]),
            # EnsureChannelFirstd(keys=["sample", "label"]),
            # # Spacingd(keys=["sample", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),     ##TODO ??
            # NormalizeIntensityd(keys=["sample"]),
            # # Resized(keys=["sample"], spatial_size=(256, 256, 128)),           ##TODO ??
            # CropForegroundd(keys=["sample", "label"], source_key="sample"),
            # SpatialPadd(keys=["sample", "label"], spatial_size=(512, 512, 224), method="symmetric", mode=("constant", "edge")),

            LoadImaged(keys=["sample", "label"]),
            EnsureChannelFirstd(keys=["sample", "label"]),
            ScaleIntensityRanged(
                keys=["sample"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["sample", "label"], source_key="sample"),
            Orientationd(keys=["sample", "label"], axcodes="RAS"),
            Spacingd(keys=["sample", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            RandCropByPosNegLabeld(
                keys=["sample", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="sample",
                image_threshold=0,
            ),
        ])

        if split == "train":
            return monai.transforms.Compose([
                *shared_transforms
            ])
            
        elif split == "val":
            return monai.transforms.Compose([                
                *shared_transforms
            ])
        
        elif split == "test":
            # TODO! Anpassen, testset nicht augmentieren
            return transforms.transforms.Compose([                
                *shared_transforms
            ])
        
        # "worked" (though sitty acc/loss):
        # if split == "train":
        #     return monai.transforms.Compose([
        #         LoadImaged(keys=["sample", "label"]),
        #         EnsureChannelFirstd(keys=["sample", "label"]),
        #         ScaleIntensityd(keys="sample"), # normalization
        #         # RandCropByPosNegLabeld(keys=["sample", "label"], label_key="label", spatial_size=[96, 96, 96], pos=1, neg=1, num_samples=4),
        #         # RandRotate90d(keys=["sample", "label"], prob=0.5, spatial_axes=[0, 2]),
        #         # Resized(keys=["sample", "label"], spatial_size=(512, 512, 215), mode=("trilinear", "nearest")),
        #         SpatialPadd(keys=["sample", "label"], spatial_size=(512, 512, 224), method="symmetric", mode=("constant", "edge")),
        #         ToTensor()
        #     ])
            
        # elif split == "val":
        #     return monai.transforms.Compose([                
        #         LoadImaged(keys=["sample", "label"]),
        #         EnsureChannelFirstd(keys=["sample", "label"]),
        #         ScaleIntensityd(keys="sample"), # normalization
        #         # RandCropByPosNegLabeld(keys=["sample", "label"], label_key="label", spatial_size=[96, 96, 96], pos=1, neg=1, num_samples=4),
        #         # RandRotate90d(keys=["sample", "label"], prob=0.5, spatial_axes=[0, 2]),
        #         # Resized(keys=["sample", "label"], spatial_size=(512, 512, 215), mode=("trilinear", "nearest")),
        #         SpatialPadd(keys=["sample", "label"], spatial_size=(512, 512, 224), method="symmetric", mode=("constant", "edge")),
        #         ToTensor()
        #     ])
        
        # elif split == "test":
        #     return transforms.transforms.Compose([                
        #         LoadImaged(keys=["sample", "label"]),
        #         EnsureChannelFirstd(keys=["sample", "label"]),
        #         ScaleIntensityd(keys="sample"), # normalization
        #         # RandCropByPosNegLabeld(keys=["sample", "label"], label_key="label", spatial_size=[96, 96, 96], pos=1, neg=1, num_samples=4),
        #         # RandRotate90d(keys=["sample", "label"], prob=0.5, spatial_axes=[0, 2]),
        #         # Resized(keys=["image", "label"], spatial_size=(512, 512, 215), mode=("trilinear", "nearest")),
        #         SpatialPadd(keys=["sample", "label"], spatial_size=(512, 512, 224), method="symmetric", mode=("constant", "edge")),
        #         ToTensor()
        #     ])

 
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]["sample"]  
        label = self.data[index]["label"]  

        sample_dict = {"sample": sample, "label": label}

        if self.transform:
            sample_dict = self.transform(sample_dict)

        return sample_dict
