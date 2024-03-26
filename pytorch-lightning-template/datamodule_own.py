import lightning.pytorch as pl
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from PIL import Image
import torch
import monai
from monai.data import CacheDataset, load_decathlon_datalist
from monai.transforms import (
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

        # train_data = load_decathlon_datalist(self.data_dir, data_list_key="training")
        # val_data = load_decathlon_datalist(self.data_dir, data_list_key="validation")
        # test_data = load_decathlon_datalist(cwd + "project/code/data.json", data_list_key="test")
        # TODO: what about test_data?

        # train_dataset = CacheDataset(data=train_data, transform=self.get_transforms("train"), cache_rate=1.0, num_workers=4)
        # val_dataset = CacheDataset(data=val_data, transform=self.get_transforms("val"), cache_rate=1.0, num_workers=4)

        # train_dataset = datasets.CIFAR100(root=self.data_root, train=True, transform=self.get_transforms("train"))
        # val_dataset = datasets.CIFASR100(root=self.data_root, train=True, transform=self.get_transforms("val"))
        # train_dataset = datasets.VisionDataset(root=self.data_dir, train=True, transform=self.get_transforms("train"))
        # val_dataset = datasets.VisionDataset(root=self.data_dir, train=True, transform=self.get_transforms("val"))
       
        # train_dataset = MONAIDecathlonToTorchvision(train_data, transform=self.get_transforms("train"))
        # val_dataset = MONAIDecathlonToTorchvision(val_data, transform=self.get_transforms("val"))

        data_transforms = transforms.Compose([
            LoadImaged(keys=["sample", "label"]),
            EnsureChannelFirstd(keys=["sample", "label"]),
            ScaleIntensityd(keys="sample"), # normalization
            # RandCropByPosNegLabeld(keys=["sample", "label"], label_key="label", spatial_size=[96, 96, 96], pos=1, neg=1, num_samples=4),
            # RandRotate90d(keys=["sample", "label"], prob=0.5, spatial_axes=[0, 2]),
            ToTensor()
        ])

        training_files = [{"sample": "/cluster/work/felixzr/TDT4265_StarterCode_2024/data/ASOCA/Diseased/CTCA/Diseased_1.nrrd", 
                           "label": "/cluster/work/felixzr/TDT4265_StarterCode_2024/data/ASOCA/Diseased/Annotations/Diseased_1.nrrd"},
                           {"sample": "/cluster/work/felixzr/TDT4265_StarterCode_2024/data/ASOCA/Diseased/CTCA/Diseased_2.nrrd", 
                           "label": "/cluster/work/felixzr/TDT4265_StarterCode_2024/data/ASOCA/Diseased/Annotations/Diseased_2.nrrd"},
                           {"sample": "/cluster/work/felixzr/TDT4265_StarterCode_2024/data/ASOCA/Diseased/CTCA/Diseased_3.nrrd", 
                           "label": "/cluster/work/felixzr/TDT4265_StarterCode_2024/data/ASOCA/Diseased/Annotations/Diseased_3.nrrd"},] 

        val_files = [{"sample": "/cluster/work/felixzr/TDT4265_StarterCode_2024/data/ASOCA/Diseased/CTCA/Diseased_4.nrrd", 
                           "label": "/cluster/work/felixzr/TDT4265_StarterCode_2024/data/ASOCA/Diseased/Annotations/Diseased_4.nrrd"},
                           {"sample": "/cluster/work/felixzr/TDT4265_StarterCode_2024/data/ASOCA/Diseased/CTCA/Diseased_5.nrrd", 
                           "label": "/cluster/work/felixzr/TDT4265_StarterCode_2024/data/ASOCA/Diseased/Annotations/Diseased_5.nrrd"},] 

        train_dataset = CustomDataset(data=training_files, transform=data_transforms)
        val_dataset = CustomDataset(data=val_files, transform=data_transforms)
        # train_dataset = CacheDataset(data=training_files, transform=data_transforms)
        # val_dataset = CacheDataset(data=val_files, transform=data_transforms)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        test_files = [{"sample": "/cluster/work/felixzr/TDT4265_StarterCode_2024/data/ASOCA/Diseased/CTCA/Diseased_6.nrrd", 
                           "label": "/cluster/work/felixzr/TDT4265_StarterCode_2024/data/ASOCA/Diseased/Annotations/Diseased_6.nrrd"}] 
        test_dataset = CustomDataset(data=test_files, transform=data_transforms)
        self.test_dataset = test_dataset
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
        
        shared_transforms = [
            transforms.ToTensor(),
            # transforms.Normalize(mean, std) 
        ]
        
        if split == "train":
            # return transforms.Compose([
            #     *shared_transforms,
            #     # LoadImaged(keys=["image", "label"]),
            #     # EnsureChannelFirstd(keys=["image", "label"]),

            #     # transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
            #     # transforms.RandomHorizontalFlip(),
            # ])
            return monai.transforms.Compose([
                *shared_transforms,
                LoadImaged(keys=["sample", "label"]),
                EnsureChannelFirstd(keys=["sample", "label"]),
                ScaleIntensityd(keys="sample"), # normalization
                Resized(keys=["sample", "label"], spatial_size=[512,512,200]),
            ])
            
        elif split == "val":
            return monai.transforms.Compose([
                *shared_transforms,
                LoadImaged(keys=["sample", "label"]),
                EnsureChannelFirstd(keys=["sample", "label"]),
                ScaleIntensityd(keys="sample"), # normalization
                Resized(keys=["sample", "label"], spatial_size=[512,512,200]),
            ])
        
        elif split == "test":
            return transforms.transforms.Compose([
                *shared_transforms,
                # ...
            ])

 
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
