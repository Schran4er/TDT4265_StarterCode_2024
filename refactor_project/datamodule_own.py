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
    ScaleIntensityRanged,
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
        # mean, std = -848.3641349994796, 1201.188331923214
        
        shared_transforms = ([
            EnsureChannelFirstd(keys=["sample", "label"]),
            ScaleIntensityRanged(keys=["sample"], a_min=-500, a_max=300, b_min=0.0, b_max=1.0, clip=True), # increase contrast # TODO: values determined using itk_snap -> tools -> image_layer_inspector
            CropForegroundd(keys=["sample", "label"], source_key="sample"), # crop to region of interest

            Orientationd(keys=["sample", "label"], axcodes="LPS"),
            # Spacingd(keys=["sample", "label"], pixdim=[1.5, 1.5, 2.0]), # resolution, maybe not, according to paper?
            # Resized(keys=["sample", "label"], spatial_size=(512, 512, 215), mode=("trilinear", "nearest")), # adjust so all images of equal size
            SpatialPadd(keys=["sample", "label"], spatial_size=(512, 512, 224), method="symmetric", mode=("constant", "edge")),
        ])

        if split == "train":
            return monai.transforms.Compose([
                LoadImaged(keys=["sample", "label"]),
                *shared_transforms,                
                # RandCropByPosNegLabeld(keys=["sample", "label"], label_key="label", spatial_size=[96, 96, 50], pos=1, neg=1, num_samples=4),
                # RandRotate90d(keys=["sample", "label"], prob=0.5, spatial_axes=[0, 2]),
                ToTensor()
            ])
            
        elif split == "val":
            return monai.transforms.Compose([   
                LoadImaged(keys=["sample", "label"]),
                *shared_transforms,
                ToTensor()
            ])
        
        elif split == "test":
            return transforms.transforms.Compose([   
                LoadImaged(keys=["sample", "label"]),
                *shared_transforms,
                ToTensor()
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
