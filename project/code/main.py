import torch
import os
import monai
from monai.networks.nets import UNet
from monai.data import CacheDataset, DataLoader, decollate_batch, load_decathlon_datalist
from monai.inferers import sliding_window_inference
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
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
    Orientationd
)

import matplotlib.pyplot as plt

if __name__ == '__main__':
    cwd = "/cluster/work/felixzr/TDT4265_StarterCode_2024/"

    train_data = load_decathlon_datalist(cwd + "project/code/data.json", data_list_key="training")
    val_data = load_decathlon_datalist(cwd + "project/code/data.json", data_list_key="validation")
    test_data = load_decathlon_datalist(cwd + "project/code/data.json", data_list_key="test")
    
    # todo: verstehen was die einzelnen compose Dinge tun
    train_transforms = Compose(
        [
            LoadImaged(keys=["sample", "label"]),
            EnsureChannelFirstd(keys=["sample", "label"]),
            # Orientationd(keys=["sample", "label"]), # unifies the data orientation based on the affine matrix
            ScaleIntensityd(keys="sample"), # normalization
            # RandCropByPosNegLabeld(
            #     keys=["sample", "label"], label_key="label", spatial_size=[96, 96, 96], pos=1, neg=1, num_samples=4
            # ), # what does this do?
            # RandRotate90d(keys=["sample", "label"], prob=0.5, spatial_axes=[0, 2]),
            Resized(keys=["sample", "label"], spatial_size=[512,512,200]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["sample", "label"]),
            EnsureChannelFirstd(keys=["sample", "label"]),
            # Orientationd(keys=["sample", "label"]), # unifies the data orientation based on the affine matrix
            ScaleIntensityd(keys="sample"), # normalization
            Resized(keys=["sample", "label"], spatial_size=[512,512,200]),
        ]
    )

    # set fixed seed for comparability
    train_transforms.set_random_state(seed=0)
    val_transforms.set_random_state(seed=0)
    monai.utils.set_determinism(seed=0, additional_settings=None)

    # train_ds = CacheDataset(data=train_data, transform=train_transforms, num_workers=4)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    # train_loader = DataLoader(train_ds, batch_size=2, num_workers=4, collate_fn=list_data_collate)
    # train_data = monai.utils.misc.first(train_loader)


    train_ds = CacheDataset(data=train_data, transform=train_transforms, cache_rate=1.0, num_workers=4)

    # use batch_size=2 to load images and use RandCropByPosNegLabeld
    # to generate 2 x 4 images for network training
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

    val_ds = CacheDataset(data=val_data, transform=val_transforms, cache_rate=1.0, num_workers=4)
    # val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)


    # standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda:0")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    max_epochs = 2
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["sample"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["sample"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = (160, 160, 160)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(cwd, "best_metric_model.pth"))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )

    print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")

    print("asd")