from datamodule_own import ASOCADataModule
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import torch
from torchmetrics import Accuracy
import munch
import yaml
from pathlib import Path

from monai.metrics import DiceMetric, hausdorff_distance
from monai.losses import DiceLoss, DiceFocalLoss
from monai.networks.nets import UNet
from monai.networks.layers import Norm


torch.set_float32_matmul_precision('medium')

cwd = "/cluster/work/felixzr/TDT4265_StarterCode_2024/pytorch-lightning-template/"
config = munch.munchify(yaml.load(open(cwd + "config.yaml"), Loader=yaml.FullLoader))

# DEVICE = torch.device("cuda:0")
DEVICE = "cuda"

class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # weights = ResNet50_Weights.DEFAULT if config.use_pretrained_weights else None
        # self.model = resnet50(weights=weights)
        # self.model.fc = nn.Linear(2048, self.config.num_classes)

        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(DEVICE)
        
        # self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = DiceLoss(to_onehot_y=True, softmax=True)
        self.loss_fn = DiceFocalLoss(sigmoid=True)
        # self.acc_fn = Accuracy(task="multiclass", num_classes=self.config.num_classes)      # todo HD95 and/or Dice
        self.acc_fn = DiceMetric(include_background=False, reduction="mean")
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(model.parameters(), 0.01)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.config.max_lr, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.max_epochs)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"} ]

    def forward(self, x):
        y_hat = torch.sigmoid(self.model(x))
        try: y_hat_thresholded = torch.round(y_hat).to(dtype=dtype)
        except: y_hat_thresholded = torch.round(y_hat).to(dtype=torch.float32)
        return y_hat_thresholded

    def training_step(self, batch, batch_idx):
        x, y = (
            batch["sample"].to(DEVICE),
            batch["label"].to(DEVICE),
        )
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.acc_fn(y_hat, y).mean()
        self.log_dict({
            "train/loss": loss,
            "train/acc": acc
        },on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = (
            batch["sample"].to(DEVICE),
            batch["label"].to(DEVICE),
        )
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.acc_fn(y_hat, y).mean()
        self.log_dict({
            "val/loss":loss,
            "val/acc": acc
        },on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        x, y = (
            batch["sample"].to(DEVICE),
            batch["label"].to(DEVICE),
        )
        y_hat = self.forward(x)
        acc = self.acc_fn(y_hat, y).mean()
        self.log_dict({
            "test/acc": acc,
        },on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)

if __name__ == "__main__":
    dtype = torch.float32
    torch.set_grad_enabled(True)
    pl.seed_everything(42)

    dm = ASOCADataModule(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_split_ratio=config.train_split_ratio,
        data_root=config.data_root
    )

    torch.cuda.empty_cache() # to prevent CUDA Out Of Memory
    if config.checkpoint_path:
        model = LitModel.load_from_checkpoint(checkpoint_path=config.checkpoint_path, config=config)
        model = model.to(dtype)
        print("Loading weights from checkpoint...")
    else:
        model = LitModel(config)
        model = model.to(dtype)

    trainer = pl.Trainer(
        devices=config.devices, 
        max_epochs=config.max_epochs, 
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        enable_progress_bar=config.enable_progress_bar,
        precision="16-mixed",
        log_every_n_steps=4,
        # deterministic=True,
        logger=WandbLogger(project=config.wandb_project, name=config.wandb_experiment_name, config=config),
        callbacks=[
            EarlyStopping(monitor="val/acc", patience=config.early_stopping_patience, mode="max", verbose=True),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(dirpath=Path(config.checkpoint_folder, config.wandb_project, config.wandb_experiment_name), 
                            filename='best_model:epoch={epoch:02d}-val_acc={val/acc:.4f}',
                            auto_insert_metric_name=False,
                            save_weights_only=True,
                            save_top_k=1),
        ])
    if not config.test_model:
        trainer.fit(model, datamodule=dm)
    
    trainer.test(model, datamodule=dm)
