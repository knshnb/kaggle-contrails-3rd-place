from __future__ import annotations

import argparse
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import transformers
import wandb
import yaml
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from config.config import Config, load_config
from src.dataset import ContrailDataset
from src.nn import Segmentor2d, Segmentor25d, SegmentorMid25d, SegmentorMid25dDouble


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training for Kaggle Contrail")
    parser.add_argument("--out_base_dir", default="result")
    parser.add_argument("--in_base_dir", default="input")
    parser.add_argument("--exp_name", default="tmp")
    parser.add_argument("--project_name", default="kaggle-contrail")
    parser.add_argument("--load_snapshot", action="store_true")
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--wandb_logger", action="store_true")
    parser.add_argument("--config_path", default="config/debug.yaml")
    return parser.parse_args()


class ContrailDataModule(LightningDataModule):
    def __init__(
        self,
        cfg: Config,
        fold: int,
        use_val_for_train: bool,
        data_dir: str,
    ):
        super().__init__()
        self.cfg = cfg
        self.fold = fold
        self.n_gpus = torch.cuda.device_count()
        train_df = pd.DataFrame(
            {"image_dir": [f"{data_dir}/train/{dir}" for dir in sorted(os.listdir(f"{data_dir}/train"))]}
        )
        self.val_df = pd.DataFrame(
            {"image_dir": [f"{data_dir}/valid/{dir}" for dir in sorted(os.listdir(f"{data_dir}/valid"))]}
        )
        if fold < 0:
            self.train_df = train_df
            self.oof_df = train_df[:0]
        else:
            splitter = KFold(cfg.n_splits, shuffle=True, random_state=0)
            train_idx, oof_idx = list(splitter.split(train_df))[fold]
            self.train_df = train_df.iloc[train_idx].copy()
            self.oof_df = train_df.iloc[oof_idx].copy()
        if use_val_for_train:
            self.train_df = pd.concat([self.train_df, self.val_df])
            self.val_df = self.val_df[:0]
        if cfg.n_data != -1:
            self.train_df = self.train_df[: cfg.n_data]
        print(self.train_df)
        print(f"train: {len(self.train_df)}, oof: {len(self.oof_df)}, val: {len(self.val_df)}")

    def train_dataloader(self):
        return DataLoader(
            ContrailDataset(self.train_df, self.cfg, True),
            batch_size=self.cfg.batch_size // self.n_gpus,
            num_workers=4,
            shuffle=True,
            persistent_workers=True,
            drop_last=True,
        )

    def val_dataloader(self):
        val_loader = DataLoader(
            ContrailDataset(self.val_df, self.cfg, False),
            batch_size=self.cfg.batch_size // self.n_gpus,
            num_workers=4,
            persistent_workers=True,
        )
        oof_loader = DataLoader(
            ContrailDataset(self.oof_df, self.cfg, False),
            batch_size=self.cfg.batch_size // self.n_gpus,
            num_workers=4,
            persistent_workers=True,
        )
        return val_loader, oof_loader


def calc_dice_coef(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    # (N, C, H, W) -> (1,)
    pred = pred.flatten()
    label = label.flatten()
    intersection = (label * pred).sum()
    return 2.0 * intersection / (label.sum() + pred.sum())


def custom_dice_loss(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    # (N, C, H, W) -> (1,)
    pred = pred.flatten()
    label = label.flatten()
    intersection = (label * pred).sum()
    return -(2.0 * intersection + 700000) / (label.sum() + pred.sum() + 1000000)


def percentile(x: torch.Tensor, percentile: float) -> float:
    x = x.flatten()
    idx = x.argsort(descending=True)
    return x[idx[round(len(idx) * percentile)]].item()


class ContrailModel(LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        if not isinstance(cfg, Config):
            cfg = Config(cfg)
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.segmentor: Segmentor2d | Segmentor25d | SegmentorMid25d | SegmentorMid25dDouble
        if cfg.frame_first == cfg.frame_last:
            self.segmentor = Segmentor2d(cfg)
        else:
            if cfg.segmentor_type == "25d":
                self.segmentor = Segmentor25d(cfg)
            elif cfg.segmentor_type == "mid-25d":
                self.segmentor = SegmentorMid25d(cfg)
            elif cfg.segmentor_type == "mid-25d-double":
                self.segmentor = SegmentorMid25dDouble(cfg)
            else:
                raise ValueError()

    def forward(self, x: torch.Tensor, apply_sigmoid: bool = True) -> torch.Tensor:
        out = self.segmentor(x)
        if apply_sigmoid:
            out = torch.sigmoid(out)
        return out

    def _loss_func(self, logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        if logit.shape[0] == 0:
            return torch.zeros(1, device=logit.device)
        dice = custom_dice_loss(torch.sigmoid(logit), label)
        bce = F.binary_cross_entropy_with_logits(logit, label)
        return (dice + bce) / 2

    def training_step(self, batch, batch_idx):
        all_logit = self(batch["image"], False)
        has_soft_mask = batch["has_soft_mask"]
        has_soft_ratio = has_soft_mask.to(torch.float16).mean()
        loss = (
            custom_dice_loss(torch.sigmoid(all_logit[:, 0]), batch["mask"])
            + self._loss_func(all_logit[has_soft_mask, 1], batch["soft_mask"][has_soft_mask]) * has_soft_ratio
        ) / 2
        if self.cfg.pseudo_frame_aux:
            before_1 = batch["mask_before_1"][:, 0, 0] >= 0
            loss_1 = -custom_dice_loss(torch.sigmoid(all_logit[before_1, 2]), batch["mask_before_1"][before_1])
            before_2 = batch["mask_before_2"][:, 0, 0] >= 0
            loss_2 = -custom_dice_loss(torch.sigmoid(all_logit[before_2, 3]), batch["mask_before_2"][before_2])
            aux_loss = (loss_1 * before_1.to(torch.float16).mean() + loss_2 * before_2.to(torch.float16).mean()) / 2
            loss = (loss + aux_loss) / 2
        return {
            "loss": loss,
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        all_logit = self(batch["image"], False)
        loss = self._loss_func(all_logit[:, 0], batch["mask"].to(torch.float32))
        preds = torchvision.transforms.functional.resize(torch.sigmoid(all_logit[:, :2]), size=(256, 256))
        return {
            "label": batch["orig_mask"].detach().cpu(),
            "preds": preds.detach().cpu(),
            "loss": loss.detach().cpu(),
        }

    def _gather_devices_and_steps(self, outputs: list[dict[str, torch.Tensor]]) -> Optional[dict[str, torch.Tensor]]:
        outputs = self.all_gather(outputs)
        assert self.trainer is not None
        if self.trainer.global_rank != 0 or len(outputs) == 0:
            return None

        epoch_results: dict[str, torch.Tensor] = {}
        for key in outputs[0].keys():
            if self.trainer.num_devices > 1:
                result = torch.cat(
                    [(x[key].unsqueeze(1) if x[key].dim() == 1 else x[key]) for x in outputs], dim=1
                ).flatten(end_dim=1)
            else:
                result = torch.cat([(x[key].unsqueeze(0) if x[key].dim() == 0 else x[key]) for x in outputs], dim=0)
            epoch_results[key] = result.detach().cpu()
        return epoch_results

    def _epoch_end(self, step_outputs: list[dict[str, torch.Tensor]], phase: str):
        epoch_results = self._gather_devices_and_steps(step_outputs)
        if epoch_results is None:
            return

        d = {
            f"{phase}/loss": epoch_results["loss"].mean().cpu(),
        }
        if phase != "train":
            preds = epoch_results["preds"].cuda()
            label = epoch_results["label"].cuda()
            for i, pred in enumerate([preds[:, 0], preds[:, 1], preds.mean(1)]):
                # Threshold search.
                thresholds = np.arange(0.1, 1.0, 0.01)
                dice_coefs = [calc_dice_coef(pred > threshold, label).item() for threshold in thresholds]
                max_idx = np.argmax(dice_coefs)
                d[f"{phase}/best_gobal_dice_coef_{i}"] = dice_coefs[max_idx]
                # d[f"{phase}/best_threshold"] = thresholds[max_idx]
                d[f"{phase}/gobal_dice_coef_018_{i}"] = calc_dice_coef(pred > percentile(pred, 0.0018), label).item()
        print(d)
        self.log_dict(d, on_epoch=True)

    def training_epoch_end(self, training_step_outputs) -> None:
        self._epoch_end(training_step_outputs, "train")

    def validation_epoch_end(self, validation_step_outputs):
        self._epoch_end(validation_step_outputs[0], "val")
        self._epoch_end(validation_step_outputs[1], "oof")

    def _get_total_steps(self) -> int:
        if not hasattr(self, "_total_steps"):
            train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()
            accum = max(1, self.trainer.num_devices) * self.trainer.accumulate_grad_batches
            self._total_steps = len(train_loader) // accum * self.trainer.max_epochs
        return self._total_steps

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.cfg.lr)
        total_steps = self._get_total_steps()
        warmup_steps = round(total_steps * self.hparams.warmup_steps_ratio)
        print(f"lr warmup step: {warmup_steps} / {total_steps}")
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


def train(
    args: argparse.Namespace,
    cfg: Config,
    fold: int,
    use_val_for_train: bool,
) -> float:
    out_dir = f"{args.out_base_dir}/{args.exp_name}/{fold}"
    if use_val_for_train:
        out_dir += "-all"
    model = ContrailModel(cfg)
    if cfg.pretrained_model_path is not None:
        state_dict = torch.load(cfg.pretrained_model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    model.segmentor.set_grad_checkpointing(cfg.grad_checkpointing)

    data_module = ContrailDataModule(cfg, fold, use_val_for_train, args.in_base_dir)
    loggers: list[pl_loggers.LightningLoggerBase] = [pl_loggers.CSVLogger(out_dir)]
    if args.wandb_logger:
        loggers.append(
            pl_loggers.WandbLogger(
                project=args.project_name,
                group=args.exp_name,
                name=f"{args.exp_name}/{fold}",
                save_dir=out_dir,
            )
        )
    callbacks = [LearningRateMonitor("epoch")]
    if args.save_checkpoint:
        callbacks.append(ModelCheckpoint(out_dir, save_last=True, save_top_k=0))
    n_gpus = torch.cuda.device_count()
    trainer = Trainer(
        gpus=n_gpus,
        max_epochs=cfg.max_epochs,
        logger=loggers,
        callbacks=callbacks,
        enable_checkpointing=args.save_checkpoint,
        precision=cfg.precision,
        gradient_clip_val=0.7,
        strategy="ddp_find_unused_parameters_false" if n_gpus > 1 else None,
    )
    ckpt_path: Optional[str] = f"{out_dir}/last.ckpt"
    if not os.path.exists(ckpt_path) or not args.load_snapshot:
        ckpt_path = None
    trainer.fit(model, ckpt_path=ckpt_path, datamodule=data_module)
    if args.save_model:
        torch.save(model.state_dict(), f"{out_dir}/model.pt")
    with open(f"{out_dir}/config.yaml", "w") as f:
        yaml.dump(dict(cfg), f)

    if args.wandb_logger:
        wandb.finish()


def main():
    args = parse()
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    cfg = load_config(args.config_path, "config/default.yaml")
    print(cfg)
    train(args, cfg, -1, False)


if __name__ == "__main__":
    main()
