import os

import segmentation_models_pytorch as smp
import torch

from config.config import Config

is_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ


class Segmentor2d(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        encoder_weights = None if is_kaggle else "imagenet"
        self.backbone = smp.Unet(
            encoder_name=cfg.model_name,
            encoder_weights=encoder_weights,
            in_channels=cfg.in_ch,
            classes=2,
            decoder_channels=[ch * cfg.decoder_ch_coef for ch in (256, 128, 64, 32, 16)],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x.squeeze(2))

    def set_grad_checkpointing(self, enable: bool = True):
        self.backbone.encoder.model.set_grad_checkpointing(enable)


class Segmentor25d(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.mid_ch = 128
        encoder_weights = None if is_kaggle else "imagenet"
        self.backbone = smp.Unet(
            encoder_name=cfg.model_name,
            encoder_weights=encoder_weights,
            in_channels=cfg.in_ch,
            classes=self.mid_ch,
            decoder_channels=[ch * cfg.decoder_ch_coef for ch in (256, 128, 64, 32, 16)],
        )
        self.head_3d = torch.nn.Sequential(
            torch.nn.Conv3d(self.mid_ch, self.mid_ch, (2, 9, 9), padding=(0, 4, 4), padding_mode="replicate"),
            torch.nn.BatchNorm3d(self.mid_ch),
            torch.nn.LeakyReLU(),
            torch.nn.Conv3d(self.mid_ch, self.mid_ch, (2, 9, 9), padding=(0, 4, 4), padding_mode="replicate"),
            torch.nn.BatchNorm3d(self.mid_ch),
            torch.nn.LeakyReLU(),
        )
        self.head_2d = torch.nn.Sequential(
            torch.nn.Conv2d(self.mid_ch, 2, 3, padding=1, padding_mode="replicate"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_batch, in_ch, n_frame, H, W = x.shape
        x = x.transpose(1, 2).reshape(n_batch * n_frame, in_ch, H, W)
        out = self.backbone(x)
        out = out.reshape(n_batch, n_frame, self.mid_ch, H, W).transpose(1, 2)
        out = self.head_3d(out).squeeze(2)
        return self.head_2d(out)

    def set_grad_checkpointing(self, enable: bool = True):
        self.backbone.encoder.model.set_grad_checkpointing(enable)


class Conv3dBlock(torch.nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: tuple[int, int, int], padding: tuple[int, int, int]
    ):
        super().__init__(
            torch.nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, padding_mode="replicate"),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.LeakyReLU(),
        )


class SegmentorMid25d(torch.nn.Module):
    def __init__(self, cfg: Config, out_ch: int = 2):
        super().__init__()
        self.n_frames = 3
        encoder_weights = None if is_kaggle else "imagenet"
        if cfg.pseudo_frame_aux:
            out_ch += 2
        self.backbone = smp.Unet(
            encoder_name=cfg.model_name,
            encoder_weights=encoder_weights,
            in_channels=cfg.in_ch,
            classes=out_ch,
            decoder_channels=[ch * cfg.decoder_ch_coef for ch in (256, 128, 64, 32, 16)],
        )
        k = cfg.conv3d_kernel_size
        conv3ds = [
            torch.nn.Sequential(
                Conv3dBlock(ch, ch, (2, k, k), (0, k // 2, k // 2)),
                Conv3dBlock(ch, ch, (2, k, k), (0, k // 2, k // 2)),
            )
            for ch in self.backbone.encoder.out_channels[1:]
        ]
        if cfg.last_single:
            conv3ds.pop()
            channels = self.backbone.encoder.out_channels
            conv3ds.append(Conv3dBlock(channels[-1], channels[-1], (3, 3, 3), (0, 1, 1)))
        self.conv3ds = torch.nn.ModuleList(conv3ds)

    def _to2d(self, conv3d_block: torch.nn.Module, feature: torch.Tensor) -> torch.Tensor:
        total_batch, ch, H, W = feature.shape
        feat_3d = feature.reshape(total_batch // self.n_frames, self.n_frames, ch, H, W).transpose(1, 2)
        return conv3d_block(feat_3d).squeeze(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_batch, in_ch, n_frame, H, W = x.shape
        x = x.transpose(1, 2).reshape(n_batch * n_frame, in_ch, H, W)

        self.backbone.check_input_shape(x)

        features = self.backbone.encoder(x)
        features[1:] = [self._to2d(conv3d, feature) for conv3d, feature in zip(self.conv3ds, features[1:])]
        decoder_output = self.backbone.decoder(*features)

        masks = self.backbone.segmentation_head(decoder_output)
        return masks

    def set_grad_checkpointing(self, enable: bool = True):
        self.backbone.encoder.model.set_grad_checkpointing(enable)


class SegmentorMid25dDouble(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        mid_ch = 128
        self.backbone25d = SegmentorMid25d(cfg, mid_ch)
        encoder_weights = None if is_kaggle else "imagenet"
        self.backbone = smp.Unet(
            encoder_name=cfg.model_name,
            encoder_weights=encoder_weights,
            in_channels=mid_ch + 3,
            classes=2,
            decoder_channels=[ch * cfg.decoder_ch_coef for ch in (256, 128, 64, 32, 16)],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mid = self.backbone25d(x)  # (n_batch, out_ch, H, W)
        mid = torch.cat([x[:, :, 2], mid], 1)
        return self.backbone(mid)

    def set_grad_checkpointing(self, enable: bool = True):
        self.backbone25d.set_grad_checkpointing(enable)
        self.backbone.encoder.model.set_grad_checkpointing(enable)
