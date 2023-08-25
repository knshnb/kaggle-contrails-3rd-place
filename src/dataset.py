import random
from typing import Any

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2

from config.config import Config


def normalize_range(data, bounds):
    """Maps data to the range [0, 1]."""
    return (data - bounds[0]) / (bounds[1] - bounds[0])


def get_3ch_image(band11: np.ndarray, band14: np.ndarray, band15: np.ndarray) -> np.ndarray:
    _T11_BOUNDS = (243, 303)
    _CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
    _TDIFF_BOUNDS = (-4, 2)
    r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
    g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
    b = normalize_range(band14, _T11_BOUNDS)
    return np.clip(np.stack([r, g, b], axis=2), 0, 1)


class ContrailDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: Config,
        data_aug: bool,
        *,
        load_mask: bool = True,
    ) -> None:
        super().__init__()
        self.image_dirs = df.image_dir.values
        self.cfg = cfg
        self.n_frames = self.cfg.frame_last - self.cfg.frame_first + 1
        self.data_aug = data_aug
        augments = [A.Resize(cfg.image_size, cfg.image_size)]
        if data_aug:
            augments = [
                A.HorizontalFlip(p=0.1),
                A.VerticalFlip(p=0.1),
                A.RandomRotate90(p=0.4),
                A.Resize(cfg.image_size, cfg.image_size),
                A.ShiftScaleRotate(0.05, 0.1, 15, p=0.3),
                A.RandomResizedCrop(
                    cfg.image_size,
                    cfg.image_size,
                    scale=(0.75, 1.0),
                    ratio=(0.9, 1.1111111111111),
                    p=0.7,
                ),
            ]
        augments.append(ToTensorV2(transpose_mask=True))  # HWC to CHW
        self.transform = A.Compose(
            augments, additional_targets={"soft_mask": "mask", "mask_before_1": "mask", "mask_before_2": "mask"}
        )
        self._load_mask = load_mask

    def __len__(self) -> int:
        return len(self.image_dirs)

    def _read_cached(self, file_path: str):
        return np.load(file_path)

    def _read_images(self, file_path: str, frame_first: int, frame_last: int):
        return self._read_cached(file_path)[:, :, frame_first : frame_last + 1]

    def get_ash_color_images(self, image_dir: str, frame_first: int, frame_last: int) -> np.ndarray:
        # Returns (256, 256, ch, n_frame)
        band11 = self._read_images(f"{image_dir}/band_11.npy", frame_first, frame_last)
        band14 = self._read_images(f"{image_dir}/band_14.npy", frame_first, frame_last)
        band15 = self._read_images(f"{image_dir}/band_15.npy", frame_first, frame_last)
        return get_3ch_image(band11, band14, band15)

    def __getitem__(self, i: int) -> dict[str, Any]:
        image_dir = self.image_dirs[i]
        if self.data_aug and self.cfg.pseudo_dir is not None:
            frame_last = random.choice([2, 3, 5, 6, 7])
        else:
            frame_last = self.cfg.frame_last
        image = self.get_ash_color_images(image_dir, frame_last - self.n_frames + 1, frame_last)
        image = image.reshape(256, 256, self.cfg.in_ch * self.n_frames)
        ret = {}
        before_transform = {"image": image}
        if self._load_mask:
            # (256, 256)
            if self.data_aug and self.cfg.pseudo_dir is not None:
                pseudo_path = f"{self.cfg.pseudo_dir}/{image_dir.split('/')[-1]}/frame{frame_last}.npy"
                mask = self._read_cached(pseudo_path).squeeze(2).astype(np.float32)
                before_transform["mask"] = (mask > 0.6).astype(np.float32)
                before_transform["soft_mask"] = mask
                if self.cfg.hard_pseudo_label:
                    before_transform["soft_mask"] = np.round(before_transform["soft_mask"] * 4.0) / 4.0
                ret["has_soft_mask"] = True
            else:
                mask = self._read_cached(f"{image_dir}/human_pixel_masks.npy").squeeze(2).astype(np.float32)
                before_transform["mask"] = mask
                ret["orig_mask"] = mask
                if "train" in image_dir:
                    # (256, 256, 4+)
                    individual_mask = self._read_cached(f"{image_dir}/human_individual_masks.npy").squeeze(2)
                    before_transform["soft_mask"] = individual_mask.mean(2)
                else:
                    before_transform["soft_mask"] = mask
                ret["has_soft_mask"] = "train" in image_dir
            for frame_dif in range(1, 3):
                frame = frame_last - frame_dif
                if self.cfg.pseudo_frame_aux and self.data_aug and self.cfg.pseudo_dir is not None and frame >= 2:
                    pseudo_path = f"{self.cfg.pseudo_dir}/{image_dir.split('/')[-1]}/frame{frame}.npy"
                    mask = self._read_cached(pseudo_path).squeeze(2).astype(np.float32)
                else:
                    mask = np.full((256, 256), -1)
                if self.cfg.hard_pseudo_label:
                    mask = np.round(mask * 4.0) / 4.0
                before_transform[f"mask_before_{frame_dif}"] = mask
        transformed = self.transform(**before_transform)
        transformed["image"] = transformed["image"].reshape(
            self.cfg.in_ch, self.n_frames, self.cfg.image_size, self.cfg.image_size
        )
        ret.update(transformed)
        return ret
