import os

import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.config import load_config
from train import ContrailDataset, ContrailModel

is_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def infer(df: pd.DataFrame, model_dir: str, target_frame: int = 4) -> torch.Tensor:
    if is_kaggle:
        default_path = "/kaggle/input/contrail-source-code/config/default.yaml"
    else:
        default_path = "config/default.yaml"
    cfg = load_config(f"{model_dir}/config.yaml", default_path)
    cfg["frame_first"] = target_frame - (cfg.frame_last - cfg.frame_first)
    cfg["frame_last"] = target_frame
    model = ContrailModel(cfg).cuda().eval()
    model.load_state_dict(torch.load(f"{model_dir}/model.pt"))
    dataset = ContrailDataset(df, cfg, False, load_mask=False)
    inference_dtype = torch.float16 if cfg.precision == 16 else torch.float32
    batch_size = 8
    if inference_dtype == torch.float32 or cfg.image_size > 512:
        batch_size = 4
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    preds = []
    with torch.cuda.amp.autocast(enabled=True, dtype=inference_dtype):
        with torch.inference_mode():
            for batch in tqdm(loader):
                image = batch["image"].cuda()
                pred = model(image)[:, 1]
                pred = torchvision.transforms.functional.resize(pred, size=(256, 256))
                preds.append(pred.cpu())
    return torch.cat(preds)


def ensemble(df: pd.DataFrame, model_weights: list[tuple[str, float]], target_frame: int = 4) -> torch.Tensor:
    ensembled = None
    weight_sum = 0.0
    for model_dir, weight in model_weights:
        pred = infer(df, model_dir, target_frame)
        print("nan ratio:", pred.isnan().to(torch.float32).mean())
        pred = pred.nan_to_num_().to(torch.float32)
        weight_sum += weight
        if ensembled is None:
            ensembled = pred * weight
        else:
            ensembled += pred * weight
    assert ensembled is not None
    return ensembled / weight_sum
