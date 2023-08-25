import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.infer import ensemble


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make pseudo label")
    parser.add_argument("--in_base_dir", default="input")
    parser.add_argument("--out_dir", default="pseudo-label")
    parser.add_argument("--model_dirs", nargs="+", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()
    model_weights = [(model_dir, 1.0) for model_dir in args.model_dirs]
    df = pd.DataFrame(
        {"image_dir": [f"{args.in_base_dir}/train/{dir}" for dir in sorted(os.listdir(f"{args.in_base_dir}/train"))]}
    )
    names = sorted(os.listdir(f"{args.in_base_dir}/train"))
    for frame in range(2, 8):
        preds = ensemble(df, model_weights)
        print(preds.shape)
        for name, pred in tqdm(zip(names, preds)):
            dir = f"{args.out_dir}/{name}"
            os.makedirs(dir, exist_ok=True)
            np.save(f"{dir}/frame{frame}.npy", pred.reshape(256, 256, 1))
