import argparse
import os

import numpy as np
import pandas as pd

from src.infer import ensemble


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training for Kaggle Contrail")
    parser.add_argument("--in_base_dir", default="input")
    parser.add_argument("--model_dirs", nargs="+", required=True)
    return parser.parse_args()


def rle_encode(y_pred, fg_val=1):
    def list_to_string(x):
        if x:
            s = str(x).replace("[", "").replace("]", "").replace(",", "")
        else:
            s = "-"
        return s

    dots = np.where(y_pred.T.flatten() == fg_val)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return list_to_string(run_lengths)


def main(args: argparse.Namespace):
    df = pd.DataFrame(
        {"image_dir": [f"{args.in_base_dir}/test/{dir}" for dir in sorted(os.listdir(f"{args.in_base_dir}/test"))]}
    )
    model_weights = [(model_dir, 1.0) for model_dir in args.model_dirs]
    preds = ensemble(df, model_weights)

    percentile = 0.0016
    idx = preds.flatten().argsort(descending=True)
    threshold = preds.flatten()[idx[round(len(idx) * percentile)]]
    binary_preds = (preds > threshold).detach().numpy().astype(int)

    sub_df = pd.DataFrame({"record_id": sorted(os.listdir(f"{args.in_base_dir}/test"))})
    sub_df["encoded_pixels"] = [rle_encode(binary_pred) for binary_pred in binary_preds]
    print(sub_df)
    sub_df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main(parse())
