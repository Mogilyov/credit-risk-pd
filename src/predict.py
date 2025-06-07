import argparse
import os

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier


def load_features(path):
    ext = os.path.splitext(path)[-1]
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext == ".feather":
        return pd.read_feather(path)
    elif ext == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unknown file extension: {ext}")


def load_models(X):
    cat_model = CatBoostClassifier()
    cat_model.load_model("models/catboost_model.cbm")
    # TabResNet
    from src.models.train_tabresnet import TabResNet1D

    tab_model = TabResNet1D(input_dim=X.shape[1], num_blocks=[2, 2, 2, 2])
    tab_model.load_state_dict(
        torch.load("models/tabresnet_model.pt", map_location="cpu")
    )
    tab_model.eval()
    # Meta-model
    meta_model = joblib.load("models/meta_model.pkl")
    return cat_model, tab_model, meta_model


def predict_stacking(X, cat_model, tab_model, meta_model):
    # CatBoost
    cat_pred = cat_model.predict_proba(X)[:, 1]
    # TabResNet
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    with torch.no_grad():
        tab_pred = torch.sigmoid(tab_model(X_tensor)).numpy()
    # Meta
    X_meta = np.column_stack([cat_pred, tab_pred])
    meta_pred = meta_model.predict_proba(X_meta)[:, 1]
    return meta_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        required=True,
        help="Path to features file (csv, feather, parquet)",
    )
    parser.add_argument(
        "--output", default="submission.csv", help="Path to save submission"
    )
    args = parser.parse_args()

    X = load_features(args.features)
    cat_model, tab_model, meta_model = load_models(X)
    preds = predict_stacking(X, cat_model, tab_model, meta_model)
    submission = pd.DataFrame({"prediction": preds})
    submission.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
