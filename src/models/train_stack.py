import os
import sys
from pathlib import Path

import git
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from src.data.download import download_data
from src.models.train_tabresnet import TabResNet1D

# Add project root to Python path (если нужно)
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Загрузка обученных моделей (CatBoost и TabResNet-18 (1D))
# (Предполагается, что модели сохранены в models/catboost_model.cbm и models/tabresnet_model.pt)
# (Если LightGBM не используется, то используем LogisticRegression как мета-модель)


def load_models(X):
    cat_model = CatBoostClassifier()
    cat_model.load_model("models/catboost_model.cbm")
    # Загрузка TabResNet
    tab_model = TabResNet1D(input_dim=X.shape[1], num_blocks=[2, 2, 2, 2])
    tab_model.load_state_dict(
        torch.load("models/tabresnet_model.pt", map_location="cpu")
    )
    tab_model.eval()
    return cat_model, tab_model


# Функция для получения out-of-fold предсказаний (например, через KFold)
def get_oof_preds(X, y, cat_model, tab_model, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_cat = np.zeros(len(X))
    oof_tab = np.zeros(len(X))
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        # CatBoost
        oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
        # TabResNet
        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
        with torch.no_grad():
            logits = tab_model(X_val_tensor)
            probs = torch.sigmoid(logits).numpy()
        oof_tab[val_idx] = probs
    return oof_cat, oof_tab


# Функция для обучения мета-модели (LogisticRegression) на out-of-fold предсказаниях
def train_meta_model(oof_cat, oof_tab, y, random_state=42):
    X_meta = np.column_stack((oof_cat, oof_tab))
    meta_model = LogisticRegression(random_state=random_state)
    meta_model.fit(X_meta, y)
    return meta_model


# Функция для логирования метрик (например, ROC-AUC) и сохранения итоговой модели
def log_and_save_meta_model(meta_model, oof_cat, oof_tab, y, run_name="stacking_meta"):
    X_meta = np.column_stack((oof_cat, oof_tab))
    preds = meta_model.predict_proba(X_meta)[:, 1]
    auc = roc_auc_score(y, preds)
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("stacking_ensemble")
    with mlflow.start_run(run_name=run_name) as run:
        # Логируем git SHA
        repo = git.Repo(search_parent_directories=True)
        git_sha = repo.head.object.hexsha
        mlflow.set_tag("git_sha", git_sha)
        mlflow.log_metric("meta_auc", auc)
        mlflow.sklearn.log_model(meta_model, "meta_model")
        # ROC-кривая
        from sklearn.metrics import auc as calc_auc
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y, preds)
        roc_auc = calc_auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Meta-model ROC Curve")
        plt.legend(loc="lower right")
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/stacking_roc_auc.png")
        plt.close()
        mlflow.log_artifact("plots/stacking_roc_auc.png")
        # Feature importance (коэффициенты)
        plt.figure()
        plt.bar(range(X_meta.shape[1]), np.abs(meta_model.coef_[0]))
        plt.title("Meta-model Feature Importances (abs(coef))")
        plt.xlabel("Meta-feature index")
        plt.ylabel("Importance")
        plt.savefig("plots/stacking_feature_importance.png")
        plt.close()
        mlflow.log_artifact("plots/stacking_feature_importance.png")
        # Learning curve (AUC по фолдам)
        # Для этого нужно передавать auc по фолдам (если есть)
        # Здесь просто логируем одну точку, но можно расширить
    # Сохраняем метрики для DVC
    import json

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/stack_metrics.json", "w") as f:
        json.dump({"meta_auc": float(auc)}, f)
    # Экспорт в ONNX
    initial_type = [("input", FloatTensorType([None, 2]))]
    onnx_model = convert_sklearn(meta_model, initial_types=initial_type)
    os.makedirs("models", exist_ok=True)
    with open("models/meta_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())


# Основная функция для запуска стекинга
def train_stack():
    download_data()
    X, y = pd.read_csv("data/processed/X_processed.csv"), pd.read_csv(
        "data/processed/y_processed.csv"
    ).squeeze(axis=1)
    cat_model, tab_model = load_models(X)
    oof_cat, oof_tab = get_oof_preds(X, y, cat_model, tab_model)
    meta_model = train_meta_model(oof_cat, oof_tab, y)
    log_and_save_meta_model(meta_model, oof_cat, oof_tab, y)


if __name__ == "__main__":
    train_stack()
