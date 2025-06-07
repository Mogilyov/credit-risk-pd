import os
from pathlib import Path

import git
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold

from src.data.download import download_data


def load_processed_data():
    """Загружает обработанные данные (X, y) из data/processed."""
    X = pd.read_csv("data/processed/X_processed.csv").head(1000)
    y = pd.read_csv("data/processed/y_processed.csv").squeeze(axis=1).head(1000)
    return X, y


def train_baseline_lr(X, y, n_splits=3, random_state=42):
    """
    Обучает бейзлайн (Logistic Regression с L2) на всех числовых признаках (кроме таргета) с 3-fold CV.
    Логирует метрики (accuracy, ROC-AUC) и модель в MLflow.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("baseline_lr")
    model = LogisticRegression(
        penalty="l2", solver="liblinear", random_state=random_state
    )
    kf = KFold(n_splits=2, shuffle=True, random_state=random_state)
    accuracies, roc_aucs = [], []
    y_true_all, y_pred_proba_all = [], []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        acc = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        accuracies.append(acc)
        roc_aucs.append(auc)
        y_true_all.extend(y_val)
        y_pred_proba_all.extend(y_pred_proba)
        print(f"Fold {fold + 1} – Accuracy: {acc:.4f}, ROC-AUC: {auc:.4f}")
    avg_acc, avg_auc = np.mean(accuracies), np.mean(roc_aucs)
    print(
        f"Baseline LR (L2) – Average Accuracy: {avg_acc:.4f}, Average ROC-AUC: {avg_auc:.4f}"
    )
    with mlflow.start_run(run_name="baseline_lr_cv") as run:
        # Логируем git SHA
        repo = git.Repo(search_parent_directories=True)
        git_sha = repo.head.object.hexsha
        mlflow.set_tag("git_sha", git_sha)
        mlflow.log_metric("avg_accuracy", avg_acc)
        mlflow.log_metric("avg_roc_auc", avg_auc)
        mlflow.sklearn.log_model(model, "model")
        # ROC-кривая
        from sklearn.metrics import auc, roc_curve

        fpr, tpr, _ = roc_curve(y_true_all, y_pred_proba_all)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/baseline_lr_roc_auc.png")
        plt.close()
        mlflow.log_artifact("plots/baseline_lr_roc_auc.png")
        # Feature importance (коэффициенты)
        plt.figure()
        plt.bar(range(X.shape[1]), np.abs(model.coef_[0]))
        plt.title("Feature Importances (abs(coef))")
        plt.xlabel("Feature index")
        plt.ylabel("Importance")
        plt.savefig("plots/baseline_lr_feature_importance.png")
        plt.close()
        mlflow.log_artifact("plots/baseline_lr_feature_importance.png")
        # Learning curve (loss по эпохам не логируется, но можно accuracy по фолдам)
        plt.figure()
        plt.plot(
            range(1, len(accuracies) + 1), accuracies, marker="o", label="Accuracy"
        )
        plt.plot(range(1, len(roc_aucs) + 1), roc_aucs, marker="x", label="ROC-AUC")
        plt.xlabel("Fold")
        plt.ylabel("Score")
        plt.title("Learning Curve (CV folds)")
        plt.legend()
        plt.savefig("plots/baseline_lr_learning_curve.png")
        plt.close()
        mlflow.log_artifact("plots/baseline_lr_learning_curve.png")
    os.makedirs("models", exist_ok=True)
    mlflow.sklearn.save_model(model, "models/baseline_lr_model")
    return model


if __name__ == "__main__":
    download_data()
    X, y = load_processed_data()
    train_baseline_lr(X, y)
