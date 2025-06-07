import json
import os
import sys
from pathlib import Path

import git
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import onnx
import onnxmltools
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from onnxmltools.convert import convert_catboost
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import KFold

from src.data.download import download_data

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def get_git_sha():
    """Get current git commit SHA."""
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha


def load_processed_data():
    """Load processed data (X, y) from data/processed."""
    X = pd.read_csv("data/processed/X_processed.csv")
    y = pd.read_csv("data/processed/y_processed.csv").squeeze(axis=1)
    return X, y


def objective(trial, X, y, n_splits=5, random_state=42):
    """
    Optuna objective function for CatBoost hyperparameter optimization.

    Parameters to optimize:
    - depth: [4-10]
    - learning_rate: [1e-3, 0.3] (log scale)
    - l2_leaf_reg: [1-10]
    """
    # Define hyperparameter search space
    params = {
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "iterations": 1000,  # Fixed number of iterations
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": random_state,
        "verbose": False,
    }

    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Store scores for each fold
    scores = []

    # Perform k-fold cross validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Create CatBoost pools
        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)

        # Train model
        model = CatBoostClassifier(**params)
        model.fit(
            train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False
        )

        # Get predictions and score
        y_pred = model.predict_proba(val_pool)[:, 1]
        score = roc_auc_score(y_val, y_pred)
        scores.append(score)

        # Log metrics for this fold
        mlflow.log_metric(f"fold_{fold}_roc_auc", score)

    # Calculate mean score across folds
    mean_score = np.mean(scores)

    # Log hyperparameters and mean score
    mlflow.log_params(params)
    mlflow.log_metric("mean_roc_auc", mean_score)

    return mean_score


def train_catboost(X, y):
    """Train CatBoost model with Optuna optimization and log to MLflow."""
    # Initialize MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("catboost_optimization")

    # Load data

    # Create study
    study = optuna.create_study(direction="maximize")

    # Start MLflow run
    with mlflow.start_run(run_name="catboost_optuna") as run:
        # Log git SHA
        git_sha = get_git_sha()
        mlflow.log_param("git_sha", git_sha)

        # Optimize hyperparameters
        study.optimize(lambda trial: objective(trial, X, y), n_trials=1)

        # Get best parameters and train final model
        best_params = study.best_params
        best_params.update(
            {
                "iterations": 1000,
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "random_seed": 42,
                "verbose": False,
            }
        )

        # Train final model on full dataset
        final_model = CatBoostClassifier(**best_params)
        train_pool = Pool(X, y)
        final_model.fit(train_pool, verbose=False)

        # Save model in CatBoost format
        os.makedirs("models", exist_ok=True)
        model_path = "models/catboost_model.cbm"
        final_model.save_model(model_path)

        # Convert to ONNX
        onnx_model = convert_catboost(
            final_model,
            name="catboost_model",
            initial_types=[("float_input", FloatTensorType([None, X.shape[1]]))],
        )
        onnx_path = "models/catboost_model.onnx"
        onnx.save(onnx_model, onnx_path)

        # Log artifacts
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(onnx_path)

        # Log best parameters and score
        mlflow.log_params(best_params)
        mlflow.log_metric("best_roc_auc", study.best_value)

        os.makedirs("metrics", exist_ok=True)
        with open("metrics/catboost_metrics.json", "w") as f:
            json.dump({"best_roc_auc": float(study.best_value)}, f)

        print(f"Best ROC-AUC: {study.best_value:.4f}")
        print(f"Best parameters: {best_params}")
        print(f"Model saved to {model_path}")
        print(f"ONNX model saved to {onnx_path}")

        # Plot ROC curve
        y_pred = final_model.predict_proba(train_pool)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], lw=2, linestyle="--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/catboost_roc_auc.png")
        plt.close()
        mlflow.log_artifact("plots/catboost_roc_auc.png")
        # Feature importance
        importances = final_model.get_feature_importance()
        plt.figure()
        plt.bar(range(len(importances)), importances)
        plt.title("Feature Importances")
        plt.savefig("plots/catboost_feature_importance.png")
        plt.close()
        mlflow.log_artifact("plots/catboost_feature_importance.png")
        # Learning curve (loss по итерациям)
        if hasattr(final_model, "get_evals_result"):
            evals = final_model.get_evals_result()
            if evals and "learn" in evals and "Logloss" in evals["learn"]:
                plt.figure()
                plt.plot(evals["learn"]["Logloss"], label="Train Logloss")
                plt.title("Learning Curve (Logloss)")
                plt.xlabel("Iteration")
                plt.ylabel("Logloss")
                plt.legend()
                plt.savefig("plots/catboost_learning_curve.png")
                plt.close()
                mlflow.log_artifact("plots/catboost_learning_curve.png")


if __name__ == "__main__":
    download_data()
    X, y = load_processed_data()
    train_catboost(X, y)
