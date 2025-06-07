import json
import os
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import BinaryAUROC

from src.data.download import download_data


# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Dataset for tabular data
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# TabResNet-18 (1D) architecture
class BasicBlock1D(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class TabResNet1D(nn.Module):
    def __init__(self, input_dim, num_blocks, num_classes=1):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        # Динамически определяем размер выхода после layer4
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            out = F.relu(self.bn1(self.conv1(dummy)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out_dim = out.shape[1] * out.shape[2]
        self.fc = nn.Linear(out_dim, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock1D(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.flatten(1)
        out = self.fc(out)
        return out.squeeze(-1)


# LightningModule for training
class TabResNetLightning(pl.LightningModule):
    def __init__(self, input_dim, lr, weight_decay):
        super().__init__()
        self.model = TabResNet1D(input_dim, [2, 2, 2, 2])
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_auc = BinaryAUROC()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        preds = torch.sigmoid(logits)
        auc = self.val_auc(preds, y.long())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_auc", auc, prog_bar=True)
        return {"val_loss": loss, "val_auc": auc}

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


# Data loading and preprocessing
def load_processed_data():
    X = pd.read_csv("data/processed/X_processed.csv")
    y = pd.read_csv("data/processed/y_processed.csv").squeeze(axis=1)
    return X, y


def objective(trial, X, y, n_splits=5, random_state=42):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    max_epochs = 20
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        train_ds = TabularDataset(X_train, y_train)
        val_ds = TabularDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        model = TabResNetLightning(X.shape[1], lr, weight_decay)
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            enable_checkpointing=False,
            logger=False,
            enable_model_summary=False,
            accelerator="cpu",
            devices=1,
            enable_progress_bar=False,
        )
        trainer.fit(model, train_loader, val_loader)
        # Validation predictions
        model.eval()
        preds, targets = [], []
        for xb, yb in val_loader:
            with torch.no_grad():
                logits = model(xb)
                preds.append(torch.sigmoid(logits).cpu().numpy())
                targets.append(yb.cpu().numpy())
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        auc = roc_auc_score(targets, preds)
        aucs.append(auc)
    mean_auc = np.mean(aucs)
    trial.set_user_attr("mean_auc", mean_auc)
    return mean_auc


def train_tabresnet():
    set_seed(42)
    download_data()
    X, y = load_processed_data()
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("tabresnet_optimization")
    study = optuna.create_study(direction="maximize")
    with mlflow.start_run(run_name="tabresnet_optuna") as run:
        study.optimize(lambda trial: objective(trial, X, y), n_trials=30)
        best_params = study.best_params
        mlflow.log_params(best_params)
        # Train final model
        model = TabResNetLightning(
            X.shape[1], best_params["lr"], best_params["weight_decay"]
        )
        train_ds = TabularDataset(X, y)
        train_loader = DataLoader(
            train_ds, batch_size=best_params["batch_size"], shuffle=True
        )
        trainer = pl.Trainer(
            max_epochs=20,
            enable_checkpointing=False,
            logger=False,
            enable_model_summary=False,
            accelerator="cpu",
            devices=1,
        )
        trainer.fit(model, train_loader)
        # Save model
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/tabresnet_model.pt")
        mlflow.pytorch.log_model(model, "model")
        # Export to ONNX
        dummy_input = torch.randn(1, X.shape[1])
        torch.onnx.export(
            model.model,
            dummy_input.unsqueeze(1),
            "models/tabresnet_model.onnx",
            input_names=["input"],
            output_names=["output"],
            opset_version=12,
        )
        print(f"Best AUC: {study.best_value:.4f}")
        print(f"Best parameters: {best_params}")
        print(
            "Model saved to models/tabresnet_model.pt and models/tabresnet_model.onnx"
        )
        # Save metrics for DVC
        metrics = {"best_auc": float(study.best_value)}
        with open("metrics/tabresnet_metrics.json", "w") as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    train_tabresnet()
