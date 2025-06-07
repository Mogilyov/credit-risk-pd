import json

import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import hydra


# Преобразует Hydra search_space в Optuna suggest-выражения
def get_search_space(search_space_cfg):
    def suggest(trial):
        params = {}
        for name, spec in search_space_cfg.items():
            if spec["type"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            elif spec["type"] == "float":
                params[name] = trial.suggest_float(
                    name, spec["low"], spec["high"], log=spec.get("log", False)
                )
            elif spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
        return params

    return suggest


def load_processed_data():
    X = pd.read_csv("data/processed/X_processed.csv")
    y = pd.read_csv("data/processed/y_processed.csv").squeeze(axis=1)
    return X, y


def catboost_objective(trial, cfg, X, y):
    params = get_search_space(cfg.model.search_space)(trial)
    params.update(
        {
            "iterations": 100,  # ускорим для поиска
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": cfg.experiment.cv.random_state,
            "verbose": False,
        }
    )
    kf = KFold(
        n_splits=cfg.experiment.cv.n_splits,
        shuffle=True,
        random_state=cfg.experiment.cv.random_state,
    )
    scores = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)
        model = CatBoostClassifier(**params)
        model.fit(
            train_pool, eval_set=val_pool, early_stopping_rounds=20, verbose=False
        )
        y_pred = model.predict_proba(val_pool)[:, 1]
        score = roc_auc_score(y_val, y_pred)
        scores.append(score)
    mean_score = sum(scores) / len(scores)
    return mean_score


def objective(trial, cfg, X, y):
    # Выбор модели
    model_name = (
        cfg.model.get("name", "catboost")
        if isinstance(cfg.model, dict)
        else str(cfg.model)
    )
    if model_name == "catboost":
        return catboost_objective(trial, cfg, X, y)
    elif model_name == "tabresnet":
        raise NotImplementedError("TabResNet objective not implemented yet.")
    else:
        raise ValueError(f"Unknown model: {model_name}")


@hydra.main(config_path="../../configs", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    print("Запуск Optuna Runner с конфигом:")
    print(OmegaConf.to_yaml(cfg))
    X, y = load_processed_data()
    study = optuna.create_study(direction=cfg.experiment.optuna.direction)
    study.optimize(
        lambda trial: objective(trial, cfg, X, y),
        n_trials=cfg.experiment.optuna.n_trials,
    )
    print("Best params:", study.best_params)
    # Сохраняем best-params
    with open("best_params.json", "w") as f:
        json.dump(study.best_params, f)


if __name__ == "__main__":
    main()
