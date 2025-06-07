from omegaconf import DictConfig, OmegaConf

import hydra


@hydra.main(config_path="../../configs", config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    print("\n=== HYDRA CONFIGURATION ===")
    print(OmegaConf.to_yaml(cfg))
    # Пример доступа к параметрам:
    print("\n--- Пример доступа к параметрам ---")
    print("CV folds:", cfg.experiment.cv.n_splits)
    print("Optuna trials:", cfg.experiment.optuna.n_trials)
    print("Model search space:", cfg.model.search_space)
    print("Preprocess config:", cfg.preprocess)
    print("Train config:", cfg.train)
    # Здесь можно вызвать препроцессинг, тюнинг, обучение, передав параметры из cfg


if __name__ == "__main__":
    main()
