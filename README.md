# Credit Risk Prediction with MLOps

## Описание задачи

Проект посвящён задаче кредитного скоринга: предсказание вероятности дефолта клиента по табличным данным. Используются современные практики MLOps: DVC, Hydra, Optuna, MLflow, CatBoost, TabResNet, стекинг, экспорт моделей в ONNX, автоматизация через Makefile и docker-compose.

---

## Setup

1. Клонируйте репозиторий и перейдите в папку проекта:
   ```bash
   git clone <repo_url>
   cd <project_folder>
   ```
2. Установите Poetry и зависимости:
   ```bash
   poetry install
   ```
3. (Опционально) Установите DVC:
   ```bash
   poetry run pip install dvc
   ```
4. Настройте MLflow Tracking Server (по умолчанию http://127.0.0.1:8080):
   ```bash
   poetry run mlflow ui
   # или в отдельном терминале:
   make serve-mlflow
   ```

---

## Train (Обучение моделей)

1. Скачайте и предобработайте данные:
   ```bash
   poetry run python src/data/download.py
   poetry run python -m src.data.preprocessing
   ```
2. Запустите обучение бейзлайна (Logistic Regression):
   ```bash
   poetry run python src/models/baseline_lr.py
   ```
3. Запустите тюнинг CatBoost с Optuna:
   ```bash
   poetry run python -m src.models.train_cat
   # или через DVC:
   poetry run dvc repro tune_cat
   ```
4. Запустите обучение TabResNet:
   ```bash
   poetry run python -m src.models.train_tabresnet
   # или через DVC:
   poetry run dvc repro tune_tabresnet
   ```
5. Запустите стекинг:
   ```bash
   poetry run python -m src.models.train_stack
   # или через DVC:
   poetry run dvc repro tune_stack
   ```

---

## Production (Serving)

### MLflow Model Serving

Быстрый REST API для модели:
```bash
make serve-mlflow
# или вручную:
# poetry run mlflow models serve -m models/meta_model -h 0.0.0.0 -p 5001
```
- Сервер доступен по адресу: http://localhost:5001/invocations
- Пример запроса:
  ```bash
  curl -X POST -H "Content-Type: application/json" --data '{"data": [[0.1, 0.2]]}' http://localhost:5001/invocations
  ```

### Triton Inference Server

Production-ready инференс ONNX-моделей:
```bash
make serve-triton
# или вручную:
# docker-compose -f docker-compose.triton.yml up
```
- Triton доступен по портам 8000 (gRPC), 8001 (HTTP), 8002 (metrics).
- Модели должны быть размещены в папке `models/triton` в формате Triton.

---

## Infer (Инференс локально)

1. Получите фичи для инференса (например, sample.parquet):
2. Запустите скрипт инференса:
   ```bash
   poetry run python src/predict.py --features sample.parquet --output submission.csv
   ```
   или через entrypoint (если настроен):
   ```bash
   poetry run credit-risk predict --features sample.parquet
   ```
3. Результат будет сохранён в submission.csv

---

## Структура проекта

- `src/models/` — скрипты обучения моделей
- `src/data/` — загрузка и препроцессинг данных
- `src/tuning/` — Optuna Runner
- `src/predict.py` — инференс
- `configs/` — Hydra-конфиги
- `dvc.yaml` — DVC pipeline
- `Makefile` — автоматизация команд
- `docker-compose.triton.yml` — Triton Inference Server
- `plots/` — графики для MLflow
- `models/` — сохранённые модели

---

## MLflow UI

- Запустите MLflow UI:
  ```bash
  poetry run mlflow ui
  ```
- Откройте http://127.0.0.1:8080 в браузере для просмотра экспериментов, метрик, артефактов и графиков.

## Автор

Автор проекта: Могилев Георгий
