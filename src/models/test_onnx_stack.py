import os

import numpy as np
import onnxruntime as ort
import pandas as pd

# Загрузка тестовых данных (берём первые 5 строк из data/processed/X_processed.csv и y_processed.csv)
X = pd.read_csv("data/processed/X_processed.csv")
y = pd.read_csv("data/processed/y_processed.csv").squeeze(axis=1)

# Для теста: создаём искусственные oof_cat и oof_tab (например, случайные или одинаковые)
oof_cat = np.random.rand(5)
oof_tab = np.random.rand(5)
X_meta = np.column_stack((oof_cat, oof_tab)).astype(np.float32)

# Загрузка ONNX-модели
onnx_path = "models/meta_model.onnx"
sess = ort.InferenceSession(onnx_path)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# ONNX-инференс
onnx_pred = sess.run([output_name], {input_name: X_meta})[0]
print("ONNX predict_proba (first 5):", onnx_pred[:5])

# (Опционально) Сравнить с predict_proba sklearn-модели, если есть
try:
    import joblib

    skl_model = joblib.load("models/meta_model.pkl")
    skl_pred = skl_model.predict_proba(X_meta)[:, 1]
    print("Sklearn predict_proba (first 5):", skl_pred[:5])
except Exception as e:
    print("Could not compare with sklearn model:", e)
