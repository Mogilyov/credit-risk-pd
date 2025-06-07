# flake8: noqa: D103
import numpy as np
import pandas as pd
import pytest

from src.models.baseline_lr import train_baseline_lr


def test_baseline_lr_fit_predict():
    X = pd.DataFrame(np.random.randn(10, 3), columns=["f1", "f2", "f3"])
    y = pd.Series(np.random.randint(0, 2, size=10))
    model = train_baseline_lr(X, y, n_splits=2)
    preds = model.predict(X)
    assert len(preds) == len(y)
