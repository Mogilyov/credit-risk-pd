# flake8: noqa: D103
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


def test_catboost_fit_predict():
    X = pd.DataFrame(np.random.randn(10, 3), columns=["f1", "f2", "f3"])
    y = pd.Series(np.random.randint(0, 2, size=10))
    model = CatBoostClassifier(iterations=5, verbose=False)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
