# flake8: noqa: D103
import numpy as np
from sklearn.linear_model import LogisticRegression


def test_stack_fit_predict():
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, size=10)
    model = LogisticRegression().fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
