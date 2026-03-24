"""Tests for data loading."""

import numpy as np
import pytest
from qiskit_classifier.data import load_binary_iris


def test_load_binary_iris_shapes():
    X_train, X_test, y_train, y_test = load_binary_iris()
    assert X_train.ndim == 2
    assert X_test.ndim == 2
    assert X_train.shape[1] == 4  # Iris has 4 features
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]


def test_binary_labels():
    _, _, y_train, y_test = load_binary_iris()
    assert set(np.unique(y_train)).issubset({0, 1})
    assert set(np.unique(y_test)).issubset({0, 1})


def test_feature_range():
    """All features should be scaled to [0, π]."""
    import numpy as np

    X_train, X_test, _, _ = load_binary_iris()
    assert X_train.min() >= -1e-9
    assert X_train.max() <= np.pi + 1e-9
