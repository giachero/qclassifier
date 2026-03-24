"""Tests for VQCClassifier (interface & smoke tests only — no full training)."""

import numpy as np
import pytest
from qiskit_classifier.models import VQCClassifier


def test_classifier_instantiation():
    clf = VQCClassifier(num_qubits=4, max_iter=5)
    assert clf.num_qubits == 4
    assert clf.max_iter == 5


def test_predict_before_fit_raises():
    clf = VQCClassifier(num_qubits=2)
    X = np.random.rand(3, 2)
    with pytest.raises(RuntimeError, match="fit"):
        clf.predict(X)


def test_fit_returns_self():
    """fit() must return self for sklearn pipeline compatibility."""
    clf = VQCClassifier(num_qubits=2, feature_map_reps=1, ansatz_reps=1, max_iter=3)
    X = np.random.rand(6, 2) * np.pi
    y = np.array([0, 0, 0, 1, 1, 1])
    result = clf.fit(X, y)
    assert result is clf


def test_predict_shape():
    clf = VQCClassifier(num_qubits=2, feature_map_reps=1, ansatz_reps=1, max_iter=3)
    X = np.random.rand(6, 2) * np.pi
    y = np.array([0, 0, 0, 1, 1, 1])
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (6,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_score_returns_float():
    clf = VQCClassifier(num_qubits=2, feature_map_reps=1, ansatz_reps=1, max_iter=3)
    X = np.random.rand(6, 2) * np.pi
    y = np.array([0, 0, 0, 1, 1, 1])
    clf.fit(X, y)
    acc = clf.score(X, y)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
