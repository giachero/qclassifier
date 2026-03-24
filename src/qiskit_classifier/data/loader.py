"""Data loading and preprocessing utilities."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_binary_iris(
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a binarised version of the Iris dataset (classes 0 and 1 only).

    Features are scaled to [0, π] to be suitable as rotation angles in
    parametrised quantum circuits.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    iris = load_iris()
    # Keep only the first two classes (binary problem)
    mask = iris.target < 2
    X, y = iris.data[mask], iris.target[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
