"""Variational Quantum Classifier (VQC) model."""

from __future__ import annotations

import numpy as np
from qiskit_machine_learning.algorithms import VQC
from qiskit_aer import AerSimulator
from qiskit.primitives import StatevectorSampler
from sklearn.base import BaseEstimator, ClassifierMixin

from qiskit_classifier.circuits.feature_map import build_ansatz, build_feature_map


class VQCClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn–compatible wrapper around Qiskit's VQC.

    Parameters
    ----------
    num_qubits:
        Number of qubits. Must equal the number of input features.
    feature_map_reps:
        Repetitions of the ZZFeatureMap encoding layer.
    ansatz_reps:
        Repetitions of the RealAmplitudes variational layer.
    max_iter:
        Maximum optimiser iterations.
    """

    def __init__(
        self,
        num_qubits: int = 4,
        feature_map_reps: int = 2,
        ansatz_reps: int = 3,
        max_iter: int = 100,
    ) -> None:
        self.num_qubits = num_qubits
        self.feature_map_reps = feature_map_reps
        self.ansatz_reps = ansatz_reps
        self.max_iter = max_iter
        self._vqc: VQC | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_vqc(self) -> VQC:
        feature_map = build_feature_map(self.num_qubits, self.feature_map_reps)
        ansatz = build_ansatz(self.num_qubits, self.ansatz_reps)
        sampler = StatevectorSampler()
        return VQC(
            sampler=sampler,
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer={"maxiter": self.max_iter},
        )

    # ------------------------------------------------------------------
    # Scikit-learn interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VQCClassifier":
        """Train the VQC on labelled data."""
        self._vqc = self._build_vqc()
        self._vqc.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for *X*."""
        if self._vqc is None:
            raise RuntimeError("Call fit() before predict().")
        return self._vqc.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy on the given test data."""
        return float(np.mean(self.predict(X) == y))
