"""Reusable circuit building blocks: feature maps and ansätze."""

from __future__ import annotations

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes


def build_feature_map(num_features: int, reps: int = 2) -> ZZFeatureMap:
    """Return a ZZFeatureMap for *num_features* qubits.

    The ZZFeatureMap encodes classical data into quantum states via
    second-order Pauli interactions.

    Parameters
    ----------
    num_features:
        Number of input features (= number of qubits).
    reps:
        Number of repetition layers in the feature map.
    """
    return ZZFeatureMap(feature_dimension=num_features, reps=reps)


def build_ansatz(num_qubits: int, reps: int = 3) -> RealAmplitudes:
    """Return a RealAmplitudes ansatz for *num_qubits* qubits.

    RealAmplitudes uses only real-valued rotation gates, making it
    efficient for binary classification tasks.

    Parameters
    ----------
    num_qubits:
        Number of qubits (must match the feature map width).
    reps:
        Number of repetition layers in the variational form.
    """
    return RealAmplitudes(num_qubits=num_qubits, reps=reps)
