"""Tests for circuit building blocks."""

import pytest
from qiskit_classifier.circuits import build_feature_map, build_ansatz


def test_feature_map_num_qubits():
    for n in [2, 4, 6]:
        fm = build_feature_map(n)
        assert fm.num_qubits == n


def test_feature_map_reps():
    fm = build_feature_map(num_features=4, reps=3)
    assert fm.reps == 3


def test_ansatz_num_qubits():
    for n in [2, 4]:
        ans = build_ansatz(n)
        assert ans.num_qubits == n


def test_ansatz_reps():
    ans = build_ansatz(num_qubits=4, reps=2)
    assert ans.reps == 2


def test_feature_map_ansatz_compatible():
    """Feature map and ansatz must share the same qubit width."""
    n = 4
    fm = build_feature_map(n)
    ans = build_ansatz(n)
    assert fm.num_qubits == ans.num_qubits
