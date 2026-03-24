"""Visualisation helpers."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str] | None = None) -> plt.Figure:
    """Plot a simple confusion matrix."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    return fig


def draw_circuit(circuit: QuantumCircuit, filename: str | None = None) -> None:
    """Draw a Qiskit circuit using matplotlib and optionally save it."""
    fig = circuit.draw(output="mpl")
    if filename:
        fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
