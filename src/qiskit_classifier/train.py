"""End-to-end training script.

Usage
-----
    python -m qiskit_classifier.train
"""

from __future__ import annotations

from qiskit_classifier.data import load_binary_iris
from qiskit_classifier.models import VQCClassifier
from qiskit_classifier.utils import plot_confusion_matrix


def main() -> None:
    print("Loading data …")
    X_train, X_test, y_train, y_test = load_binary_iris()

    num_features = X_train.shape[1]
    print(f"Features: {num_features}, Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    print("Building and training VQC …")
    clf = VQCClassifier(num_qubits=num_features, max_iter=100)
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test  accuracy: {test_acc:.3f}")

    fig = plot_confusion_matrix(y_test, clf.predict(X_test), labels=["Class 0", "Class 1"])
    fig.savefig("confusion_matrix.png", dpi=150)
    print("Confusion matrix saved to confusion_matrix.png")


if __name__ == "__main__":
    main()
