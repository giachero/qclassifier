# Qiskit Classifier

A minimal **Variational Quantum Classifier (VQC)** built with [Qiskit](https://www.ibm.com/quantum/qiskit) and `qiskit-machine-learning`.  
The project is intentionally kept simple so it can serve as a clean starting point for quantum ML experiments.

---

## Project Structure

```
qiskit-classifier/
├── src/
│   └── qiskit_classifier/
│       ├── __init__.py
│       ├── train.py                # End-to-end training script
│       ├── circuits/
│       │   └── feature_map.py      # ZZFeatureMap + RealAmplitudes builders
│       ├── data/
│       │   └── loader.py           # Dataset loading & preprocessing
│       ├── models/
│       │   └── vqc_classifier.py   # Sklearn-compatible VQCClassifier
│       └── utils/
│           └── visualization.py    # Confusion matrix & circuit drawing
├── tests/
│   ├── test_circuits.py
│   ├── test_data.py
│   └── test_model.py
├── notebooks/
│   └── 01_exploration.ipynb        # (add your exploratory notebooks here)
├── docs/
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/yourusername/qiskit-classifier.git
cd qiskit-classifier

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
# or, without extras:
pip install -r requirements-dev.txt
```

### 2. Run the training script

```bash
python -m qiskit_classifier.train
```

### 3. Run tests

```bash
pytest
```

---

## How It Works

| Component | Qiskit primitive | Role |
|---|---|---|
| **Feature map** | `ZZFeatureMap` | Encodes classical data into quantum states |
| **Ansatz** | `RealAmplitudes` | Variational circuit whose parameters are learned |
| **Sampler** | `StatevectorSampler` | Executes circuits and returns measurement statistics |
| **Optimizer** | L-BFGS-B (default) | Updates parameters to minimise cross-entropy loss |

The `VQCClassifier` wraps everything in a scikit-learn–compatible `fit / predict / score` interface.

---

## Extending the Project

- **Custom dataset** → add a loader in `src/qiskit_classifier/data/`
- **Different feature map** → swap `ZZFeatureMap` in `circuits/feature_map.py`
- **Real hardware** → replace `StatevectorSampler` with a `QiskitRuntimeService` sampler
- **Hyperparameter search** → drop `VQCClassifier` into a `GridSearchCV` (it's sklearn-compatible)

---

## Requirements

- Python ≥ 3.10
- Qiskit ≥ 2.3
- qiskit-machine-learning ≥ 0.9
- qiskit-aer ≥ 0.17

---

## License

MIT
