# 🧬 Qiskit Classifier — Project Structure Guide

> An educational walkthrough of a Variational Quantum Classifier (VQC) project built with Qiskit.  
> This document explains **what each file does**, **why it exists**, and **how the pieces fit together**.

---

## 📚 Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [Big Picture: How a VQC Works](#2-big-picture-how-a-vqc-works)
3. [Project Layout](#3-project-layout)
4. [Layer by Layer: The Source Code](#4-layer-by-layer-the-source-code)
   - [4.1 data/ — Loading & Preprocessing](#41-data--loading--preprocessing)
   - [4.2 circuits/ — Quantum Building Blocks](#42-circuits--quantum-building-blocks)
   - [4.3 models/ — The Classifier](#43-models--the-classifier)
   - [4.4 utils/ — Visualisation](#44-utils--visualisation)
   - [4.5 train.py — Putting It All Together](#45-trainpy--putting-it-all-together)
5. [Tests](#5-tests)
6. [Packaging Files Explained](#6-packaging-files-explained)
   - [6.1 pyproject.toml](#61-pyprojecttoml)
   - [6.2 requirements.txt & requirements-dev.txt](#62-requirementstxt--requirements-devtxt)
7. [GitHub & CI](#7-github--ci)
8. [Data Flow Diagram](#8-data-flow-diagram)
9. [Key Concepts Glossary](#9-key-concepts-glossary)
10. [Where to Go Next](#10-where-to-go-next)

---

## 1. What Is This Project?

This project implements a **Variational Quantum Classifier (VQC)** — a hybrid quantum-classical machine learning algorithm that uses a parameterised quantum circuit to classify data.

It is intentionally minimal so that it can serve as a **learning scaffold**:

- Every directory has one clear responsibility.
- Every file is short and focused.
- The classifier is scikit-learn–compatible, so you can experiment with it using familiar tools.

The example task is **binary classification on the Iris dataset** (distinguishing between two species of iris flower), but the architecture is general-purpose.

---

## 2. Big Picture: How a VQC Works

A VQC is a **hybrid algorithm** — it combines quantum circuits with a classical optimiser.

```
Classical data (features)
        │
        ▼
┌───────────────────┐
│   Feature Map     │  ← Encodes data into a quantum state
│   (ZZFeatureMap)  │
└────────┬──────────┘
         │ quantum state
         ▼
┌───────────────────┐
│     Ansatz        │  ← Applies trainable rotations (parameters θ)
│  (RealAmplitudes) │
└────────┬──────────┘
         │ measurement
         ▼
┌───────────────────┐
│  Classical Loss   │  ← Computes cross-entropy between output and true labels
│  & Optimiser      │
└────────┬──────────┘
         │ updated θ
         └──────────────► repeat until convergence
```

**Training loop in plain English:**

1. Encode input features into a quantum state using the **feature map**.
2. Apply the **ansatz** — a circuit with tunable rotation angles (the "weights" of the model).
3. Measure the output qubits to get class probabilities.
4. Compute how wrong the prediction is (the **loss**).
5. A classical optimiser adjusts the rotation angles to reduce the loss.
6. Repeat steps 1–5 until the model converges.

---

## 3. Project Layout

```
qiskit-classifier/
│
├── src/                            # All installable source code lives here
│   └── qiskit_classifier/          # The Python package
│       ├── __init__.py             # Public API of the package
│       ├── train.py                # End-to-end training script
│       │
│       ├── data/                   # Data loading & preprocessing
│       │   ├── __init__.py
│       │   └── loader.py
│       │
│       ├── circuits/               # Quantum circuit building blocks
│       │   ├── __init__.py
│       │   └── feature_map.py
│       │
│       ├── models/                 # ML model wrappers
│       │   ├── __init__.py
│       │   └── vqc_classifier.py
│       │
│       └── utils/                  # Visualisation helpers
│           ├── __init__.py
│           └── visualization.py
│
├── tests/                          # Automated tests (mirrors src structure)
│   ├── __init__.py
│   ├── test_circuits.py
│   ├── test_data.py
│   └── test_model.py
│
├── notebooks/                      # Jupyter notebooks for exploration
│
├── docs/                           # Documentation (expand as needed)
│
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI pipeline
│
├── pyproject.toml                  # Package metadata + tool configuration
├── requirements.txt                # Runtime dependencies
├── requirements-dev.txt            # Dev + test dependencies
├── .gitignore                      # Files Git should ignore
├── LICENSE                         # MIT License
└── README.md                       # Project overview (start here)
```

> **Why `src/` layout?**  
> Placing the package inside `src/` prevents Python from accidentally importing your
> local code instead of the installed version during tests. It is the [recommended
> modern layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/).

---

## 4. Layer by Layer: The Source Code

### 4.1 `data/` — Loading & Preprocessing

**File:** `src/qiskit_classifier/data/loader.py`

This module is responsible for getting data into the shape the quantum circuit expects.

```python
def load_binary_iris(test_size=0.2, random_state=42):
    ...
```

**What it does, step by step:**

| Step | Operation | Why |
|------|-----------|-----|
| 1 | Load Iris dataset from scikit-learn | Iris is built-in — no download needed |
| 2 | Filter to classes 0 and 1 only | VQC in this project handles binary classification |
| 3 | Train/test split (80/20) | Standard ML practice to evaluate generalisation |
| 4 | Scale features to **[0, π]** | Rotation gates take angles; π is the natural range |

> **Why scale to [0, π]?**  
> Quantum rotation gates (like `Ry`) accept an angle in radians. Scaling your input
> features to this range means they map directly onto physically meaningful rotations,
> avoiding redundant circuit configurations.

---

### 4.2 `circuits/` — Quantum Building Blocks

**File:** `src/qiskit_classifier/circuits/feature_map.py`

This module provides factory functions that build the two main circuit components.

#### `build_feature_map(num_features, reps=2)` → `ZZFeatureMap`

The **feature map** is the encoding layer. It takes a classical vector **x** and
produces a quantum state |φ(x)⟩.

```
Input: x = [x₀, x₁, x₂, x₃]  (4 Iris features)
           │
     ┌─────▼──────────────────────────────┐
     │  ZZFeatureMap  (4 qubits, 2 reps)  │
     │                                    │
     │  H ── Rz(x₀) ──●── Rz(2·x₀·x₁) ──│── ...
     │  H ── Rz(x₁) ──┼──●── ...          │
     │  ...                               │
     └────────────────────────────────────┘
           │
     quantum state |φ(x)⟩
```

The `ZZFeatureMap` uses second-order Pauli interactions (the `ZZ` cross terms), which
can capture correlations between features — something a linear map cannot.

#### `build_ansatz(num_qubits, reps=3)` → `RealAmplitudes`

The **ansatz** is the trainable part of the circuit. It applies parameterised rotations
whose angles (θ) are updated by the classical optimiser.

```
     ┌────────────────────────────────────────────┐
     │  RealAmplitudes  (4 qubits, 3 reps)        │
     │                                            │
     │  Ry(θ₀) ──●────── Ry(θ₄) ──●────── ...   │
     │  Ry(θ₁) ──┼──●─── Ry(θ₅) ──┼──●── ...   │
     │  ...                                       │
     └────────────────────────────────────────────┘
           │
     measurement → class probabilities
```

`RealAmplitudes` uses only real-valued gates (no complex phases). This is computationally
efficient and sufficient for binary classification tasks.

> **Rule of thumb:** `feature_map.num_qubits` must always equal `ansatz.num_qubits`.
> Both must equal the number of input features. This constraint is enforced in `VQCClassifier`.

---

### 4.3 `models/` — The Classifier

**File:** `src/qiskit_classifier/models/vqc_classifier.py`

This is the core of the project. `VQCClassifier` wraps Qiskit's `VQC` algorithm in a
**scikit-learn–compatible interface** by inheriting from `BaseEstimator` and `ClassifierMixin`.

```python
class VQCClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y)    → self
    def predict(self, X)   → np.ndarray
    def score(self, X, y)  → float
```

**Why scikit-learn compatibility matters:**

Because `VQCClassifier` follows the sklearn contract, you can immediately use it with:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {"ansatz_reps": [1, 2, 3], "max_iter": [50, 100]}
search = GridSearchCV(VQCClassifier(num_qubits=4), param_grid, cv=3)
search.fit(X_train, y_train)
```

**Internal flow of `fit()`:**

```
fit(X, y)
  │
  ├─ build_feature_map(num_qubits)   ← from circuits/
  ├─ build_ansatz(num_qubits)        ← from circuits/
  ├─ create StatevectorSampler       ← exact quantum simulator
  └─ VQC.fit(X, y)                  ← Qiskit optimises θ via L-BFGS-B
```

> **StatevectorSampler vs real hardware:**  
> `StatevectorSampler` simulates the quantum circuit exactly on your CPU.
> To run on real IBM Quantum hardware, swap it for a `QiskitRuntimeService` sampler —
> the rest of the code stays the same.

---

### 4.4 `utils/` — Visualisation

**File:** `src/qiskit_classifier/utils/visualization.py`

Two helper functions:

| Function | Output |
|----------|--------|
| `plot_confusion_matrix(y_true, y_pred, labels)` | Matplotlib figure of the confusion matrix |
| `draw_circuit(circuit, filename)` | Renders a Qiskit circuit diagram |

These are intentionally kept simple so you can replace them with your preferred plotting library without touching any model code.

---

### 4.5 `train.py` — Putting It All Together

**File:** `src/qiskit_classifier/train.py`

This script is the **glue layer** — it calls each module in sequence and demonstrates
the full end-to-end workflow.

```python
# 1. Load data
X_train, X_test, y_train, y_test = load_binary_iris()

# 2. Build and train the classifier
clf = VQCClassifier(num_qubits=4, max_iter=100)
clf.fit(X_train, y_train)

# 3. Evaluate
print(clf.score(X_test, y_test))

# 4. Visualise
plot_confusion_matrix(y_test, clf.predict(X_test))
```

Run it with:

```bash
python -m qiskit_classifier.train
```

The `-m` flag tells Python to run the module as a script, which works regardless of
your current directory (as long as the package is installed with `pip install -e .`).

---

## 5. Tests

The `tests/` directory mirrors the structure of `src/`, with one test file per module.

| Test file | What it covers |
|-----------|----------------|
| `test_data.py` | Shape of train/test splits, label values, feature scaling range |
| `test_circuits.py` | Correct qubit counts, `reps` propagation, feature map ↔ ansatz compatibility |
| `test_model.py` | sklearn interface (fit returns self, predict shape, score range), error before fit |

**Design philosophy:** these tests are **interface tests**, not integration tests.
They verify that the API contract holds without running an expensive full training loop.
Full training is tested manually (or in a separate slow-test suite).

Run all tests:

```bash
pytest                          # run everything
pytest -v                       # verbose output
pytest tests/test_data.py       # single file
pytest --cov=qiskit_classifier  # with coverage report
```

---

## 6. Packaging Files Explained

### 6.1 `pyproject.toml`

This single file replaces `setup.py`, `setup.cfg`, and individual config files for
`pytest`, `ruff`, and `mypy`. It is the modern Python standard ([PEP 517](https://peps.python.org/pep-0517/)/[518](https://peps.python.org/pep-0518/)).

```toml
[build-system]          # tells pip HOW to build the package
[project]               # metadata: name, version, description, authors
[project.dependencies]  # runtime packages (what users need)
[project.optional-dependencies]
  dev = [...]           # testing + linting tools (what developers need)
  notebooks = [...]     # Jupyter, ipywidgets
[tool.ruff]             # linting configuration
[tool.pytest.ini_options]  # test configuration
[tool.mypy]             # type checking configuration
```

### 6.2 `requirements.txt` & `requirements-dev.txt`

These exist alongside `pyproject.toml` for compatibility with tools that prefer
plain requirements files (Docker, some CI systems, legacy tooling).

| File | Used by |
|------|---------|
| `requirements.txt` | Production deployments, Docker `RUN pip install -r requirements.txt` |
| `requirements-dev.txt` | Developers: starts with `-r requirements.txt` then adds dev tools |
| `pyproject.toml` `[project.dependencies]` | `pip install -e .` and `pip install qiskit-classifier` |

> **They should always be kept in sync.** If you add a dependency to one, add it to all three.

---

## 7. GitHub & CI

### `.gitignore`

Tells Git which files to **never track**. Key entries for this project:

- `__pycache__/`, `*.pyc` — compiled Python bytecode (machine-specific, regenerated automatically)
- `.venv/` — virtual environment (never commit this; it's huge and machine-specific)
- `.coverage`, `htmlcov/` — test coverage reports (generated, not source)
- `qiskit-ibm.json` — IBM Quantum credentials (secret! never commit this)
- `*.ipynb_checkpoints/` — Jupyter autosave files

### `.github/workflows/ci.yml`

Every time you push to `main` or open a pull request, GitHub automatically:

1. **Lints** the code with `ruff` (fast Python linter)
2. **Tests** on Python 3.10, 3.11, and 3.12 in parallel
3. **Uploads** coverage data to Codecov

```
Push / PR
   │
   ├── lint job ────────── ruff check src/ tests/
   │
   └── test job (matrix)
         ├── Python 3.10 ── pytest --cov
         ├── Python 3.11 ── pytest --cov
         └── Python 3.12 ── pytest --cov
                                │
                          codecov upload
```

This ensures that every change is automatically validated, and that the project works
across all supported Python versions.

---

## 8. Data Flow Diagram

Here is how data moves through the entire system at training time:

```
sklearn Iris dataset
        │
        ▼
load_binary_iris()              [data/loader.py]
  - filter to 2 classes
  - train/test split
  - scale to [0, π]
        │
        ▼
X_train (shape: N×4)
y_train (shape: N,)
        │
        ▼
VQCClassifier.fit()             [models/vqc_classifier.py]
  │
  ├─ build_feature_map(4)       [circuits/feature_map.py]
  │     → ZZFeatureMap (4 qubits, 2 reps)
  │
  ├─ build_ansatz(4)            [circuits/feature_map.py]
  │     → RealAmplitudes (4 qubits, 3 reps)
  │
  └─ VQC(sampler, feature_map, ansatz)
        │
        ▼
     For each training sample:
        encode x → |φ(x)⟩ → apply ansatz(θ) → measure → p(class)
        compute loss → update θ via L-BFGS-B
        │
        ▼
     Converged model (trained θ values stored in VQC object)
        │
        ▼
VQCClassifier.predict(X_test)
        │
        ▼
plot_confusion_matrix()         [utils/visualization.py]
```

---

## 9. Key Concepts Glossary

| Term | Meaning |
|------|---------|
| **Qubit** | The quantum equivalent of a bit; can exist in a superposition of 0 and 1 |
| **Feature map** | A circuit that encodes classical data into a quantum state |
| **Ansatz** | A parameterised quantum circuit whose angles are the "weights" to be trained |
| **ZZFeatureMap** | A specific feature map using second-order Pauli-Z interactions |
| **RealAmplitudes** | An ansatz that uses only real-valued Ry rotation gates and CNOT entanglement |
| **Sampler** | A Qiskit primitive that runs a circuit and returns measurement statistics |
| **StatevectorSampler** | An exact (noiseless) simulator of a quantum circuit, runs on CPU |
| **L-BFGS-B** | A quasi-Newton optimiser used to update the ansatz parameters |
| **VQC** | Variational Quantum Classifier — the full algorithm combining the above |
| **PEP 517/518** | Python Enhancement Proposals that define the `pyproject.toml` build standard |
| **src layout** | A project structure where the package lives inside a `src/` directory |

---

## 10. Where to Go Next

Once you are comfortable with this structure, here are natural extensions to explore:

**Extend the data layer**
- Add `load_from_csv()` in `data/loader.py` to train on custom datasets
- Add a multi-class loader (one-vs-rest strategy)

**Experiment with the circuits**
- Try `PauliFeatureMap` or `EfficientSU2` as alternatives
- Add noise models to simulate real hardware

**Improve the model**
- Try different optimisers (`COBYLA`, `SPSA`) — useful for noisy hardware
- Add cross-validation using sklearn's `GridSearchCV`

**Run on real hardware**
- Replace `StatevectorSampler` with `QiskitRuntimeService` and a real IBM backend
- Add error mitigation techniques from `qiskit-ibm-runtime`

**Expand the tests**
- Add a slow test suite for end-to-end training
- Use `pytest-benchmark` to track circuit execution time

---

*This document was written to accompany the `qiskit-classifier` starter project.*  
*Feel free to adapt it for your own quantum ML experiments.*
