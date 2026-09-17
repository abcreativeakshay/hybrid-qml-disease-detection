---
title: Hybrid Quantum ML Platform — Disease Detection
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# 🧬 Hybrid Quantum-Classical ML Platform for Early Disease Detection

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hybrid-qml-disease-detection.streamlit.app)

**SIH26139 — Problem Statement 3: Hybrid Quantum Machine Learning Platform for Early Disease Detection**

A comprehensive platform that benchmarks quantum-enhanced machine learning models against classical baselines for binary disease classification. Built with PennyLane, PyTorch, scikit-learn, and Streamlit.

> **Live demo:** deploy this repo to [Streamlit Cloud](https://share.streamlit.io/) — no training required (Quick Demo Mode loads pre-computed results instantly).

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the dashboard (Quick Demo Mode — instant results)
streamlit run app.py

# 3. (Optional) Regenerate pre-computed results
python -m src.demo.precompute
```

---

## 🏗️ Architecture

```
Streamlit UI (5 tabs)
   └─ Data Layer: clean → impute → StandardScaler → PCA(n_qubits)
        ├─ Classical Baselines:
        │     ├─ Logistic Regression
        │     ├─ Random Forest (100 trees)
        │     └─ SVM (RBF kernel)
        └─ Quantum Models:
              ├─ Variational Quantum Classifier (VQC):
              │     Linear → Tanh → AngleEmbedding(Y) → StronglyEntanglingLayers
              │     → expval(PauliZ) → Linear → Sigmoid
              │     Trained end-to-end with Adam optimizer + BCELoss
              └─ Quantum SVM (QSVM):
                    Fidelity kernel: AngleEmbedding(x₁) → adjoint(AngleEmbedding)(x₂)
                    → P(|00...0⟩) = kernel value
                    Kernel matrix → sklearn SVC(kernel='precomputed')
   └─ Evaluation: accuracy, precision, recall, specificity, F1, ROC-AUC, confusion matrix
   └─ Explainability: permutation importance + quantum circuit diagrams
   └─ Decision Support: probability → risk band + adjustable threshold slider
```

## 📂 Project Structure

```
hybrid-qml-disease-detection/
├── app.py                          # Streamlit dashboard (5 tabs)
├── src/
│   ├── data/
│   │   ├── loader.py               # Breast cancer + CSV data loader
│   │   └── preprocessing.py        # Impute → Scale → PCA pipeline
│   ├── models/
│   │   ├── classical.py            # LR, RF, SVC baselines
│   │   ├── quantum_vqc.py          # Dressed quantum circuit (PennyLane + PyTorch)
│   │   └── quantum_kernel.py       # Fidelity kernel QSVM
│   ├── evaluation/
│   │   ├── metrics.py              # All metrics + CV + timing
│   │   └── explainability.py       # Permutation importance + circuit diagrams
│   └── demo/
│       └── precompute.py           # Generates results/ for Quick Demo Mode
├── results/                        # Pre-computed metrics/checkpoints
├── requirements.txt
├── README.md
└── DEMO_SCRIPT.md
```

## 🔬 Tech Stack

| Component | Technology |
|-----------|-----------|
| Quantum Circuits | PennyLane (default.qubit simulator) |
| Hybrid Model | PyTorch + PennyLane TorchLayer |
| Classical ML | scikit-learn |
| Dashboard | Streamlit |
| Visualizations | Plotly + Matplotlib (circuit diagrams) |

## 📊 Dataset

**Default**: Wisconsin Breast Cancer Dataset (sklearn built-in)
- 569 samples, 30 features, binary classification (malignant/benign)
- Fully offline — no network calls required

**Custom**: Upload any CSV with a binary target column via the dashboard sidebar.

## ⚙️ Configuration

All configurable via the Streamlit sidebar:

| Parameter | Default | Range |
|-----------|---------|-------|
| Qubits (PCA components) | 5 | 4–8 |
| VQC Layers | 2 | 1–4 |
| VQC Epochs | 30 | 10–100 |
| QSVM Subsample | 100 | 50–300 |

## 🏃 Modes

### Quick Demo Mode (Default)
Loads pre-computed results from `results/` instantly. No training required.
Toggle on in the sidebar.

### Live Training Mode
Trains all models from scratch. Classical models train in < 1s, VQC in ~30-90s, QSVM in ~60-120s depending on CPU.

## 🔄 Regenerating Results

```bash
python -m src.demo.precompute
```

This runs all 5 models on the breast cancer dataset, computes metrics, permutation importance, and saves everything to `results/`.

## 📋 Requirements

- Python 3.11+
- No GPU required (runs on CPU simulator)
- No internet required at demo time
- ~500MB disk space (mostly PyTorch)

## 🏆 Key Features

1. **Quantum-Classical Benchmarking**: Side-by-side comparison on identical data
2. **Interactive Threshold Slider**: Live-updates sensitivity/specificity and confusion matrix
3. **Explainability**: Model-agnostic permutation importance for all models
4. **Quantum Circuit Visualization**: Auto-rendered VQC and kernel circuit diagrams
5. **Risk Bands**: Low/Medium/High probability-based risk assessment
6. **Quick Demo Mode**: Pre-computed results for reliable live demos
7. **CSV Upload**: Generic binary classification beyond breast cancer

---

*Built for AICTE Smart India Hackathon 2026 — Problem Statement 3 (SIH26139)*
