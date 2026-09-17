# 🎤 Demo Script — Hybrid Quantum ML Platform

**Duration**: 2–3 minutes  
**Audience**: SIH judges evaluating PS3 (SIH26139)

**Live Demo URL (Hugging Face Spaces):** `https://huggingface.co/spaces/<YOUR_HF_USERNAME>/hybrid-qml-disease-detection`  
**Firebase Demo:** *(TBD — future production deployment)*

> **Role Switcher:** Use the sidebar "View as" dropdown to switch between **Researcher** (full technical dashboard), **Clinician** (patient risk assessment), and **Admin** (system status) views during the demo.

---

## Opening (15 seconds)

> "We built a hybrid quantum-classical machine learning platform for early disease detection. It runs entirely offline, requires no quantum hardware, and benchmarks quantum-enhanced models against classical baselines on the same clinical data."

## Tab 1: Data Explorer (20 seconds)

> "We're using the Wisconsin Breast Cancer dataset — 569 patient samples with 30 clinical measurements. Our pipeline reduces these to 5 principal components via PCA, which map directly to our 5-qubit quantum circuits. The PCA captures about 85% of the original variance."

**Show**: Class distribution chart, PCA variance plot

## Tab 2: Training (30 seconds)

> "The platform supports two modes. In Quick Demo Mode, results load instantly from pre-computed checkpoints. In Live Training Mode, you can see the actual quantum circuit training in real time."

> "Here's the VQC — a dressed quantum circuit. Classical layers encode features, then 5 qubits with strongly entangling layers extract quantum features, and a final classical layer outputs disease probability. It trains in about 30 seconds on CPU."

**Show**: VQC loss curve, circuit diagrams for both VQC and QSVM

## Tab 3: Benchmark (40 seconds)

> "This is where it gets interesting. We benchmark 3 classical models against 2 quantum models on identical test data."

> "The quantum VQC achieves **95.6% accuracy** and **0.993 ROC-AUC** — the best of all 5 models. It outperforms Logistic Regression and SVM (both at 94.7%). The VQC's F1 score of 0.966 is also the highest."

> "The QSVM uses a fidelity quantum kernel — it computes patient similarity in quantum state space. It achieves 94.7% accuracy, matching the classical SVM. Notice the training time difference — the VQC takes 0.3s and QSVM takes 1.9s versus milliseconds for classical models. On a real quantum processor, this gap narrows significantly."

**Show**: Metrics comparison table (highlight best per metric), ROC curves, training time chart

## Tab 4: Predict — Decision Support (30 seconds)

> "For clinical use, raw accuracy isn't enough. Our Predict tab lets clinicians adjust the decision threshold to balance sensitivity versus specificity."

**Demo**: Move the threshold slider from 0.5 → 0.3

> "Lowering the threshold increases sensitivity — catching more true positives at the cost of more false alarms. For cancer screening, high sensitivity is critical. The confusion matrix and metrics update in real time."

> "Each patient gets a probability score mapped to a risk band — Low, Medium, or High — giving clinicians an actionable decision support signal."

**Show**: Threshold slider, confusion matrix updating, risk badge

## Tab 5: Explainability (20 seconds)

> "Finally, explainability. We use model-agnostic permutation importance — shuffling each feature and measuring accuracy drop. This works identically for classical and quantum models."

> "We also render the actual quantum circuits — judges can see exactly what's happening inside the quantum model. No black boxes."

**Show**: Permutation importance chart, circuit diagram

## Closing (15 seconds)

> "To summarize: this platform demonstrates that hybrid quantum-classical approaches can achieve competitive performance for early disease detection. It runs entirely on a CPU quantum simulator, supports any binary clinical dataset, and provides the explainability and decision support tools that clinicians need."

> "The code is modular — when real quantum hardware becomes accessible, the only change needed is swapping the device from `default.qubit` to a hardware backend."

---

## Anticipated Judge Questions

**Q: Why not use real quantum hardware?**
> A: Our platform uses PennyLane's `default.qubit` simulator for reliability and reproducibility. The architecture is hardware-agnostic — swapping to IBM or AWS Braket requires changing one line (`qml.device("default.qubit")` → `qml.device("qiskit.ibmq")`). For a 5-qubit circuit, the simulator is actually more practical than noisy near-term hardware.

**Q: Does quantum provide any advantage here?**
> A: On this dataset, quantum models achieve competitive (not superior) accuracy. The theoretical advantage emerges with (1) larger feature spaces where quantum kernels can capture classically-intractable patterns, and (2) real quantum hardware with enough qubits. Our platform is built to scale — increasing qubits is a slider change.

**Q: How does this handle new datasets?**
> A: Upload any CSV with a binary target column. The platform auto-detects features, applies PCA to match the qubit count, and trains all models from scratch.

**Q: What about multi-class problems?**
> A: The current version is binary classification, matching the malignant/benign cancer detection use case. Extending to multi-class requires one-vs-rest or modifying the VQC output layer — a straightforward next step.

**Q: Can this run offline?**
> A: Yes, 100%. The dataset is built into scikit-learn, models run on a local CPU simulator, and Quick Demo Mode uses cached results. Zero network calls.
