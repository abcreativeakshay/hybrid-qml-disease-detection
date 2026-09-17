"""
Explainability module: permutation importance, SHAP explanations (classical),
quantum parameter sensitivity, plain-language explanation generator, and
quantum circuit visualization.

Model-agnostic — works for both classical and quantum models.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Streamlit
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def permutation_importance(predict_fn, X, y, feature_names=None, n_repeats=10, seed=42):
    """
    Compute permutation importance (model-agnostic).
    
    For each feature: shuffle the column, measure accuracy drop.
    Works identically for classical and quantum models since it only needs predict(X).
    
    Args:
        predict_fn: Callable that takes X (np.ndarray) and returns predictions (np.ndarray)
        X (np.ndarray): Test data (n_samples, n_features)
        y (np.ndarray): True labels
        feature_names (list, optional): Names for each feature
        n_repeats (int): Number of shuffle repeats per feature
        seed (int): Random seed
    
    Returns:
        list of dict: Sorted by importance descending.
            Each dict has keys: 'feature', 'importance_mean', 'importance_std'
    """
    rng = np.random.RandomState(seed)
    
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(X.shape[1])]
    
    # Baseline accuracy
    baseline_acc = accuracy_score(y, predict_fn(X))
    
    results = []
    for col in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[:, col] = rng.permutation(X_permuted[:, col])
            perm_acc = accuracy_score(y, predict_fn(X_permuted))
            scores.append(baseline_acc - perm_acc)
        
        results.append({
            'feature': feature_names[col],
            'importance_mean': float(np.mean(scores)),
            'importance_std': float(np.std(scores)),
        })
    
    # Sort by mean importance descending
    results.sort(key=lambda x: x['importance_mean'], reverse=True)
    return results


# ────────────────────────────────────────────────────
# SHAP explanations for classical models (FR-6.3)
# ────────────────────────────────────────────────────

def compute_shap_explanations(model_name, model, X_train, X_test, feature_names=None):
    """
    Compute SHAP feature importance for classical models.
    
    Uses the appropriate explainer per model type:
    - Random Forest → TreeExplainer
    - Logistic Regression → LinearExplainer
    - SVM (RBF) → KernelExplainer (with background subsample for speed)
    
    Args:
        model_name (str): Name of the model (used to select explainer type)
        model: ClassicalModel wrapper (has .get_model() method)
        X_train (np.ndarray): Training data (for background distribution)
        X_test (np.ndarray): Test data to explain
        feature_names (list, optional): Feature names for output
    
    Returns:
        dict with keys:
            'shap_values': np.ndarray of SHAP values (n_test_samples, n_features)
            'feature_names': list of feature names
            'mean_abs_shap': list of dicts [{feature, importance_mean}] sorted descending
    """
    import shap
    
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(X_test.shape[1])]
    
    sklearn_model = model.get_model()
    
    def _extract_shap_values(raw):
        """Extract numpy array from SHAP output (handles Explanation objects and lists)."""
        # SHAP >= 0.42 returns Explanation objects with .values attribute
        if hasattr(raw, 'values'):
            vals = raw.values
        elif isinstance(raw, list):
            # Older SHAP: [class0_shap, class1_shap]
            vals = raw[1] if len(raw) > 1 else raw[0]
            if hasattr(vals, 'values'):
                vals = vals.values
        else:
            vals = raw
        
        vals = np.array(vals, dtype=float)
        
        # Handle 3D output (n_samples, n_features, n_classes) → take class 1
        if vals.ndim == 3:
            vals = vals[:, :, 1]
        
        return vals
    
    if 'Random Forest' in model_name:
        explainer = shap.TreeExplainer(sklearn_model)
        shap_values_raw = explainer.shap_values(X_test)
        shap_vals = _extract_shap_values(shap_values_raw)
    elif 'Logistic' in model_name:
        explainer = shap.LinearExplainer(sklearn_model, X_train)
        shap_values_raw = explainer.shap_values(X_test)
        shap_vals = _extract_shap_values(shap_values_raw)
    else:
        # SVM or fallback: KernelExplainer (slower, subsample for speed)
        background = shap.sample(X_train, min(50, len(X_train)))
        
        def predict_proba_wrapper(X):
            return model.predict_proba(X)
        
        explainer = shap.KernelExplainer(predict_proba_wrapper, background)
        shap_values_raw = explainer.shap_values(X_test, nsamples=100)
        shap_vals = _extract_shap_values(shap_values_raw)
    
    # Ensure 2D
    if shap_vals.ndim == 1:
        shap_vals = shap_vals.reshape(1, -1)
    
    # Compute mean |SHAP| per feature
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    
    importance_list = []
    for i, fname in enumerate(feature_names):
        importance_list.append({
            'feature': fname,
            'importance_mean': float(mean_abs[i]),
        })
    importance_list.sort(key=lambda x: x['importance_mean'], reverse=True)
    
    return {
        'shap_values': shap_vals,
        'feature_names': feature_names,
        'mean_abs_shap': importance_list,
    }


# ────────────────────────────────────────────────────
# Quantum parameter sensitivity (FR-6.4)
# ────────────────────────────────────────────────────

def compute_quantum_sensitivity(predict_proba_fn, X_test, feature_names=None, delta=0.1):
    """
    Compute quantum parameter sensitivity by perturbing each PCA input feature.
    
    For each feature, perturb by ±delta, re-run inference, and measure how much
    the output probability shifts. This is NOT SHAP — it is explicitly labeled
    as "quantum parameter sensitivity" (per SRS 2.5, calling it "quantum SHAP"
    is a non-goal).
    
    Args:
        predict_proba_fn: Callable that takes X and returns 2-column probability array
        X_test (np.ndarray): Test data (n_samples, n_features)
        feature_names (list, optional): Names for each feature
        delta (float): Perturbation magnitude
    
    Returns:
        list of dict: Sorted by sensitivity descending.
            Each dict has keys: 'feature', 'sensitivity_mean', 'sensitivity_std'
    """
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(X_test.shape[1])]
    
    # Baseline probabilities (positive class)
    base_proba = predict_proba_fn(X_test)[:, 1]
    
    results = []
    for col in range(X_test.shape[1]):
        shifts = []
        for direction in [+delta, -delta]:
            X_perturbed = X_test.copy()
            X_perturbed[:, col] += direction
            perturbed_proba = predict_proba_fn(X_perturbed)[:, 1]
            shift = np.abs(perturbed_proba - base_proba)
            shifts.append(shift)
        
        # Average across both directions and all samples
        all_shifts = np.concatenate(shifts)
        results.append({
            'feature': feature_names[col],
            'sensitivity_mean': float(np.mean(all_shifts)),
            'sensitivity_std': float(np.std(all_shifts)),
        })
    
    results.sort(key=lambda x: x['sensitivity_mean'], reverse=True)
    return results


# ────────────────────────────────────────────────────
# Plain-language explanation generator (FR-6.1)
# ────────────────────────────────────────────────────

def generate_plain_language_explanation(importances, risk_label, feature_names_pca=None, top_k=3):
    """
    Generate a conservative, plain-language explanation sentence.
    
    Takes the top contributing features (from SHAP for classical or
    parameter-sensitivity for quantum) and produces one sentence suitable
    for the Clinician view. No diagnostic claims or certainty language.
    
    Args:
        importances (list of dict): Sorted list with 'feature' and 
            'importance_mean' or 'sensitivity_mean' keys
        risk_label (str): "Low", "Medium", or "High"
        feature_names_pca (list, optional): PCA feature names for context
        top_k (int): Number of top features to mention
    
    Returns:
        str: A single plain-language explanation sentence
    """
    # Extract top features
    top_features = importances[:min(top_k, len(importances))]
    
    # Get feature names from whichever key is present
    feature_list = []
    for f in top_features:
        feature_list.append(f['feature'])
    
    if len(feature_list) == 0:
        return "Insufficient data to generate an explanation."
    
    # Format the feature names nicely
    if len(feature_list) == 1:
        feature_str = feature_list[0]
    elif len(feature_list) == 2:
        feature_str = f"{feature_list[0]} and {feature_list[1]}"
    else:
        feature_str = ", ".join(feature_list[:-1]) + f", and {feature_list[-1]}"
    
    # Risk-specific phrasing
    risk_lower = risk_label.lower()
    if risk_lower == "high":
        context = "which are elevated relative to typical low-risk cases"
    elif risk_lower == "medium":
        context = "which are at levels requiring further evaluation"
    else:
        context = "which are within ranges typical of low-risk cases"
    
    return (
        f"This assessment is most influenced by {feature_str}, {context}. "
        f"This information is provided for decision-support purposes only and "
        f"does not constitute a clinical diagnosis."
    )


# ────────────────────────────────────────────────────
# Quantum circuit diagrams (unchanged)
# ────────────────────────────────────────────────────

def draw_quantum_circuit(n_qubits=5, n_layers=2):
    """
    Render the VQC quantum circuit as a matplotlib figure.
    
    Args:
        n_qubits (int): Number of qubits
        n_layers (int): Number of StronglyEntanglingLayers
    
    Returns:
        matplotlib.figure.Figure: The circuit diagram
    """
    import pennylane as qml
    
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    # Create dummy inputs for drawing
    dummy_inputs = np.zeros(n_qubits)
    dummy_weights = np.zeros((n_layers, n_qubits, 3))
    
    fig, ax = qml.draw_mpl(circuit, style="pennylane")(dummy_inputs, dummy_weights)
    fig.set_size_inches(12, max(3, n_qubits * 0.8))
    fig.suptitle("Variational Quantum Circuit (VQC)", fontsize=14, fontweight='bold')
    # plt.tight_layout()  # Removed to suppress Streamlit/Matplotlib warnings
    
    return fig


def draw_kernel_circuit(n_qubits=5):
    """
    Render the QSVM fidelity kernel circuit as a matplotlib figure.
    
    Args:
        n_qubits (int): Number of qubits
    
    Returns:
        matplotlib.figure.Figure: The circuit diagram
    """
    import pennylane as qml
    
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        qml.AngleEmbedding(x1, wires=range(n_qubits), rotation='Y')
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits), rotation='Y')
        return qml.probs(wires=range(n_qubits))
    
    dummy_x1 = np.zeros(n_qubits)
    dummy_x2 = np.zeros(n_qubits)
    
    fig, ax = qml.draw_mpl(kernel_circuit, style="pennylane")(dummy_x1, dummy_x2)
    fig.set_size_inches(10, max(3, n_qubits * 0.8))
    fig.suptitle("Quantum Fidelity Kernel Circuit (QSVM)", fontsize=14, fontweight='bold')
    # plt.tight_layout()  # Removed to suppress Streamlit/Matplotlib warnings
    
    return fig
