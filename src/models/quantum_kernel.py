"""
Quantum Support Vector Machine using a fidelity quantum kernel.
Kernel: AngleEmbedding(x1) → adjoint(AngleEmbedding)(x2) → P(|00...0⟩).
The kernel matrix is fed to sklearn SVC(kernel='precomputed').
"""

import os
import time
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pennylane as qml
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler


def create_kernel_circuit(n_qubits=5):
    """
    Build the fidelity kernel circuit.
    
    The kernel value k(x1, x2) = |⟨φ(x2)|φ(x1)⟩|² is computed as the
    probability of measuring the all-zeros state after:
        AngleEmbedding(x1) → adjoint(AngleEmbedding)(x2)
    
    Args:
        n_qubits (int): Number of qubits
    
Returns:
        callable: kernel function k(x1, x2) → float in [0, 1]
        qml.QNode: The underlying circuit (works for single AND batched inputs)
    """
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        qml.AngleEmbedding(x1, wires=range(n_qubits), rotation='Y')
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits), rotation='Y')
        return qml.probs(wires=range(n_qubits))

    def kernel(x1, x2):
        """Compute the fidelity kernel value between two samples."""
        return float(kernel_circuit(x1, x2)[0])

    return kernel, kernel_circuit


def compute_kernel_matrix(X1, X2, kernel_circuit):
    """
    Compute the kernel matrix between two sets of samples (vectorized).
    The full (n1, n2) grid is evaluated in a single batched circuit call.

    Args:
        X1 (np.ndarray): First set of samples (n1, d)
        X2 (np.ndarray): Second set of samples (n2, d)
        kernel_circuit: qnode computing k(x1, x2); accepts batched inputs

    Returns:
        np.ndarray: Kernel matrix of shape (n1, n2)
    """
    n1, n2 = len(X1), len(X2)
    if n1 == 0 or n2 == 0:
        return np.zeros((n1, n2))
    x1_exp = np.repeat(X1, n2, axis=0)
    x2_exp = np.tile(X2, (n1, 1))
    K = np.asarray(kernel_circuit(x1_exp, x2_exp)[:, 0]).reshape(n1, n2)
    return K


def compute_square_kernel_matrix(X, kernel_circuit):
    """
    Compute the symmetric kernel matrix for a single set (vectorized).
    The full (n, n) grid is evaluated in one batched circuit call, then
    symmetrized with the diagonal forced to 1 (fidelity kernel k(x,x)=1).

    Args:
        X (np.ndarray): Samples (n, d)
        kernel_circuit: qnode accepting batched inputs

    Returns:
        np.ndarray: Symmetric kernel matrix (n, n) with 1s on diagonal
    """
    n = len(X)
    if n == 0:
        return np.zeros((0, 0))
    x1_exp = np.repeat(X, n, axis=0)
    x2_exp = np.tile(X, (n, 1))
    K = np.asarray(kernel_circuit(x1_exp, x2_exp)[:, 0]).reshape(n, n)
    K = (K + K.T) / 2.0
    np.fill_diagonal(K, 1.0)
    return K


class QuantumKernelSVM:
    """
    QSVM using a fidelity quantum kernel.
    Provides the same interface as ClassicalModel.
    """
    
    def __init__(self, n_qubits=5, subsample=100, seed=42):
        self.n_qubits = n_qubits
        self.subsample = subsample
        self.seed = seed
        self.train_time = 0.0
        self.is_fitted = False
        self.name = "Quantum SVM (QSVM)"
        
        np.random.seed(seed)
        self.kernel_fn, self.kernel_circuit = create_kernel_circuit(n_qubits)
        self.svm = SVC(kernel='precomputed', random_state=seed)
        
        # Scale features to [0, π] for AngleEmbedding
        self.feature_scaler = MinMaxScaler(feature_range=(0, np.pi))
        
        # Stored for prediction
        self.X_train_sub = None
        self.K_train = None
        
        # Cache for test predictions to avoid redundant recomputation
        self._last_X_test_hash = None
        self._last_K_test = None

    def _get_K_test(self, X_test):
        """Helper to get K_test with simple caching."""
        # Simple hash using bytes of the array
        x_hash = hash(X_test.tobytes())
        if self._last_X_test_hash == x_hash and self._last_K_test is not None:
            return self._last_K_test

        X_test_scaled = self.feature_scaler.transform(X_test)
        K_test = compute_kernel_matrix(X_test_scaled, self.X_train_sub, self.kernel_circuit)

        self._last_X_test_hash = x_hash
        self._last_K_test = K_test
        return K_test

    def fit(self, X_train, y_train, progress_callback=None):
        """
        Build kernel matrix and train SVC.

        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Binary labels
            progress_callback: Optional callable(step, total) for progress

        Returns:
            float: Training time in seconds
        """
        np.random.seed(self.seed)
        start = time.time()

        # Scale features to [0, π] for optimal angle embedding
        X_train_scaled = self.feature_scaler.fit_transform(X_train)

        # Subsample if needed
        n = len(X_train_scaled)
        if self.subsample and n > self.subsample:
            idx = np.random.choice(n, self.subsample, replace=False)
            self.X_train_sub = X_train_scaled[idx]
            y_sub = y_train[idx]
        else:
            self.X_train_sub = X_train_scaled.copy()
            y_sub = y_train.copy()

        n_sub = len(self.X_train_sub)

        # Compute training kernel matrix (vectorized)
        total_pairs = n_sub * (n_sub - 1) // 2
        self.K_train = compute_square_kernel_matrix(self.X_train_sub, self.kernel_circuit)

        if progress_callback:
            progress_callback(total_pairs, total_pairs)

        # Fit SVM
        self.svm.fit(self.K_train, y_sub)
        self.train_time = time.time() - start
        self.is_fitted = True

        return self.train_time
    
    def predict(self, X_test):
        """Return class predictions for test data."""
        K_test = self._get_K_test(X_test)
        return self.svm.predict(K_test)
    
    def predict_proba(self, X_test):
        """
        Return approximate probability estimates using decision function + sigmoid.
        This avoids the expensive internal CV that SVC(probability=True) requires
        with precomputed kernels.
        """
        K_test = self._get_K_test(X_test)
        decision = self.svm.decision_function(K_test)
        # Sigmoid transform for probability estimates
        proba_1 = 1.0 / (1.0 + np.exp(-decision))
        proba_0 = 1.0 - proba_1
        return np.column_stack([proba_0, proba_1])
    
    def get_kernel_matrices(self):
        """Return stored kernel matrices for serialization."""
        return self.K_train, self.X_train_sub
