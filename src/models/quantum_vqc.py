"""
Hybrid Variational Quantum Classifier (VQC).
Dressed quantum circuit: Linear → Tanh → AngleEmbedding → StronglyEntanglingLayers → Linear → Sigmoid.
Uses PennyLane's TorchLayer for end-to-end differentiable training with PyTorch.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml


def _set_seeds(seed=42):
    """Fix all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_vqc_model(n_qubits=5, n_layers=2, seed=42):
    """
    Build the dressed quantum circuit as a torch.nn.Sequential.
    
    Architecture:
        Linear(n_qubits, n_qubits) → Tanh
        → AngleEmbedding(Y) → StronglyEntanglingLayers → expval(PauliZ) per wire
        → Linear(n_qubits, 1) → Sigmoid
    
    Args:
        n_qubits (int): Number of qubits and PCA components
        n_layers (int): Number of StronglyEntanglingLayers
        seed (int): Random seed
    
    Returns:
        nn.Sequential: The full hybrid model
        qml.QNode: The quantum circuit (for drawing)
    """
    _set_seeds(seed)
    
    # Quantum device
    dev = qml.device("lightning.qubit", wires=n_qubits)
    
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    # Weight shapes for TorchLayer
    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    
    qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
    
    model = nn.Sequential(
        nn.Linear(n_qubits, n_qubits),
        nn.Tanh(),
        qlayer,
        nn.Linear(n_qubits, 1),
        nn.Sigmoid(),
    )
    
    return model, circuit


class HybridVQC:
    """
    High-level wrapper for the VQC with training and prediction methods.
    Provides the same interface as ClassicalModel for interoperability.
    """
    
    def __init__(self, n_qubits=5, n_layers=2, lr=0.01, epochs=30, seed=42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.train_time = 0.0
        self.loss_history = []
        self.is_fitted = False
        self.name = "Quantum VQC"
        
        _set_seeds(seed)
        self.model, self.qnode = create_vqc_model(n_qubits, n_layers, seed)
    
    def fit(self, X_train, y_train, progress_callback=None):
        """
        Train the VQC model.
        
        Args:
            X_train (np.ndarray): Training features (n_samples, n_qubits)
            y_train (np.ndarray): Binary labels
            progress_callback: Optional callable(epoch, loss) for Streamlit progress
        
        Returns:
            float: Training time in seconds
        """
        _set_seeds(self.seed)
        
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.BCELoss()
        
        self.loss_history = []
        self.model.train()
        
        start = time.time()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            y_pred = self.model(X_tensor)
            loss = loss_fn(y_pred, y_tensor)
            loss.backward()
            optimizer.step()
            
            loss_val = loss.item()
            self.loss_history.append(loss_val)
            
            if progress_callback:
                progress_callback(epoch, loss_val)
        
        self.train_time = time.time() - start
        self.is_fitted = True
        return self.train_time
    
    def predict_proba(self, X):
        """
        Return probability estimates as a 2-column array [P(0), P(1)].
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            proba_1 = self.model(X_tensor).numpy().flatten()
        proba_0 = 1.0 - proba_1
        return np.column_stack([proba_0, proba_1])
    
    def predict(self, X, threshold=0.5):
        """Return class predictions at the given threshold."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
    
    def get_state_dict(self):
        """Return model state dict for serialization."""
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load model weights from a state dict."""
        self.model.load_state_dict(state_dict)
        self.is_fitted = True
    
    def get_circuit_drawer(self):
        """Return (qnode, n_qubits, n_layers) for circuit visualization."""
        return self.qnode, self.n_qubits, self.n_layers
