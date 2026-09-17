"""
Classical ML baselines with a uniform interface.
All models support fit, predict, and predict_proba.
"""

import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class ClassicalModel:
    """Wrapper providing a uniform interface for sklearn classifiers."""
    
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.train_time = 0.0
        self.is_fitted = False
    
    def fit(self, X_train, y_train):
        """
        Train the model and record wall-clock time.
        
        Returns:
            float: Training time in seconds
        """
        start = time.time()
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - start
        self.is_fitted = True
        return self.train_time
    
    def predict(self, X):
        """Return class predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Return probability estimates as a 2-column array [P(class=0), P(class=1)].
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Fallback for models without predict_proba
            decision = self.model.decision_function(X)
            # Sigmoid transform
            proba_1 = 1.0 / (1.0 + np.exp(-decision))
            proba_0 = 1.0 - proba_1
            return np.column_stack([proba_0, proba_1])
    
    def get_model(self):
        """Return the underlying sklearn estimator."""
        return self.model


def get_classical_models(random_state=42):
    """
    Factory function returning a dict of classical baseline models.
    
    Returns:
        dict: {name: ClassicalModel} for LR, RF, SVM
    """
    models = {
        'Logistic Regression': ClassicalModel(
            'Logistic Regression',
            LogisticRegression(max_iter=1000, random_state=random_state)
        ),
        'Random Forest': ClassicalModel(
            'Random Forest',
            RandomForestClassifier(n_estimators=100, random_state=random_state)
        ),
        'SVM (RBF)': ClassicalModel(
            'SVM (RBF)',
            SVC(kernel='rbf', probability=True, random_state=random_state)
        ),
    }
    return models
