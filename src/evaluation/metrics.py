"""
Metrics module: accuracy, precision, recall, specificity, F1, ROC-AUC,
confusion matrix, cross-validation, threshold-based evaluation, and timing.
"""

import time
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold


def compute_metrics(y_true, y_pred, y_proba=None):
    """
    Compute a full suite of classification metrics.
    
    Args:
        y_true (np.ndarray): Ground truth labels
        y_pred (np.ndarray): Predicted labels
        y_proba (np.ndarray, optional): Probability of positive class (1D)
    
    Returns:
        dict: All metrics including accuracy, precision, recall (sensitivity),
              specificity, f1, roc_auc, confusion_matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),  # sensitivity
        'sensitivity': recall_score(y_true, y_pred, zero_division=0),
        'specificity': specificity,
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': cm.tolist(),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }
    
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['roc_auc'] = 0.0
    else:
        metrics['roc_auc'] = 0.0
    
    return metrics


def compute_metrics_at_threshold(y_true, y_proba, threshold=0.5):
    """
    Compute metrics at a specific probability threshold.
    Used by the threshold slider in the Predict tab.
    
    Args:
        y_true (np.ndarray): Ground truth labels
        y_proba (np.ndarray): Probability of positive class (1D)
        threshold (float): Classification threshold
    
    Returns:
        dict: Metrics at the given threshold, including y_pred
    """
    y_pred = (y_proba >= threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_proba)
    metrics['threshold'] = threshold
    metrics['y_pred'] = y_pred.tolist()
    return metrics


def compute_cv_scores(model_class, X, y, cv=5, random_state=42, **model_kwargs):
    """
    Stratified K-Fold cross-validation for classical models.
    
    Args:
        model_class: Callable that returns a fitted model with .predict and .predict_proba
        X (np.ndarray): Full feature set
        y (np.ndarray): Full label set
        cv (int): Number of folds
        random_state (int): Random seed
    
    Returns:
        dict: {metric: {'mean': float, 'std': float}} for accuracy, f1, roc_auc
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    scores = {'accuracy': [], 'f1': [], 'roc_auc': []}
    
    for train_idx, val_idx in skf.split(X, y):
        X_t, X_v = X[train_idx], X[val_idx]
        y_t, y_v = y[train_idx], y[val_idx]
        
        model = model_class(**model_kwargs)
        model.fit(X_t, y_t)
        
        y_pred = model.predict(X_v)
        
        scores['accuracy'].append(accuracy_score(y_v, y_pred))
        scores['f1'].append(f1_score(y_v, y_pred, zero_division=0))
        
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_v)[:, 1]
                scores['roc_auc'].append(roc_auc_score(y_v, y_proba))
            except (ValueError, IndexError):
                scores['roc_auc'].append(0.0)
        else:
            scores['roc_auc'].append(0.0)
    
    results = {}
    for metric, vals in scores.items():
        results[metric] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
        }
    
    return results


def time_inference(predict_fn, X, n_runs=5):
    """
    Measure average inference time.
    
    Args:
        predict_fn: Callable that takes X and returns predictions
        X (np.ndarray): Input data
        n_runs (int): Number of runs to average
    
    Returns:
        float: Average inference time in seconds
    """
    times = []
    for _ in range(n_runs):
        start = time.time()
        predict_fn(X)
        times.append(time.time() - start)
    return float(np.mean(times))
