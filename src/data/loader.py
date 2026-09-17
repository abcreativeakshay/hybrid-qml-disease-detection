"""
Data loader module.
Supports built-in breast cancer dataset and generic CSV upload.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_default_dataset():
    """
    Load the sklearn breast cancer dataset.
    
    Returns:
        X (np.ndarray): Feature matrix (569, 30)
        y (np.ndarray): Binary labels (0=malignant, 1=benign)
        feature_names (list): Names of the 30 features
        target_names (list): ['malignant', 'benign']
    """
    data = load_breast_cancer()
    X = data.data.astype(np.float64)
    y = data.target.astype(np.int64)
    feature_names = list(data.feature_names)
    target_names = list(data.target_names)
    return X, y, feature_names, target_names


def load_csv_dataset(file_or_path, target_col=None):
    """
    Load a dataset from a CSV file or file-like object.
    
    Args:
        file_or_path: File path string or file-like object (e.g., Streamlit UploadedFile)
        target_col: Name of the target column. If None, uses the last column.
    
    Returns:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Binary labels (mapped to 0/1)
        feature_names (list): Column names used as features
        target_names (list): Unique class labels as strings
    
    Raises:
        ValueError: If the target column has more than 2 unique values
    """
    df = pd.read_csv(file_or_path)
    
    # Determine target column
    if target_col is None:
        target_col = df.columns[-1]
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")
    
    # Separate features and target
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values.astype(np.float64)
    
    # Map target to binary 0/1
    unique_labels = sorted(df[target_col].unique())
    if len(unique_labels) != 2:
        raise ValueError(
            f"Expected binary classification (2 classes), got {len(unique_labels)} "
            f"unique values in '{target_col}': {unique_labels}"
        )
    
    label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
    y = df[target_col].map(label_map).values.astype(np.int64)
    
    feature_names = list(feature_cols)
    target_names = [str(unique_labels[0]), str(unique_labels[1])]
    
    return X, y, feature_names, target_names
