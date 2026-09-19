"""
Preprocessing pipeline: impute → scale → feature selection → PCA → train/test split.
All transformers are returned for re-use in the Predict tab.
"""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split


def preprocess(X, y, n_qubits=5, test_size=0.2, random_state=42, k_features=15):
    """
    Full preprocessing pipeline.
    
    Args:
        X (np.ndarray): Raw feature matrix
        y (np.ndarray): Binary labels
        n_qubits (int): Number of PCA components (= number of qubits)
        test_size (float): Fraction reserved for testing
        random_state (int): Random seed for reproducibility
        k_features (int): Number of features to select before PCA
    
    Returns:
        dict with keys:
            X_train, X_test, y_train, y_test: processed splits
            pca_variance_ratio: explained variance ratio per component
            imputer, scaler, selector, pca: fitted sklearn transformers
            feature_names_pca: ['PC1', 'PC2', ..., 'PC{n_qubits}']
    """
    # Fix seed
    np.random.seed(random_state)
    
    # 1. Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # 2. Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # 3. Explicit Feature Selection
    # Select top k_features (or fewer if X has fewer features)
    k = min(k_features, X_scaled.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X_scaled, y)
    
    # 4. PCA to n_qubits dimensions
    n_components = min(n_qubits, X_selected.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_selected)
    
    # 5. Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    feature_names_pca = [f'PC{i+1}' for i in range(n_components)]
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'pca_variance_ratio': pca.explained_variance_ratio_,
        'imputer': imputer,
        'scaler': scaler,
        'selector': selector,
        'pca': pca,
        'feature_names_pca': feature_names_pca,
    }


def transform_new_data(X_new, imputer, scaler, selector, pca):
    """
    Apply the fitted preprocessing pipeline to new data (for prediction).
    
    Args:
        X_new (np.ndarray): Raw feature matrix (same shape as training features)
        imputer, scaler, selector, pca: Fitted transformers from preprocess()
    
    Returns:
        np.ndarray: Transformed data ready for model input
    """
    X_imputed = imputer.transform(X_new)
    X_scaled = scaler.transform(X_imputed)
    X_selected = selector.transform(X_scaled)
    X_pca = pca.transform(X_selected)
    return X_pca
