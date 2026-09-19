"""
Data loader module.
Supports built-in breast cancer dataset and generic CSV upload.
"""

import numpy as np
import pandas as pd
import zipfile
import tempfile
import os
from sklearn.datasets import load_breast_cancer

try:
    import pydicom
except ImportError:
    pydicom = None

try:
    from Bio import SeqIO
    from Bio.SeqUtils import gc_fraction
except ImportError:
    SeqIO = None



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


def _extract_and_parse_zip(zip_path):
    """Helper to extract a zip and organize files by parent directory (label)."""
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
        
    file_map = {}
    labels_set = set()
    for root, _, files in os.walk(temp_dir):
        for f in files:
            if f.startswith('.') or f.startswith('__MACOSX'):
                continue # Skip hidden files
            
            # Use immediate parent directory as the class label
            label = os.path.basename(root)
            if label and label != os.path.basename(temp_dir):
                full_path = os.path.join(root, f)
                if label not in file_map:
                    file_map[label] = []
                file_map[label].append(full_path)
                labels_set.add(label)
                
    if len(labels_set) != 2:
        raise ValueError(f"ZIP must contain exactly 2 folders for binary classification. Found {len(labels_set)}: {labels_set}")
        
    return file_map, sorted(list(labels_set))


def load_dicom_zip(zip_path):
    """
    Extracts a ZIP of DICOM files organized by class folders.
    Extracts 1D image statistics to be used as features.
    """
    if pydicom is None:
        raise ImportError("pydicom is not installed.")
        
    file_map, target_names = _extract_and_parse_zip(zip_path)
    
    X_list = []
    y_list = []
    
    # Simple feature extraction from pixels
    feature_names = [
        'pixel_mean', 'pixel_std', 'pixel_max', 'pixel_min',
        'pixel_median', 'pixel_p25', 'pixel_p75'
    ]
    
    for label_idx, label in enumerate(target_names):
        for file_path in file_map[label]:
            try:
                ds = pydicom.dcmread(file_path, force=True)
                if not hasattr(ds, 'pixel_array'):
                    continue
                pixels = ds.pixel_array.astype(float).flatten()
                
                # Compute statistics
                f_mean = np.mean(pixels)
                f_std = np.std(pixels)
                f_max = np.max(pixels)
                f_min = np.min(pixels)
                f_median = np.median(pixels)
                f_p25 = np.percentile(pixels, 25)
                f_p75 = np.percentile(pixels, 75)
                
                X_list.append([f_mean, f_std, f_max, f_min, f_median, f_p25, f_p75])
                y_list.append(label_idx)
            except Exception as e:
                print(f"Failed to read DICOM {file_path}: {e}")
                
    if not X_list:
        raise ValueError("No valid DICOM files found in ZIP.")
        
    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int64)
    
    return X, y, feature_names, target_names


def load_fasta_zip(zip_path):
    """
    Extracts a ZIP of FASTA files organized by class folders.
    Extracts 1D genomic features (GC content, length, basic k-mer counts).
    """
    if SeqIO is None:
        raise ImportError("biopython is not installed.")
        
    file_map, target_names = _extract_and_parse_zip(zip_path)
    
    X_list = []
    y_list = []
    
    # 3-mer counting for key codons as a simple feature set
    kmers = ['ATG', 'TAA', 'TAG', 'TGA', 'AAA', 'CCC', 'GGG', 'TTT']
    feature_names = ['seq_length', 'gc_fraction'] + [f'count_{k}' for k in kmers]
    
    for label_idx, label in enumerate(target_names):
        for file_path in file_map[label]:
            try:
                for record in SeqIO.parse(file_path, "fasta"):
                    seq_str = str(record.seq).upper()
                    
                    f_len = len(seq_str)
                    f_gc = gc_fraction(seq_str)
                    
                    features = [f_len, f_gc]
                    for k in kmers:
                        features.append(seq_str.count(k))
                        
                    X_list.append(features)
                    y_list.append(label_idx)
            except Exception as e:
                print(f"Failed to read FASTA {file_path}: {e}")
                
    if not X_list:
        raise ValueError("No valid FASTA records found in ZIP.")
        
    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int64)
    
    return X, y, feature_names, target_names
