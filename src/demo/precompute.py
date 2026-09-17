"""
Pre-compute all model results for Quick Demo Mode.
Run: python -m src.demo.precompute

Saves trained models, metrics, predictions, kernel matrices,
SHAP explanations, and quantum sensitivity to results/.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import joblib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.loader import load_default_dataset
from src.data.preprocessing import preprocess
from src.models.classical import get_classical_models
from src.models.quantum_vqc import HybridVQC
from src.models.quantum_kernel import QuantumKernelSVM
from src.evaluation.metrics import compute_metrics, time_inference, compute_metrics_at_threshold
from src.evaluation.explainability import (
    permutation_importance, compute_shap_explanations,
    compute_quantum_sensitivity, generate_plain_language_explanation
)


# Configuration
CONFIG = {
    'n_qubits': 5,
    'n_layers': 2,
    'vqc_epochs': 30,
    'vqc_lr': 0.01,
    'qsvm_subsample': 100,
    'test_size': 0.2,
    'random_state': 42,
}


def main():
    print("=" * 60)
    print("Hybrid QML Platform — Pre-computing Demo Results")
    print("=" * 60)
    
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'results'
    )
    os.makedirs(results_dir, exist_ok=True)
    
    # ── 1. Load and preprocess data ──
    print("\n[1/7] Loading and preprocessing data...")
    X_raw, y, feature_names, target_names = load_default_dataset()
    data = preprocess(X_raw, y, n_qubits=CONFIG['n_qubits'],
                      test_size=CONFIG['test_size'],
                      random_state=CONFIG['random_state'])
    
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"   PCA variance explained: {sum(data['pca_variance_ratio']):.2%}")
    
    # Save preprocessing objects
    joblib.dump({
        'imputer': data['imputer'],
        'scaler': data['scaler'],
        'pca': data['pca'],
        'feature_names': feature_names,
        'target_names': target_names,
        'feature_names_pca': data['feature_names_pca'],
        'pca_variance_ratio': data['pca_variance_ratio'].tolist(),
    }, os.path.join(results_dir, 'preprocessing.pkl'))
    
    # Save test data
    np.savez(os.path.join(results_dir, 'test_data.npz'),
             X_train=X_train, X_test=X_test,
             y_train=y_train, y_test=y_test)
    
    all_metrics = {}
    all_predictions = {}
    all_probabilities = {}
    train_times = {}
    inference_times = {}
    
    # ── 2. Train classical models ──
    print("\n[2/7] Training classical models...")
    classical_models = get_classical_models(random_state=CONFIG['random_state'])
    
    for name, model in classical_models.items():
        t = model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = compute_metrics(y_test, y_pred, y_proba)
        inf_time = time_inference(model.predict, X_test)
        
        all_metrics[name] = metrics
        all_predictions[name] = y_pred.tolist()
        all_probabilities[name] = y_proba.tolist()
        train_times[name] = t
        inference_times[name] = inf_time
        
        print(f"   {name}: acc={metrics['accuracy']:.3f}, "
              f"F1={metrics['f1']:.3f}, AUC={metrics['roc_auc']:.3f}, "
              f"train={t:.4f}s")
    
    # Save classical models
    joblib.dump(classical_models, os.path.join(results_dir, 'classical_models.pkl'))
    
    # ── 3. Train VQC ──
    print(f"\n[3/7] Training Quantum VQC ({CONFIG['vqc_epochs']} epochs)...")
    vqc = HybridVQC(
        n_qubits=CONFIG['n_qubits'],
        n_layers=CONFIG['n_layers'],
        lr=CONFIG['vqc_lr'],
        epochs=CONFIG['vqc_epochs'],
        seed=CONFIG['random_state'],
    )
    
    def vqc_progress(epoch, loss):
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{CONFIG['vqc_epochs']}: loss={loss:.4f}")
    
    t = vqc.fit(X_train, y_train, progress_callback=vqc_progress)
    y_pred = vqc.predict(X_test)
    y_proba = vqc.predict_proba(X_test)[:, 1]
    
    metrics = compute_metrics(y_test, y_pred, y_proba)
    inf_time = time_inference(vqc.predict, X_test)
    
    all_metrics['Quantum VQC'] = metrics
    all_predictions['Quantum VQC'] = y_pred.tolist()
    all_probabilities['Quantum VQC'] = y_proba.tolist()
    train_times['Quantum VQC'] = t
    inference_times['Quantum VQC'] = inf_time
    
    print(f"   VQC: acc={metrics['accuracy']:.3f}, "
          f"F1={metrics['f1']:.3f}, AUC={metrics['roc_auc']:.3f}, "
          f"train={t:.1f}s")
    
    # Save VQC state
    torch.save({
        'state_dict': vqc.get_state_dict(),
        'loss_history': vqc.loss_history,
        'config': {
            'n_qubits': CONFIG['n_qubits'],
            'n_layers': CONFIG['n_layers'],
        }
    }, os.path.join(results_dir, 'vqc_state.pt'))
    
    # ── 4. Train QSVM ──
    print(f"\n[4/7] Training Quantum SVM (subsample={CONFIG['qsvm_subsample']})...")
    qsvm = QuantumKernelSVM(
        n_qubits=CONFIG['n_qubits'],
        subsample=CONFIG['qsvm_subsample'],
        seed=CONFIG['random_state'],
    )
    
    def qsvm_progress(step, total):
        pct = step / total * 100
        if pct % 20 < (100 / total * 100):
            print(f"   Kernel matrix: {pct:.0f}%")
    
    t = qsvm.fit(X_train, y_train, progress_callback=qsvm_progress)
    y_pred = qsvm.predict(X_test)
    y_proba = qsvm.predict_proba(X_test)[:, 1]
    
    metrics = compute_metrics(y_test, y_pred, y_proba)
    inf_time = time_inference(qsvm.predict, X_test)
    
    all_metrics['Quantum SVM (QSVM)'] = metrics
    all_predictions['Quantum SVM (QSVM)'] = y_pred.tolist()
    all_probabilities['Quantum SVM (QSVM)'] = y_proba.tolist()
    train_times['Quantum SVM (QSVM)'] = t
    inference_times['Quantum SVM (QSVM)'] = inf_time
    
    print(f"   QSVM: acc={metrics['accuracy']:.3f}, "
          f"F1={metrics['f1']:.3f}, AUC={metrics['roc_auc']:.3f}, "
          f"train={t:.1f}s")
    
    # Save QSVM kernel matrices
    K_train, X_train_sub = qsvm.get_kernel_matrices()
    np.savez(os.path.join(results_dir, 'qsvm_kernel.npz'),
             K_train=K_train, X_train_sub=X_train_sub)
    # Note: QSVM object contains closures that can't be pickled.
    # Kernel matrices are saved in qsvm_kernel.npz for demo mode.
    
    # ── 5. Compute permutation importance ──
    print("\n[5/7] Computing permutation importance...")
    perm_importance = {}
    
    for name, model in classical_models.items():
        pi = permutation_importance(
            model.predict, X_test, y_test,
            feature_names=data['feature_names_pca'],
            n_repeats=10, seed=CONFIG['random_state']
        )
        perm_importance[name] = pi
        print(f"   {name}: top feature = {pi[0]['feature']} ({pi[0]['importance_mean']:.4f})")
    
    # VQC permutation importance
    pi = permutation_importance(
        vqc.predict, X_test, y_test,
        feature_names=data['feature_names_pca'],
        n_repeats=10, seed=CONFIG['random_state']
    )
    perm_importance['Quantum VQC'] = pi
    print(f"   VQC: top feature = {pi[0]['feature']} ({pi[0]['importance_mean']:.4f})")
    
    # QSVM permutation importance
    pi = permutation_importance(
        qsvm.predict, X_test, y_test,
        feature_names=data['feature_names_pca'],
        n_repeats=10, seed=CONFIG['random_state']
    )
    perm_importance['Quantum SVM (QSVM)'] = pi
    print(f"   QSVM: top feature = {pi[0]['feature']} ({pi[0]['importance_mean']:.4f})")
    
    # ── 6. Compute SHAP + quantum sensitivity ──
    print("\n[6/7] Computing SHAP explanations and quantum sensitivity...")
    shap_importance = {}
    quantum_sensitivity = {}
    
    # SHAP for classical models
    for name, model in classical_models.items():
        print(f"   SHAP for {name}...")
        try:
            shap_result = compute_shap_explanations(
                name, model, X_train, X_test,
                feature_names=data['feature_names_pca']
            )
            # Store only the serializable mean_abs_shap (not the full shap_values array)
            shap_importance[name] = shap_result['mean_abs_shap']
            print(f"     Top SHAP feature: {shap_result['mean_abs_shap'][0]['feature']} "
                  f"({shap_result['mean_abs_shap'][0]['importance_mean']:.4f})")
        except Exception as e:
            print(f"     SHAP failed for {name}: {e}")
            shap_importance[name] = []
    
    # Quantum parameter sensitivity
    print("   Quantum sensitivity for VQC...")
    qs_vqc = compute_quantum_sensitivity(
        vqc.predict_proba, X_test,
        feature_names=data['feature_names_pca'], delta=0.1
    )
    quantum_sensitivity['Quantum VQC'] = qs_vqc
    print(f"     Top sensitive feature: {qs_vqc[0]['feature']} ({qs_vqc[0]['sensitivity_mean']:.4f})")
    
    print("   Quantum sensitivity for QSVM...")
    qs_qsvm = compute_quantum_sensitivity(
        qsvm.predict_proba, X_test,
        feature_names=data['feature_names_pca'], delta=0.1
    )
    quantum_sensitivity['Quantum SVM (QSVM)'] = qs_qsvm
    print(f"     Top sensitive feature: {qs_qsvm[0]['feature']} ({qs_qsvm[0]['sensitivity_mean']:.4f})")
    
    # ── 7. Save all results ──
    print("\n[7/7] Saving results...")
    
    # Metrics summary
    results_data = {
        'config': CONFIG,
        'metrics': all_metrics,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'train_times': train_times,
        'inference_times': inference_times,
        'permutation_importance': perm_importance,
        'shap_importance': shap_importance,
        'quantum_sensitivity': quantum_sensitivity,
        'vqc_loss_history': vqc.loss_history,
        'dataset_info': {
            'name': 'Breast Cancer Wisconsin',
            'n_samples': len(y),
            'n_features_raw': X_raw.shape[1],
            'n_features_pca': CONFIG['n_qubits'],
            'n_train': len(y_train),
            'n_test': len(y_test),
            'class_distribution': {
                target_names[0]: int((y == 0).sum()),
                target_names[1]: int((y == 1).sum()),
            },
            'pca_variance_explained': data['pca_variance_ratio'].tolist(),
        }
    }
    
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    # Save config separately for easy access
    with open(os.path.join(results_dir, 'config.json'), 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    # ── Summary ──
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'Accuracy':>8} {'F1':>8} {'ROC-AUC':>8} {'Train(s)':>10}")
    print("-" * 60)
    for name in all_metrics:
        m = all_metrics[name]
        print(f"{name:<25} {m['accuracy']:>8.3f} {m['f1']:>8.3f} "
              f"{m['roc_auc']:>8.3f} {train_times[name]:>10.3f}")
    print("-" * 60)
    print(f"\nResults saved to: {results_dir}/")
    print("Files: metrics.json, config.json, preprocessing.pkl,")
    print("       vqc_state.pt, qsvm_kernel.npz,")
    print("       classical_models.pkl, test_data.npz")
    print("\nNew in this version:")
    print("  - SHAP importance for classical models (in metrics.json)")
    print("  - Quantum parameter sensitivity (in metrics.json)")
    print("\nReady for Quick Demo Mode! Run: streamlit run app.py")


if __name__ == '__main__':
    main()
