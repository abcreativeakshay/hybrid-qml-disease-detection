import os
import json
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.data.loader import load_default_dataset, load_csv_dataset, load_dicom_zip, load_fasta_zip
from src.data.preprocessing import preprocess
from src.evaluation.metrics import compute_metrics, time_inference
from src.evaluation.explainability import permutation_importance, compute_quantum_sensitivity

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

class DummyObj:
    def progress(self, *a, **k):
        print("Progress:", k.get('text', a[1] if len(a)>1 else ''))
    def text(self, *a, **k):
        print("Status:", a[0] if a else '')
    def empty(self, *a, **k):
        pass

def train_all_models(processed_data, n_qubits, n_layers, vqc_epochs, qsvm_subsample):
    from src.models.classical import get_classical_models
    from src.models.quantum_vqc import HybridVQC
    from src.models.quantum_kernel import QuantumKernelSVM
    from src.evaluation.explainability import compute_shap_explanations

    X_train = processed_data['X_train']
    X_test = processed_data['X_test']
    y_train = processed_data['y_train']
    y_test = processed_data['y_test']
    
    all_results = {
        'metrics': {},
        'predictions': {},
        'probabilities': {},
        'train_times': {},
        'inference_times': {},
        'permutation_importance': {},
        'shap_importance': {},
        'quantum_sensitivity': {},
        'vqc_loss_history': [],
        'models': {},
    }
    
    classical_models = get_classical_models()
    
    for i, (name, model) in enumerate(classical_models.items()):
        print(f"Training {name}...")
        t = model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = compute_metrics(y_test, y_pred, y_proba)
        inf_time = time_inference(model.predict, X_test)
        
        all_results['metrics'][name] = metrics
        all_results['predictions'][name] = y_pred.tolist()
        all_results['probabilities'][name] = y_proba.tolist()
        all_results['train_times'][name] = t
        all_results['inference_times'][name] = inf_time
        all_results['models'][name] = model
        
        pi = permutation_importance(
            model.predict, X_test, y_test,
            feature_names=processed_data['feature_names_pca'],
            n_repeats=10
        )
        all_results['permutation_importance'][name] = pi
        
        try:
            shap_result = compute_shap_explanations(
                name, model, X_train, X_test,
                feature_names=processed_data['feature_names_pca']
            )
            all_results['shap_importance'][name] = shap_result['mean_abs_shap']
        except Exception:
            all_results['shap_importance'][name] = []
            
    print("Training Quantum VQC...")
    vqc = HybridVQC(n_qubits=n_qubits, n_layers=n_layers, lr=0.01, epochs=vqc_epochs)
    
    def vqc_cb(epoch, loss):
        if (epoch + 1) % 2 == 0:
            print(f"VQC Epoch {epoch+1}/{vqc_epochs} — Loss: {loss:.4f}")
            
    t = vqc.fit(X_train, y_train, progress_callback=vqc_cb)
    y_pred = vqc.predict(X_test)
    y_proba = vqc.predict_proba(X_test)[:, 1]
    
    all_results['metrics']['Quantum VQC'] = compute_metrics(y_test, y_pred, y_proba)
    all_results['predictions']['Quantum VQC'] = y_pred.tolist()
    all_results['probabilities']['Quantum VQC'] = y_proba.tolist()
    all_results['train_times']['Quantum VQC'] = t
    all_results['inference_times']['Quantum VQC'] = time_inference(vqc.predict, X_test)
    all_results['vqc_loss_history'] = vqc.loss_history
    all_results['models']['Quantum VQC'] = vqc
    
    all_results['permutation_importance']['Quantum VQC'] = permutation_importance(
        vqc.predict, X_test, y_test,
        feature_names=processed_data['feature_names_pca'],
        n_repeats=10
    )
    all_results['quantum_sensitivity']['Quantum VQC'] = compute_quantum_sensitivity(
        vqc.predict_proba, X_test,
        feature_names=processed_data['feature_names_pca'], delta=0.1
    )
    
    print(f"Training Quantum SVM (subsample={qsvm_subsample})...")
    qsvm = QuantumKernelSVM(n_qubits=n_qubits, subsample=qsvm_subsample)
    
    def qsvm_cb(step, total):
        pass
        
    t = qsvm.fit(X_train, y_train, progress_callback=qsvm_cb)
    y_pred = qsvm.predict(X_test)
    y_proba = qsvm.predict_proba(X_test)[:, 1]

    all_results['metrics']['Quantum SVM (QSVM)'] = compute_metrics(y_test, y_pred, y_proba)
    all_results['predictions']['Quantum SVM (QSVM)'] = y_pred.tolist()
    all_results['probabilities']['Quantum SVM (QSVM)'] = y_proba.tolist()
    all_results['train_times']['Quantum SVM (QSVM)'] = t
    all_results['inference_times']['Quantum SVM (QSVM)'] = time_inference(qsvm.predict, X_test)
    all_results['models']['Quantum SVM (QSVM)'] = qsvm

    all_results['permutation_importance']['Quantum SVM (QSVM)'] = permutation_importance(
        qsvm.predict, X_test, y_test,
        feature_names=processed_data['feature_names_pca'],
        n_repeats=10
    )
    all_results['quantum_sensitivity']['Quantum SVM (QSVM)'] = compute_quantum_sensitivity(
        qsvm.predict_proba, X_test,
        feature_names=processed_data['feature_names_pca'], delta=0.1
    )
    
    return all_results


def run_precompute():
    datasets = {
        'sample_dicom': lambda: load_dicom_zip('src/data/datasets/sample_dicom.zip'),
        'sample_fasta': lambda: load_fasta_zip('src/data/datasets/sample_fasta.zip')
    }
    
    n_qubits = 5
    n_layers = 2
    vqc_epochs = 30
    qsvm_subsample = 100
    
    for name, loader_func in datasets.items():
        print(f"\n======================================")
        print(f"Precomputing {name}...")
        print(f"======================================")
        
        path = os.path.join(RESULTS_DIR, name)
        os.makedirs(path, exist_ok=True)
        
        # Load and preprocess
        X, y, feature_names, target_names = loader_func()
        data = preprocess(X, y, n_qubits=n_qubits)
        
        # Save preprocessing
        preprocessing_dict = {
            'imputer': data['imputer'],
            'scaler': data['scaler'],
            'selector': data['selector'],
            'pca': data['pca'],
            'feature_names_pca': data['feature_names_pca'],
            'target_names': target_names,
        }
        joblib.dump(preprocessing_dict, os.path.join(path, 'preprocessing.pkl'))
        
        np.savez(os.path.join(path, 'test_data.npz'),
                 X_train=data['X_train'], X_test=data['X_test'],
                 y_train=data['y_train'], y_test=data['y_test'])
                 
        # Train
        results = train_all_models(data, n_qubits, n_layers, vqc_epochs, qsvm_subsample)
        
        # Remove un-serializable models from JSON payload
        del results['models']
        
        with open(os.path.join(path, 'metrics.json'), 'w') as f:
            json.dump(results, f, indent=4)
            
        print(f"Finished {name}. Saved to {path}\n")

if __name__ == '__main__':
    run_precompute()
