export interface FeatureImportance {
  feature: string;
  importance_mean: number;
  importance_std?: number;
  sensitivity_mean?: number;
  sensitivity_std?: number;
}

export interface MetricSet {
  accuracy: number;
  precision?: number;
  recall?: number;
  sensitivity?: number;
  specificity?: number;
  f1: number;
  roc_auc: number;
  confusion_matrix?: number[][];
  tp?: number;
  tn?: number;
  fp?: number;
  fn?: number;
  permutation_importance?: FeatureImportance[];
  shap_importance?: FeatureImportance[];
  quantum_sensitivity?: FeatureImportance[];
  fpr?: number[];
  tpr?: number[];
  thresholds?: number[];
}

export interface DatasetData {
  config: Record<string, any>;
  metrics: MetricsData;
  predictions: Record<string, number[]>;
  probabilities: Record<string, number[]>;
  train_times: Record<string, number>;
  inference_times: Record<string, number>;
  permutation_importance: Record<string, FeatureImportance[]>;
  shap_importance: Record<string, FeatureImportance[]>;
  quantum_sensitivity: Record<string, FeatureImportance[]>;
  vqc_loss_history: number[];
  dataset_info?: {
    name: string;
    n_samples: number;
    n_features_raw: number;
    n_features_pca: number;
    n_train: number;
    n_test: number;
    class_distribution: Record<string, number>;
    pca_variance_explained: number[];
  };
}

export interface MetricsData {
  "Logistic Regression": MetricSet;
  "Random Forest": MetricSet;
  "SVM (RBF)": MetricSet;
  "Quantum VQC": MetricSet;
  "Quantum SVM (QSVM)": MetricSet;
}

export type Role = 'Researcher' | 'Clinician' | 'Admin';

export type DatasetKey = 'breast_cancer' | 'heart_disease' | 'parkinsons';

export const DATASET_LABELS: Record<DatasetKey, string> = {
  breast_cancer: '🧬 Breast Cancer',
  heart_disease: '❤️ Heart Disease',
  parkinsons: '🧠 Parkinson\'s',
};

export const DATASET_TARGET_NAMES: Record<DatasetKey, [string, string]> = {
  breast_cancer: ['Benign', 'Malignant'],
  heart_disease: ['No Disease', 'Disease'],
  parkinsons: ['Healthy', 'Parkinson\'s'],
};
