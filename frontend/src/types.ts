export interface FeatureImportance {
  feature: string;
  importance_mean: number;
}

export interface MetricSet {
  accuracy: number;
  f1: number;
  roc_auc: number;
  permutation_importance?: FeatureImportance[];
  shap_importance?: FeatureImportance[];
  quantum_sensitivity?: FeatureImportance[];
  fpr?: number[];
  tpr?: number[];
  thresholds?: number[];
}

export interface RootData {
  config: Record<string, any>;
  metrics: MetricsData;
}

export interface MetricsData {
  "Logistic Regression": MetricSet;
  "Random Forest": MetricSet;
  "SVM (RBF)": MetricSet;
  "Quantum VQC": MetricSet;
  "Quantum SVM (QSVM)": MetricSet;
}

export type Role = 'Researcher' | 'Clinician' | 'Admin';
