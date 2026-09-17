import { useState } from 'react';
import type { RootData } from '../types';
import { SHAPBarChart } from './charts/SHAPBarChart';

interface Props {
  data: RootData;
}

export const ResearcherDashboard: React.FC<Props> = ({ data }) => {
  const [activeTab, setActiveTab] = useState<'Benchmark' | 'Explainability'>('Benchmark');

  return (
    <div className="animate-fade-in">
      <header style={{ marginBottom: '2rem' }}>
        <h1 style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>Technical Dashboard</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Advanced model metrics, benchmarking, and quantum explainability.</p>
      </header>

      <div className="tabs-container">
        <button 
          className={`tab-button ${activeTab === 'Benchmark' ? 'active' : ''}`}
          onClick={() => setActiveTab('Benchmark')}
        >
          Model Benchmark
        </button>
        <button 
          className={`tab-button ${activeTab === 'Explainability' ? 'active' : ''}`}
          onClick={() => setActiveTab('Explainability')}
        >
          Explainability
        </button>
      </div>

      {activeTab === 'Benchmark' && (
        <div className="glass-panel">
          <h2 style={{ fontSize: '1.25rem', marginBottom: '1.5rem', color: 'var(--accent-primary)' }}>Performance Comparison</h2>
          <table className="data-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>F1 Score</th>
                <th>ROC AUC</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(data.metrics).map(([name, metrics]) => (
                <tr key={name} className={name.includes('Quantum') ? 'highlight-row' : ''}>
                  <td style={{ fontWeight: 500, color: name.includes('Quantum') ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
                    {name.replace(/_/g, ' ')}
                  </td>
                  <td>{(metrics.accuracy * 100).toFixed(1)}%</td>
                  <td>{(metrics.f1 * 100).toFixed(1)}%</td>
                  <td>{metrics.roc_auc.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {activeTab === 'Explainability' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
          <div className="glass-panel">
            <h2 style={{ fontSize: '1.25rem', marginBottom: '0.5rem', color: 'var(--accent-primary)' }}>Classical SHAP</h2>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>Logistic Regression (LinearExplainer)</p>
            {data.metrics["Logistic Regression"]?.shap_importance && (
              <SHAPBarChart data={data.metrics["Logistic Regression"].shap_importance} color="var(--accent-primary)" />
            )}
          </div>
          
          <div className="glass-panel" style={{ border: '1px solid var(--border-focus)' }}>
            <h2 style={{ fontSize: '1.25rem', marginBottom: '0.5rem', color: 'var(--accent-quantum1)' }}>Quantum Sensitivity</h2>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>VQC Parameter Perturbation Analysis</p>
            {data.metrics["Quantum VQC"]?.quantum_sensitivity && (
              <SHAPBarChart data={data.metrics["Quantum VQC"].quantum_sensitivity} color="var(--accent-quantum1)" />
            )}
          </div>
        </div>
      )}
    </div>
  );
};
