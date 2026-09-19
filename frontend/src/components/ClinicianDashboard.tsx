import React, { useState, useCallback } from 'react';
import type { RootData } from '../types';
import { Activity, UploadCloud, FileText, Loader2, AlertCircle } from 'lucide-react';

interface PatientPrediction {
  sample_id: string;
  risk_label: 'critical' | 'high' | 'moderate' | 'low';
  probability_score: number;
  model_name: string;
}

interface Props {
  data: RootData;
}

export const ClinicianDashboard: React.FC<Props> = ({ data }) => {
  const [predictions, setPredictions] = useState<PatientPrediction[]>([
    { sample_id: 'P-1042', risk_label: 'high', probability_score: 0.87, model_name: 'Hybrid QML (QSVM)' },
    { sample_id: 'P-8831', risk_label: 'low', probability_score: 0.12, model_name: 'Hybrid QML (QSVM)' },
  ]);
  const [selectedSample, setSelectedSample] = useState<string>('P-1042');
  
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setSelectedFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleAnalyze = () => {
    if (!selectedFile) return;
    setIsAnalyzing(true);
    
    setTimeout(() => {
      const newId = `SAMPLE_NEW_${Math.floor(1000 + Math.random() * 9000)}`;
      const score = Math.random();
      let label: 'critical' | 'high' | 'moderate' | 'low' = 'low';
      if (score > 0.85) label = 'critical';
      else if (score > 0.6) label = 'high';
      else if (score > 0.3) label = 'moderate';

      const newPatient: PatientPrediction = {
        sample_id: newId,
        risk_label: label,
        probability_score: score,
        model_name: 'Hybrid QML Ensemble'
      };

      setPredictions(prev => [newPatient, ...prev]);
      setSelectedSample(newId);
      setIsAnalyzing(false);
      setSelectedFile(null);
    }, 1800);
  };

  // Stats
  const total = predictions.length;
  const critical = predictions.filter(p => p.risk_label === 'critical').length;
  const high = predictions.filter(p => p.risk_label === 'high').length;
  const lowMod = predictions.filter(p => p.risk_label === 'low' || p.risk_label === 'moderate').length;

  const activePrediction = predictions.find(p => p.sample_id === selectedSample);

  // Demographics logic
  const getDemographics = (id: string) => {
    const hash = id.split('').reduce((a, b) => { a = ((a << 5) - a) + b.charCodeAt(0); return a & a }, 0);
    return {
      age: 30 + Math.abs(hash % 50),
      gender: (hash % 2 === 0) ? 'Male' : 'Female',
      blood: ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-'][Math.abs(hash) % 8],
      comorbidities: ['None', 'Hypertension', 'Type 2 Diabetes', 'Asthma', 'Hyperlipidemia', 'Hypertension, Diabetes'][Math.abs(hash) % 6]
    };
  };

  const activeDemo = activePrediction ? getDemographics(activePrediction.sample_id) : null;

  const getPlainLanguage = (label: string, score: number) => {
    const pct = Math.round(score * 100);
    switch (label) {
      case 'critical':
        return <>This sample shows a <strong style={{color:'var(--text-primary)'}}>{pct}%</strong> probability of malignancy. Immediate clinical review is recommended.</>;
      case 'high':
        return <>This sample shows a <strong style={{color:'var(--text-primary)'}}>{pct}%</strong> elevated risk. Further diagnostic testing is advised.</>;
      case 'moderate':
        return <>This sample shows a <strong style={{color:'var(--text-primary)'}}>{pct}%</strong> moderate risk. Consider follow-up screening within the standard timeline.</>;
      default:
        return <>This sample shows a <strong style={{color:'var(--text-primary)'}}>{pct}%</strong> low probability. Routine monitoring is sufficient.</>;
    }
  };

  const getRecommendation = (label: string) => {
    switch (label) {
      case 'critical':
        return 'Schedule urgent biopsy or advanced imaging. Flag for priority review by oncology team. The AI model identifies strong indicators across multiple features.';
      case 'high':
        return 'Refer for additional diagnostic imaging (ultrasound/MRI). Schedule follow-up within 2 weeks. Consider genetic testing if family history is present.';
      case 'moderate':
        return 'Continue standard screening schedule. Schedule follow-up in 3-6 months. Monitor for any symptom changes.';
      default:
        return 'No immediate action required. Continue routine annual screening. Results are consistent with benign classification.';
    }
  };
  
  const getRiskColor = (label: string) => {
    switch (label) {
      case 'critical': return '#ef4444'; // red
      case 'high': return '#f97316'; // orange
      case 'moderate': return '#eab308'; // yellow
      default: return '#10b981'; // green
    }
  };

  const getRiskBg = (label: string) => {
    switch (label) {
      case 'critical': return 'rgba(239, 68, 68, 0.1)';
      case 'high': return 'rgba(249, 115, 22, 0.1)';
      case 'moderate': return 'rgba(234, 179, 8, 0.1)';
      default: return 'rgba(16, 185, 129, 0.1)';
    }
  };

  const qsvmMetrics = data.metrics["Quantum SVM (QSVM)"];
  let explanation = `Prediction generated by ${activePrediction?.model_name || 'AI'}. No anomalies detected in standard metric ranges outside primary risk factors.`;
  if (activePrediction?.sample_id === 'P-1042' && qsvmMetrics?.quantum_sensitivity && qsvmMetrics.quantum_sensitivity.length >= 2) {
    const topFeatures = [...qsvmMetrics.quantum_sensitivity]
      .sort((a, b) => b.importance_mean - a.importance_mean)
      .slice(0, 2)
      .map(f => f.feature.replace(/_/g, ' '));
    explanation = `This assessment is most influenced by ${topFeatures.join(' and ')}, which align with the historical profile for this risk band.`;
  }

  return (
    <div className="animate-fade-in" style={{ maxWidth: '1000px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      
      {/* Header and Upload Section */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
        
        {/* Stats */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
          <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', padding: '1.25rem' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Total Patients</span>
            <span style={{ fontSize: '2rem', fontWeight: 600, color: 'var(--text-primary)' }}>{total}</span>
          </div>
          <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', padding: '1.25rem', borderLeft: '3px solid #ef4444' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Critical Risk</span>
            <span style={{ fontSize: '2rem', fontWeight: 600, color: '#ef4444' }}>{critical}</span>
          </div>
          <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', padding: '1.25rem', borderLeft: '3px solid #f97316' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>High Risk</span>
            <span style={{ fontSize: '2rem', fontWeight: 600, color: '#f97316' }}>{high}</span>
          </div>
          <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', padding: '1.25rem', borderLeft: '3px solid #10b981' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Low / Mod</span>
            <span style={{ fontSize: '2rem', fontWeight: 600, color: '#10b981' }}>{lowMod}</span>
          </div>
        </div>

        {/* Upload Dropzone */}
        <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', padding: '1.25rem' }}>
          <h3 style={{ fontSize: '1rem', marginBottom: '0.75rem', color: 'var(--text-primary)' }}>New Assessment</h3>
          <label 
            htmlFor="clinician-file-upload"
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            style={{ 
              flex: 1,
              border: `2px dashed ${isDragging ? 'var(--accent-primary)' : 'var(--border-color)'}`,
              borderRadius: '12px',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              background: isDragging ? 'rgba(99, 110, 250, 0.05)' : 'transparent',
              transition: 'all 0.2s ease',
              padding: '1rem',
              gap: '0.5rem'
            }}
          >
            {selectedFile ? (
              <>
                <FileText size={28} color="var(--accent-secondary)" />
                <div style={{ textAlign: 'center' }}>
                  <p style={{ color: 'var(--text-primary)', fontWeight: 500, fontSize: '0.9rem' }}>{selectedFile.name}</p>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>{(selectedFile.size / 1024).toFixed(1)} KB ready</p>
                </div>
              </>
            ) : (
              <>
                <UploadCloud size={28} color="var(--text-secondary)" />
                <div style={{ textAlign: 'center' }}>
                  <p style={{ color: 'var(--text-primary)', fontSize: '0.9rem', fontWeight: 500 }}>Drag & drop patient file</p>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>or click to browse (.csv)</p>
                </div>
              </>
            )}
            <input 
              id="clinician-file-upload" 
              type="file" 
              style={{ display: 'none' }} 
              onChange={handleFileChange}
              accept=".csv,.json"
            />
          </label>
          <button 
            onClick={handleAnalyze}
            disabled={!selectedFile || isAnalyzing}
            style={{
              marginTop: '0.75rem',
              width: '100%',
              padding: '0.6rem',
              background: (!selectedFile || isAnalyzing) ? 'rgba(255,255,255,0.05)' : 'var(--accent-primary)',
              color: (!selectedFile || isAnalyzing) ? 'var(--text-secondary)' : '#fff',
              border: 'none',
              borderRadius: '8px',
              fontWeight: 600,
              cursor: (!selectedFile || isAnalyzing) ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.5rem',
              transition: 'all 0.2s'
            }}
          >
            {isAnalyzing ? (
              <><Loader2 size={16} className="animate-spin" /> Running...</>
            ) : 'Run Assessment'}
          </button>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '250px 1fr', gap: '1.5rem', alignItems: 'start' }}>
        
        {/* Patient List */}
        <div className="glass-panel" style={{ padding: '1rem', maxHeight: '500px', overflowY: 'auto' }}>
          <h3 style={{ fontSize: '1rem', marginBottom: '1rem', color: 'var(--text-secondary)', paddingLeft: '0.5rem' }}>Recent Assessments</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {predictions.map(p => (
              <div 
                key={p.sample_id}
                onClick={() => setSelectedSample(p.sample_id)}
                style={{
                  padding: '0.8rem',
                  borderRadius: '10px',
                  background: selectedSample === p.sample_id ? 'rgba(255,255,255,0.08)' : 'transparent',
                  border: `1px solid ${selectedSample === p.sample_id ? 'rgba(255,255,255,0.1)' : 'transparent'}`,
                  cursor: 'pointer',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.4rem',
                  transition: 'all 0.2s'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontWeight: 500, color: 'var(--text-primary)', fontSize: '0.9rem' }}>{p.sample_id}</span>
                  <span style={{
                    fontSize: '0.65rem',
                    textTransform: 'uppercase',
                    fontWeight: 700,
                    padding: '0.2rem 0.4rem',
                    borderRadius: '4px',
                    background: getRiskBg(p.risk_label),
                    color: getRiskColor(p.risk_label),
                  }}>
                    {p.risk_label}
                  </span>
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                  Score: {(p.probability_score * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Selected Patient View */}
        {activePrediction && activeDemo ? (
          <div className="glass-panel" style={{ padding: '2rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem' }}>
              <div>
                <h2 style={{ fontSize: '1.6rem', color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
                  Patient {activePrediction.sample_id}
                </h2>
                <div style={{ display: 'flex', gap: '1.5rem', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                  <span><strong style={{ color: 'var(--text-primary)', fontWeight: 500 }}>Age:</strong> {activeDemo.age}</span>
                  <span><strong style={{ color: 'var(--text-primary)', fontWeight: 500 }}>Gender:</strong> {activeDemo.gender}</span>
                  <span><strong style={{ color: 'var(--text-primary)', fontWeight: 500 }}>Blood:</strong> {activeDemo.blood}</span>
                </div>
                <div style={{ marginTop: '0.5rem', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                  <strong style={{ color: 'var(--text-primary)', fontWeight: 500 }}>Comorbidities:</strong> {activeDemo.comorbidities}
                </div>
              </div>
              
              <div style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                width: '90px',
                height: '90px',
                borderRadius: '50%',
                background: getRiskBg(activePrediction.risk_label),
                border: `3px solid ${getRiskColor(activePrediction.risk_label)}`,
                boxShadow: `0 0 20px ${getRiskBg(activePrediction.risk_label)}`
              }}>
                <span style={{ fontSize: '1.6rem', fontWeight: 700, color: getRiskColor(activePrediction.risk_label) }}>
                  {Math.round(activePrediction.probability_score * 100)}%
                </span>
              </div>
            </div>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '1.25rem' }}>
              {/* Plain Language Interpretation */}
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1.5rem', borderRadius: '12px', border: '1px solid var(--border-color)' }}>
                <h4 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '1rem', color: 'var(--text-primary)', marginBottom: '0.75rem' }}>
                  <Activity size={18} color="var(--accent-primary)" /> Clinical Interpretation
                </h4>
                <p style={{ color: 'var(--text-secondary)', lineHeight: 1.6, fontSize: '0.95rem' }}>
                  {getPlainLanguage(activePrediction.risk_label, activePrediction.probability_score)}
                </p>
                <div style={{ marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                  <h5 style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.4rem', fontWeight: 500 }}>Key Finding</h5>
                  <p style={{ color: 'var(--text-tertiary)', fontSize: '0.9rem' }}>
                    {explanation}
                  </p>
                </div>
              </div>

              {/* Recommendation */}
              <div style={{ 
                background: getRiskBg(activePrediction.risk_label), 
                padding: '1.25rem', 
                borderRadius: '12px', 
                border: `1px solid ${getRiskColor(activePrediction.risk_label)}33`
              }}>
                <h4 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '1rem', color: getRiskColor(activePrediction.risk_label), marginBottom: '0.5rem' }}>
                  <AlertCircle size={18} /> Recommended Action
                </h4>
                <p style={{ color: 'var(--text-secondary)', lineHeight: 1.5, fontSize: '0.9rem' }}>
                  {getRecommendation(activePrediction.risk_label)}
                </p>
              </div>
            </div>

          </div>
        ) : (
          <div className="glass-panel" style={{ padding: '2rem', display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '300px' }}>
            <p style={{ color: 'var(--text-tertiary)' }}>Select a patient from the list to view details.</p>
          </div>
        )}
      </div>
    </div>
  );
};
