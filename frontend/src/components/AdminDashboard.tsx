import React from 'react';
import type { RootData } from '../types';
import { Server, Database, Shield, Clock } from 'lucide-react';

interface Props {
  data: RootData;
}

export const AdminDashboard: React.FC<Props> = () => {
  return (
    <div className="animate-fade-in">
      <header style={{ marginBottom: '2rem' }}>
        <h1 style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>System Status</h1>
        <p style={{ color: 'var(--text-secondary)' }}>Overview of operational health and precomputed data.</p>
      </header>

      <div className="metric-grid">
        <div className="glass-panel metric-card">
          <Server size={24} color="var(--accent-primary)" style={{ marginBottom: '0.5rem' }} />
          <div className="metric-value" style={{ fontSize: '1.5rem' }}>HF Static SDK</div>
          <div className="metric-label">Deployment Target</div>
        </div>
        <div className="glass-panel metric-card">
          <Database size={24} color="var(--accent-secondary)" style={{ marginBottom: '0.5rem' }} />
          <div className="metric-value" style={{ fontSize: '1.5rem' }}>Loaded</div>
          <div className="metric-label">metrics.json</div>
        </div>
        <div className="glass-panel metric-card">
          <Shield size={24} color="var(--accent-quantum1)" style={{ marginBottom: '0.5rem' }} />
          <div className="metric-value" style={{ fontSize: '1.5rem' }}>5 Models</div>
          <div className="metric-label">Precomputed</div>
        </div>
      </div>

      <div className="glass-panel">
        <h2 style={{ fontSize: '1.25rem', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Clock size={20} color="var(--text-tertiary)" />
          Session Action Log
        </h2>
        <div style={{ background: 'rgba(0,0,0,0.2)', borderRadius: '8px', padding: '1rem', border: '1px solid var(--border-color)', minHeight: '200px', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <div style={{ display: 'flex', gap: '1rem', fontSize: '0.9rem' }}>
            <span style={{ color: 'var(--text-tertiary)', width: '80px' }}>14:02:45</span>
            <span style={{ color: 'var(--text-secondary)' }}>[ROUTING] Switched to Admin view</span>
          </div>
          <div style={{ display: 'flex', gap: '1rem', fontSize: '0.9rem' }}>
            <span style={{ color: 'var(--text-tertiary)', width: '80px' }}>14:02:12</span>
            <span style={{ color: 'var(--text-secondary)' }}>[ROUTING] Switched to Clinician view</span>
          </div>
          <div style={{ display: 'flex', gap: '1rem', fontSize: '0.9rem' }}>
            <span style={{ color: 'var(--text-tertiary)', width: '80px' }}>14:01:50</span>
            <span style={{ color: 'var(--accent-secondary)' }}>[DATA] Fetched metrics.json payload</span>
          </div>
          <div style={{ display: 'flex', gap: '1rem', fontSize: '0.9rem' }}>
            <span style={{ color: 'var(--text-tertiary)', width: '80px' }}>14:01:45</span>
            <span style={{ color: 'var(--accent-primary)' }}>[SYSTEM] Application mounted in Static Demo Mode</span>
          </div>
        </div>
      </div>
    </div>
  );
};
