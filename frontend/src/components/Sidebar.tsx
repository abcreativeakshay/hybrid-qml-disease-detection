import React from 'react';
import type { Role } from '../types';
import { Activity, ShieldCheck, Microscope, User, Settings, Database } from 'lucide-react';

interface SidebarProps {
  currentRole: Role;
  setRole: (role: Role) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ currentRole, setRole }) => {
  return (
    <aside className="sidebar">
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
        <div style={{ background: 'var(--accent-primary)', padding: '0.5rem', borderRadius: '10px' }}>
          <Activity size={24} color="white" />
        </div>
        <h2 style={{ fontSize: '1.25rem', margin: 0, fontFamily: 'Outfit' }}>Neutron <span className="text-gradient-primary">QML</span></h2>
      </div>

      <div className="glass-panel" style={{ padding: '1rem', background: 'rgba(255,255,255,0.03)' }}>
        <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>
          View As (Demo Role)
        </label>
        <select 
          value={currentRole} 
          onChange={(e) => setRole(e.target.value as Role)}
        >
          <option value="Researcher">👨‍🔬 Researcher</option>
          <option value="Clinician">🩺 Clinician</option>
          <option value="Admin">🛡️ System Admin</option>
        </select>
      </div>

      <nav style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '1rem' }}>
        <div style={{ padding: '0.75rem 1rem', display: 'flex', alignItems: 'center', gap: '0.75rem', color: currentRole === 'Researcher' ? 'var(--text-primary)' : 'var(--text-tertiary)', background: currentRole === 'Researcher' ? 'rgba(255,255,255,0.05)' : 'transparent', borderRadius: '8px', transition: 'all 0.2s' }}>
          <Microscope size={18} />
          <span style={{ fontSize: '0.95rem', fontWeight: 500 }}>Technical Dashboard</span>
        </div>
        <div style={{ padding: '0.75rem 1rem', display: 'flex', alignItems: 'center', gap: '0.75rem', color: currentRole === 'Clinician' ? 'var(--text-primary)' : 'var(--text-tertiary)', background: currentRole === 'Clinician' ? 'rgba(255,255,255,0.05)' : 'transparent', borderRadius: '8px', transition: 'all 0.2s' }}>
          <User size={18} />
          <span style={{ fontSize: '0.95rem', fontWeight: 500 }}>Patient Assessment</span>
        </div>
        <div style={{ padding: '0.75rem 1rem', display: 'flex', alignItems: 'center', gap: '0.75rem', color: currentRole === 'Admin' ? 'var(--text-primary)' : 'var(--text-tertiary)', background: currentRole === 'Admin' ? 'rgba(255,255,255,0.05)' : 'transparent', borderRadius: '8px', transition: 'all 0.2s' }}>
          <ShieldCheck size={18} />
          <span style={{ fontSize: '0.95rem', fontWeight: 500 }}>System Status</span>
        </div>
      </nav>

      <div style={{ marginTop: 'auto', paddingTop: '2rem', borderTop: '1px solid var(--border-color)', fontSize: '0.8rem', color: 'var(--text-tertiary)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
          <Database size={14} />
          <span>Local Pre-compute Engine</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Settings size={14} />
          <span>VQC Config: 5 Qubits, 2 Layers</span>
        </div>
      </div>
    </aside>
  );
};
