import { useState, useEffect } from 'react';
import type { Role, RootData } from './types';
import { Sidebar } from './components/Sidebar';
import { ResearcherDashboard } from './components/ResearcherDashboard';
import { ClinicianDashboard } from './components/ClinicianDashboard';
import { AdminDashboard } from './components/AdminDashboard';
import metricsRaw from './assets/metrics.json';

function App() {
  const [role, setRole] = useState<Role>('Researcher');
  const [data, setData] = useState<RootData | null>(null);

  useEffect(() => {
    // Simulate loading to show off smooth UI transition if needed
    // In reality, it's statically imported so it's instant.
    setData(metricsRaw as RootData);
  }, []);

  if (!data) return (
    <div style={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div style={{ fontSize: '1.25rem', color: 'var(--accent-primary)' }}>Loading Models...</div>
    </div>
  );

  return (
    <div className="app-container">
      <Sidebar currentRole={role} setRole={setRole} />
      
      <main className="main-content">
        {role === 'Researcher' && <ResearcherDashboard data={data} />}
        {role === 'Clinician' && <ClinicianDashboard data={data} />}
        {role === 'Admin' && <AdminDashboard data={data} />}
      </main>
    </div>
  );
}

export default App;
