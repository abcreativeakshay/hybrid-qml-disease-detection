import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import type { FeatureImportance } from '../../types';

interface SHAPBarChartProps {
  data: FeatureImportance[];
  title?: string;
  color?: string;
}

export const SHAPBarChart: React.FC<SHAPBarChartProps> = ({ data, title = "Feature Importance", color = "var(--accent-primary)" }) => {
  // Sort data ascending for horizontal bar chart so highest is at top
  const sortedData = [...data].sort((a, b) => a.importance_mean - b.importance_mean);

  return (
    <div style={{ width: '100%', height: '300px' }}>
      <h3 style={{ fontSize: '1rem', color: 'var(--text-secondary)', marginBottom: '1rem', textAlign: 'center' }}>{title}</h3>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          layout="vertical"
          data={sortedData}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <XAxis type="number" hide />
          <YAxis dataKey="feature" type="category" axisLine={false} tickLine={false} tick={{ fill: 'var(--text-secondary)', fontSize: 12 }} width={80} />
          <Tooltip 
            cursor={{ fill: 'rgba(255, 255, 255, 0.05)' }}
            contentStyle={{ backgroundColor: 'var(--bg-panel)', border: '1px solid var(--border-color)', borderRadius: '8px', color: 'var(--text-primary)' }}
            formatter={(value: any) => [Number(value).toFixed(4), 'Importance']}
          />
          <Bar dataKey="importance_mean" radius={[0, 4, 4, 0]}>
            {sortedData.map((_entry, index) => (
              <Cell key={`cell-${index}`} fill={index === sortedData.length - 1 ? color : 'var(--text-tertiary)'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
