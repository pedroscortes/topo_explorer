import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const MetricsPanel = ({ metrics, type }) => {
  const getChartData = () => {
    if (type === 'training') {
      return {
        title: 'Training Progress',
        data: metrics.training || [],
        lines: [
          { key: 'value_loss', name: 'Value Loss', stroke: '#8884d8' },
          { key: 'policy_loss', name: 'Policy Loss', stroke: '#82ca9d' }
        ]
      };
    } else if (type === 'curvature') {
      return {
        title: 'Curvature Distribution',
        data: metrics.curvature || [],
        lines: [
          { key: 'value', name: 'Curvature', stroke: '#ffc658' }
        ]
      };
    }
  };

  const chartConfig = getChartData();

  return (
    <div className="w-full">
      <h3 className="text-lg font-semibold mb-4">{chartConfig.title}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartConfig.data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="step" />
          <YAxis />
          <Tooltip />
          <Legend />
          {chartConfig.lines.map(line => (
            <Line
              key={line.key}
              type="monotone"
              dataKey={line.key}
              name={line.name}
              stroke={line.stroke}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default MetricsPanel;