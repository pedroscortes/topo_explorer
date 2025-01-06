import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const MetricsPanel = ({ metrics, type }) => {
  console.log('MetricsPanel render:', { type, metrics }); 

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
    return null;
  };

  const chartConfig = getChartData();

  if (!chartConfig) {
    return (
      <div className="w-full h-64 flex items-center justify-center text-gray-500">
        Invalid metrics type
      </div>
    );
  }

  if (!chartConfig.data || chartConfig.data.length === 0) {
    return (
      <div className="w-full h-64 flex items-center justify-center text-gray-500">
        {type === 'training' 
          ? 'Start training to see metrics...'
          : 'No curvature data available yet...'}
      </div>
    );
  }

  return (
    <div className="w-full">
      <h3 className="text-lg font-semibold mb-4">{chartConfig.title}</h3>
      <div className="h-64"> {/* Fixed height container */}
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartConfig.data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="step" 
              label={{ value: 'Steps', position: 'bottom' }}
            />
            <YAxis 
              label={{ 
                value: type === 'training' ? 'Loss' : 'Curvature', 
                angle: -90, 
                position: 'insideLeft' 
              }}
            />
            <Tooltip 
              formatter={(value, name) => [value.toFixed(4), name]}
              labelFormatter={(label) => `Step ${label}`}
            />
            <Legend verticalAlign="top" height={36} />
            {chartConfig.lines.map(line => (
              <Line
                key={line.key}
                type="monotone"
                dataKey={line.key}
                name={line.name}
                stroke={line.stroke}
                dot={false}
                strokeWidth={2}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default MetricsPanel;