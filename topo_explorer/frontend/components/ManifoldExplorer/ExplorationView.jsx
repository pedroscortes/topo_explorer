import React from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const ExplorationView = ({ explorationData }) => {
  console.log('ExplorationView:', { dataPoints: explorationData?.length });

  const processData = (points) => {
    if (!points?.length) return [];
    
    return points.map((point, index) => {
      try {
        const [x, y, z] = point;
        const r = Math.sqrt(x*x + y*y + z*z);
        const theta = Math.atan2(y, x);
        const phi = Math.acos(z/r);
        return {
          theta: theta * 180 / Math.PI,
          phi: phi * 180 / Math.PI,
          value: 1,
          id: index
        };
      } catch (error) {
        console.error('Error processing point:', point, error);
        return null;
      }
    }).filter(Boolean); 
  };

  const data = processData(explorationData);

  if (!data.length) {
    return (
      <div className="bg-white p-4 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Exploration Density</h3>
        <div className="h-64 flex items-center justify-center text-gray-500">
          No exploration data available yet
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white p-4 rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-4">Exploration Density</h3>
      <ResponsiveContainer width="100%" height={400}>
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="theta"
            name="θ"
            unit="°"
            domain={[-180, 180]}
            label={{ value: 'Longitude (θ)', position: 'bottom' }}
          />
          <YAxis
            type="number"
            dataKey="phi"
            name="φ"
            unit="°"
            domain={[0, 180]}
            label={{ value: 'Latitude (φ)', angle: -90, position: 'left' }}
          />
          <ZAxis
            type="number"
            dataKey="value"
            range={[50, 400]}
          />
          <Tooltip
            cursor={{ strokeDasharray: '3 3' }}
            formatter={(value, name) => {
              if (name === 'θ') return [`${value.toFixed(1)}°`, 'Longitude'];
              if (name === 'φ') return [`${value.toFixed(1)}°`, 'Latitude'];
              return [value, name];
            }}
          />
          <Scatter
            data={data}
            fill="#8884d8"
            opacity={0.6}
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
};

export default ExplorationView;