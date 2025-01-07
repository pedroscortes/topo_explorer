import React, { useState, useEffect, useRef, useCallback } from 'react';
import GeometryView from './GeometryView';
import MetricsPanel from './MetricsPanel';
import ExplorationView from './ExplorationView';

const Button = ({ children, variant = 'default', onClick, className = '' }) => (
  <button
    onClick={onClick}
    className={`px-4 py-2 rounded-md font-medium transition-colors 
    ${variant === 'default' 
      ? 'bg-blue-600 text-white hover:bg-blue-700' 
      : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'} 
    ${className}`}
  >
    {children}
  </button>
);

const Card = ({ children, className = '' }) => (
  <div className={`bg-white rounded-lg shadow ${className}`}>
    {children}
  </div>
);

const CardContent = ({ children, className = '' }) => (
  <div className={`p-6 ${className}`}>
    {children}
  </div>
);

const WS_URL = 'ws://localhost:8765';
const WS_RETRY_DELAY = 5000;
const WS_CLOSE_NORMAL = 1000;

const MANIFOLDS = [
  { id: 'sphere', label: 'Sphere' },
  { id: 'torus', label: 'Torus' },
  { id: 'mobius', label: 'Möbius Strip' },
  { id: 'klein', label: 'Klein Bottle' },
  { id: 'hyperbolic', label: 'Hyperbolic' },
  { id: 'projective', label: 'Projective' }
];

const ManifoldExplorer = () => {
  const wsRef = useRef(null);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [error, setError] = useState(null);
  const [viewMode, setViewMode] = useState('3d');
  const [isPlaying, setIsPlaying] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [selectedManifold, setSelectedManifold] = useState('sphere');
  const [manifoldData, setManifoldData] = useState({ type: 'sphere', surface: null });
  const [trajectory, setTrajectory] = useState([]);
  const [metrics, setMetrics] = useState({
    exploration: [],
    curvature: [],
    training: []
  });

  const handleManifoldChange = useCallback((manifoldId) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      setSelectedManifold(manifoldId);
      
      const message = {
        type: 'control',
        data: {
          action: 'change_manifold',
          manifold: manifoldId
        }
      };
      console.log('Sending manifold change message:', message);
      wsRef.current.send(JSON.stringify(message));
      
      // Reset visualization state
      setTrajectory([]);
      setMetrics({
        exploration: [],
        curvature: [],
        training: []
      });
    }
  }, []);

  const ManifoldSelector = () => (
    <div className="bg-white rounded-lg shadow p-4 space-y-2">
      {MANIFOLDS.map(({ id, label }) => (
        <label 
          key={id} 
          className="flex items-center space-x-2 cursor-pointer"
        >
          <input
            type="radio"
            name="manifold"
            value={id}
            checked={selectedManifold === id}
            onChange={() => handleManifoldChange(id)}
            className="form-radio text-blue-600"
          />
          <span className="text-sm">{label}</span>
        </label>
      ))}
    </div>
  );

  const handleVisualizationUpdate = useCallback((message) => {
    if (!message.type || !message.data) {
      console.warn('Invalid message format:', message);
      return;
    }

    try {
      console.log('Processing message:', message.type);
      switch (message.type) {
        case 'manifold':
          setManifoldData(message.data);
          break;

        case 'trajectory':
          if (Array.isArray(message.data.points)) {
            setTrajectory(message.data.points);
          }
          break;

        case 'metrics':
          const { data } = message;
          setMetrics(prevMetrics => {
            const newMetrics = { ...prevMetrics };
            
            if (data.training) {
              newMetrics.training = [
                ...prevMetrics.training,
                data.training
              ].slice(-100);
            }
            
            if (data.curvature) {
              newMetrics.curvature = [
                ...prevMetrics.curvature,
                ...data.curvature
              ].slice(-100);
            }
            
            return newMetrics;
          });
          break;

        case 'status':
          if ('training' in message.data) {
            setIsTraining(message.data.training);
            console.log('Training status updated:', message.data.training);
          }
          break;

        default:
          console.warn('Unknown message type:', message.type);
      }
    } catch (error) {
      console.error('Error handling message:', error);
    }
  }, []);

  const handleTraining = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const message = {
        type: 'control',
        data: {
          action: isTraining ? 'stop_training' : 'start_training'
        }
      };
      wsRef.current.send(JSON.stringify(message));
    }
  }, [isTraining]);

  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.close();
    }

    console.log('Establishing WebSocket connection...');
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket Connected');
      setConnectionStatus('connected');
      setError(null);

      // Send init message
      const initMessage = {
        type: 'init',
        data: { 
          clientId: Math.random().toString(36).substr(2, 9),
          manifold: selectedManifold
        }
      };
      ws.send(JSON.stringify(initMessage));
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setConnectionStatus('disconnected');
      wsRef.current = null;
      
      // Attempt to reconnect after delay
      setTimeout(connectWebSocket, WS_RETRY_DELAY);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('Connection error occurred');
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        handleVisualizationUpdate(message);
      } catch (error) {
        console.error('Error processing message:', error);
      }
    };
  }, [handleVisualizationUpdate, selectedManifold]);

  // Single WebSocket connection effect
  useEffect(() => {
    console.log('Component mounted, establishing WebSocket connection');
    connectWebSocket();

    return () => {
      console.log('Component unmounting, cleaning up WebSocket');
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.close(WS_CLOSE_NORMAL, 'Component unmounting');
      }
    };
  }, [connectWebSocket]);

  const renderMainView = () => {
    switch (viewMode) {
      case '3d':
        return (
          <Card>
            <CardContent className="p-0">
              <GeometryView 
                manifoldData={manifoldData}
                trajectory={trajectory}
                isPlaying={isPlaying}
              />
            </CardContent>
          </Card>
        );
      case 'curvature':
      case 'training':
        return (
          <Card>
            <CardContent>
              <MetricsPanel 
                metrics={metrics}
                type={viewMode}
              />
            </CardContent>
          </Card>
        );
      default:
        return null;
    }
  };

  return (
    <div className="w-full space-y-4">
      {connectionStatus === 'disconnected' && (
        <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4">
          <div className="flex">
            <div className="ml-3">
              <p className="text-sm text-yellow-700">
                Connecting to visualization server...
              </p>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border-l-4 border-red-400 p-4">
          <div className="flex">
            <div className="ml-3">
              <p className="text-sm text-red-700">{error}</p>
            </div>
          </div>
        </div>
      )}
      
      <div className="flex flex-col md:flex-row md:items-start gap-4">
        <div className="md:w-64">
          <ManifoldSelector />
          <div className="mt-2 flex items-center gap-2 px-4">
            <div className={`h-2 w-2 rounded-full ${
              connectionStatus === 'connected' ? 'bg-green-500' : 'bg-red-500'
            }`} />
            <span className="text-sm text-gray-500">
              {connectionStatus === 'connected' ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>
  
        <div className="flex-1">
          <div className="flex justify-end space-x-2 mb-4">
            <Button 
              onClick={() => setViewMode('3d')}
              variant={viewMode === '3d' ? 'default' : 'outline'}
            >
              3D View
            </Button>
            <Button 
              onClick={() => setViewMode('curvature')}
              variant={viewMode === 'curvature' ? 'default' : 'outline'}
            >
              Curvature
            </Button>
            <Button 
              onClick={() => setViewMode('training')}
              variant={viewMode === 'training' ? 'default' : 'outline'}
            >
              Training
            </Button>
            <Button 
              onClick={() => setIsPlaying(!isPlaying)}
              variant={isPlaying ? 'default' : 'outline'}
            >
              {isPlaying ? 'Pause' : 'Play'}
            </Button>
            <Button 
              onClick={handleTraining}
              variant={isTraining ? 'default' : 'outline'}
            >
              {isTraining ? 'Stop Training' : 'Start Training'}
            </Button>
          </div>
  
          {renderMainView()}
  
          <Card className="mt-4">
            <CardContent>
              <ExplorationView 
                explorationData={trajectory}
              />
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default ManifoldExplorer;