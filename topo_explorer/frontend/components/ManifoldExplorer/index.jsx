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

const ManifoldExplorer = () => {
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [error, setError] = useState(null);
  const [viewMode, setViewMode] = useState('3d');
  const [isPlaying, setIsPlaying] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [selectedManifold, setSelectedManifold] = useState('sphere');
  const [manifoldData, setManifoldData] = useState({
    type: 'sphere',
    surface: null
  });
  const [trajectory, setTrajectory] = useState([]);
  const [metrics, setMetrics] = useState({
    exploration: [],
    curvature: [],
    training: []
  });
  const [isChangingManifold, setIsChangingManifold] = useState(false);

  const ManifoldSelector = () => {
    const manifolds = [
      { id: 'sphere', label: 'Sphere' },
      { id: 'torus', label: 'Torus' },
      { id: 'mobius', label: 'Möbius Strip' },
      { id: 'klein', label: 'Klein Bottle' },
      { id: 'hyperbolic', label: 'Hyperbolic' },
      { id: 'projective', label: 'Projective' }
    ];

    const handleManifoldChange = (manifold) => {
      if (isChangingManifold) return;

      setIsChangingManifold(true);
      console.log('Changing manifold to:', manifold);
      setSelectedManifold(manifold);

      if (wsRef.current?.readyState === WebSocket.OPEN) {
        const message = {
          type: 'set_manifold',
          data: {
            type: manifold
          }
        };
        console.log('Sending manifold change message:', message);
        wsRef.current.send(JSON.stringify(message));
        
        setTrajectory([]);
        setMetrics({
          exploration: [],
          curvature: [],
          training: []
        });

        setTimeout(() => {
          setIsChangingManifold(false);
        }, 500);
      } else {
        console.warn('WebSocket not connected, state:', wsRef.current?.readyState);
        setError('Connection lost. Trying to reconnect...');
        connectWebSocket();
        setIsChangingManifold(false);
      }
    };

    return (
      <div className="bg-white rounded-lg shadow p-4">
        <div className="space-y-2">
          {manifolds.map((manifold) => (
            <div key={manifold.id} className="flex items-center">
              <input
                type="radio"
                id={manifold.id}
                name="manifold"
                value={manifold.id}
                checked={selectedManifold === manifold.id}
                onChange={() => handleManifoldChange(manifold.id)}
                className="w-4 h-4 text-blue-600 cursor-pointer"
                disabled={isChangingManifold}
              />
              <label
                htmlFor={manifold.id}
                className="ml-2 text-sm font-medium text-gray-700 cursor-pointer"
              >
                {manifold.label}
              </label>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const handlePlayToggle = () => {
    setIsPlaying(prev => !prev);
  };

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
          if (message.data.hasOwnProperty('training')) {
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
  
    try {
      console.log('Attempting WebSocket connection...');
      const ws = new WebSocket(WS_URL);
      let connectionTimeout = setTimeout(() => {
        console.log('Connection timeout, retrying...');
        ws.close();
      }, 5000); 
  
      ws.onopen = () => {
        clearTimeout(connectionTimeout);
        console.log('WebSocket Connected');
        setConnectionStatus('connected');
        setError(null);
        wsRef.current = ws;
  
        try {
          const initMessage = {
            type: 'init',
            data: { 
              clientId: Math.random().toString(36).substr(2, 9),
              manifold: selectedManifold
            }
          };
          ws.send(JSON.stringify(initMessage));
          console.log('Sent init message:', initMessage);
        } catch (err) {
          console.error('Failed to send init message:', err);
        }
      };
  
      ws.onclose = (event) => {
        clearTimeout(connectionTimeout);
        console.log('WebSocket closed:', event.code, event.reason);
        setConnectionStatus('disconnected');
        wsRef.current = null;
  
        if (event.code !== WS_CLOSE_NORMAL) {
          console.log('Attempting to reconnect in 5 seconds...');
          setTimeout(connectWebSocket, WS_RETRY_DELAY);
        }
      };
  
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setError('Connection error. Retrying...');
        clearTimeout(connectionTimeout);
        
        if (ws.readyState !== WebSocket.CLOSED) {
          ws.close();
        }
      };
  
      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          console.log('Received message:', message);
          handleVisualizationUpdate(message);
        } catch (error) {
          console.error('Error processing message:', error);
        }
      };
  
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      setError('Connection failed. Retrying...');
      setTimeout(connectWebSocket, WS_RETRY_DELAY);
    }
  
    return () => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.close(WS_CLOSE_NORMAL);
      }
    };
  }, [handleVisualizationUpdate, selectedManifold]);

  useEffect(() => {
    connectWebSocket();
    
    return () => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.close(WS_CLOSE_NORMAL);
      }
    };
  }, [connectWebSocket]);

  useEffect(() => {
    console.log('Component mounted, establishing WebSocket connection');
    connectWebSocket();

    return () => {
      console.log('Component unmounting, cleaning up WebSocket');
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
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
              onClick={handlePlayToggle}
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
              <ExplorationView explorationData={trajectory} />
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default ManifoldExplorer;