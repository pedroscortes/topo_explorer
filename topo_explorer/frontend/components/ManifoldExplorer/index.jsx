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
  // WebSocket state
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [error, setError] = useState(null);

  // UI state
  const [viewMode, setViewMode] = useState('3d');
  const [isPlaying, setIsPlaying] = useState(false);
  const [isTraining, setIsTraining] = useState(false);

  // Data state
  const [manifoldData, setManifoldData] = useState(null);
  const [trajectory, setTrajectory] = useState([]);
  const [metrics, setMetrics] = useState({
    exploration: [],
    curvature: [],
    training: []
  });

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
              ].slice(-100); // Keep last 100 points
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
      console.log('Sending training control:', message);
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected, cannot send training control');
    }
  }, [isTraining]);

  const connectWebSocket = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    // Close existing connection if any
    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch (err) {
        console.warn('Error closing existing connection:', err);
      }
    }

    try {
      console.log('Attempting WebSocket connection...');
      wsRef.current = new WebSocket(WS_URL);
      
      wsRef.current.onopen = () => {
        console.log('WebSocket Connected');
        setConnectionStatus('connected');
        setError(null);

        // Send init message with retry
        const sendInit = () => {
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            try {
              const initMessage = {
                type: 'init',
                data: { clientId: Math.random().toString(36).substr(2, 9) }
              };
              wsRef.current.send(JSON.stringify(initMessage));
              console.log('Sent init message:', initMessage);
            } catch (err) {
              console.error('Error sending init message:', err);
              setTimeout(sendInit, 1000);
            }
          }
        };
        sendInit();
      };
      
      wsRef.current.onclose = (event) => {
        console.log(`WebSocket Disconnected: ${event.code} - ${event.reason}`);
        setConnectionStatus('disconnected');
        wsRef.current = null;

        if (event.code !== WS_CLOSE_NORMAL) {
          console.log(`Scheduling reconnection in ${WS_RETRY_DELAY}ms...`);
          reconnectTimeoutRef.current = setTimeout(connectWebSocket, WS_RETRY_DELAY);
        }
      };
      
      wsRef.current.onerror = (error) => {
        console.error('WebSocket Error:', error);
        setError('Connection error occurred');
      };
      
      wsRef.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          handleVisualizationUpdate(message);
        } catch (error) {
          console.error('Error processing message:', error);
        }
      };
    } catch (error) {
      console.error('WebSocket connection error:', error);
      setError('Failed to establish connection');
      reconnectTimeoutRef.current = setTimeout(connectWebSocket, WS_RETRY_DELAY);
    }
  }, [handleVisualizationUpdate]);

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
      {error && (
        <div className="bg-red-50 border-l-4 border-red-400 p-4">
          <div className="flex">
            <div className="ml-3">
              <p className="text-sm text-red-700">
                {error}
              </p>
            </div>
          </div>
        </div>
      )}
      
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className={`h-2 w-2 rounded-full ${
            connectionStatus === 'connected' ? 'bg-green-500' : 'bg-red-500'
          }`} />
          <span className="text-sm text-gray-500">
            {connectionStatus === 'connected' ? 'Connected' : 'Disconnected'}
          </span>
        </div>

        <div className="space-x-2">
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
      </div>

      {renderMainView()}

      <Card>
        <CardContent>
          <ExplorationView 
            explorationData={trajectory}
          />
        </CardContent>
      </Card>
    </div>
  );
};

export default ManifoldExplorer;