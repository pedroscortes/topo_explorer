"""Integration between existing visualizers and WebSocket server."""

import asyncio
import time
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
from .websocket_server import ManifoldVisWebSocket

@dataclass
class VisualizationData:
    type: str
    data: Dict[str, Any]
    timestamp: float

class WebSocketManifoldVisualizer:
    """Combines ManifoldVisualizer and TrainingVisualizer with WebSocket streaming."""
    
    def __init__(self, manifold_vis, training_vis=None, websocket_server: Optional[ManifoldVisWebSocket] = None):
        """Initialize the integrated visualizer."""
        self.manifold_vis = manifold_vis
        self.training_vis = training_vis
        self.ws_server = websocket_server or ManifoldVisWebSocket()
        self._start_server()
        
    def _start_server(self):
        """Start WebSocket server in background."""
        import threading
        self.server_thread = threading.Thread(
            target=self.ws_server.start_server,
            daemon=True
        )
        self.server_thread.start()

    async def _stream_update(self, data_type: str, data: dict):
        """Stream update through WebSocket."""
        if not self.ws_server.running:
            return
            
        vis_data = VisualizationData(
            type=data_type,
            data=data,
            timestamp=time.time()
        )
        await self.ws_server.broadcast(vis_data)

    def plot_trajectory(self, trajectory: np.ndarray, curvatures: Optional[np.ndarray] = None):
        """Plot trajectory and stream data."""
        self.manifold_vis.plot_trajectory(trajectory, curvatures)
        
        data = {
            'trajectory': trajectory.tolist() if isinstance(trajectory, np.ndarray) else trajectory,
            'curvatures': curvatures.tolist() if isinstance(curvatures, np.ndarray) else curvatures
        }
        asyncio.run(self._stream_update('trajectory', data))

    def plot_geometric_view(self, env, agent, trajectory):
        """Plot geometric visualization and stream data."""
        self.manifold_vis.plot_geometric_view(env, agent, trajectory)
        
        r = env.params['radius']
        theta = np.linspace(0, 2*np.pi, 30)
        phi = np.linspace(0, np.pi, 30)
        THETA, PHI = np.meshgrid(theta, phi)
        
        X = r * np.sin(PHI) * np.cos(THETA)
        Y = r * np.sin(PHI) * np.sin(THETA)
        Z = r * np.cos(PHI)
        
        data = {
            'surface': {
                'x': X.tolist(),
                'y': Y.tolist(),
                'z': Z.tolist()
            },
            'trajectory': trajectory.tolist() if isinstance(trajectory, np.ndarray) else trajectory
        }
        
        asyncio.run(self._stream_update('geometric', data))

    def update_training(self, metrics: Dict[str, float]):
        """Update training metrics and stream data."""
        if self.training_vis:
            self.training_vis.update(metrics)
            self.training_vis.plot_metrics()
        
        asyncio.run(self._stream_update('metrics', metrics))

    def __getattr__(self, name):
        """Delegate unknown attributes to manifold visualizer."""
        return getattr(self.manifold_vis, name)