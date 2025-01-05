"""WebSocket server for streaming manifold visualization data."""

import asyncio
import websockets
import json
import numpy as np
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from topo_explorer.visualization.test_data_generator import TestDataGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VisualizationData:
    """Data structure for visualization updates."""
    type: str  
    data: Dict[str, Any]
    timestamp: float

class ManifoldVisWebSocket:
    """WebSocket server for streaming manifold visualization data."""
    
    def __init__(self, host: str = 'localhost', port: int = 8765):
        """Initialize the WebSocket server."""
        self.host = host
        self.port = port
        self.clients = set()
        self.running = False
        self._last_data = {}
        self.logger = logger
        self.generator = None

    async def handle_control_message(self, data: dict):
        """Handle control messages from clients."""
        action = data.get('data', {}).get('action')
        self.logger.info(f"Processing control action: {action}")
        
        if action == 'start_training':
            if self.generator:
                self.generator.start_training()
                await self.broadcast(VisualizationData(
                    type='status',
                    data={'training': True},
                    timestamp=time.time()
                ))
                self.logger.info("Training started")
                
        elif action == 'stop_training':
            if self.generator:
                self.generator.stop_training()
                await self.broadcast(VisualizationData(
                    type='status',
                    data={'training': False},
                    timestamp=time.time()
                ))
                self.logger.info("Training stopped")
        
    async def register(self, websocket):
        """Register a new client connection."""
        self.clients.add(websocket)
        self.logger.info(f"New client connected. Total clients: {len(self.clients)}")
        
        if self._last_data:
            try:
                await websocket.send(json.dumps({
                    'type': 'full_state',
                    'data': self._last_data
                }))
            except Exception as e:
                self.logger.error(f"Error sending initial state: {e}")
    
    async def unregister(self, websocket):
        """Unregister a client connection."""
        if websocket in self.clients:
            self.clients.remove(websocket)
            self.logger.info(f"Client disconnected. Remaining clients: {len(self.clients)}")
    
    async def broadcast(self, data: VisualizationData):
        """Broadcast data to all connected clients."""
        if not self.clients:
            return
            
        processed_data = self._process_data(data)
        
        self._last_data[processed_data['type']] = processed_data['data']
        
        message = json.dumps(processed_data)
        logger.debug(f"Broadcasting message type: {processed_data['type']}")
        
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message)
                logger.debug(f"Sent {processed_data['type']} data to client")
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"Client disconnected during broadcast")
                disconnected.add(client)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.add(client)
                
        for client in disconnected:
            await self.unregister(client)
    
    def _process_data(self, data: VisualizationData) -> Dict:
        """Process data for JSON serialization."""
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return obj
            
        processed = asdict(data)
        processed['data'] = convert_numpy(processed['data'])
        return processed
    
    async def handler(self, websocket):
        """Handle incoming WebSocket connections."""
        try:
            await self.register(websocket)
            async for message in websocket:
                try:
                    data = json.loads(message)
                    self.logger.info(f"Received message: {data}")
                    
                    if data.get('type') == 'init':
                        client_id = data.get('data', {}).get('clientId')
                        self.logger.info(f"Client initialized: {client_id}")
                        
                        if self._last_data:
                            await websocket.send(json.dumps({
                                'type': 'full_state',
                                'data': self._last_data
                            }))
                        
                        await websocket.send(json.dumps({
                            'type': 'ack',
                            'data': {
                                'status': 'connected',
                                'clientId': client_id
                            }
                        }))
                    elif data.get('type') == 'control':
                        await self.handle_control_message(data)
                            
                except json.JSONDecodeError:
                    self.logger.error("Invalid JSON received")
                except Exception as e:
                    self.logger.error(f"Error handling message: {e}")
        except websockets.exceptions.ConnectionClosed:
            self.logger.info("Connection closed normally")
        except Exception as e:
            self.logger.error(f"Error in handler: {e}")
        finally:
            await self.unregister(websocket)
    
    async def start(self):
        """Start the WebSocket server."""
        self.running = True
        logger.info(f"Starting WebSocket server on ws://{self.host}:{self.port}")
        async with websockets.serve(
            self.handler,
            self.host,
            self.port,
            ping_interval=None,  
        ) as server:
            logger.info("WebSocket server is running")
            await asyncio.Future() 
    
    async def stop(self):
        """Stop the WebSocket server."""
        self.running = False
        for client in self.clients.copy():
            await client.close()
        self.clients.clear()
        logger.info("WebSocket server stopped")

async def run_test_server():
    """Run a test server that generates visualization data."""
    logger.info("Initializing test server...")
    generator = TestDataGenerator()
    server = ManifoldVisWebSocket()
    server.generator = generator
    
    async def send_test_data():
        """Send test data to connected clients."""
        logger.info("Starting test data generation...")
        while server.running:
            try:
                if server.clients:
                    traj_data = generator.generate_trajectory_point()
                    await server.broadcast(VisualizationData(
                        type='trajectory',
                        data=traj_data,
                        timestamp=time.time()
                    ))
                    logger.debug("Sent trajectory update")

                    metrics_data = generator.generate_metrics()
                    await server.broadcast(VisualizationData(
                        type='metrics',
                        data=metrics_data['data'],
                        timestamp=time.time()
                    ))
                    logger.debug("Sent metrics update")

                    manifold_data = generator.generate_manifold_data()
                    await server.broadcast(VisualizationData(
                        type='manifold',
                        data=manifold_data,
                        timestamp=time.time()
                    ))
                    logger.debug("Sent manifold update")

            except Exception as e:
                logger.error(f"Error in send_test_data: {e}")
                
            await asyncio.sleep(0.1)  

    server.running = True
    try:
        logger.info("Starting server and data generator...")
        await asyncio.gather(
            server.start(),
            send_test_data()
        )
    except Exception as e:
        logger.error(f"Error in test server: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting WebSocket test server script...")
    asyncio.run(run_test_server())