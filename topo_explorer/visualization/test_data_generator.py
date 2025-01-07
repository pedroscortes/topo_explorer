"""Test data generator with RL agent integration."""
import numpy as np
import time
import math
import logging
import torch
from ..environments.manifold_env import ManifoldEnvironment

logger = logging.getLogger(__name__)

class TestDataGenerator:
    """Generates test data for visualization."""
    
    def __init__(self):
        """Initialize the test data generator."""
        logger.info("Initializing TestDataGenerator")
        self.trajectory = []
        self.step = 0
        self.last_time = time.time()
        self.curvature_history = []
        self.is_training = False
        
        try:
            self.current_manifold = 'sphere' 
            logger.info("Creating initial ManifoldEnvironment")
            self.env = ManifoldEnvironment(manifold_type=self.current_manifold)
            state = self.env.reset()[0]
            self.trajectory = [state['position']]
            logger.info("TestDataGenerator initialization complete")
        except Exception as e:
            logger.error(f"Error initializing TestDataGenerator: {e}")
            raise
        
    def generate_trajectory_point(self) -> dict:
        """Generate a point on a manifold."""
        try:
            current_time = time.time()
            t = current_time - self.last_time
            
            if self.is_training:
                action = np.array([
                    np.cos(t),
                    np.sin(t),
                    0.5 * np.sin(2*t)
                ])
                state, _, _, _, _ = self.env.step(action)
                point = state['position']
            else:
                theta = t % (2 * math.pi)
                phi = (t * 0.5) % math.pi
                point = np.array([
                    math.sin(phi) * math.cos(theta),
                    math.sin(phi) * math.sin(theta),
                    math.cos(phi)
                ]) * 2.0  
            
            if len(self.trajectory) > 50:
                self.trajectory = self.trajectory[-49:]
            self.trajectory.append(point)
            
            return {
                'points': np.array(self.trajectory).tolist()  
            }
        except Exception as e:
            logger.error(f"Error generating trajectory point: {e}")
            return {'points': []}
        
    def generate_metrics(self) -> dict:
        """Generate training metrics."""
        current_time = time.time()
        t = current_time - self.last_time
        self.step += 1
        
        state = self.env.get_state()
        point = state['position']
        curvature = self.env.manifold.gaussian_curvature(point)
        
        self.curvature_history.append({
            'step': self.step,
            'value': curvature
        })
        if len(self.curvature_history) > 50:
            self.curvature_history = self.curvature_history[-50:]
            
        return {
            'type': 'metrics',
            'data': {
                'training': {
                    'step': self.step,
                    'value_loss': math.sin(t) * 0.5 + 0.5,
                    'policy_loss': math.cos(t) * 0.3 + 0.5,
                    'coverage': len(self.trajectory) / 50.0
                },
                'curvature': self.curvature_history
            }
        }
        
    def generate_manifold_data(self) -> dict:
        """Generate manifold visualization data."""
        try:
            vis_data = self.env.get_visualization_data()
            surface_data = vis_data.get('surface')
            
            if isinstance(surface_data, tuple) and len(surface_data) == 3:
                x, y, z = surface_data
                return {
                    'type': self.current_manifold,
                    'radius': 2.0,
                    'surface': {
                        'x': x.tolist() if hasattr(x, 'tolist') else x,
                        'y': y.tolist() if hasattr(y, 'tolist') else y,
                        'z': z.tolist() if hasattr(z, 'tolist') else z,
                    },
                    'wireframe_params': {
                        'color': 'gray',
                        'alpha': 0.2
                    }
                }
            
            return {
                'type': self.current_manifold,
                'radius': 2.0,
                'wireframe_params': {
                    'color': 'gray',
                    'alpha': 0.2
                }
            }
        except Exception as e:
            logger.error(f"Error generating manifold data: {e}")
            return {
                'type': 'sphere',
                'radius': 2.0,
                'wireframe_params': {
                    'color': 'gray',
                    'alpha': 0.2
                }
            }
        
    def start_training(self):
        """Start the training process."""
        logger.info("Starting training")
        self.is_training = True
    
    def stop_training(self):
        """Stop the training process."""
        logger.info("Stopping training")
        self.is_training = False

    def set_manifold_type(self, manifold_type: str):
        """Change the current manifold type."""
        try:
            logger.info(f"Changing manifold type from {self.current_manifold} to {manifold_type}")
            self.current_manifold = manifold_type
            self.env = ManifoldEnvironment(manifold_type=manifold_type)
            state = self.env.reset()[0]
            self.trajectory = [state['position']]
            self.curvature_history = []
            self.step = 0
            logger.info("Manifold change successful")
        except Exception as e:
            logger.error(f"Error changing manifold: {e}")
            raise
        
    def reset_trajectory(self):
        """Reset the trajectory for a new manifold."""
        pass