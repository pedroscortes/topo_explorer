"""Test data generator for visualization."""
import numpy as np
import time
import math
import logging
logger = logging.getLogger(__name__)

class TestDataGenerator:
    """Generates test data for visualization."""
    def __init__(self):
        self.trajectory = []
        self.step = 0
        self.last_time = time.time()
        self.curvature_history = []
        self.is_training = False  
        
    def start_training(self):
        """Start the training process."""
        logger.info("Starting training")
        self.is_training = True
        
    def stop_training(self):
        """Stop the training process."""
        logger.info("Stopping training")
        self.is_training = False
        
    def generate_trajectory_point(self) -> dict:
        """Generate a point on a sphere."""
        current_time = time.time()
        t = current_time - self.last_time
        theta = t % (2 * math.pi)
        phi = (t * 0.5) % math.pi
        point = np.array([
            math.sin(phi) * math.cos(theta),
            math.sin(phi) * math.sin(theta),
            math.cos(phi)
        ])
        
        if len(self.trajectory) > 50:
            self.trajectory = self.trajectory[-49:]
        self.trajectory.append(point)
        logger.debug(f"Generated trajectory point {len(self.trajectory)}")
        
        return {
            'points': np.array(self.trajectory)
        }
        
    def calculate_gaussian_curvature(self, point) -> float:
        """Calculate Gaussian curvature at a point on a unit sphere.
        For a sphere, the Gaussian curvature is constant: K = 1/R^2
        where R is the radius (1 in our case)."""
        return 1.0
        
    def generate_metrics(self) -> dict:
        """Generate training metrics."""
        current_time = time.time()
        t = current_time - self.last_time
        
        if self.is_training:
            self.step += 1
            value_loss = math.sin(t) * 0.5 + 0.5
            policy_loss = math.cos(t) * 0.3 + 0.5
        else:
            value_loss = 0
            policy_loss = 0
            
        if self.trajectory:
            current_point = self.trajectory[-1]
            curvature = self.calculate_gaussian_curvature(current_point)
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
                    'value_loss': value_loss,
                    'policy_loss': policy_loss,
                    'coverage': len(self.trajectory) / 50.0
                },
                'curvature': self.curvature_history
            }
        }
        
    def generate_manifold_data(self) -> dict:
        """Generate manifold geometric data."""
        return {
            'type': 'sphere',
            'radius': 1.0,
            'resolution': 32,
            'properties': {
                'color': 0x156289,
                'opacity': 0.7,
            }
        }