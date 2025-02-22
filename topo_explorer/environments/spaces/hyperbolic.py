"""Hyperbolic manifold implementation using the Poincaré disk model."""

import numpy as np
from typing import Dict, Optional, Tuple
from .base_manifold import BaseManifold

class HyperbolicManifold(BaseManifold):
    """
    Implementation of a hyperbolic space using the Poincaré disk model.
    
    This class represents the hyperbolic plane H² embedded in R³,
    using the Poincaré disk model where the disk |z| < 1 represents
    the entire hyperbolic plane.
    """

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)

    def should_terminate(self, point: np.ndarray, step_count: int, total_reward: float) -> bool:
        """Determine if episode should end."""
        distance_from_start = np.linalg.norm(point[:2] - self.initial_point[:2])
        r = np.linalg.norm(point[:2])
        near_boundary = r > self.params['boundary_threshold']
        return (step_count >= 200 or
                total_reward < -50.0 or
                near_boundary or
                (total_reward > 50.0 and distance_from_start > 0.5))    
    
    def _default_params(self) -> Dict:
        return {
            'k': -1.0,   
            'boundary_threshold': 0.99  
        }
    
    def random_point(self) -> np.ndarray:
        """Generate random point in Poincaré disk."""
        r = np.random.uniform(0, 0.9)  
        theta = np.random.uniform(0, 2 * np.pi)
        return np.array([
            r * np.cos(theta),
            r * np.sin(theta),
            0
        ])
    
    def initial_frame(self, point: np.ndarray) -> np.ndarray:
        """Create orthonormal frame using hyperbolic metric."""
        pos = point[:2]
        theta = np.arctan2(pos[1], pos[0])
        
        e1 = np.array([np.cos(theta), np.sin(theta), 0])
        e2 = np.array([-np.sin(theta), np.cos(theta), 0])
        
        scale = 1 / (1 - np.sum(pos**2))
        return scale * np.stack([e1, e2])
    
    def parallel_transport(self, 
                         frame: np.ndarray, 
                         point: np.ndarray,
                         displacement: np.ndarray) -> np.ndarray:
        """Parallel transport in hyperbolic space."""
        new_pos = self.project_to_manifold(point + displacement)
        scale = 1 / (1 - np.sum(new_pos[:2]**2))
        
        new_frame = []
        for vec in frame:
            transported = scale * vec
            if np.linalg.norm(transported) > 0:
                transported = transported / np.linalg.norm(transported)
            new_frame.append(transported)
            
        return np.stack(new_frame)
    
    def gaussian_curvature(self, point: np.ndarray) -> float:
        """Return constant negative curvature."""
        return self.params['k']
    
    def project_to_manifold(self, point: np.ndarray) -> np.ndarray:
        """Project point onto Poincaré disk."""
        point = point.copy()
        r = np.linalg.norm(point[:2])
        if r >= self.params['boundary_threshold']:
            point[:2] = point[:2] / r * (self.params['boundary_threshold'])
        point[2] = 0  
        return point
    
    def project_to_tangent(self, 
                          point: np.ndarray, 
                          vector: np.ndarray) -> np.ndarray:
        """Project vector onto tangent space of hyperbolic plane."""
        vector = vector.copy()
        vector[2] = 0  
        return vector
    
    def get_step_size(self, point: np.ndarray) -> float:
        """Return step size that decreases near boundary."""
        r = np.linalg.norm(point[:2])
        return 0.1 * (1 - r)  
    
    def compute_reward(self, 
                      old_pos: np.ndarray, 
                      new_pos: np.ndarray) -> float:
        """
        Compute reward encouraging exploration while respecting boundary.
        """
        distance_moved = np.linalg.norm(new_pos - old_pos)
        
        r = np.linalg.norm(new_pos[:2])
        boundary_reward = r / (1 - r)  
        
        x1, y1 = old_pos[:2]
        x2, y2 = new_pos[:2]
        d = 2 * np.arccosh(1 + 2 * ((x2-x1)**2 + (y2-y1)**2) / 
                          ((1-x1**2-y1**2)*(1-x2**2-y2**2)))
        
        return distance_moved + 0.3 * boundary_reward + 0.2 * d
    
    def get_visualization_data(self) -> Dict:
        """Return data for visualizing the Poincaré disk."""
        return {
            'type': 'hyperbolic',
            'boundary_circle': {'radius': 1.0, 'color': 'gray', 'fill': False},
            'frame_scale': 0.2,
            'limits': {'x': (-1.1, 1.1), 'y': (-1.1, 1.1)}
        }