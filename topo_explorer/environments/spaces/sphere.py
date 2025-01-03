"""Spherical manifold implementation."""

import numpy as np
from typing import Dict, Optional, Tuple
from .base_manifold import BaseManifold

class SphereManifold(BaseManifold):
    """
    Implementation of a spherical manifold.
    
    This class represents a 2-sphere (S²) embedded in R³.
    """
    
    def _default_params(self) -> Dict:
        return {'radius': 2.0}
    
    def random_point(self) -> np.ndarray:
        """Generate a random point on the sphere using uniform spherical coordinates."""
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.arccos(np.random.uniform(-1, 1))
        r = self.params['radius']
        
        return r * np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ])
    
    def initial_frame(self, point: np.ndarray) -> np.ndarray:
        """Create orthonormal frame at given point using spherical coordinates."""
        r = self.params['radius']
        theta = np.arctan2(point[1], point[0])
        phi = np.arccos(point[2] / r)
        
        e1 = np.array([np.cos(theta) * np.cos(phi),
                      np.sin(theta) * np.cos(phi),
                      -np.sin(phi)])
        e2 = np.array([-np.sin(theta),
                      np.cos(theta),
                      0])
        
        return np.stack([e1, e2])
    
    def parallel_transport(self, 
                         frame: np.ndarray, 
                         point: np.ndarray,
                         displacement: np.ndarray) -> np.ndarray:
        """Parallel transport frame along geodesic."""
        new_pos = self.project_to_manifold(point + displacement)
        normal = new_pos / np.linalg.norm(new_pos)
        
        new_frame = []
        for vec in frame:
            transported = vec - np.dot(vec, normal) * normal
            transported = transported / np.linalg.norm(transported)
            new_frame.append(transported)
            
        return np.stack(new_frame)
    
    def gaussian_curvature(self, point: np.ndarray) -> float:
        """Return constant curvature 1/r²."""
        return 1.0 / (self.params['radius'] ** 2)
    
    def project_to_manifold(self, point: np.ndarray) -> np.ndarray:
        """Project point onto sphere by normalizing."""
        return point * self.params['radius'] / np.linalg.norm(point)
    
    def project_to_tangent(self, 
                          point: np.ndarray, 
                          vector: np.ndarray) -> np.ndarray:
        """Project vector onto tangent space using normal projection."""
        normal = point / np.linalg.norm(point)
        return vector - np.dot(vector, normal) * normal
    
    def get_step_size(self, point: np.ndarray) -> float:
        """Return constant step size relative to radius."""
        return 0.1 * self.params['radius']
    
    def compute_reward(self, old_pos: np.ndarray, new_pos: np.ndarray) -> float:
        """Compute reward based on distance moved and curvature discovery."""
        distance_moved = np.linalg.norm(new_pos - old_pos)
        angular_distance = np.arccos(np.clip(np.dot(old_pos, new_pos) / (self.params['radius'] ** 2), -1.0, 1.0))
        
        curvature = self.gaussian_curvature(new_pos)
        exploration_bonus = 1.0 if not self._is_previously_visited(new_pos) else 0.0
        geodesic_alignment = np.abs(np.dot(new_pos - old_pos, self.project_to_tangent(old_pos, new_pos - old_pos)))
        
        return (0.3 * distance_moved + 
                0.3 * angular_distance + 
                0.2 * curvature + 
                0.1 * exploration_bonus +
                0.1 * geodesic_alignment)
    
    def should_terminate(self, point: np.ndarray, step_count: int, total_reward: float) -> bool:
        distance_from_start = np.linalg.norm(point - self.initial_point)
        has_explored = distance_from_start > 0.5 * self.params['radius']
        return (step_count >= 200 or  
                total_reward < -50.0 or  
                (total_reward > 50.0 and has_explored)) 
    
    def get_visualization_data(self) -> Dict:
        """Return data for visualizing the sphere."""
        r = self.params['radius']
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        
        x = r * np.cos(u) * np.sin(v)
        y = r * np.sin(u) * np.sin(v)
        z = r * np.cos(v)
        
        return {
            'type': 'sphere',
            'surface': (x, y, z),
            'frame_scale': 0.5,
            'wireframe_params': {'color': 'gray', 'alpha': 0.2}
        }