"""Toroidal manifold implementation."""

import numpy as np
from typing import Dict, Optional, Tuple
from .base_manifold import BaseManifold

class TorusManifold(BaseManifold):
    """
    Implementation of a torus manifold.
    
    This class represents a torus embedded in R³, defined by two radii:
    R (major radius, distance from the center of the tube to the center of the torus)
    r (minor radius, radius of the tube)
    """
    
    def _default_params(self) -> Dict:
        return {
            'R': 3.0,  
            'r': 1.0   
        }
    
    def random_point(self) -> np.ndarray:
        """Generate a random point on the torus using uniform parameters."""
        u = np.random.uniform(0, 2 * np.pi)  
        v = np.random.uniform(0, 2 * np.pi)  
        R, r = self.params['R'], self.params['r']
        
        return np.array([
            (R + r * np.cos(v)) * np.cos(u),
            (R + r * np.cos(v)) * np.sin(u),
            r * np.sin(v)
        ])
    
    def initial_frame(self, point: np.ndarray) -> np.ndarray:
        """Create orthonormal frame using the natural torus coordinates."""
        R, r = self.params['R'], self.params['r']
        u = np.arctan2(point[1], point[0])
        v = np.arctan2(point[2], np.sqrt(point[0]**2 + point[1]**2) - R)
        
        e1 = np.array([-np.sin(u), np.cos(u), 0])
        
        e2 = np.array([
            np.cos(u) * (-np.sin(v)),
            np.sin(u) * (-np.sin(v)),
            np.cos(v)
        ])
        
        return np.stack([e1, e2])
    
    def parallel_transport(self, 
                         frame: np.ndarray, 
                         point: np.ndarray,
                         displacement: np.ndarray) -> np.ndarray:
        """Parallel transport frame along geodesic."""
        new_pos = self.project_to_manifold(point + displacement)
        R, r = self.params['R'], self.params['r']
        
        u = np.arctan2(new_pos[1], new_pos[0])
        d = np.sqrt(new_pos[0]**2 + new_pos[1]**2)
        v = np.arctan2(new_pos[2], d - R)
        
        normal = np.array([
            np.cos(u) * np.cos(v),
            np.sin(u) * np.cos(v),
            np.sin(v)
        ])
        
        new_frame = []
        for vec in frame:
            transported = vec - np.dot(vec, normal) * normal
            transported = transported / np.linalg.norm(transported)
            new_frame.append(transported)
            
        return np.stack(new_frame)
    
    def gaussian_curvature(self, point: np.ndarray) -> float:
        """Compute Gaussian curvature which varies with position."""
        R, r = self.params['R'], self.params['r']
        u = np.arctan2(point[1], point[0])
        v = np.arctan2(point[2], np.sqrt(point[0]**2 + point[1]**2) - R)
        return np.cos(v) / (r * (R + r * np.cos(v)))
    
    def project_to_manifold(self, point: np.ndarray) -> np.ndarray:
        """Project point onto torus surface."""
        R, r = self.params['R'], self.params['r']
        u = np.arctan2(point[1], point[0])
        d = np.sqrt(point[0]**2 + point[1]**2)
        v = np.arctan2(point[2], d - R)
        
        return np.array([
            (R + r * np.cos(v)) * np.cos(u),
            (R + r * np.cos(v)) * np.sin(u),
            r * np.sin(v)
        ])
    
    def project_to_tangent(self, 
                          point: np.ndarray, 
                          vector: np.ndarray) -> np.ndarray:
        """Project vector onto tangent space of torus at point."""
        R, r = self.params['R'], self.params['r']
        u = np.arctan2(point[1], point[0])
        v = np.arctan2(point[2], np.sqrt(point[0]**2 + point[1]**2) - R)
        
        normal = np.array([
            np.cos(u) * np.cos(v),
            np.sin(u) * np.cos(v),
            np.sin(v)
        ])
        
        return vector - np.dot(vector, normal) * normal
    
    def get_step_size(self, point: np.ndarray) -> float:
        """Return appropriate step size based on torus parameters."""
        return 0.2 * min(self.params['R'], self.params['r'])
    
    def compute_reward(self, 
                      old_pos: np.ndarray, 
                      new_pos: np.ndarray) -> float:
        """
        Compute reward encouraging exploration of both major and minor circles.
        """
        distance_moved = np.linalg.norm(new_pos - old_pos)
        
        u_old = np.arctan2(old_pos[1], old_pos[0])
        u_new = np.arctan2(new_pos[1], new_pos[0])
        major_circle_progress = abs(u_new - u_old)
        
        R = self.params['R']
        v_old = np.arctan2(old_pos[2], np.sqrt(old_pos[0]**2 + old_pos[1]**2) - R)
        v_new = np.arctan2(new_pos[2], np.sqrt(new_pos[0]**2 + new_pos[1]**2) - R)
        minor_circle_progress = abs(v_new - v_old)
        
        return distance_moved + 0.5 * major_circle_progress + 0.3 * minor_circle_progress
    
    def get_visualization_data(self) -> Dict:
        """Return data for visualizing the torus."""
        R, r = self.params['R'], self.params['r']
        u, v = np.mgrid[0:2*np.pi:30j, 0:2*np.pi:20j]
        
        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v)
        
        return {
            'type': 'torus',
            'surface': (x, y, z),
            'frame_scale': 0.3,
            'wireframe_params': {'color': 'gray', 'alpha': 0.2}
        }