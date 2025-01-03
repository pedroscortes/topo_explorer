"""Klein bottle manifold implementation."""

import numpy as np
from typing import Dict, Optional, Tuple
from .base_manifold import BaseManifold

class KleinBottleManifold(BaseManifold):
    """
    Implementation of the Klein bottle.
    
    This class represents the Klein bottle as an immersion in R³,
    although it cannot be embedded without self-intersection.
    We use the "figure-8" immersion.
    """
    
    def _default_params(self) -> Dict:
        return {
            'R': 2.0,  
            'r': 0.5,   
            'twist_rate': 2.0  
        }
    
    def random_point(self) -> np.ndarray:
        """Generate random point on Klein bottle."""
        u = np.random.uniform(0, 2 * np.pi)  
        v = np.random.uniform(0, 2 * np.pi)  
        
        return self._parametric_point(u, v)
    
    def _parametric_point(self, u: float, v: float) -> np.ndarray:
        """Convert from parameter space to R³ coordinates."""
        R, r = self.params['R'], self.params['r']
        
        cosu, sinu = np.cos(u), np.sin(u)
        cosv, sinv = np.cos(v), np.sin(v)
        
        x = (R + r * cosv * cosu) * np.cos(u/2)
        y = (R + r * cosv * cosu) * np.sin(u/2)
        z = r * sinv * cosu
        
        return np.array([x, y, z])
    
    def _get_local_parameters(self, point: np.ndarray) -> Tuple[float, float]:
        """Convert from R³ coordinates to parameter space."""
        x, y, z = point
        
        u = 2 * np.arctan2(y, x)
        
        R = self.params['R']
        proj_dist = np.sqrt(x*x + y*y) - R
        v = np.arctan2(z, proj_dist)
        
        return u, v
    
    def initial_frame(self, point: np.ndarray) -> np.ndarray:
        """Create orthonormal frame using local parameters."""
        u, v = self._get_local_parameters(point)
        
        du = np.array([
            -np.sin(u/2),
            np.cos(u/2),
            0
        ])
        
        dv = np.array([
            -np.sin(v) * np.cos(u/2),
            -np.sin(v) * np.sin(u/2),
            np.cos(v)
        ])
        
        du = du / np.linalg.norm(du)
        dv = dv / np.linalg.norm(dv)
        
        return np.stack([du, dv])
    
    def parallel_transport(self, 
                         frame: np.ndarray, 
                         point: np.ndarray,
                         displacement: np.ndarray) -> np.ndarray:
        """Parallel transport frame along geodesic."""
        old_u, old_v = self._get_local_parameters(point)
        new_pos = self.project_to_manifold(point + displacement)
        new_u, new_v = self._get_local_parameters(new_pos)
        
        crossed_identification = abs(new_u - old_u) > np.pi
        
        new_frame = []
        for vec in frame:
            transported = vec.copy()
            
            if crossed_identification:
                transported = -transported
                
            transported = transported / np.linalg.norm(transported)
            new_frame.append(transported)
            
        return np.stack(new_frame)
    
    def gaussian_curvature(self, point: np.ndarray) -> float:
        """Compute Gaussian curvature at point."""
        u, v = self._get_local_parameters(point)
        R, r = self.params['R'], self.params['r']
        
        cos_u = np.cos(u)
        return -cos_u / (r * (R + r * cos_u))
    
    def project_to_manifold(self, point: np.ndarray) -> np.ndarray:
        """Project point onto Klein bottle surface."""
        u, v = self._get_local_parameters(point)
        
        return self._parametric_point(u, v)
    
    def project_to_tangent(self, 
                          point: np.ndarray, 
                          vector: np.ndarray) -> np.ndarray:
        """Project vector onto tangent space."""
        frame = self.initial_frame(point)
        
        coeffs = np.array([np.dot(vector, basis) for basis in frame])
        
        return coeffs[0] * frame[0] + coeffs[1] * frame[1]
    
    def get_step_size(self, point: np.ndarray) -> float:
        """Return step size based on local geometry."""
        return 0.1 * min(self.params['R'], self.params['r'])
    
    def compute_reward(self, 
                      old_pos: np.ndarray, 
                      new_pos: np.ndarray) -> float:
        """
        Compute reward encouraging exploration of the non-orientable structure.
        """
        distance = np.linalg.norm(new_pos - old_pos)
        
        old_u, old_v = self._get_local_parameters(old_pos)
        new_u, new_v = self._get_local_parameters(new_pos)
        
        u_progress = abs(new_u - old_u)
        
        v_progress = abs(new_v - old_v)
        
        identification_reward = 1.0 if abs(new_u - old_u) > np.pi else 0.0
        
        return distance + 0.3 * u_progress + 0.2 * v_progress + 0.5 * identification_reward
    
    def get_visualization_data(self) -> Dict:
        """Return data for visualizing Klein bottle."""
        R, r = self.params['R'], self.params['r']
        u = np.linspace(0, 2*np.pi, 50)
        v = np.linspace(0, 2*np.pi, 25)
        u, v = np.meshgrid(u, v)
        
        x = (R + r * np.cos(v) * np.cos(u)) * np.cos(u/2)
        y = (R + r * np.cos(v) * np.cos(u)) * np.sin(u/2)
        z = r * np.sin(v) * np.cos(u)
        
        return {
            'type': 'klein',
            'surface': (x, y, z),
            'frame_scale': 0.2,
            'wireframe_params': {'color': 'gray', 'alpha': 0.2}
        }