"""Möbius strip manifold implementation."""

import numpy as np
from typing import Dict, Optional, Tuple
from .base_manifold import BaseManifold

class MobiusManifold(BaseManifold):
    """
    Implementation of the Möbius strip.
    
    This class represents the Möbius strip as a non-orientable surface
    embedded in R³, parameterized by a length coordinate u and a width
    coordinate v, with a twist as u goes around the strip.
    """
    
    def _default_params(self) -> Dict:
        return {
            'R': 2.0,    # Major radius
            'width': 1.0  # Width of strip
        }
    
    def random_point(self) -> np.ndarray:
        """Generate random point on Möbius strip."""
        u = np.random.uniform(0, 2 * np.pi)  # Position around strip
        v = np.random.uniform(-1, 1)         # Position across width
        
        return self._parametric_point(u, v)
    
    def _parametric_point(self, u: float, v: float) -> np.ndarray:
        """Convert from parameter space to R³ coordinates."""
        R = self.params['R']
        w = self.params['width']
        
        # Standard parameterization of Möbius strip
        x = (R + w * v * np.cos(u/2)) * np.cos(u)
        y = (R + w * v * np.cos(u/2)) * np.sin(u)
        z = w * v * np.sin(u/2)
        
        return np.array([x, y, z])
    
    def _get_local_parameters(self, point: np.ndarray) -> Tuple[float, float]:
        """Convert from R³ coordinates to parameter space."""
        x, y, z = point
        R = self.params['R']
        w = self.params['width']
        
        # Get u parameter (angle around strip)
        u = np.arctan2(y, x)
        if u < 0:
            u += 2 * np.pi
            
        # Get v parameter (position across width)
        r = np.sqrt(x*x + y*y)
        v = z / (w * np.sin(u/2)) if abs(np.sin(u/2)) > 1e-6 else \
            (r - R) / (w * np.cos(u/2))
            
        return u, v
    
    def initial_frame(self, point: np.ndarray) -> np.ndarray:
        """Create orthonormal frame using local parameters."""
        u, v = self._get_local_parameters(point)
        R = self.params['R']
        w = self.params['width']
        
        # Tangent vector in u direction (around strip)
        du = np.array([
            -R * np.sin(u) - w * v * (0.5 * np.sin(u/2) * np.cos(u) + np.cos(u/2) * np.sin(u)),
            R * np.cos(u) + w * v * (0.5 * np.sin(u/2) * np.sin(u) - np.cos(u/2) * np.cos(u)),
            0.5 * w * v * np.cos(u/2)
        ])
        
        # Tangent vector in v direction (across width)
        dv = np.array([
            w * np.cos(u/2) * np.cos(u),
            w * np.cos(u/2) * np.sin(u),
            w * np.sin(u/2)
        ])
        
        # Normalize frame vectors
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
        
        # Check if we've gone around the strip
        crossed_identification = abs(new_u - old_u) > np.pi
        
        new_frame = []
        for vec in frame:
            # Basic parallel transport
            transported = vec.copy()
            
            # If we crossed the identification, apply reflection
            if crossed_identification:
                transported = -transported
                
            transported = transported / np.linalg.norm(transported)
            new_frame.append(transported)
            
        return np.stack(new_frame)
    
    def gaussian_curvature(self, point: np.ndarray) -> float:
        """Compute Gaussian curvature at point."""
        u, v = self._get_local_parameters(point)
        R = self.params['R']
        w = self.params['width']
        
        # Curvature formula for Möbius strip
        denom = R + w * v * np.cos(u/2)
        return -np.cos(u/2) / (denom * (1 + (w*v/(2*denom))**2)**2)
    
    def project_to_manifold(self, point: np.ndarray) -> np.ndarray:
        """Project point onto Möbius strip surface."""
        u, v = self._get_local_parameters(point)
        # Clamp v to strip width
        v = np.clip(v, -1, 1)
        return self._parametric_point(u, v)
    
    def project_to_tangent(self, 
                          point: np.ndarray, 
                          vector: np.ndarray) -> np.ndarray:
        """Project vector onto tangent space."""
        frame = self.initial_frame(point)
        coeffs = np.array([np.dot(vector, basis) for basis in frame])
        return coeffs[0] * frame[0] + coeffs[1] * frame[1]
    
    def get_step_size(self, point: np.ndarray) -> float:
        """Return step size based on geometry."""
        return 0.1 * min(self.params['R'], self.params['width'])
    
    def compute_reward(self, 
                      old_pos: np.ndarray, 
                      new_pos: np.ndarray) -> float:
        """
        Compute reward encouraging exploration of the non-orientable structure.
        """
        # Basic distance reward
        distance = np.linalg.norm(new_pos - old_pos)
        
        # Parameters
        old_u, old_v = self._get_local_parameters(old_pos)
        new_u, new_v = self._get_local_parameters(new_pos)
        
        # Reward for exploring length of strip
        u_progress = abs(new_u - old_u)
        
        # Reward for exploring width of strip
        v_progress = abs(new_v - old_v)
        
        # Extra reward for crossing identification (completing a loop)
        identification_reward = 1.0 if abs(new_u - old_u) > np.pi else 0.0
        
        return distance + 0.3 * u_progress + 0.2 * v_progress + \
               0.5 * identification_reward
    
    def get_visualization_data(self) -> Dict:
        """Return data for visualizing the Möbius strip."""
        R = self.params['R']
        w = self.params['width']
        
        # Create a grid in parameter space
        u = np.linspace(0, 2*np.pi, 100)
        v = np.linspace(-1, 1, 20)
        u, v = np.meshgrid(u, v)
        
        # Convert to cartesian coordinates
        x = (R + w * v * np.cos(u/2)) * np.cos(u)
        y = (R + w * v * np.cos(u/2)) * np.sin(u)
        z = w * v * np.sin(u/2)
        
        return {
            'type': 'mobius',
            'surface': (x, y, z),
            'frame_scale': 0.2,
            'wireframe_params': {
                'color': 'gray',
                'alpha': 0.2,
                'edge_colors': 'black'
            },
            'colormap_data': {
                'values': v,  # Use v coordinate for coloring
                'cmap': 'viridis',  # Color scheme
                'label': 'Width Parameter'
            }
        }
        
    def get_geodesics(self, 
                     start_point: np.ndarray, 
                     end_point: np.ndarray, 
                     num_points: int = 100) -> np.ndarray:
        """
        Compute geodesic between two points on the Möbius strip.
        
        Args:
            start_point: Starting point on the strip
            end_point: Ending point on the strip
            num_points: Number of points to sample along geodesic
            
        Returns:
            Array of points along the geodesic
        """
        # Get parameters for start and end points
        u1, v1 = self._get_local_parameters(start_point)
        u2, v2 = self._get_local_parameters(end_point)
        
        # Handle the case where the geodesic crosses the identification
        if abs(u2 - u1) > np.pi:
            if u2 > u1:
                u1 += 2*np.pi
            else:
                u2 += 2*np.pi
        
        # Linear interpolation in parameter space
        t = np.linspace(0, 1, num_points)
        u = u1 + t*(u2 - u1)
        v = v1 + t*(v2 - v1)
        
        # Convert to cartesian coordinates
        points = np.array([self._parametric_point(ui, vi) 
                          for ui, vi in zip(u, v)])
        
        return points