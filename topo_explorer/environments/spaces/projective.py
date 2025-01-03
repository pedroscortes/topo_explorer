"""Real projective space implementation."""

import numpy as np
from typing import Dict, Optional, Tuple
from .base_manifold import BaseManifold

class ProjectiveManifold(BaseManifold):
    """
    Implementation of real projective space RP².
    
    This class represents RP² as a quotient of the sphere S² by the antipodal map.
    Points (x,y,z) and (-x,-y,-z) are identified as the same point in RP².
    """
    
    def _default_params(self) -> Dict:
        return {'radius': 1.0}  # Unit sphere for canonical representation
    
    def random_point(self) -> np.ndarray:
        """Generate random point in RP² using hemisphere representation."""
        # Generate point on unit sphere
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.arccos(np.random.uniform(-1, 1))
        point = np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ])
        
        # Project to upper hemisphere (z ≥ 0)
        if point[2] < 0:
            point = -point
            
        return self.params['radius'] * point
    
    def initial_frame(self, point: np.ndarray) -> np.ndarray:
        """Create orthonormal frame at point."""
        # Normalize point to unit sphere
        p = point / np.linalg.norm(point)
        
        # First basis vector in theta direction
        theta = np.arctan2(p[1], p[0])
        e1 = np.array([-np.sin(theta), np.cos(theta), 0])
        
        # Second basis vector in phi direction
        e2 = np.cross(p, e1)
        
        return np.stack([e1, e2])
    
    def parallel_transport(self, 
                         frame: np.ndarray, 
                         point: np.ndarray,
                         displacement: np.ndarray) -> np.ndarray:
        """Parallel transport frame along geodesic."""
        old_p = point / np.linalg.norm(point)
        new_pos = self.project_to_manifold(point + displacement)
        new_p = new_pos / np.linalg.norm(new_pos)
        
        # If crossed to lower hemisphere, flip coordinates
        if np.dot(old_p, new_p) < 0:
            new_p = -new_p
            new_pos = -new_pos
        
        new_frame = []
        for vec in frame:
            # Parallel transport on sphere
            transported = vec - (np.dot(vec, new_p) * new_p)
            transported = transported / np.linalg.norm(transported)
            new_frame.append(transported)
            
        return np.stack(new_frame)
    
    def gaussian_curvature(self, point: np.ndarray) -> float:
        """Return constant positive curvature."""
        return 1.0 / (self.params['radius'] ** 2)
    
    def project_to_manifold(self, point: np.ndarray) -> np.ndarray:
        """Project point onto RP² (represented as hemisphere)."""
        p = point / np.linalg.norm(point)
        if p[2] < 0:
            p = -p
        return self.params['radius'] * p
    
    def project_to_tangent(self, 
                          point: np.ndarray, 
                          vector: np.ndarray) -> np.ndarray:
        """Project vector onto tangent space."""
        p = point / np.linalg.norm(point)
        return vector - np.dot(vector, p) * p
    
    def get_step_size(self, point: np.ndarray) -> float:
        """Return constant step size relative to radius."""
        return 0.1 * self.params['radius']
    
    def compute_reward(self, 
                      old_pos: np.ndarray, 
                      new_pos: np.ndarray) -> float:
        """Compute reward based on geodesic distance."""
        old_p = old_pos / np.linalg.norm(old_pos)
        new_p = new_pos / np.linalg.norm(new_pos)
        
        # Compute angle between points (considering antipodal identification)
        cos_angle = abs(np.dot(old_p, new_p))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return angle * self.params['radius']
    
    def get_visualization_data(self) -> Dict:
        """Return data for visualizing RP² as a hemisphere."""
        r = self.params['radius']
        u = np.linspace(0, 2*np.pi, 30)
        v = np.linspace(0, np.pi/2, 15)  # Only upper hemisphere
        u, v = np.meshgrid(u, v)
        
        x = r * np.cos(u) * np.sin(v)
        y = r * np.sin(u) * np.sin(v)
        z = r * np.cos(v)
        
        return {
            'type': 'projective',
            'surface': (x, y, z),
            'frame_scale': 0.3,
            'wireframe_params': {'color': 'gray', 'alpha': 0.2}
        }