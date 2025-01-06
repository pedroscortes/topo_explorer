"""N-dimensional torus manifold implementation."""

import numpy as np
from typing import Dict, Optional, Tuple, List
from .base_manifold import BaseManifold

class NTorusManifold(BaseManifold):
    """
    Implementation of an n-dimensional torus T^n.
    
    This class represents the n-torus as a product of circles S¹ × ... × S¹.
    Each circle can have its own radius. The manifold is embedded in R^(2n).
    """

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)

    def should_terminate(self, point: np.ndarray, step_count: int, total_reward: float) -> bool:
        """Determine if episode should end."""
        n = self.params['dimension']
        angles = self._embedding_to_angles(point)
        initial_angles = self._embedding_to_angles(self.initial_point)
        angle_diffs = np.abs(angles - initial_angles)
        has_explored = np.mean(angle_diffs) > np.pi/2
        return (step_count >= 200 or
                total_reward < -50.0 or
                (total_reward > 50.0 and has_explored))    
    
    def _default_params(self) -> Dict:
        return {
            'dimension': 2,  
            'radii': [1.0, 1.0]  
        }
    
    def __init__(self, params: Optional[Dict] = None):
        """Initialize with dimension check."""
        super().__init__(params)
        assert len(self.params['radii']) == self.params['dimension'], \
            "Number of radii must match dimension"
    
    def random_point(self) -> np.ndarray:
        """Generate random point on n-torus."""
        n = self.params['dimension']
        angles = np.random.uniform(0, 2*np.pi, n)
        return self._angles_to_embedding(angles)
    
    def _angles_to_embedding(self, angles: np.ndarray) -> np.ndarray:
        """Convert from angle parameters to embedded coordinates."""
        n = self.params['dimension']
        radii = self.params['radii']
        coords = np.zeros(2*n)
        
        for i in range(n):
            coords[2*i] = radii[i] * np.cos(angles[i])
            coords[2*i + 1] = radii[i] * np.sin(angles[i])
            
        return coords
    
    def _embedding_to_angles(self, point: np.ndarray) -> np.ndarray:
        """Convert from embedded coordinates to angle parameters."""
        n = self.params['dimension']
        angles = np.zeros(n)
        
        for i in range(n):
            angles[i] = np.arctan2(point[2*i + 1], point[2*i])
            
        return angles
    
    def initial_frame(self, point: np.ndarray) -> np.ndarray:
        """Create orthonormal frame using product structure."""
        n = self.params['dimension']
        radii = self.params['radii']
        angles = self._embedding_to_angles(point)
        frame = []
        
        for i in range(n):
            basis = np.zeros(2*n)
            basis[2*i] = -radii[i] * np.sin(angles[i])
            basis[2*i + 1] = radii[i] * np.cos(angles[i])
            frame.append(basis / np.linalg.norm(basis))
            
        return np.stack(frame)
    
    def parallel_transport(self, 
                         frame: np.ndarray, 
                         point: np.ndarray,
                         displacement: np.ndarray) -> np.ndarray:
        """Parallel transport frame using product structure."""
        n = self.params['dimension']
        new_pos = self.project_to_manifold(point + displacement)
        
        old_angles = self._embedding_to_angles(point)
        new_angles = self._embedding_to_angles(new_pos)
        angle_diffs = new_angles - old_angles
        
        new_frame = []
        for vec in frame:
            transported = np.zeros_like(vec)
            for i in range(n):
                cos_d = np.cos(angle_diffs[i])
                sin_d = np.sin(angle_diffs[i])
                transported[2*i] = cos_d * vec[2*i] - sin_d * vec[2*i + 1]
                transported[2*i + 1] = sin_d * vec[2*i] + cos_d * vec[2*i + 1]
            
            new_frame.append(transported / np.linalg.norm(transported))
            
        return np.stack(new_frame)
    
    def gaussian_curvature(self, point: np.ndarray) -> float:
        """Return zero (flat metric on torus)."""
        return 0.0
    
    def project_to_manifold(self, point: np.ndarray) -> np.ndarray:
        """Project point onto n-torus."""
        n = self.params['dimension']
        radii = self.params['radii']
        angles = self._embedding_to_angles(point)
        return self._angles_to_embedding(angles)
    
    def project_to_tangent(self, 
                          point: np.ndarray, 
                          vector: np.ndarray) -> np.ndarray:
        """Project vector onto tangent space."""
        frame = self.initial_frame(point)
        coeffs = np.array([np.dot(vector, basis) for basis in frame])
        return sum(c * basis for c, basis in zip(coeffs, frame))
    
    def get_step_size(self, point: np.ndarray) -> float:
        """Return step size based on smallest radius."""
        return 0.1 * min(self.params['radii'])
    
    def compute_reward(self, 
                      old_pos: np.ndarray, 
                      new_pos: np.ndarray) -> float:
        """Compute reward encouraging exploration of all factors."""
        distance = np.linalg.norm(new_pos - old_pos)
        
        old_angles = self._embedding_to_angles(old_pos)
        new_angles = self._embedding_to_angles(new_pos)
        angle_progress = np.abs(new_angles - old_angles)
        
        return distance + 0.2 * np.sum(angle_progress)
    
    def get_visualization_data(self) -> Dict:
        """Return data for visualizing (only for 2-torus)."""
        if self.params['dimension'] != 2:
            raise NotImplementedError(
                "Visualization only implemented for 2-torus")
            
        R, r = self.params['radii']
        u = np.linspace(0, 2*np.pi, 30)
        v = np.linspace(0, 2*np.pi, 20)
        u, v = np.meshgrid(u, v)
        
        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v)
        
        return {
            'type': 'torus',
            'surface': (x, y, z),
            'frame_scale': 0.3,
            'wireframe_params': {'color': 'gray', 'alpha': 0.2}
        }