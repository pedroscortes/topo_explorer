"""Base class for manifold spaces."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, List
import numpy as np

class BaseManifold(ABC):
    """Abstract base class for manifold spaces."""
    
    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize the manifold.

        Args:
            params: Dictionary of manifold-specific parameters
        """
        self.params = params or self._default_params()
        self._visited_points = set()  
        self._visit_threshold = 0.1
        self.initial_point = self.random_point()

    def _is_previously_visited(self, point: np.ndarray) -> bool:
        """Check if a point has been previously visited within threshold."""
        point = np.array(point)
        for visited_point in self._visited_points:
            if np.linalg.norm(point - visited_point) < self._visit_threshold:
                return True
        return False

    def _mark_as_visited(self, point: np.ndarray):
        """Mark a point as visited."""
        self._visited_points.add(tuple(point)) 
    
    @abstractmethod
    def _default_params(self) -> Dict:
        """Return default parameters for this manifold."""
        pass
    
    @abstractmethod
    def random_point(self) -> np.ndarray:
        """Generate a random point on the manifold."""
        pass
    
    @abstractmethod
    def initial_frame(self, point: np.ndarray) -> np.ndarray:
        """Create an initial orthonormal frame at given point."""
        pass
    
    @abstractmethod
    def parallel_transport(self, 
                          frame: np.ndarray, 
                          point: np.ndarray,
                          displacement: np.ndarray) -> np.ndarray:
        """Parallel transport a frame along a displacement vector."""
        pass
    
    @abstractmethod
    def gaussian_curvature(self, point: np.ndarray) -> float:
        """Compute Gaussian curvature at a point."""
        pass
    
    @abstractmethod
    def project_to_manifold(self, point: np.ndarray) -> np.ndarray:
        """Project a point in ambient space onto the manifold."""
        pass
    
    @abstractmethod
    def project_to_tangent(self, 
                          point: np.ndarray, 
                          vector: np.ndarray) -> np.ndarray:
        """Project a vector onto the tangent space at a point."""
        pass
    
    @abstractmethod
    def get_step_size(self, point: np.ndarray) -> float:
        """Get appropriate step size for current position."""
        pass
    
    @abstractmethod
    def compute_reward(self, 
                      old_pos: np.ndarray, 
                      new_pos: np.ndarray) -> float:
        """Compute reward for moving from old_pos to new_pos."""
        pass
    
    @abstractmethod
    def get_visualization_data(self) -> Dict:
        """Return data needed for visualization."""
        pass

    @abstractmethod
    def should_terminate(self, point: np.ndarray, step_count: int, total_reward: float) -> bool:
        """Determine if episode should end."""
        pass