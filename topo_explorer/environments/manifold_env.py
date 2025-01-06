"""
ManifoldEnvironment: A Gymnasium environment for learning on geometric manifolds.
Provides a framework for RL agents to explore different geometric spaces including
spheres, tori, and hyperbolic spaces.
"""

from typing import Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

from .spaces import get_manifold

logger = logging.getLogger(__name__)

class ManifoldEnvironment(gym.Env):
    """A geometric environment that simulates movement on various manifolds."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, 
                 manifold_type: str = 'sphere', 
                 params: Optional[Dict] = None,
                 render_mode: Optional[str] = None):
        """Initialize the environment."""
        super().__init__()
        
        try:
            self.manifold = get_manifold(manifold_type, params)
        except ValueError as e:
            logger.error(f"Failed to create manifold: {e}")
            raise
            
        self.manifold_type = manifold_type
        self.render_mode = render_mode
        self.params = self.manifold.params
        
        self.current_position = self.manifold.random_point()
        self.trajectory = [self.current_position]
        self.frame = self.manifold.initial_frame(self.current_position)
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Dict({
            'position': spaces.Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(3,),
                dtype=np.float32
            ),
            'frame': spaces.Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(2, 3),
                dtype=np.float32
            ),
            'curvature': spaces.Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(1,),
                dtype=np.float32
            )
        })
        self.rewards = []

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Take a step in the environment."""
        old_position = self.current_position.copy()
        
        action = self.manifold.project_to_tangent(old_position, action)
        
        step_size = self.manifold.get_step_size(old_position)
        
        new_position = self.manifold.project_to_manifold(
            old_position + step_size * action
        )
        
        self.current_position = new_position
        self.trajectory.append(new_position)
        
        self.frame = self.manifold.parallel_transport(
            self.frame, 
            old_position,
            new_position - old_position
        )
        
        reward = self.manifold.compute_reward(old_position, new_position)
        self.rewards.append(reward)
        
        observation = self.get_state()
        
        terminated = len(self.trajectory) >= 500
        truncated = False
        
        info = {
            'step_size': step_size,
            'curvature': self.manifold.gaussian_curvature(self.current_position)
        }
        
        return observation, reward, terminated, truncated, info

    def reset(self, 
              seed: Optional[int] = None, 
              options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        self.current_position = self.manifold.random_point()
        self.trajectory = [self.current_position]
        self.frame = self.manifold.initial_frame(self.current_position)
        self.rewards = []
        
        return self.get_state(), {}

    def get_state(self) -> Dict:
        """Get current state dictionary."""
        return {
            'position': self.current_position,
            'frame': self.frame,
            'curvature': np.array([
                self.manifold.gaussian_curvature(self.current_position)
            ])
        }

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self.visualize_trajectory()

    def close(self):
        """Clean up resources."""
        plt.close()

    def _get_coverage(self) -> float:
        """Compute exploration coverage."""
        points = np.array(self.trajectory)
        dist_matrix = np.linalg.norm(points[:, None] - points, axis=2)
        max_possible = 2 * np.max(list(self.params.values()))
        return np.mean(dist_matrix) / max_possible

    def visualize_trajectory(self):
        """Visualize the trajectory and manifold."""
        trajectory = np.array(self.trajectory)
        fig = plt.figure(figsize=(15, 5))
        
        vis_data = self.manifold.get_visualization_data()
        
        if vis_data['type'] in ['sphere', 'torus', 'klein', 'mobius']:
            ax = fig.add_subplot(121, projection='3d')
            x, y, z = vis_data['surface']
            ax.plot_wireframe(x, y, z, **vis_data['wireframe_params'])
            
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                   'r-', label='Path', linewidth=2)
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                      color='red', s=100, label='Current')
            
            frame_scale = vis_data['frame_scale']
            pos = self.current_position
            frame = self.frame
            ax.quiver(pos[0], pos[1], pos[2], 
                     frame[0, 0], frame[0, 1], frame[0, 2],
                     color='blue', length=frame_scale, label='e1')
            ax.quiver(pos[0], pos[1], pos[2],
                     frame[1, 0], frame[1, 1], frame[1, 2],
                     color='green', length=frame_scale, label='e2')
            
        else:  
            ax = fig.add_subplot(121)
            if 'boundary_circle' in vis_data:
                circle = plt.Circle((0, 0), vis_data['boundary_circle']['radius'],
                                 fill=False, color='gray')
                ax.add_artist(circle)
            
            ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', 
                   label='Path', linewidth=2)
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                      color='red', s=100, label='Current')
            
            pos = self.current_position
            frame = self.frame
            frame_scale = vis_data['frame_scale']
            ax.quiver(pos[0], pos[1], frame[0, 0], frame[0, 1],
                     color='blue', scale=1/frame_scale, label='e1')
            ax.quiver(pos[0], pos[1], frame[1, 0], frame[1, 1],
                     color='green', scale=1/frame_scale, label='e2')
            
            if 'limits' in vis_data:
                ax.set_xlim(vis_data['limits']['x'])
                ax.set_ylim(vis_data['limits']['y'])
            ax.set_aspect('equal')
        
        ax.set_title(f'{self.manifold_type.capitalize()} Manifold')
        ax.legend()
        
        ax2 = fig.add_subplot(122)
        curvatures = [self.manifold.gaussian_curvature(pos) 
                     for pos in trajectory]
        ax2.plot(curvatures, linewidth=2)
        ax2.set_title('Gaussian Curvature along Path')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Curvature')
        
        plt.tight_layout()
        plt.show()

    def get_visualization_data(self) -> Dict:
        """Get data for visualization."""
        return self.manifold.get_visualization_data()

if __name__ == "__main__":
    env = ManifoldEnvironment(manifold_type='sphere', render_mode='human')
    obs = env.reset()
    env.render()