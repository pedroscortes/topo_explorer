"""
ManifoldEnvironment: A Gymnasium environment for learning on geometric manifolds.
Provides a framework for RL agents to explore different geometric spaces including
spheres, tori, and hyperbolic spaces.
"""

# Standard library imports
from typing import Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

class ManifoldEnvironment(gym.Env):
    """
    A geometric environment that simulates movement on various manifolds.
    Implements the Gymnasium interface for reinforcement learning.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, 
                 manifold_type: str = 'sphere', 
                 params: Optional[Dict] = None,
                 render_mode: Optional[str] = None):
        """
        Initialize the manifold environment.

        Args:
            manifold_type (str): Type of manifold ('sphere', 'torus', or 'hyperbolic')
            params (dict, optional): Parameters defining the manifold geometry
            render_mode (str, optional): Mode for rendering ('human' or 'rgb_array')
        """
        super().__init__()
        
        self.manifold_type = manifold_type
        self.params = params or self._default_params()
        self.render_mode = render_mode
        
        # Initialize state
        self.current_position = self._random_point()
        self.trajectory = [self.current_position]
        self.frame = self._initial_frame()
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),  # 3D movement
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

    def _default_params(self) -> dict:
        """Define default parameters for each manifold type."""
        defaults = {
            'sphere': {'radius': 2.0},
            'torus': {'R': 3.0, 'r': 1.0},
            'hyperbolic': {'k': -1.0}
        }
        return defaults[self.manifold_type]
    
    def _random_point(self) -> np.ndarray:
        """Generate a random point on the manifold."""
        if self.manifold_type == 'sphere':
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.arccos(np.random.uniform(-1, 1))
            return self.params['radius'] * np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])
        
        elif self.manifold_type == 'torus':
            u = np.random.uniform(0, 2 * np.pi)
            v = np.random.uniform(0, 2 * np.pi)
            R, r = self.params['R'], self.params['r']
            return np.array([
                (R + r * np.cos(v)) * np.cos(u),
                (R + r * np.cos(v)) * np.sin(u),
                r * np.sin(v)
            ])
        
        elif self.manifold_type == 'hyperbolic':
            r = np.random.uniform(0, 0.9)
            theta = np.random.uniform(0, 2 * np.pi)
            return np.array([r * np.cos(theta), r * np.sin(theta), 0])

    def _initial_frame(self) -> np.ndarray:
        """Initialize the parallel transport frame at the current position."""
        if self.manifold_type == 'sphere':
            pos = self.current_position
            r = self.params['radius']
            theta = np.arctan2(pos[1], pos[0])
            phi = np.arccos(pos[2] / r)
            
            e1 = np.array([np.cos(theta) * np.cos(phi),
                          np.sin(theta) * np.cos(phi),
                          -np.sin(phi)])
            e2 = np.array([-np.sin(theta),
                          np.cos(theta),
                          0])
            return np.stack([e1, e2])
        
        elif self.manifold_type == 'torus':
            pos = self.current_position
            R, r = self.params['R'], self.params['r']
            u = np.arctan2(pos[1], pos[0])
            v = np.arctan2(pos[2], np.sqrt(pos[0]**2 + pos[1]**2) - R)
            
            e1 = np.array([-np.sin(u), np.cos(u), 0])
            e2 = np.array([np.cos(u) * (-np.sin(v)),
                          np.sin(u) * (-np.sin(v)),
                          np.cos(v)])
            return np.stack([e1, e2])
        
        elif self.manifold_type == 'hyperbolic':
            pos = self.current_position[:2]
            theta = np.arctan2(pos[1], pos[0])
            e1 = np.array([np.cos(theta), np.sin(theta), 0])
            e2 = np.array([-np.sin(theta), np.cos(theta), 0])
            scale = 1 / (1 - np.sum(pos**2))
            return scale * np.stack([e1, e2])

    def parallel_transport(self, frame: np.ndarray, displacement: np.ndarray) -> np.ndarray:
        """
        Parallel transport a frame along a geodesic defined by displacement.

        Args:
            frame (np.ndarray): Current orthonormal frame
            displacement (np.ndarray): Displacement vector

        Returns:
            np.ndarray: Transported frame
        """
        if self.manifold_type == 'sphere':
            new_pos = self.current_position + displacement
            new_pos = new_pos / np.linalg.norm(new_pos) * self.params['radius']
            normal = new_pos / np.linalg.norm(new_pos)
            
            new_frame = []
            for vec in frame:
                transported = vec - np.dot(vec, normal) * normal
                transported = transported / np.linalg.norm(transported)
                new_frame.append(transported)
            return np.stack(new_frame)
            
        elif self.manifold_type == 'torus':
            new_pos = self.current_position + displacement
            R, r = self.params['R'], self.params['r']
            u = np.arctan2(new_pos[1], new_pos[0])
            d = np.sqrt(new_pos[0]**2 + new_pos[1]**2)
            v = np.arctan2(new_pos[2], d - R)
            normal = np.array([np.cos(u) * np.cos(v),
                            np.sin(u) * np.cos(v),
                            np.sin(v)])
            
            new_frame = []
            for vec in frame:
                transported = vec - np.dot(vec, normal) * normal
                transported = transported / np.linalg.norm(transported)
                new_frame.append(transported)
            return np.stack(new_frame)
            
        elif self.manifold_type == 'hyperbolic':
            new_pos = self.current_position + displacement
            scale = 1 / (1 - np.sum(new_pos[:2]**2))
            new_frame = []
            for vec in frame:
                transported = scale * vec
                if np.linalg.norm(transported) > 0:
                    transported = transported / np.linalg.norm(transported)
                new_frame.append(transported)
            return np.stack(new_frame)

    def gaussian_curvature(self, point: np.ndarray) -> float:
        """
        Compute the Gaussian curvature at a point on the manifold.

        Args:
            point (np.ndarray): Point on the manifold

        Returns:
            float: Gaussian curvature at the point
        """
        if self.manifold_type == 'sphere':
            return 1.0 / (self.params['radius'] ** 2)
        
        elif self.manifold_type == 'torus':
            R, r = self.params['R'], self.params['r']
            u = np.arctan2(point[1], point[0])
            v = np.arctan2(point[2], np.sqrt(point[0]**2 + point[1]**2) - R)
            return np.cos(v) / (r * (R + r * np.cos(v)))
        
        elif self.manifold_type == 'hyperbolic':
            return self.params['k']

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Take a step in the environment.

        Args:
            action: Action vector to take

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Store current position for reward computation
        old_position = self.current_position.copy()
        
        # Project action to tangent space and take step
        if self.manifold_type == 'sphere':
            # Project action to tangent space
            normal = old_position / np.linalg.norm(old_position)
            action = action - np.dot(action, normal) * normal
            action = action / (np.linalg.norm(action) + 1e-8)
            
            # Adaptive step size based on exploration history
            min_dist_to_history = float('inf')
            if len(self.trajectory) > 1:
                dists = np.linalg.norm(
                    np.array(self.trajectory) - old_position, axis=1)
                min_dist_to_history = np.min(dists)
            
            # Larger steps when in well-explored areas
            step_size = 0.2 * self.params['radius'] * (1.0 + 0.5 * np.exp(-min_dist_to_history))
            
            # Take step
            new_position = old_position + step_size * action
            # Project back to sphere
            new_position = new_position * self.params['radius'] / np.linalg.norm(new_position)
            
        elif self.manifold_type == 'torus':
            # Implementation for torus
            pass
            
        elif self.manifold_type == 'hyperbolic':
            # Implementation for hyperbolic space
            pass
            
        # Update state
        self.current_position = new_position
        self.trajectory.append(new_position)
        
        # Update frame via parallel transport
        self.frame = self.parallel_transport(self.frame, new_position - old_position)
        
        # Compute reward
        reward = self.compute_reward(old_position, new_position)
        
        # Get observation
        observation = {
            'position': self.current_position,
            'frame': self.frame,
            'curvature': np.array([self.gaussian_curvature(self.current_position)])
        }
        
        # Check if episode should end
        terminated = len(self.trajectory) >= 500  # Maximum episode length
        truncated = False
        
        # Additional info for visualization and analysis
        info = {
            'step_size': step_size if 'step_size' in locals() else 0.0,
            'curvature': self.gaussian_curvature(self.current_position),
            'exploration_score': min_dist_to_history if 'min_dist_to_history' in locals() else float('inf')
        }
        
        return observation, reward, terminated, truncated, info
        
    def _safe_angle(self, v1: np.ndarray, v2: np.ndarray, eps: float = 1e-8) -> float:
        """Safely compute angle between vectors."""
        norm_prod = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_prod < eps:
            return 0.0
        cos_angle = np.clip(np.dot(v1, v2) / norm_prod, -1.0 + eps, 1.0 - eps)
        return np.arccos(cos_angle)
    
    def compute_reward(self, old_pos: np.ndarray, new_pos: np.ndarray) -> float:
        """
        Compute reward based on exploration and geometric properties.
        
        Args:
            old_pos: Previous position on manifold
            new_pos: New position on manifold
        
        Returns:
            float: Computed reward value
        """
        if self.manifold_type == 'sphere':
            r = self.params['radius']
            eps = 1e-8
            
            # 1. Movement reward (geodesic distance)
            norm_prod = np.linalg.norm(old_pos) * np.linalg.norm(new_pos)
            if norm_prod < eps:
                movement_reward = 0.0
            else:
                cos_angle = np.clip(np.dot(old_pos, new_pos) / norm_prod, -1.0 + eps, 1.0 - eps)
                geodesic_dist = r * np.arccos(cos_angle)
                movement_reward = geodesic_dist
            
            # 2. Exploration reward
            exploration_reward = 0.0
            coverage_reward = 0.0
            
            if len(self.trajectory) > 1:
                trajectory_array = np.array(self.trajectory[:-1])
                dists_to_history = np.linalg.norm(trajectory_array - new_pos, axis=1)
                min_dist = np.min(dists_to_history)
                mean_dist = np.mean(dists_to_history)
                exploration_reward = min_dist * 2.0 + mean_dist * 0.5
                
                if len(self.trajectory) > 2:
                    prev_dir = self.trajectory[-2] - self.trajectory[-3]
                    curr_dir = new_pos - self.trajectory[-2]
                    angle = self._safe_angle(prev_dir, curr_dir)
                    coverage_reward = np.sin(angle)
            
            # 3. Smoothness penalty
            smoothness_penalty = 0.0
            if len(self.trajectory) > 2:
                last_three = np.array(self.trajectory[-3:] + [new_pos])
                diffs = np.diff(last_three, axis=0)
                angles = []
                for i in range(len(diffs) - 1):
                    angle = self._safe_angle(diffs[i], diffs[i+1])
                    angles.append(angle)
                smoothness_penalty = np.sum(angles) * 0.1
            
            # 4. Progress penalty
            progress_penalty = 1.0 if movement_reward < 0.1 * r else 0.0
            
            # 5. Area coverage reward
            area_reward = 0.0
            if len(self.trajectory) > 10:
                recent_positions = np.array(self.trajectory[-10:] + [new_pos])
                pdist = np.zeros((len(recent_positions), len(recent_positions)))
                for i in range(len(recent_positions)):
                    for j in range(i+1, len(recent_positions)):
                        angle = self._safe_angle(recent_positions[i], recent_positions[j])
                        pdist[i,j] = pdist[j,i] = angle
                area_reward = np.mean(pdist) * r * 0.5
            
            # Combine rewards with weights
            total_reward = (
                movement_reward * 1.0 +       # Base movement
                exploration_reward * 2.0 +    # Exploring new areas
                coverage_reward * 1.0 +       # Angular coverage
                area_reward * 1.0 -           # Area coverage
                smoothness_penalty * 1.0 -    # Smoothness
                progress_penalty * 0.5        # Progress
            )
            
            # Clip reward for stability
            total_reward = np.clip(total_reward, -10.0, 10.0)
            return float(total_reward)
            
        return 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment to initial state.

        Args:
            seed (int, optional): Random seed
            options (dict, optional): Additional options

        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        self.current_position = self._random_point()
        self.trajectory = [self.current_position]
        self.frame = self._initial_frame()
        
        observation = {
            'position': self.current_position,
            'frame': self.frame,
            'curvature': np.array([self.gaussian_curvature(self.current_position)])
        }
        
        return observation, {}

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self.visualize_trajectory()

    def close(self):
        """Clean up resources."""
        plt.close()
 
    def get_visualization_data(self) -> dict:
        """Get data needed for visualization."""
        if self.manifold_type == 'sphere':
            r = self.params['radius']
            # Increase resolution of sphere wireframe
            u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
            x = r * np.cos(u) * np.sin(v)
            y = r * np.sin(u) * np.sin(v)
            z = r * np.cos(v)
            return {
                'type': 'sphere',
                'surface': (x, y, z),
                'frame_scale': 0.3,
                'wireframe_params': {
                    'color': 'gray',
                    'alpha': 0.1,
                    'linewidth': 0.5
                }
            }
            
        elif self.manifold_type == 'torus':
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
            
        elif self.manifold_type == 'hyperbolic':
            return {
                'type': 'hyperbolic',
                'boundary_circle': {'radius': 1.0, 'color': 'gray'},
                'limits': {'x': (-1.1, 1.1), 'y': (-1.1, 1.1)},
                'frame_scale': 0.2
            }
            
        else:
            raise NotImplementedError(
                f"Visualization not implemented for {self.manifold_type}"
            )
        
    def visualize_trajectory(self):
        trajectory = np.array(self.trajectory)
        fig = plt.figure(figsize=(15, 5))
        
        if self.manifold_type in ['sphere', 'torus']:
            ax = fig.add_subplot(121, projection='3d')
            
            # Set frame scale based on manifold
            if self.manifold_type == 'sphere':
                frame_scale = 0.5
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                r = self.params['radius']
                x = r * np.cos(u) * np.sin(v)
                y = r * np.sin(u) * np.sin(v)
                z = r * np.cos(v)
                ax.plot_wireframe(x, y, z, color='gray', alpha=0.2)
            else:  # torus
                frame_scale = 0.3
                u, v = np.mgrid[0:2*np.pi:30j, 0:2*np.pi:20j]
                R, r = self.params['R'], self.params['r']
                x = (R + r * np.cos(v)) * np.cos(u)
                y = (R + r * np.cos(v)) * np.sin(u)
                z = r * np.sin(v)
                ax.plot_wireframe(x, y, z, color='gray', alpha=0.2)
            
            # Plot trajectory and frame
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'r-', 
                   label='Path', linewidth=2)
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                      color='red', s=100, label='Current')
            
            pos = self.current_position
            frame = self.frame
            ax.quiver(pos[0], pos[1], pos[2], 
                     frame[0, 0], frame[0, 1], frame[0, 2],
                     color='blue', length=frame_scale, label='e1')
            ax.quiver(pos[0], pos[1], pos[2],
                     frame[1, 0], frame[1, 1], frame[1, 2],
                     color='green', length=frame_scale, label='e2')
            
        else:  # Hyperbolic plane
            ax = fig.add_subplot(121)
            circle = plt.Circle((0, 0), 1, fill=False, color='gray')
            ax.add_artist(circle)
            ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', 
                   label='Path', linewidth=2)
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                      color='red', s=100, label='Current')
            
            pos = self.current_position
            frame = self.frame
            r = np.linalg.norm(pos[:2])
            frame_scale = 0.2 * (1 - r)  # Scale with distance from boundary
            
            ax.quiver(pos[0], pos[1], frame[0, 0], frame[0, 1],
                     color='blue', scale=1/frame_scale, label='e1')
            ax.quiver(pos[0], pos[1], frame[1, 0], frame[1, 1],
                     color='green', scale=1/frame_scale, label='e2')
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_aspect('equal')
        
        ax.set_title(f'{self.manifold_type.capitalize()} Manifold')
        ax.legend()
        
        # Plot curvature
        ax2 = fig.add_subplot(122)
        curvatures = [self.gaussian_curvature(pos) for pos in trajectory]
        ax2.plot(curvatures, linewidth=2)
        ax2.set_title('Gaussian Curvature along Path')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Curvature')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Example usage
    env = ManifoldEnvironment(manifold_type='sphere', render_mode='human')
    obs = env.reset()
    env.render()