"""Visualization module for manifold environments."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Optional, List, Tuple
import matplotlib.animation as animation
from scipy import stats

class ManifoldVisualizer:
    """Visualizer for manifold geometry and agent trajectories."""
    
    def __init__(self, figsize: Tuple[int, int] = (20, 6)):
        """Initialize visualizer."""
        self.figsize = figsize
        self.fig = None
        self.ax_3d = None
        self.ax_curvature = None
        self.ax_metrics = None
        self.last_vis_data = None  
    
    def setup_plot(self):
        """Setup matplotlib figure with more informative plots."""
        plt.close('all')
        self.fig = plt.figure(figsize=(20, 12))
        gs = self.fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.3, wspace=0.3)
        
        self.ax_3d = self.fig.add_subplot(gs[0, 0], projection='3d')
        self.ax_learning = self.fig.add_subplot(gs[0, 1]) 
        self.ax_exploration = self.fig.add_subplot(gs[0, 2])  
        
        self.ax_curvature = self.fig.add_subplot(gs[1, 0], projection='3d')
        self.ax_geodesics = self.fig.add_subplot(gs[1, 1], projection='3d')
        self.ax_value = self.fig.add_subplot(gs[1, 2])
        
        self.ax_3d.set_title("Current Trajectory")
        self.ax_learning.set_title("Training Progress")
        self.ax_exploration.set_title("Exploration Density")
        self.ax_curvature.set_title("Curvature Distribution")
        self.ax_geodesics.set_title("Recent Geodesics")
        self.ax_value.set_title("Value Function")
        
    def plot_manifold(self, vis_data: Dict):
        """Plot manifold surface."""
        self.last_vis_data = vis_data  
        if not self.fig:
            self.setup_plot()
        
        manifold_type = vis_data['type']
        
        if manifold_type == 'sphere':
            x, y, z = vis_data['surface']
            self.ax_3d.plot_wireframe(x, y, z, 
                                    color='gray',
                                    alpha=0.3,
                                    linewidth=0.5,
                                    rstride=2,
                                    cstride=2)
            
            self.ax_3d.view_init(elev=20, azim=45)
            self.ax_3d.set_box_aspect([1,1,1])
            
            self.ax_3d.grid(False)
            self.ax_3d.set_facecolor('white')
            self.ax_3d.xaxis.pane.fill = False
            self.ax_3d.yaxis.pane.fill = False
            self.ax_3d.zaxis.pane.fill = False
            
            self.ax_3d.set_title("Sphere Manifold", pad=10)
    
    def plot_trajectory(self, 
                       trajectory: np.ndarray,
                       curvatures: Optional[np.ndarray] = None):
        """Plot agent trajectory and curvatures."""
        if len(trajectory) < 2:
            return
            
        points = trajectory[-50:]  
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(points)-1))
        
        for i in range(len(points)-1):
            self.ax_3d.plot3D(points[i:i+2, 0],
                            points[i:i+2, 1],
                            points[i:i+2, 2],
                            color=colors[i],
                            linewidth=2,
                            alpha=0.8)
        
        self.ax_3d.scatter(points[-1, 0],
                          points[-1, 1],
                          points[-1, 2],
                          color='red',
                          s=100,
                          label='Current')
        
        if curvatures is not None and len(curvatures) > 0:
            self.ax_curvature.clear()
            steps = np.arange(len(curvatures))
            self.ax_curvature.plot(steps, curvatures, 'b-', linewidth=2)
            self.ax_curvature.fill_between(steps, 0, curvatures, alpha=0.2)
            self.ax_curvature.set_title('Gaussian Curvature along Path')
            self.ax_curvature.set_xlabel('Step')
            self.ax_curvature.set_ylabel('Curvature')
            self.ax_curvature.grid(True, alpha=0.3)

    def _get_state_representation(self, point, env):
        """Match the state dimension expected by the agent (11)"""
        old_pos = env.current_position
        env.current_position = point
        frame = env._initial_frame()
        env.current_position = old_pos
        
        return np.concatenate([
            point,  
            [env.gaussian_curvature(point)],  
            frame.flatten(),  
            [0.0]  
        ])    

    def plot_geodesics(self, points, num_steps=50):
        """Plot geodesic paths between points"""
        u = np.linspace(0, 1, num_steps)
        paths = []
        
        for i in range(len(points)-1):
            start, end = points[i], points[i+1]
            start = start / np.linalg.norm(start)
            end = end / np.linalg.norm(end)
            
            omega = np.arccos(np.clip(np.dot(start, end), -1.0, 1.0))
            if omega < 1e-6:
                continue
                
            path = np.array([(np.sin((1-t)*omega)*start + np.sin(t*omega)*end)/np.sin(omega) 
                        for t in u])
            paths.append(path)
        
        return paths

    def plot_curvature_heatmap(self, manifold, resolution=50):
        theta = np.linspace(0, 2*np.pi, resolution)
        phi = np.linspace(0, np.pi, resolution)
        theta, phi = np.meshgrid(theta, phi)
        
        r = manifold.params['radius']
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        curvature = np.ones_like(theta) * (1.0 / (r * r))
        return x, y, z, curvature

    def plot_value_function(self, agent, env):
        theta = np.linspace(0, 2*np.pi, 50)
        phi = np.linspace(0, np.pi, 50)
        values = np.zeros((len(theta), len(phi)))
        
        for i, th in enumerate(theta):
            for j, ph in enumerate(phi):
                point = env.params['radius'] * np.array([
                    np.sin(ph) * np.cos(th),
                    np.sin(ph) * np.sin(th),
                    np.cos(ph)
                ])
                state = self._get_state_representation(point, env)
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    _, value = agent.forward(state_tensor)
                    values[i,j] = value.item()
        
        values = (values - values.min()) / (values.max() - values.min() + 1e-8)
        return theta, phi, values     
    
    def plot_metrics(self, metrics):
        """Plot learning curves on the middle plots."""
        if 'value_loss' in metrics:
            self.ax_curvature.clear()
            self.ax_curvature.plot(metrics['value_loss'], label='Value Loss')
            self.ax_curvature.set_yscale('log')
            self.ax_curvature.set_xlabel('Steps')
            self.ax_curvature.grid(True)
            self.ax_curvature.legend()

    def plot_frame(self, position: np.ndarray, frame: np.ndarray, scale: float = 0.3):
        """Plot parallel transport frame at current position.
        
        Args:
            position: Current position on manifold
            frame: Current frame vectors
            scale: Scale factor for frame vectors
        """
        colors = ['blue', 'green']
        labels = ['e1', 'e2']
        
        for i, vec in enumerate(frame):
            scaled_vec = scale * vec
            
            self.ax_3d.quiver(position[0], position[1], position[2],
                            scaled_vec[0], scaled_vec[1], scaled_vec[2],
                            color=colors[i],
                            linewidth=2,
                            label=labels[i])
            
    def plot_geometric_view(self, env, agent, trajectory):
        """Plot geometric visualization without using PolyCollection."""
        if not self.fig:
            self.setup_plot()
            
        r = env.params['radius']
        theta = np.linspace(0, 2*np.pi, 30)
        phi = np.linspace(0, np.pi, 30)
        THETA, PHI = np.meshgrid(theta, phi)
        
        X = r * np.sin(PHI) * np.cos(THETA)
        Y = r * np.sin(PHI) * np.sin(THETA)
        Z = r * np.cos(PHI)
        
        points = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
        colors = plt.cm.viridis(np.ones(len(points))/2)  
        self.ax_curvature.scatter(points[:, 0], points[:, 1], points[:, 2], 
                                c=colors, alpha=0.1, s=1)
        self.ax_curvature.set_title('Curvature Map')
        
        self.ax_geodesics.plot_wireframe(X, Y, Z, color='gray', alpha=0.1, rstride=2, cstride=2)
        
        recent_points = trajectory[-10:]
        for i in range(len(recent_points)-1):
            path = self.plot_geodesics([recent_points[i], recent_points[i+1]])[0]
            self.ax_geodesics.plot(path[:,0], path[:,1], path[:,2], 'r-', linewidth=2)
            self.ax_geodesics.scatter(recent_points[i][0], recent_points[i][1], 
                                recent_points[i][2], color='blue', s=50)
        
        self.ax_curvature.view_init(elev=30, azim=45)
        self.ax_geodesics.view_init(elev=30, azim=45)
        
        theta, phi, values = self.plot_value_function(agent, env)
        im = self.ax_value.imshow(values.T, origin='lower', 
                            extent=[0, 2*np.pi, 0, np.pi],
                            aspect='auto', cmap='viridis')
        plt.colorbar(im, ax=self.ax_value)
        self.ax_value.set_xlabel('θ')
        self.ax_value.set_ylabel('φ')

        plt.tight_layout()

    def plot_exploration_density(self, trajectory, ax=None):
        """Plot density of explored regions using scatter plot instead of hexbin"""
        if ax is None:
            ax = self.ax_exploration

        if ax is None:
            return

        ax.clear()
        points = np.array(trajectory)
        xy = np.arctan2(points[:,1], points[:,0])
        z = np.arccos(np.clip(points[:,2] / np.linalg.norm(points, axis=1), -1.0, 1.0))
        
        density = stats.gaussian_kde([xy, z])(np.vstack([xy, z]))
        idx = np.argsort(density)
        
        scatter = ax.scatter(xy[idx], z[idx], c=density[idx], 
                            cmap='viridis', s=50, alpha=0.5)
        plt.colorbar(scatter, ax=ax, label='Visit Density')
        
        ax.set_title('Exploration Density')
        ax.set_xlabel('θ')
        ax.set_ylabel('φ')
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(0, np.pi)   

    def plot_learning_progress(self, metrics):
        """Plot learning curves and statistics."""
        self.ax_learning.clear()
        
        value_losses = metrics.get('value_loss', [])
        if value_losses:
            self.ax_learning.plot(value_losses, label='Value Loss', color='blue')
            self.ax_learning.set_yscale('log')
        
        ax2 = self.ax_learning.twinx()
        policy_losses = metrics.get('policy_loss', [])
        if policy_losses:
            ax2.plot(policy_losses, label='Policy Loss', color='red', linestyle='--')
        
        self.ax_learning.set_xlabel('Steps')
        self.ax_learning.set_ylabel('Value Loss')
        ax2.set_ylabel('Policy Loss')
        
        lines1, labels1 = self.ax_learning.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')          
    
    def show(self):
        """Display the visualization."""
        if self.fig:
            plt.draw()
            plt.pause(0.1)  
            
    def save(self, filename: str):
        """Save current figure."""
        if self.fig:
            self.fig.savefig(filename)

    def create_animation(self, 
                         trajectory: List[np.ndarray],
                         frames: List[np.ndarray],
                         curvatures: List[float],
                         interval: int = 50) -> animation.FuncAnimation:
        """Create animation of agent's movement on manifold."""
        trajectory = np.array(trajectory)
        frames = np.array(frames)
        curvatures = np.array(curvatures)
        
        plt.close('all')
        self.fig = plt.figure(figsize=(20, 6), dpi=150)
        gs = self.fig.add_gridspec(1, 3)
        self.ax_3d = self.fig.add_subplot(gs[0], projection='3d')
        self.ax_curvature = self.fig.add_subplot(gs[1])
        self.ax_metrics = self.fig.add_subplot(gs[2])
        
        def update(frame_idx):
            self.ax_3d.cla()
            self.ax_curvature.cla()
            self.ax_metrics.cla()
            
            if self.last_vis_data['type'] == 'sphere':
                x, y, z = self.last_vis_data['surface']
                self.ax_3d.plot_wireframe(x, y, z,
                                        color='lightgray',
                                        alpha=0.3,
                                        linewidth=0.5,
                                        rstride=2,
                                        cstride=2)
            
            if len(trajectory) > 1:
                for i in range(len(trajectory)-1):
                    alpha = min(1.0, (i - frame_idx + 50) / 50) if i <= frame_idx else 0.0
                    if alpha > 0:
                        color = plt.cm.viridis(i/len(trajectory))
                        self.ax_3d.plot3D(trajectory[i:i+2, 0],
                                        trajectory[i:i+2, 1],
                                        trajectory[i:i+2, 2],
                                        color=color,
                                        linewidth=2,
                                        alpha=alpha)
            
            if frame_idx < len(trajectory):
                self.ax_3d.scatter(*trajectory[frame_idx],
                                 color='red',
                                 s=100,
                                 edgecolor='white',
                                 linewidth=2,
                                 label='Current Position')
                
                if frame_idx < len(frames):
                    current_frame = frames[frame_idx]
                    colors = ['blue', 'green']
                    labels = ['e₁', 'e₂']
                    scale = 0.3
                    
                    for i, vec in enumerate(current_frame):
                        scaled_vec = scale * vec
                        self.ax_3d.quiver(trajectory[frame_idx, 0],
                                        trajectory[frame_idx, 1],
                                        trajectory[frame_idx, 2],
                                        scaled_vec[0], scaled_vec[1], scaled_vec[2],
                                        color=colors[i],
                                        linewidth=2,
                                        label=labels[i])
            
            self.ax_3d.view_init(elev=20, azim=(45 + frame_idx/2) % 360)  # Rotate view
            self.ax_3d.set_title("Manifold Exploration\nStep: {:d}".format(frame_idx))
            self.ax_3d.set_box_aspect([1,1,1])
            self.ax_3d.grid(False)
            self.ax_3d.legend()
            
            current_curvatures = curvatures[:frame_idx+1]
            if len(current_curvatures) > 0:
                steps = np.arange(len(current_curvatures))
                self.ax_curvature.plot(steps, current_curvatures, 'b-', linewidth=2)
                self.ax_curvature.fill_between(steps, 0, current_curvatures, 
                                             alpha=0.2, color='blue')
                self.ax_curvature.set_title('Gaussian Curvature along Path')
                self.ax_curvature.set_xlabel('Step')
                self.ax_curvature.set_ylabel('Curvature')
                self.ax_curvature.grid(True, alpha=0.3)
            
            if frame_idx > 0:
                phi = np.linspace(0, np.pi, 20)
                theta = np.linspace(0, 2*np.pi, 40)
                phi, theta = np.meshgrid(phi, theta)
                
                r = self.last_vis_data['surface'][0].shape[0]/2
                x = r * np.sin(phi) * np.cos(theta)
                y = r * np.sin(phi) * np.sin(theta)
                z = r * np.cos(phi)
                
                points = np.stack([x.flatten(), y.flatten(), z.flatten()]).T
                current_traj = trajectory[:frame_idx+1]
                
                distances = np.min([np.linalg.norm(points - p, axis=1) 
                                  for p in current_traj], axis=0)
                coverage = np.exp(-distances)
                
                self.ax_metrics.hist(coverage, bins=30, density=True,
                                   alpha=0.6, color='green')
                self.ax_metrics.set_title('Coverage Distribution')
                self.ax_metrics.set_xlabel('Coverage')
                self.ax_metrics.set_ylabel('Density')
                self.ax_metrics.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return self.ax_3d, self.ax_curvature, self.ax_metrics
        
        anim = animation.FuncAnimation(
            self.fig, 
            update,
            frames=len(trajectory),
            interval=interval,
            blit=False
        )
        
        return anim
    
    def save_animation(self, anim: animation.FuncAnimation, filename: str):
        """Save animation to file."""
        print(f"Saving animation to {filename}...")
        writer = animation.PillowWriter(fps=30)
        anim.save(filename, writer=writer)
        print(f"Animation saved successfully!")