"""Visualization module for manifold environments."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Optional, List, Tuple
import matplotlib.animation as animation

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
        """Setup matplotlib figure with subplots."""
        plt.close('all')  
        self.fig = plt.figure(figsize=self.figsize)
        gs = self.fig.add_gridspec(1, 3)
        self.ax_3d = self.fig.add_subplot(gs[0], projection='3d')
        self.ax_curvature = self.fig.add_subplot(gs[1])
        self.ax_metrics = self.fig.add_subplot(gs[2])
        plt.tight_layout(pad=3.0)
    
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
    
    def plot_metrics(self, metrics: Dict[str, List[float]]):
        """Plot training metrics."""
        if not metrics or not any(len(v) > 0 for v in metrics.values()):
            return
            
        self.ax_metrics.clear()
        
        for name, values in metrics.items():
            if not values:
                continue
                
            steps = np.arange(len(values))
            self.ax_metrics.plot(steps, values, label=name, alpha=0.8)
            
            if len(values) > 10:
                window = min(len(values) // 5, 20)
                window = max(window, 2)
                smoothed = np.convolve(values, 
                                     np.ones(window)/window,
                                     mode='valid')
                steps_smooth = np.arange(len(smoothed))
                self.ax_metrics.plot(steps_smooth,
                                   smoothed,
                                   '--',
                                   label=f'{name} (smoothed)',
                                   alpha=0.5)
        
        self.ax_metrics.set_title('Training Metrics')
        self.ax_metrics.set_xlabel('Step')
        self.ax_metrics.grid(True, alpha=0.3)
        self.ax_metrics.legend()

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