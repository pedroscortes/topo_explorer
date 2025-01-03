"""Training visualization module."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from collections import defaultdict

class TrainingVisualizer:
    """Visualizer for training progress and metrics."""
    
    def __init__(self):
        """Initialize metric storage."""
        self.metrics = defaultdict(list)
        self.fig = None
        self.axes = None
        
    def update(self, metrics: Dict[str, float]):
        """Update metrics with new values."""
        for key, value in metrics.items():
            self.metrics[key].append(value)
            
    def plot_metrics(self, figsize: tuple = (15, 10)):
        """Plot all tracked metrics."""
        # Only create plot if we have metrics
        if not self.metrics:
            return
            
        # Get number of metrics that have data
        metrics_with_data = [(k, v) for k, v in self.metrics.items() if len(v) > 0]
        if not metrics_with_data:
            return
            
        n_metrics = len(metrics_with_data)
        rows = max((n_metrics + 1) // 2, 1)  # At least 1 row
        
        # Create new figure
        plt.close('all')
        self.fig, self.axes = plt.subplots(rows, 2, figsize=figsize, squeeze=False)
        self.axes = self.axes.flatten()
        
        # Plot each metric
        for i, (key, values) in enumerate(metrics_with_data):
            ax = self.axes[i]
            
            # Plot raw values
            steps = np.arange(len(values))
            ax.plot(steps, values, alpha=0.3, label='Raw')
            
            # Plot smoothed values if we have enough data
            if len(values) > 10:
                window = min(len(values) // 10 + 1, 20)
                window = max(window, 2)  # Ensure window size is at least 2
                smoothed = np.convolve(values, 
                                     np.ones(window)/window,
                                     mode='valid')
                smooth_steps = np.arange(len(smoothed))
                ax.plot(smooth_steps, smoothed, 
                       label=f'Smoothed (window={window})',
                       linewidth=2)
            
            ax.set_title(key.replace('_', ' ').title())
            ax.set_xlabel('Step')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Hide unused subplots
        for i in range(len(metrics_with_data), len(self.axes)):
            self.axes[i].set_visible(False)
            
        plt.tight_layout()
        
    def create_training_summary(self, figsize: tuple = (15, 10)):
        """Create comprehensive training summary plot."""
        if not self.metrics:
            return
            
        plt.close('all')
        self.fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Learning curve
        if 'reward' in self.metrics and len(self.metrics['reward']) > 0:
            ax = axes[0]
            values = self.metrics['reward']
            steps = np.arange(len(values))
            ax.plot(steps, values, alpha=0.3, label='Raw')
            
            if len(values) > 10:
                window = min(len(values) // 10 + 1, 20)
                smoothed = np.convolve(values,
                                     np.ones(window)/window,
                                     mode='valid')
                smooth_steps = np.arange(len(smoothed))
                ax.plot(smooth_steps, smoothed, 
                       label='Smoothed',
                       linewidth=2)
            ax.set_title('Learning Curve')
            ax.set_xlabel('Step')
            ax.set_ylabel('Reward')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Loss curves
        if 'policy_loss' in self.metrics and len(self.metrics['policy_loss']) > 0:
            ax = axes[1]
            steps = np.arange(len(self.metrics['policy_loss']))
            ax.plot(steps, self.metrics['policy_loss'], 
                   label='Policy Loss',
                   alpha=0.8)
            if 'value_loss' in self.metrics:
                ax.plot(steps, self.metrics['value_loss'], 
                       label='Value Loss',
                       alpha=0.8)
            ax.set_title('Training Losses')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Exploration metric
        if 'exploration_score' in self.metrics and len(self.metrics['exploration_score']) > 0:
            ax = axes[2]
            values = self.metrics['exploration_score']
            steps = np.arange(len(values))
            ax.plot(steps, values, alpha=0.8)
            ax.set_title('Exploration Score')
            ax.set_xlabel('Step')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    def show(self):
        """Display current plots."""
        if self.fig:
            plt.draw()
            plt.pause(0.1)  # Small pause to ensure display