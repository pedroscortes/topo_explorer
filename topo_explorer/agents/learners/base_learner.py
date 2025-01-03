"""Base learner class for managing agent training."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Optional, Any
from ...environments import ManifoldEnvironment
from ..base_agent import BaseAgent

class BaseLearner(ABC):
    """
    Abstract base class for training agents.
    Handles interaction between agent and environment.
    """
    
    def __init__(self, 
                 env: ManifoldEnvironment, 
                 agent: BaseAgent,
                 buffer_size: int = 1000,
                 batch_size: int = 64,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95):
        """
        Initialize learner.
        
        Args:
            env: Training environment
            agent: Agent to train
            buffer_size: Size of replay buffer
            batch_size: Size of training batches
            gamma: Discount factor
            gae_lambda: GAE parameter
        """
        self.env = env
        self.agent = agent
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.reset_buffer()
        
    def reset_buffer(self) -> None:
        """Reset experience buffer."""
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'values': [],
            'log_probs': []
        }
        
    @abstractmethod
    def collect_experience(self, steps: int) -> Dict[str, float]:
        """
        Collect experience by running agent in environment.
        
        Args:
            steps: Number of steps to collect
            
        Returns:
            Dictionary of metrics
        """
        pass
    
    @abstractmethod
    def compute_advantages(self) -> tuple:
        """
        Compute advantages and returns for collected experience.
        
        Returns:
            Tuple of (advantages, returns)
        """
        pass
    
    @abstractmethod
    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step.
        
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate current agent.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        self.agent.save(path)
        
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        self.agent.load(path)