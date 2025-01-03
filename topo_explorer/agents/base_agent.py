"""Base agent class for topological space exploration."""

from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Any, Optional

class BaseAgent(ABC, nn.Module):
    """
    Abstract base class for agents exploring manifolds.
    Inherits from both ABC (for abstract methods) and nn.Module (for PyTorch functionality).
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        """
        Initialize base agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    @abstractmethod
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the neural network.
        
        Args:
            state: Current state tensor
            
        Returns:
            Tuple of (action_probs, state_value)
        """
        pass
    
    @abstractmethod
    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select an action given the current state.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def evaluate_actions(self, 
                        states: torch.Tensor, 
                        actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions given states.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            
        Returns:
            Tuple of (action_log_probs, state_values, entropy)
        """
        pass
    
    def save(self, path: str) -> None:
        """Save agent state."""
        torch.save(self.state_dict(), path)
    
    def load(self, path: str) -> None:
        """Load agent state."""
        self.load_state_dict(torch.load(path))
        
    @abstractmethod
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Update agent parameters.
        
        Args:
            batch: Dictionary containing training data
            
        Returns:
            Dictionary of training metrics
        """
        pass