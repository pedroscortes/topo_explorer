"""Geometric agent for learning on manifolds."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any, Optional

class GeometricAgent(nn.Module):
    """Agent for learning on geometric manifolds."""
    
    def __init__(self, 
                 state_dim: int = 11,  # position (3) + curvature (1) + frame (6) + exploration (1)
                 action_dim: int = 3,   # tangent vector in R³
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 learning_rate: float = 3e-4):
        """Initialize geometric agent."""
        super().__init__()
        
        # Build network layers
        layers = []
        current_dim = state_dim
        
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            current_dim = hidden_dim
            
        self.embedding = nn.Sequential(*layers)
        
        # Policy head (for actions)
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        self.policy_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Value head (for critic)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Initialize parameters with small values
        for name, param in self.named_parameters():
            if 'policy' in name or 'value' in name:
                nn.init.normal_(param, mean=0.0, std=0.1)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # For numerical stability
        self.eps = 1e-8
        self.clip_grad_norm = 0.5
        self.min_std = 1e-6
    
    def forward(self, state: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward pass of the network."""
        # Ensure input is float tensor
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
            
        # Get embedding
        embedding = self.embedding(state)
        
        # Get action distribution parameters
        action_mean = torch.tanh(self.policy_mean(embedding))
        action_logstd = torch.clamp(self.policy_logstd, min=np.log(self.min_std))
        
        # Get state value
        state_value = self.value_head(embedding)
        
        return (action_mean, action_logstd), state_value
    
    def get_distribution(self, state: torch.Tensor) -> torch.distributions.Normal:
        """Get action distribution for state."""
        (mean, logstd), _ = self.forward(state)
        std = torch.exp(logstd) + self.min_std
        return torch.distributions.Normal(mean, std)
    
    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action given state."""
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state)
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
                
            distribution = self.get_distribution(state)
            
            if deterministic:
                action = distribution.mean
            else:
                action = distribution.sample()
            
            # Ensure action is valid
            action = torch.clamp(action, -1.0, 1.0)
            return action.squeeze(0).numpy()
    
    def evaluate_actions(self, 
                        states: torch.Tensor, 
                        actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions given states."""
        distribution = self.get_distribution(states)
        
        # Get log probabilities
        log_probs = distribution.log_prob(actions).sum(dim=-1)
        
        # Get entropy
        entropy = distribution.entropy().sum(dim=-1)
        
        # Get state values
        _, state_values = self.forward(states)
        
        return log_probs, state_values.squeeze(-1), entropy
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Update agent parameters using PPO."""
        # Convert to tensors if needed
        states = torch.FloatTensor(batch['states'])
        actions = torch.FloatTensor(batch['actions'])
        old_log_probs = torch.FloatTensor(batch['old_log_probs'])
        advantages = torch.FloatTensor(batch['advantages'])
        returns = torch.FloatTensor(batch['returns'])
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)
        
        # Get current log probs and values
        log_probs, values, entropy = self.evaluate_actions(states, actions)
        
        # Compute policy loss (PPO clipped objective)
        ratio = torch.exp(log_probs - old_log_probs)
        clip_ratio = torch.clamp(ratio, 0.8, 1.2)
        policy_loss = -torch.min(
            ratio * advantages,
            clip_ratio * advantages
        ).mean()
        
        # Compute value loss
        value_pred = values
        value_target = returns
        value_loss = F.mse_loss(value_pred, value_target)
        
        # Compute entropy bonus
        entropy_loss = -entropy.mean()
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
        
        # Update parameters
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': loss.item()
        }