"""Geometric learner implementation for manifold environments."""

import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

from ...environments import ManifoldEnvironment
from ..geometric_agent import GeometricAgent
from .base_learner import BaseLearner

class GeometricLearner(BaseLearner):
    """
    Learner specialized for geometric environments.
    Implements manifold-aware PPO with parallel transport and curvature adaptation.
    """
    
    def __init__(self,
                 env: ManifoldEnvironment,
                 agent: GeometricAgent,
                 buffer_size: int = 1000,
                 batch_size: int = 64,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 target_kl: float = 0.01,
                 num_epochs: int = 10):
        """
        Initialize geometric learner.
        
        Args:
            env: Manifold environment
            agent: Geometric agent to train
            buffer_size: Size of experience buffer
            batch_size: Size of training batches
            gamma: Discount factor
            gae_lambda: GAE parameter
            clip_ratio: PPO clipping parameter
            target_kl: Target KL divergence for early stopping
            num_epochs: Number of training epochs per update
        """
        super().__init__(env, agent, buffer_size, batch_size, gamma, gae_lambda)
        
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.num_epochs = num_epochs
        
        self.curvature_history = []
        self._last_positions = np.zeros((10, 3)) 
        
    def collect_experience(self, steps: int) -> Dict[str, float]:
        """
        Collect experience by running agent in environment.
        Handles parallel transport of actions between steps.
        
        Args:
            steps: Number of steps to collect
            
        Returns:
            Dictionary of collection metrics
        """
        metrics = defaultdict(list)
        state = self.env.reset()[0]  
        
        for _ in range(steps):
            state_tensor = torch.FloatTensor(self._get_state_representation(state))
            
            with torch.no_grad():
                distribution = self.agent.get_distribution(state_tensor.unsqueeze(0))
                action = distribution.sample().squeeze(0)
                value = self.agent.forward(state_tensor.unsqueeze(0))[1].squeeze()
                log_prob = distribution.log_prob(action).sum()
            
            next_state, reward, terminated, truncated, info = self.env.step(action.numpy())
            done = terminated or truncated
            
            self.buffer['states'].append(state)
            self.buffer['actions'].append(action.numpy())
            self.buffer['rewards'].append(reward)
            self.buffer['next_states'].append(next_state)
            self.buffer['dones'].append(done)
            self.buffer['values'].append(value.item())
            self.buffer['log_probs'].append(log_prob.item())
            
            metrics['reward'].append(reward)
            metrics['curvature'].append(self.env.gaussian_curvature(state['position']))
            
            state = next_state
            
            self._last_positions = np.roll(self._last_positions, 1, axis=0)
            self._last_positions[0] = state['position']
            
            if done:
                state = self.env.reset()[0]
        
        return {k: np.mean(v) for k, v in metrics.items()}
    
    def _get_state_representation(self, state: Dict) -> np.ndarray:
        """
        Create state representation including geometric information.
        
        Args:
            state: Environment state dictionary
            
        Returns:
            State representation array
        """
        position = state['position'].flatten()  
        curvature = np.array(state['curvature']).flatten()  
        frame = state['frame'].flatten()  
        
        exploration = np.mean(np.linalg.norm(
            self._last_positions - position, axis=1))
        exploration = np.array([exploration])  
        
        return np.concatenate([
            position,          
            curvature,        
            frame,            
            exploration      
        ])  
    
    def compute_advantages(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute advantages using GAE, accounting for manifold structure.
        
        Returns:
            Tuple of (advantages, returns)
        """
        rewards = np.array(self.buffer['rewards'])
        values = np.array(self.buffer['values'])
        dones = np.array(self.buffer['dones'])
        
        last_state = self.buffer['next_states'][-1]
        last_state_tensor = torch.FloatTensor(
            self._get_state_representation(last_state))
        with torch.no_grad():
            last_value = self.agent.forward(
                last_state_tensor.unsqueeze(0))[1].squeeze().numpy()
        
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        running_advantage = 0
        running_return = last_value
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
                next_not_done = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_not_done = 1.0 - dones[t]
                
            delta = rewards[t] + self.gamma * next_value * next_not_done - values[t]
            running_advantage = delta + \
                self.gamma * self.gae_lambda * next_not_done * running_advantage
            running_return = rewards[t] + \
                self.gamma * next_not_done * running_return
            
            advantages[t] = running_advantage
            returns[t] = running_return
            
        return advantages, returns
    
    def train_step(self) -> Dict[str, float]:
        """Perform one training step using PPO."""
        if len(self.buffer['states']) < self.batch_size:
            return {}
            
        advantages, returns = self.compute_advantages()
        
        states = np.array([self._get_state_representation(s) 
                        for s in self.buffer['states']])
        actions = np.array(self.buffer['actions'])
        old_log_probs = np.array(self.buffer['log_probs'])
        
        dataset = {
            'states': states,
            'actions': actions,
            'old_log_probs': old_log_probs,
            'advantages': advantages,
            'returns': returns,
            'values': np.array(self.buffer['values'])
        }
        
        metrics = defaultdict(list)
        early_stop = False
        
        for epoch in range(self.num_epochs):
            if early_stop:
                break
                
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), self.batch_size):
                batch_indices = indices[start:min(start + self.batch_size, len(states))]
                batch = {k: v[batch_indices] for k, v in dataset.items()}
                
                update_metrics = self.agent.update(batch)
                
                states_tensor = torch.FloatTensor(batch['states'])
                with torch.no_grad():
                    old_dist = self.agent.get_distribution(states_tensor)
                    new_dist = self.agent.get_distribution(states_tensor)
                    kl = torch.distributions.kl_divergence(old_dist, new_dist).mean()
                    
                    if kl > self.target_kl:
                        early_stop = True
                        break
                        
                for k, v in update_metrics.items():
                    metrics[k].append(v)
        
        avg_metrics = {
            k: np.mean(v) for k, v in metrics.items() 
            if len(v) > 0  
        }
        
        self.reset_buffer()
        return avg_metrics
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate current agent.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = defaultdict(list)
        
        for _ in range(num_episodes):
            state = self.env.reset()[0]
            episode_reward = 0
            episode_curvatures = []
            episode_length = 0
            done = False
            
            while not done:
                state_tensor = torch.FloatTensor(
                    self._get_state_representation(state))
                action = self.agent.act(state_tensor, deterministic=True)
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_curvatures.append(
                    self.env.gaussian_curvature(state['position']))
                episode_length += 1
                
                state = next_state
                
            metrics['eval_reward'].append(episode_reward)
            metrics['eval_length'].append(episode_length)
            metrics['eval_curvature_mean'].append(np.mean(episode_curvatures))
            metrics['eval_curvature_std'].append(np.std(episode_curvatures))
        
        return {k: np.mean(v) for k, v in metrics.items()}