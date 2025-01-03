"""Evaluation script for trained agents."""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

from topo_explorer.environments import ManifoldEnvironment
from topo_explorer.agents.geometric_agent import GeometricAgent
from topo_explorer.config import load_config

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained agent')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to training config file')
    parser.add_argument('--num-episodes', type=int, default=100,
                      help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true',
                      help='Render environment')
    return parser.parse_args()

def evaluate_agent(env: ManifoldEnvironment, 
                  agent: GeometricAgent,
                  num_episodes: int,
                  render: bool = False) -> dict:
    """Run evaluation episodes."""
    episode_rewards = []
    episode_lengths = []
    curvatures = []
    exploration_metrics = []
    
    for episode in range(num_episodes):
        state = env.reset()[0]
        episode_reward = 0
        episode_curvatures = []
        positions = []
        
        done = False
        while not done:
            # Get deterministic action
            action = agent.act(state, deterministic=True)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update metrics
            episode_reward += reward
            episode_curvatures.append(env.gaussian_curvature(state['position']))
            positions.append(state['position'])
            
            if render:
                env.render()
            
            state = next_state
        
        # Compute episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(len(positions))
        curvatures.extend(episode_curvatures)
        
        # Compute exploration metric (average pairwise distance)
        positions = np.array(positions)
        dists = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :], axis=-1)
        exploration_metrics.append(np.mean(dists))
        
        print(f"Episode {episode + 1}/{num_episodes} | "
              f"Reward: {episode_reward:.2f} | "
              f"Length: {len(positions)}")
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_curvature': np.mean(curvatures),
        'mean_exploration': np.mean(exploration_metrics)
    }

def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create environment
    env = ManifoldEnvironment(
        manifold_type=config['environment']['manifold_type'],
        params=config['environment']['params'],
        render_mode="human" if args.render else None
    )
    
    # Create agent
    agent = GeometricAgent(
        state_dim=config['agent']['state_dim'],
        action_dim=config['agent']['action_dim'],
        hidden_dim=config['agent']['hidden_dim'],
        num_layers=config['agent']['num_layers']
    )
    
    # Load checkpoint
    agent.load(args.checkpoint)
    
    # Run evaluation
    metrics = evaluate_agent(env, agent, args.num_episodes, args.render)
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 50)
    for key, value in metrics.items():
        print(f"{key}: {value:.3f}")

if __name__ == "__main__":
    main()