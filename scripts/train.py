"""Main training script for Topological Space Explorer."""

import os
import argparse
import yaml
import time
from datetime import datetime
import numpy as np
import torch
from pathlib import Path

from topo_explorer.environments import ManifoldEnvironment
from topo_explorer.agents.geometric_agent import GeometricAgent
from topo_explorer.agents.learners.geometric_learner import GeometricLearner
from topo_explorer.config import load_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train an agent on a manifold')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                      help='Path to config file')
    parser.add_argument('--manifold', type=str, default=None,
                      help='Override manifold type from config')
    parser.add_argument('--log-dir', type=str, default='logs',
                      help='Directory for logs')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    return parser.parse_args()

def setup_logging(log_dir: str, config: dict) -> str:
    """Setup logging directory and save config."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(log_dir) / f"{config['environment']['manifold_type']}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(log_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
        
    return str(log_dir)

def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def train(config: dict, log_dir: str):
    """Main training loop."""
    # Create environment
    env = ManifoldEnvironment(
        manifold_type=config['environment']['manifold_type'],
        params=config['environment']['params'],
        render_mode=config['visualization']['render_mode']
    )
    
    # Create agent
    agent = GeometricAgent(
        state_dim=config['agent']['state_dim'],
        action_dim=config['agent']['action_dim'],
        hidden_dim=config['agent']['hidden_dim'],
        num_layers=config['agent']['num_layers'],
        learning_rate=config['training']['learning_rate']
    )
    
    # Create learner
    learner = GeometricLearner(
        env=env,
        agent=agent,
        buffer_size=config['training']['buffer_size'],
        batch_size=config['training']['batch_size'],
        gamma=config['training']['gamma'],
        gae_lambda=config['training']['gae_lambda'],
        clip_ratio=config['training']['clip_ratio'],
        target_kl=config['training']['target_kl'],
        num_epochs=config['training']['num_epochs']
    )
    
    # Setup logging
    log_file = open(os.path.join(log_dir, 'training.csv'), 'w')
    log_file.write('episode,total_reward,avg_curvature,exploration_metric\n')
    
    # Training loop
    best_reward = float('-inf')
    episode = 0
    total_steps = 0
    
    while total_steps < config['training']['total_steps']:
        # Collect experience
        collect_metrics = learner.collect_experience(
            config['training']['steps_per_epoch'])
        total_steps += config['training']['steps_per_epoch']
        
        # Train agent
        train_metrics = learner.train_step()
        
        # Evaluate agent
        eval_metrics = learner.evaluate(
            num_episodes=config['training']['eval_episodes'])
        
        # Log metrics
        log_file.write(f"{episode},{eval_metrics['eval_reward']},"
                      f"{eval_metrics['eval_curvature_mean']},"
                      f"{collect_metrics['exploration']}\n")
        log_file.flush()
        
        # Save best model
        if eval_metrics['eval_reward'] > best_reward:
            best_reward = eval_metrics['eval_reward']
            learner.save_checkpoint(os.path.join(log_dir, 'best_model.pt'))
        
        # Regular checkpoint
        if episode % config['training']['save_frequency'] == 0:
            learner.save_checkpoint(
                os.path.join(log_dir, f'checkpoint_{episode}.pt'))
        
        # Print progress
        if episode % config['training']['print_frequency'] == 0:
            print(f"Episode {episode} | Steps: {total_steps}")
            print(f"Eval Reward: {eval_metrics['eval_reward']:.2f}")
            print(f"Train Loss: {train_metrics['total_loss']:.4f}")
            print("-" * 50)
        
        episode += 1
    
    # Final save
    learner.save_checkpoint(os.path.join(log_dir, 'final_model.pt'))
    log_file.close()

def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override manifold type if specified
    if args.manifold is not None:
        config['environment']['manifold_type'] = args.manifold
    
    # Setup
    log_dir = setup_logging(args.log_dir, config)
    set_random_seed(args.seed)
    
    # Train
    train(config, log_dir)

if __name__ == "__main__":
    main()