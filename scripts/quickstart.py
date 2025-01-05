"""Quick start example of training an agent on a sphere."""

import torch
import numpy as np
import time
import sys
from topo_explorer.environments import ManifoldEnvironment
from topo_explorer.agents.geometric_agent import GeometricAgent
from topo_explorer.agents.learners.geometric_learner import GeometricLearner
from topo_explorer.visualization import ManifoldVisualizer, TrainingVisualizer
import matplotlib.pyplot as plt

def main():
    print("Initializing training environment...")
    
    env = ManifoldEnvironment(
        manifold_type='sphere',
        params={'radius': 2.0},
        render_mode='human'
    )
    
    manifold_vis = ManifoldVisualizer()
    training_vis = TrainingVisualizer()
    
    state_dim = 3 + 1 + 6 + 1  
    
    agent = GeometricAgent(
        state_dim=state_dim,
        action_dim=3,  
        hidden_dim=64,
        num_layers=2
    )
    
    learner = GeometricLearner(
        env=env,
        agent=agent,
        buffer_size=2048,
        batch_size=128
    )
    
    print("Starting training loop...")
    print("Total training steps: 10000")
    print("=" * 50)
    
    total_steps = 0
    episode_rewards = []
    episode_lengths = []
    episode_values = []    
    trajectory = []
    curvatures = []
    frames = []
    start_time = time.time()
    
    metrics = {
        'reward': [],
        'exploration_score': [],
        'policy_loss': [],
        'value_loss': []
    }
    current_metrics = {
        'reward': 0.0,
        'exploration_score': 0.0,
        'policy_loss': 0.0,
        'value_loss': 0.0
    }
    
    while total_steps < 10000:
        collect_metrics = learner.collect_experience(100)
        total_steps += 100
        
        elapsed_time = time.time() - start_time
        steps_per_second = total_steps / elapsed_time if elapsed_time > 0 else 0
        remaining_steps = 10000 - total_steps
        estimated_time = remaining_steps / steps_per_second if steps_per_second > 0 else 0
        
        print('\033[K', end='')  
        print(f"Progress: {total_steps}/10000 steps "
              f"({(total_steps/10000)*100:.1f}%) | "
              f"Speed: {steps_per_second:.1f} steps/s | "
              f"ETA: {estimated_time:.1f}s", end='\r')
        sys.stdout.flush()
        
        trajectory.append(env.current_position)
        curvatures.append(env.gaussian_curvature(env.current_position))
        frames.append(env.frame)
        
        if collect_metrics:
            current_metrics.update({
                k: float(v) for k, v in collect_metrics.items() 
                if k in current_metrics
            })
            for k, v in collect_metrics.items():
                if k in metrics:
                    metrics[k].append(float(v))
        
        train_metrics = learner.train_step()
        if train_metrics:
            current_metrics.update({
                k: float(v) for k, v in train_metrics.items() 
                if k in current_metrics
            })
            for k, v in train_metrics.items():
                if k in metrics:
                    metrics[k].append(float(v))
        
        if total_steps % 1000 == 0:
            print("\n\nDetailed Evaluation:")
            eval_metrics = learner.evaluate(num_episodes=5)
            episode_rewards.append(eval_metrics['eval_reward'])
            episode_lengths.append(eval_metrics['eval_length'])
            
            print(f"Step {total_steps}/10000")
            print(f"Average Reward: {eval_metrics['eval_reward']:.2f}")
            print(f"Average Episode Length: {eval_metrics['eval_length']:.2f}")
            if 'policy_loss' in current_metrics:
                print(f"Policy Loss: {current_metrics['policy_loss']:.4f}")
            if 'value_loss' in current_metrics:
                print(f"Value Loss: {current_metrics['value_loss']:.4f}")
            print("-" * 50)
    
    print("\nTraining completed!")
    print(f"Total time: {time.time() - start_time:.1f} seconds")
    print("=" * 50)
    
    print("\nFinal Training Statistics:")
    print(f"Value Loss Progress: {metrics['value_loss'][0]:.2f} → {metrics['value_loss'][-1]:.2f}")
    print(f"Final Policy Loss: {metrics['policy_loss'][-1]:.4f}")
    print(f"Reward Range: {min(episode_rewards):.2f} - {max(episode_rewards):.2f}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    
    print("\nCreating training summary...")
    plt.figure(figsize=(20, 10))
    
    plt.subplot(231)
    plt.plot(episode_rewards, 'b-', label='Reward')
    plt.fill_between(range(len(episode_rewards)), 
                    np.array(episode_rewards) - np.std(episode_rewards),
                    np.array(episode_rewards) + np.std(episode_rewards),
                    alpha=0.2)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    plt.subplot(232)
    plt.plot(metrics['value_loss'], 'r-', label='Value Loss')
    plt.yscale('log')
    plt.title('Value Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(233)
    plt.plot(metrics['policy_loss'], 'g-', label='Policy Loss')
    plt.title('Policy Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(234)
    points = np.array(trajectory)
    theta = np.arctan2(points[:,1], points[:,0])
    phi = np.arccos(points[:,2] / np.linalg.norm(points, axis=1))
    plt.hexbin(theta, phi, gridsize=30, cmap='viridis')
    plt.colorbar(label='Density')
    plt.title('State Space Coverage')
    plt.xlabel('θ')
    plt.ylabel('φ')
    
    plt.subplot(235)
    plt.plot(episode_lengths, 'c-', label='Length')
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True)
    
    plt.subplot(236)
    plt.hist(episode_rewards, bins=20, color='purple', alpha=0.7)
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Count')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Creating animation...")
    manifold_vis.setup_plot()
    manifold_vis.plot_manifold(env.get_visualization_data())  
    
    print("Saving training animation...")
    try:
        anim = manifold_vis.create_animation(
            trajectory=trajectory,
            frames=frames,
            curvatures=curvatures,
            interval=50
        )
        manifold_vis.save_animation(anim, "training_animation.gif")
        print("Animation saved successfully!")
    except Exception as e:
        print(f"Warning: Could not save animation due to error: {str(e)}")
    finally:
        plt.close('all')

if __name__ == "__main__":
    main()