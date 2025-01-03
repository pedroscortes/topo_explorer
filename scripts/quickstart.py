"""Quick start example of training an agent on a sphere."""

import torch
import numpy as np
import time
import sys
from topo_explorer.environments import ManifoldEnvironment
from topo_explorer.agents.geometric_agent import GeometricAgent
from topo_explorer.agents.learners.geometric_learner import GeometricLearner
from topo_explorer.visualization import ManifoldVisualizer, TrainingVisualizer

def main():
    print("Initializing training environment...")
    
    # Create environment and visualizers
    env = ManifoldEnvironment(
        manifold_type='sphere',
        params={'radius': 2.0},
        render_mode='human'
    )
    
    manifold_vis = ManifoldVisualizer()
    training_vis = TrainingVisualizer()
    
    # Calculate state dimension
    state_dim = 3 + 1 + 6 + 1  # position (3) + curvature (1) + frame (6) + exploration (1)
    
    # Create agent and learner
    agent = GeometricAgent(
        state_dim=state_dim,
        action_dim=3,  # tangent vector in R³
        hidden_dim=64,
        num_layers=2
    )
    
    learner = GeometricLearner(
        env=env,
        agent=agent,
        buffer_size=1000,
        batch_size=64
    )
    
    print("Starting training loop...")
    print("Total training steps: 10000")
    print("=" * 50)
    
    # Training loop with visualization
    total_steps = 0
    episode_rewards = []
    episode_lengths = []
    trajectory = []
    curvatures = []
    frames = []
    start_time = time.time()
    
    # Track metrics
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
    
    while total_steps < 10000:  # Train for 10k steps
        # Collect experience
        collect_metrics = learner.collect_experience(100)
        total_steps += 100
        
        # Quick progress update every 100 steps
        elapsed_time = time.time() - start_time
        steps_per_second = total_steps / elapsed_time if elapsed_time > 0 else 0
        remaining_steps = 10000 - total_steps
        estimated_time = remaining_steps / steps_per_second if steps_per_second > 0 else 0
        
        # Clear previous line and print progress
        print('\033[K', end='')  # Clear line
        print(f"Progress: {total_steps}/10000 steps "
              f"({(total_steps/10000)*100:.1f}%) | "
              f"Speed: {steps_per_second:.1f} steps/s | "
              f"ETA: {estimated_time:.1f}s", end='\r')
        sys.stdout.flush()
        
        # Track trajectory and curvatures
        trajectory.append(env.current_position)
        curvatures.append(env.gaussian_curvature(env.current_position))
        frames.append(env.frame)
        
        # Update current metrics from collection
        if collect_metrics:
            current_metrics.update({
                k: float(v) for k, v in collect_metrics.items() 
                if k in current_metrics
            })
        
        # Train agent
        train_metrics = learner.train_step()
        if train_metrics:
            current_metrics.update({
                k: float(v) for k, v in train_metrics.items() 
                if k in current_metrics
            })
        
        # Detailed evaluation and visualization every 1000 steps
        if total_steps % 1000 == 0:
            print("\n\nDetailed Evaluation:")
            # Evaluate agent
            eval_metrics = learner.evaluate(num_episodes=5)
            episode_rewards.append(eval_metrics['eval_reward'])
            episode_lengths.append(eval_metrics['eval_length'])
            
            # Update metrics with evaluation results
            current_metrics['reward'] = eval_metrics['eval_reward']
            
            print(f"Step {total_steps}/{10000}")
            print(f"Average Reward: {eval_metrics['eval_reward']:.2f}")
            print(f"Average Episode Length: {eval_metrics['eval_length']:.2f}")
            if 'policy_loss' in current_metrics:
                print(f"Policy Loss: {current_metrics['policy_loss']:.4f}")
            if 'value_loss' in current_metrics:
                print(f"Value Loss: {current_metrics['value_loss']:.4f}")
            print("-" * 50)
            
            # Update training visualizer with current metrics
            training_vis.update(current_metrics)
            
            # Visualize
            manifold_vis.setup_plot()
            manifold_vis.plot_manifold(env.get_visualization_data())
            manifold_vis.plot_trajectory(np.array(trajectory), np.array(curvatures))
            manifold_vis.plot_metrics(metrics)
            manifold_vis.show()
            
            # Plot training progress
            training_vis.plot_metrics()
            training_vis.show()
    
    print("\nTraining completed!")
    print(f"Total time: {time.time() - start_time:.1f} seconds")
    print("=" * 50)
    
    # Final evaluation
    final_metrics = learner.evaluate(num_episodes=10)
    print("\nFinal Results:")
    print(f"Average Reward: {final_metrics.get('eval_reward', 0.0):.2f}")
    print(f"Average Episode Length: {final_metrics.get('eval_length', 0.0):.2f}")
    
    # Create final visualization summary
    manifold_vis.setup_plot()
    manifold_vis.plot_manifold(env.get_visualization_data())
    manifold_vis.plot_trajectory(np.array(trajectory), np.array(curvatures))
    manifold_vis.plot_frame(trajectory[-1], frames[-1])
    manifold_vis.show()
    
    # Create training summary if we have metrics
    training_vis.create_training_summary()
    training_vis.show()
    
    print("\nSaving training animation...")
    # Create and save animation
    anim = manifold_vis.create_animation(
        trajectory=trajectory,
        frames=frames,
        curvatures=curvatures,
        interval=50
    )
    manifold_vis.save_animation(anim, "training_animation.gif")
    print("Animation saved as 'training_animation.gif'")

if __name__ == "__main__":
    main()