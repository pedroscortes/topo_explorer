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
        buffer_size=1000,
        batch_size=64
    )
    
    print("Starting training loop...")
    print("Total training steps: 10000")
    print("=" * 50)
    
    total_steps = 0
    episode_rewards = []
    episode_lengths = []
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
        
        train_metrics = learner.train_step()
        if train_metrics:
            current_metrics.update({
                k: float(v) for k, v in train_metrics.items() 
                if k in current_metrics
            })
        
        if total_steps % 1000 == 0:
            print("\n\nDetailed Evaluation:")
            eval_metrics = learner.evaluate(num_episodes=5)
            episode_rewards.append(eval_metrics['eval_reward'])
            episode_lengths.append(eval_metrics['eval_length'])
            
            current_metrics['reward'] = eval_metrics['eval_reward']
            
            print(f"Step {total_steps}/{10000}")
            print(f"Average Reward: {eval_metrics['eval_reward']:.2f}")
            print(f"Average Episode Length: {eval_metrics['eval_length']:.2f}")
            if 'policy_loss' in current_metrics:
                print(f"Policy Loss: {current_metrics['policy_loss']:.4f}")
            if 'value_loss' in current_metrics:
                print(f"Value Loss: {current_metrics['value_loss']:.4f}")
            print("-" * 50)
            
            training_vis.update(current_metrics)
            
            manifold_vis.setup_plot()
            manifold_vis.plot_manifold(env.get_visualization_data())
            manifold_vis.plot_trajectory(np.array(trajectory), np.array(curvatures))
            manifold_vis.plot_metrics(metrics)
            manifold_vis.show()
            
            training_vis.plot_metrics()
            training_vis.show()
    
    print("\nTraining completed!")
    print(f"Total time: {time.time() - start_time:.1f} seconds")
    print("=" * 50)
    
    final_metrics = learner.evaluate(num_episodes=10)
    print("\nFinal Results:")
    print(f"Average Reward: {final_metrics.get('eval_reward', 0.0):.2f}")
    print(f"Average Episode Length: {final_metrics.get('eval_length', 0.0):.2f}")
    
    manifold_vis.setup_plot()
    manifold_vis.plot_manifold(env.get_visualization_data())
    manifold_vis.plot_trajectory(np.array(trajectory), np.array(curvatures))
    manifold_vis.plot_frame(trajectory[-1], frames[-1])
    manifold_vis.show()
    
    training_vis.create_training_summary()
    training_vis.show()
    
    print("\nSaving training animation...")
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