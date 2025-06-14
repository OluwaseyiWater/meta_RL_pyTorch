import torch
import torch.nn as nn
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pickle
from typing import Dict, Any
import wandb

# Import PyTorch versions
from mLN.environment import DynamicSpectrumEnv
from models.recurrent_ml import (
    RecurrentPolicyNetwork, ValueNetwork, train_recurrent_maml_ppo,
    ObservationNormalizer, RewardNormalizer
)


def save_model(save_path: str, params: Dict[str, Any], optimizer_state: Any = None) -> None:
    """Save model parameters and optional optimizer state"""
    save_dict = {
        'policy_state_dict': params[0],  # Policy network state dict
        'value_state_dict': params[1],   # Value network state dict
    }
    if optimizer_state is not None:
        save_dict['optimizer_state'] = optimizer_state
    
    torch.save(save_dict, save_path)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seeds
    seed = cfg.seed if hasattr(cfg, 'seed') else 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create environment using Hydra's instantiate
    env_config = OmegaConf.to_container(cfg.env, resolve=True)
    
    # Add device to env config
    env_config['device'] = device.type
    
    # Create environment manually since we need to handle device
    env = DynamicSpectrumEnv(**env_config)
    
    # Ensure compatibility by setting additional attributes if needed
    if not hasattr(env, 'bandwidth_hz'):
        env.bandwidth_hz = getattr(env, 'bandwidth_hz', 10e6)
    if not hasattr(env, 'noise_figure_db'):
        env.noise_figure_db = getattr(env, 'noise_figure_db', 7.0)
    if not hasattr(env, 'thermal_noise_dbm_hz'):
        env.thermal_noise_dbm_hz = getattr(env, 'thermal_noise_dbm_hz', -174.0)
    
    # Handle attribute name compatibility
    if hasattr(env, 'max_power_dbm'):
        env.max_power = env.max_power_dbm
    elif hasattr(env, 'max_power'):
        env.max_power_dbm = env.max_power
        
    if hasattr(env, 'power_levels_dbm'):
        env.power_levels = torch.tensor(env.power_levels_dbm, device=device)
    elif hasattr(env, 'power_levels'):
        env.power_levels_dbm = env.power_levels.cpu().numpy()
        
    if hasattr(env, 'max_external_interference_mW'):
        env.max_interference = env.max_external_interference_mW
    elif hasattr(env, 'max_interference'):
        env.max_external_interference_mW = env.max_interference
        
    if hasattr(env, 'min_sinr_db'):
        env.min_sinr = env.min_sinr_db
    elif hasattr(env, 'min_sinr'):
        env.min_sinr_db = env.min_sinr
    
    # Get environment dimensions
    num_bs = env.num_bs
    num_bands = env.num_bands
    num_power_levels = env.num_power_levels
    obs_dim = env.observation_space.shape[0]
    
    print(f"\nEnvironment Configuration:")
    print(f"  - Observation dimension: {obs_dim}")
    print(f"  - Number of base stations: {num_bs}")
    print(f"  - Number of bands: {num_bands}")
    print(f"  - Number of power levels: {num_power_levels}")
    
    # Initialize recurrent networks
    policy_net = RecurrentPolicyNetwork(
        obs_dim=obs_dim,
        num_bs=num_bs,
        num_bands=num_bands,
        num_power_levels=num_power_levels,
        hidden_dim=cfg.recurrent.get('hidden_dim', 64),
        lstm_hidden_dim=cfg.recurrent.get('lstm_hidden_dim', 32)
    ).to(device)
    
    value_net = ValueNetwork(
        obs_dim=obs_dim,
        hidden_dim=cfg.recurrent.get('hidden_dim', 64),
        num_blocks=cfg.recurrent.get('num_blocks', 2)
    ).to(device)
    
    print(f"\nNetwork Architecture:")
    print(f"  - Recurrent Policy network parameters: {sum(p.numel() for p in policy_net.parameters())}")
    print(f"  - Value network parameters: {sum(p.numel() for p in value_net.parameters())}")
    print(f"  - LSTM hidden dimension: {cfg.recurrent.get('lstm_hidden_dim', 32)}")
    
    # Training configuration
    train_config = {
        'num_tasks': cfg.recurrent.num_meta_tasks,
        'inner_lr': cfg.recurrent.inner_lr,
        'inner_steps': cfg.recurrent.inner_steps,
        'meta_lr': cfg.recurrent.meta_lr,
        'num_iterations': cfg.recurrent.num_meta_iterations,
        'clip_ratio': cfg.recurrent.get('clip_ratio', 0.1),
        'eval_interval': cfg.recurrent.get('eval_interval', 10),
        'num_eval_tasks': cfg.recurrent.get('num_eval_tasks', 5),
        'rollout_length': cfg.recurrent.get('rollout_len', 20),
        'use_wandb': cfg.recurrent.get('use_wandb', True),
        'wandb_project': cfg.recurrent.get('wandb_project', 'recurrent-maml-ppo'),
        'wandb_name': cfg.recurrent.get('wandb_name', None),
        'device': device
    }
    
    print(f"\nTraining Configuration:")
    for key, value in train_config.items():
        if key != 'device':
            print(f"  - {key}: {value}")
    
    # Train the model
    print(f"\nStarting Recurrent MAML-PPO training...")
    trained_params, history = train_recurrent_maml_ppo(
        env=env,
        policy_net=policy_net,
        value_net=value_net,
        **train_config
    )
    
    # Extract the trained networks
    trained_policy, trained_value = trained_params
    
    # Get the output directory managed by Hydra
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the trained parameters
    save_path = os.path.join(model_dir, "trained_params.pth")
    save_model(
        save_path,
        (trained_policy.state_dict(), trained_value.state_dict()),
        None
    )
    print(f"✓ Trained parameters saved to {save_path}")
    
    # Save the complete models (for easier loading)
    torch.save(trained_policy, os.path.join(model_dir, "recurrent_policy_network.pth"))
    torch.save(trained_value, os.path.join(model_dir, "value_network.pth"))
    print(f"✓ Complete models saved to {model_dir}")
    
    # Save the training history
    history_path = os.path.join(model_dir, "history.pkl")
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"✓ Training history saved to {history_path}")
    
    # Save configuration
    config_path = os.path.join(model_dir, "config.yaml")
    OmegaConf.save(cfg, config_path)
    print(f"✓ Configuration saved to {config_path}")
    
    # Print training summary
    print("\n=== Training Summary ===")
    print(f"Environment details:")
    print(f"  - Number of base stations: {env.num_bs}")
    print(f"  - Number of bands: {env.num_bands}")
    print(f"  - Number of power levels: {env.num_power_levels}")
    print(f"  - Bandwidth: {env.bandwidth_hz/1e6:.1f} MHz")
    print(f"  - Noise figure: {env.noise_figure_db} dB")
    
    if 'training_stats' in history:
        stats = history['training_stats']
        print(f"\nTraining statistics:")
        print(f"  - Successful iterations: {stats['successful_iterations']}/{stats['total_iterations']}")
        print(f"  - Success rate: {stats['successful_iterations']/stats['total_iterations']*100:.1f}%")
        print(f"  - Final consecutive failures: {stats['final_consecutive_failures']}")
    
    if 'eval_avg_post_reward' in history and history['eval_avg_post_reward']:
        print(f"\nPerformance improvements:")
        print(f"  - Initial avg reward: {history['eval_avg_pre_reward'][0]:.3f}")
        print(f"  - Final avg reward: {history['eval_avg_post_reward'][-1]:.3f}")
        print(f"  - Improvement: {(history['eval_avg_post_reward'][-1] - history['eval_avg_pre_reward'][0]):.3f}")
        
        if 'eval_avg_sinr_improvement' in history and history['eval_avg_sinr_improvement']:
            print(f"\nSINR violation improvements:")
            print(f"  - Initial avg violations: {history['eval_avg_pre_sinr_violation'][0]:.3f}")
            print(f"  - Final avg violations: {history['eval_avg_post_sinr_violation'][-1]:.3f}")
            print(f"  - Reduction: {history['eval_avg_sinr_improvement'][-1]:.3f}")
            
        if 'eval_avg_qos_improvement' in history and history['eval_avg_qos_improvement']:
            print(f"\nQoS violation improvements:")
            print(f"  - Initial avg violations: {history['eval_avg_pre_qos_violation'][0]:.3f}")
            print(f"  - Final avg violations: {history['eval_avg_post_qos_violation'][-1]:.3f}")
            print(f"  - Reduction: {history['eval_avg_qos_improvement'][-1]:.3f}")
    
    # Plot training curves if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Recurrent MAML-PPO Training Progress', fontsize=16)
        
        # Meta loss
        if 'meta_losses' in history and history['meta_losses']:
            ax = axes[0, 0]
            valid_losses = [l for l in history['meta_losses'] if not np.isnan(l)]
            if valid_losses:
                ax.plot(valid_losses)
                ax.set_title('Meta Loss')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Loss')
                ax.grid(True, alpha=0.3)
        
        # Reward improvement
        if 'eval_avg_pre_reward' in history and history['eval_avg_pre_reward']:
            ax = axes[0, 1]
            eval_iters = np.arange(0, len(history['eval_avg_pre_reward'])) * train_config['eval_interval']
            ax.plot(eval_iters, history['eval_avg_pre_reward'], 'b-', label='Pre-adaptation', linewidth=2)
            ax.plot(eval_iters, history['eval_avg_post_reward'], 'r-', label='Post-adaptation', linewidth=2)
            ax.fill_between(eval_iters, history['eval_avg_pre_reward'], history['eval_avg_post_reward'], 
                           alpha=0.3, color='green')
            ax.set_title('Average Rewards')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Reward')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # SINR violations
        if 'eval_avg_pre_sinr_violation' in history and history['eval_avg_pre_sinr_violation']:
            ax = axes[1, 0]
            eval_iters = np.arange(0, len(history['eval_avg_pre_sinr_violation'])) * train_config['eval_interval']
            ax.plot(eval_iters, history['eval_avg_pre_sinr_violation'], 'b-', label='Pre-adaptation', linewidth=2)
            ax.plot(eval_iters, history['eval_avg_post_sinr_violation'], 'r-', label='Post-adaptation', linewidth=2)
            ax.set_title('SINR Violations')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Average Violations')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # QoS violations
        if 'eval_avg_pre_qos_violation' in history and history['eval_avg_pre_qos_violation']:
            ax = axes[1, 1]
            eval_iters = np.arange(0, len(history['eval_avg_pre_qos_violation'])) * train_config['eval_interval']
            ax.plot(eval_iters, history['eval_avg_pre_qos_violation'], 'b-', label='Pre-adaptation', linewidth=2)
            ax.plot(eval_iters, history['eval_avg_post_qos_violation'], 'r-', label='Post-adaptation', linewidth=2)
            ax.set_title('QoS Violations')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Average Violations')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Training curves saved to {plot_path}")
        plt.close()
        
        # Additional plot for reward improvements over time
        if 'eval_avg_reward_improvement' in history and history['eval_avg_reward_improvement']:
            plt.figure(figsize=(10, 6))
            eval_iters = np.arange(0, len(history['eval_avg_reward_improvement'])) * train_config['eval_interval']
            plt.plot(eval_iters, history['eval_avg_reward_improvement'], 'g-', linewidth=2)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            plt.fill_between(eval_iters, 0, history['eval_avg_reward_improvement'], 
                           where=np.array(history['eval_avg_reward_improvement']) > 0, 
                           alpha=0.3, color='green', label='Improvement')
            plt.fill_between(eval_iters, 0, history['eval_avg_reward_improvement'], 
                           where=np.array(history['eval_avg_reward_improvement']) <= 0, 
                           alpha=0.3, color='red', label='Degradation')
            plt.title('Reward Improvement from Adaptation')
            plt.xlabel('Iteration')
            plt.ylabel('Reward Improvement')
            plt.legend()
            plt.grid(True, alpha=0.3)
            improvement_plot_path = os.path.join(output_dir, "reward_improvement.png")
            plt.savefig(improvement_plot_path, dpi=150, bbox_inches='tight')
            print(f"✓ Reward improvement plot saved to {improvement_plot_path}")
            plt.close()
        
    except ImportError:
        print("Note: matplotlib not available, skipping training curves")
    
    print("\nTraining completed successfully!")
    print(f"All outputs saved to: {output_dir}")
    
    # Create a summary report
    summary_path = os.path.join(output_dir, "training_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("RECURRENT MAML-PPO TRAINING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  - Algorithm: Recurrent MAML with PPO\n")
        f.write(f"  - Device: {device}\n")
        f.write(f"  - Random seed: {seed}\n")
        f.write(f"  - Number of meta-tasks: {train_config['num_tasks']}\n")
        f.write(f"  - Inner learning rate: {train_config['inner_lr']}\n")
        f.write(f"  - Inner steps: {train_config['inner_steps']}\n")
        f.write(f"  - Meta learning rate: {train_config['meta_lr']}\n")
        f.write(f"  - PPO clip ratio: {train_config['clip_ratio']}\n")
        f.write(f"  - Rollout length: {train_config['rollout_length']}\n")
        f.write(f"  - Total iterations: {train_config['num_iterations']}\n\n")
        
        f.write("Environment:\n")
        f.write(f"  - Base stations: {env.num_bs}\n")
        f.write(f"  - Frequency bands: {env.num_bands}\n")
        f.write(f"  - Power levels: {env.num_power_levels}\n")
        f.write(f"  - Observation dimension: {obs_dim}\n")
        f.write(f"  - Bandwidth: {env.bandwidth_hz/1e6:.1f} MHz\n")
        f.write(f"  - Noise figure: {env.noise_figure_db} dB\n\n")
        
        f.write("Network Architecture:\n")
        f.write(f"  - Policy network: Recurrent (LSTM)\n")
        f.write(f"  - LSTM hidden dimension: {cfg.recurrent.get('lstm_hidden_dim', 32)}\n")
        f.write(f"  - Hidden dimension: {cfg.recurrent.get('hidden_dim', 64)}\n")
        f.write(f"  - Policy parameters: {sum(p.numel() for p in policy_net.parameters())}\n")
        f.write(f"  - Value parameters: {sum(p.numel() for p in value_net.parameters())}\n\n")
        
        if 'training_stats' in history:
            stats = history['training_stats']
            f.write("Training Results:\n")
            f.write(f"  - Successful iterations: {stats['successful_iterations']}/{stats['total_iterations']}\n")
            f.write(f"  - Success rate: {stats['successful_iterations']/stats['total_iterations']*100:.1f}%\n")
            f.write(f"  - Final consecutive failures: {stats['final_consecutive_failures']}\n\n")
        
        if 'eval_avg_post_reward' in history and history['eval_avg_post_reward']:
            f.write("Performance Metrics:\n")
            f.write(f"  - Initial average reward: {history['eval_avg_pre_reward'][0]:.3f}\n")
            f.write(f"  - Final average reward: {history['eval_avg_post_reward'][-1]:.3f}\n")
            f.write(f"  - Total improvement: {(history['eval_avg_post_reward'][-1] - history['eval_avg_pre_reward'][0]):.3f}\n")
            f.write(f"  - Relative improvement: {((history['eval_avg_post_reward'][-1] - history['eval_avg_pre_reward'][0])/abs(history['eval_avg_pre_reward'][0])*100):.1f}%\n\n")
            
            if 'eval_avg_sinr_improvement' in history and history['eval_avg_sinr_improvement']:
                f.write(f"  - Initial SINR violations: {history['eval_avg_pre_sinr_violation'][0]:.3f}\n")
                f.write(f"  - Final SINR violations: {history['eval_avg_post_sinr_violation'][-1]:.3f}\n")
                f.write(f"  - Violation reduction: {history['eval_avg_sinr_improvement'][-1]:.3f}\n\n")
                
            if 'eval_avg_qos_improvement' in history and history['eval_avg_qos_improvement']:
                f.write(f"  - Initial QoS violations: {history['eval_avg_pre_qos_violation'][0]:.3f}\n")
                f.write(f"  - Final QoS violations: {history['eval_avg_post_qos_violation'][-1]:.3f}\n")
                f.write(f"  - Violation reduction: {history['eval_avg_qos_improvement'][-1]:.3f}\n")
    
    print(f"✓ Training summary saved to {summary_path}")


if __name__ == "__main__":
    main()
