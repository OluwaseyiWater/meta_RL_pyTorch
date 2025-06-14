import torch
import torch.nn as nn
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pickle
from typing import Dict, Any
import wandb
import warnings

# Suppress gymnasium warnings about plugins
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.envs.registration")

# Import PyTorch versions
from mLN.environment import DynamicSpectrumEnv
from models.maml import (
    MLPNetwork, ValueNetwork, train_maml,
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
    
    # Remove Hydra-specific keys
    if '_target_' in env_config:
        env_config.pop('_target_')
    
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
    
    # Initialize networks
    policy_net = MLPNetwork(
        obs_dim=obs_dim,
        num_bs=num_bs,
        num_bands=num_bands,
        num_power_levels=num_power_levels,
        hidden_dim=cfg.maml.get('hidden_dim', 64),
        num_blocks=cfg.maml.get('num_blocks', 2)
    ).to(device)
    
    value_net = ValueNetwork(
        obs_dim=obs_dim,
        hidden_dim=cfg.maml.get('hidden_dim', 64),
        num_blocks=cfg.maml.get('num_blocks', 2)
    ).to(device)
    
    print(f"\nNetwork Architecture:")
    print(f"  - Policy network parameters: {sum(p.numel() for p in policy_net.parameters())}")
    print(f"  - Value network parameters: {sum(p.numel() for p in value_net.parameters())}")
    
    # Training configuration
    train_config = {
        'num_tasks': cfg.maml.num_meta_tasks,
        'inner_lr': cfg.maml.inner_lr,
        'inner_steps': cfg.maml.inner_steps,
        'meta_lr': cfg.maml.meta_lr,
        'num_iterations': cfg.maml.num_meta_iterations,
        'eval_interval': cfg.maml.get('eval_interval', 10),
        'num_eval_tasks': cfg.maml.get('num_eval_tasks', 5),
        'rollout_length': cfg.maml.get('rollout_len', 50),
        'use_wandb': cfg.maml.get('use_wandb', True),
        'wandb_project': cfg.maml.get('wandb_project', 'maml-training'),
        'wandb_name': cfg.maml.get('wandb_name', None),
        'device': device
    }
    
    print(f"\nTraining Configuration:")
    for key, value in train_config.items():
        if key != 'device':
            print(f"  - {key}: {value}")
    
    # Train the model
    print(f"\nStarting MAML training...")
    trained_params, history = train_maml(
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
    torch.save(trained_policy, os.path.join(model_dir, "policy_network.pth"))
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
        
        # Meta loss
        if 'meta_losses' in history and history['meta_losses']:
            ax = axes[0, 0]
            valid_losses = [l for l in history['meta_losses'] if not np.isnan(l)]
            if valid_losses:
                ax.plot(valid_losses)
                ax.set_title('Meta Loss')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Loss')
                ax.grid(True)
        
        # Reward improvement
        if 'eval_avg_pre_reward' in history and history['eval_avg_pre_reward']:
            ax = axes[0, 1]
            eval_iters = np.arange(0, len(history['eval_avg_pre_reward'])) * train_config['eval_interval']
            ax.plot(eval_iters, history['eval_avg_pre_reward'], label='Pre-adaptation')
            ax.plot(eval_iters, history['eval_avg_post_reward'], label='Post-adaptation')
            ax.set_title('Average Rewards')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Reward')
            ax.legend()
            ax.grid(True)
        
        # SINR violations
        if 'eval_avg_pre_sinr_violation' in history and history['eval_avg_pre_sinr_violation']:
            ax = axes[1, 0]
            eval_iters = np.arange(0, len(history['eval_avg_pre_sinr_violation'])) * train_config['eval_interval']
            ax.plot(eval_iters, history['eval_avg_pre_sinr_violation'], label='Pre-adaptation')
            ax.plot(eval_iters, history['eval_avg_post_sinr_violation'], label='Post-adaptation')
            ax.set_title('SINR Violations')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Violations')
            ax.legend()
            ax.grid(True)
        
        # QoS violations
        if 'eval_avg_pre_qos_violation' in history and history['eval_avg_pre_qos_violation']:
            ax = axes[1, 1]
            eval_iters = np.arange(0, len(history['eval_avg_pre_qos_violation'])) * train_config['eval_interval']
            ax.plot(eval_iters, history['eval_avg_pre_qos_violation'], label='Pre-adaptation')
            ax.plot(eval_iters, history['eval_avg_post_qos_violation'], label='Post-adaptation')
            ax.set_title('QoS Violations')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Violations')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=150)
        print(f"✓ Training curves saved to {plot_path}")
        plt.close()
        
    except ImportError:
        print("Note: matplotlib not available, skipping training curves")
    
    print("\nTraining completed successfully!")
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
