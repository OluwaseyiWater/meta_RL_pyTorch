import pickle
import torch
import torch.nn as nn
import numpy as np
import wandb
from graphviz import Digraph
import tempfile
import os
from typing import Dict, Tuple, Any, Optional


def save_model(path: str, params: Tuple[Any, Any], optimizer_state: Optional[Dict] = None) -> None:
    """
    Save model parameters and optional optimizer state.
    
    Args:
        path: Path to save the model
        params: Tuple of (policy_state_dict, value_state_dict) or (policy_net, value_net)
        optimizer_state: Optional optimizer state dictionary
    """
    # Check if params are state dictionaries or full models
    if hasattr(params[0], 'state_dict'):
        # Full models provided
        save_dict = {
            'policy_state_dict': params[0].state_dict(),
            'value_state_dict': params[1].state_dict(),
            'policy_architecture': params[0].__class__.__name__,
            'value_architecture': params[1].__class__.__name__,
        }
    else:
        # State dictionaries provided
        save_dict = {
            'policy_state_dict': params[0],
            'value_state_dict': params[1],
        }
    
    if optimizer_state is not None:
        save_dict['optimizer_state'] = optimizer_state
    
    # Save with pickle for compatibility, but could also use torch.save
    with open(path, 'wb') as f:
        pickle.dump(save_dict, f)
    
    print(f"Parameters saved to {path}")


def load_model(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Optional[Dict]]:
    """
    Load model parameters from file.
    
    Args:
        path: Path to load the model from
        
    Returns:
        Tuple of (policy_state_dict, value_state_dict, optimizer_state)
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    policy_params = data['policy_state_dict']
    value_params = data['value_state_dict']
    optimizer_state = data.get('optimizer_state', None)
    
    print(f"Parameters loaded from {path}")
    
    # Return in format compatible with JAX version
    return (policy_params, value_params), optimizer_state


def flatten_state(state) -> torch.Tensor:
    """
    Flatten a SpectrumState object into a single tensor.
    
    Args:
        state: SpectrumState object with PyTorch tensors or numpy arrays
        
    Returns:
        Flattened tensor suitable for network input
    """
    # Handle both PyTorch SpectrumState and numpy-based states
    if hasattr(state, 'to_numpy'):
        # PyTorch SpectrumState with to_numpy method
        return torch.tensor(state.to_numpy(), dtype=torch.float32)
    
    # Manual flattening for other state formats
    components = []
    
    # Helper function to convert to tensor
    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.flatten().float()
        elif isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=torch.float32).flatten()
        else:
            return torch.tensor(x, dtype=torch.float32).flatten()
    
    # Flatten each component
    if hasattr(state, 'channel_gains'):
        components.append(to_tensor(state.channel_gains))
    if hasattr(state, 'interference_map'):
        components.append(to_tensor(state.interference_map))
    if hasattr(state, 'qos_metrics'):
        components.append(to_tensor(state.qos_metrics))
    if hasattr(state, 'spectrum_alloc'):
        components.append(to_tensor(state.spectrum_alloc))
    if hasattr(state, 'tx_power'):
        components.append(to_tensor(state.tx_power))
    
    # Concatenate all components
    return torch.cat(components)


def log_network_architecture_to_wandb(
    policy_network: nn.Module,
    value_network: nn.Module,
    num_bs: int,
    num_bands: int,
    num_power_levels: int,
    is_recurrent: bool = False
) -> None:
    """
    Create a visualization of the network architecture and log it to W&B.
    
    Args:
        policy_network: Policy network module
        value_network: Value network module
        num_bs: Number of base stations
        num_bands: Number of frequency bands
        num_power_levels: Number of power levels
        is_recurrent: Whether the policy network is recurrent
    """
    
    # Create a temporary directory for the visualization
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create policy network visualization
        if is_recurrent:
            dot = Digraph(comment='Recurrent Policy Network', engine='dot')
            dot.attr(rankdir='TB')
            
            # Get network dimensions from the actual network
            obs_dim = None
            hidden_dim = None
            lstm_hidden_dim = None
            
            # Extract dimensions from the network
            for name, module in policy_network.named_modules():
                if isinstance(module, nn.Linear) and obs_dim is None:
                    obs_dim = module.in_features
                    hidden_dim = module.out_features
                elif isinstance(module, nn.LSTM):
                    lstm_hidden_dim = module.hidden_size
            
            # Add nodes with actual dimensions
            dot.node('input', f'Input\n({obs_dim})', shape='box')
            dot.node('linear1', f'Linear\n({obs_dim} → {hidden_dim})', shape='box')
            dot.node('relu1', 'ReLU', shape='box')
            dot.node('lstm', f'LSTM\n(hidden: {lstm_hidden_dim})', shape='box', style='filled', fillcolor='lightblue')
            dot.node('output', f'Linear\n({lstm_hidden_dim} → {num_bs * num_bands * num_power_levels})', shape='box')
            dot.node('reshape', f'Reshape\n({num_bs * num_bands}, {num_power_levels})', shape='box')
            dot.node('categorical', 'Categorical\nDistribution', shape='ellipse', style='filled', fillcolor='lightgreen')
            
            # Add edges
            dot.edge('input', 'linear1')
            dot.edge('linear1', 'relu1')
            dot.edge('relu1', 'lstm')
            dot.edge('lstm', 'output')
            dot.edge('output', 'reshape')
            dot.edge('reshape', 'categorical')
            
        else:
            dot = Digraph(comment='MLP Policy Network', engine='dot')
            dot.attr(rankdir='TB')
            
            # Get network dimensions
            obs_dim = None
            hidden_dim = None
            num_blocks = 0
            
            for name, module in policy_network.named_modules():
                if isinstance(module, nn.Linear) and obs_dim is None:
                    obs_dim = module.in_features
                    hidden_dim = module.out_features
                if 'residual_blocks' in name and len(name.split('.')) == 2:
                    num_blocks += 1
            
            # Add nodes
            dot.node('input', f'Input\n({obs_dim})', shape='box')
            dot.node('linear1', f'Linear\n({obs_dim} → {hidden_dim})', shape='box')
            dot.node('relu1', 'ReLU', shape='box')
            
            # Add residual blocks
            prev_node = 'relu1'
            for i in range(num_blocks):
                block_name = f'resblock{i}'
                dot.node(block_name, f'Residual Block {i+1}\n({hidden_dim})', 
                        shape='box', style='filled', fillcolor='lightyellow')
                dot.edge(prev_node, block_name)
                prev_node = block_name
            
            dot.node('output', f'Linear\n({hidden_dim} → {num_bs * num_bands * num_power_levels})', shape='box')
            dot.node('reshape', f'Reshape\n({num_bs * num_bands}, {num_power_levels})', shape='box')
            dot.node('categorical', 'Categorical\nDistribution', shape='ellipse', style='filled', fillcolor='lightgreen')
            
            # Add edges
            dot.edge('input', 'linear1')
            dot.edge('linear1', 'relu1')
            if num_blocks > 0:
                dot.edge(f'resblock{num_blocks-1}', 'output')
            else:
                dot.edge('relu1', 'output')
            dot.edge('output', 'reshape')
            dot.edge('reshape', 'categorical')
        
        # Render policy network
        policy_viz_path = os.path.join(tmpdirname, 'policy_network')
        dot.render(policy_viz_path, format='png', cleanup=True)
        
        # Create value network visualization
        dot = Digraph(comment='Value Network', engine='dot')
        dot.attr(rankdir='TB')
        
        # Get value network dimensions
        obs_dim = None
        hidden_dim = None
        num_blocks = 0
        
        for name, module in value_network.named_modules():
            if isinstance(module, nn.Linear) and obs_dim is None:
                obs_dim = module.in_features
                hidden_dim = module.out_features
            if 'residual_blocks' in name and len(name.split('.')) == 2:
                num_blocks += 1
        
        # Add nodes
        dot.node('input', f'Input\n({obs_dim})', shape='box')
        dot.node('linear1', f'Linear\n({obs_dim} → {hidden_dim})', shape='box')
        dot.node('relu1', 'ReLU', shape='box')
        
        # Add residual blocks
        prev_node = 'relu1'
        for i in range(num_blocks):
            block_name = f'resblock{i}'
            dot.node(block_name, f'Residual Block {i+1}\n({hidden_dim})', 
                    shape='box', style='filled', fillcolor='lightcyan')
            dot.edge(prev_node, block_name)
            prev_node = block_name
        
        dot.node('output', f'Linear\n({hidden_dim} → 1)', shape='box')
        dot.node('value', 'State Value', shape='ellipse', style='filled', fillcolor='lightcoral')
        
        # Add edges
        dot.edge('input', 'linear1')
        dot.edge('linear1', 'relu1')
        if num_blocks > 0:
            dot.edge(f'resblock{num_blocks-1}', 'output')
        else:
            dot.edge('relu1', 'output')
        dot.edge('output', 'value')
        
        # Render value network
        value_viz_path = os.path.join(tmpdirname, 'value_network')
        dot.render(value_viz_path, format='png', cleanup=True)
        
        # Log to W&B
        try:
            if wandb.run is not None:
                wandb.log({
                    "policy_network_architecture": wandb.Image(f"{policy_viz_path}.png"),
                    "value_network_architecture": wandb.Image(f"{value_viz_path}.png"),
                    "network_summary": {
                        "policy_parameters": sum(p.numel() for p in policy_network.parameters()),
                        "value_parameters": sum(p.numel() for p in value_network.parameters()),
                        "total_parameters": sum(p.numel() for p in policy_network.parameters()) + 
                                          sum(p.numel() for p in value_network.parameters()),
                        "is_recurrent": is_recurrent
                    }
                })
                print("Network architecture logged to W&B")
            else:
                print("W&B run not initialized, skipping architecture logging")
        except Exception as e:
            print(f"Error logging network architecture to W&B: {e}")


def get_network_summary(network: nn.Module) -> Dict[str, Any]:
    """
    Get a summary of the network architecture.
    
    Args:
        network: PyTorch network module
        
    Returns:
        Dictionary containing network summary information
    """
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    
    summary = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "layers": []
    }
    
    for name, module in network.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            layer_info = {
                "name": name,
                "type": module.__class__.__name__,
                "parameters": sum(p.numel() for p in module.parameters())
            }
            
            # Add specific information for different layer types
            if isinstance(module, nn.Linear):
                layer_info["input_features"] = module.in_features
                layer_info["output_features"] = module.out_features
            elif isinstance(module, nn.LSTM):
                layer_info["input_size"] = module.input_size
                layer_info["hidden_size"] = module.hidden_size
                layer_info["num_layers"] = module.num_layers
            
            summary["layers"].append(layer_info)
    
    return summary
