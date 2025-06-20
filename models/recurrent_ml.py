import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, Tuple, List, Optional, Any, NamedTuple
import copy
from dataclasses import dataclass
import wandb
from collections import defaultdict, namedtuple

# Import the PyTorch environment
from mLN.environment import DynamicSpectrumEnv, SpectrumState

# ==============================================================================
# CONSTANTS
# ==============================================================================
ROLLOUT_LENGTH = 20
BANDWIDTH_HZ = 10e6
NOISE_FIGURE_DB = 7.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# LSTM STATE
# ==============================================================================

LSTMState = namedtuple('LSTMState', ['hidden', 'cell'])

# ==============================================================================
# NEURAL NETWORK ARCHITECTURES
# ==============================================================================

class ResidualBlock(nn.Module):
    """Residual block for deeper networks"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.fc2(h)
        return F.relu(x + h)


class RecurrentPolicyNetwork(nn.Module):
    """Recurrent policy network with LSTM"""
    def __init__(self, obs_dim, num_bs, num_bands, num_power_levels, 
                 hidden_dim=64, lstm_hidden_dim=32):
        super().__init__()
        self.num_bs = num_bs
        self.num_bands = num_bands
        self.num_power_levels = num_power_levels
        self.lstm_hidden_dim = lstm_hidden_dim
        
        # Input processing
        self.input_layer = nn.Linear(obs_dim, hidden_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
        
        # Output layer with conservative initialization
        self.output_layer = nn.Linear(lstm_hidden_dim, num_bs * num_bands * num_power_levels)
        nn.init.normal_(self.output_layer.weight, 0, 0.01)
        nn.init.constant_(self.output_layer.bias, 0)
        
    def forward(self, obs, hidden_state=None):
        # Handle different input dimensions
        if obs.dim() == 1:
            obs = obs.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dims
        elif obs.dim() == 2:
            obs = obs.unsqueeze(1)  # Add sequence dim
            
        batch_size = obs.shape[0]
        
        # Process input
        x = F.relu(self.input_layer(obs))
        
        # Initialize hidden state if needed
        if hidden_state is None:
            h0 = torch.zeros(1, batch_size, self.lstm_hidden_dim, device=obs.device)
            c0 = torch.zeros(1, batch_size, self.lstm_hidden_dim, device=obs.device)
            hidden_state = (h0, c0)
        else:
            h, c = hidden_state
            if h.dim() == 2: 
                h = h.unsqueeze(0) 
                c = c.unsqueeze(0)
            hidden_state = (h, c)
            
        # LSTM forward
        lstm_out, (new_h, new_c) = self.lstm(x, hidden_state)
        
        # Get last output
        if lstm_out.dim() == 3:
            lstm_out = lstm_out[:, -1, :]  # Get last timestep
            
        # Output logits
        logits = self.output_layer(lstm_out)
        logits = logits.view(-1, self.num_bs * self.num_bands, self.num_power_levels)
        
        # Conservative clipping
        logits = torch.clamp(logits, -3.0, 3.0)
        new_hidden = LSTMState(new_h, new_c)
        
        return logits, new_hidden


class ValueNetwork(nn.Module):
    """Value network with residual blocks"""
    def __init__(self, obs_dim, hidden_dim=64, num_blocks=2):
        super().__init__()
        self.input_layer = nn.Linear(obs_dim, hidden_dim)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        x = F.relu(self.input_layer(x))
        for block in self.residual_blocks:
            x = block(x)
        return self.output_layer(x)

# ==============================================================================
# NORMALIZERS (Same as MAML)
# ==============================================================================

class ObservationNormalizer:
    """Running statistics for observation normalization"""
    def __init__(self, obs_dim: int):
        self.mean = torch.zeros(obs_dim)
        self.var = torch.ones(obs_dim)
        self.count = 0
        self.eps = 1e-4
        self.min_count = 50
        
    def update(self, obs_batch: torch.Tensor):
        """Update running statistics"""
        if obs_batch.dim() == 1:
            obs_batch = obs_batch.unsqueeze(0)
            
        batch_mean = obs_batch.mean(dim=0)
        batch_var = obs_batch.var(dim=0, unbiased=False)
        batch_count = obs_batch.shape[0]
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        
        self.count = total_count
        
    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observations"""
        if self.count < self.min_count:
            return torch.clamp(obs, -3.0, 3.0)
            
        normalized = (obs - self.mean) / torch.sqrt(self.var + self.eps + 1e-6)
        return torch.clamp(normalized, -3.0, 3.0)


class RewardNormalizer:
    """Running statistics for reward normalization"""
    def __init__(self):
        self.mean = torch.tensor(-300.0)
        self.var = torch.tensor(100.0)
        self.count = 0
        self.eps = 1e-4
        self.min_count = 200
        
    def update(self, rewards: torch.Tensor):
        """Update running statistics"""
        batch_mean = rewards.mean()
        batch_var = rewards.var(unbiased=False)
        batch_count = rewards.numel()
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        
        self.count = total_count
        
    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards"""
        if self.count < self.min_count:
            return rewards / 150.0
            
        normalized = (rewards - self.mean) / torch.sqrt(self.var + self.eps + 1e-6)
        return torch.clamp(normalized, -50.0, 50.0)

# ==============================================================================
# TRAJECTORY WITH RECURRENT STATES
# ==============================================================================

@dataclass
class RecurrentTrajectory:
    """Container for trajectory data with recurrent states"""
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    raw_rewards: torch.Tensor
    values: torch.Tensor
    dones: torch.Tensor
    log_probs: torch.Tensor
    sinr_violations: torch.Tensor
    qos_violations: torch.Tensor
    initial_hidden_state: LSTMState
    final_value: torch.Tensor
    final_hidden_state: LSTMState
    returns: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None


def sample_recurrent_trajectory(
    env: DynamicSpectrumEnv,
    policy_net: RecurrentPolicyNetwork,
    value_net: ValueNetwork,
    obs_normalizer: ObservationNormalizer,
    reward_normalizer: RewardNormalizer,
    num_steps: int = ROLLOUT_LENGTH,
    device: torch.device = DEVICE
) -> RecurrentTrajectory:
    """Sample a trajectory with recurrent policy"""
    
    # Reset environment
    obs_np, _ = env.reset()
    obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
    
    if not torch.all(torch.isfinite(obs)):
        print("Warning: Invalid initial observation")
        return create_empty_recurrent_trajectory(policy_net, device)
        
    norm_obs = obs_normalizer.normalize(obs)
    
    # Initialize hidden state
    with torch.no_grad():
        _, initial_hidden = policy_net(norm_obs, None)
    
    # Storage
    observations = []
    actions = []
    rewards = []
    raw_rewards = []
    values = []
    dones = []
    log_probs = []
    sinr_violations = []
    qos_violations = []
    
    hidden_state = initial_hidden
    
    for step in range(num_steps):
        with torch.no_grad():
            # Get action distribution and update hidden state
            logits, hidden_state = policy_net(norm_obs, hidden_state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Handle multi-dimensional log probs
            if log_prob.dim() > 0:
                log_prob = log_prob.sum(dim=-1).squeeze()
                
            # Get value
            value = value_net(norm_obs).squeeze()
            
        # Environment step
        action_np = action.cpu().numpy().flatten()
        next_obs_np, reward_scalar, terminated, truncated, info = env.step(action_np)
        
        # Convert to tensors
        next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
        raw_reward = torch.tensor(reward_scalar, dtype=torch.float32, device=device)
        done = terminated or truncated
        
        # Check validity
        if not torch.isfinite(raw_reward):
            raw_reward = torch.tensor(-0.1, device=device)
            
        # Normalize
        norm_reward = reward_normalizer.normalize(raw_reward)
        
        if not torch.all(torch.isfinite(next_obs)):
            print("Warning: Invalid observation encountered")
            done = True
            
        next_norm_obs = obs_normalizer.normalize(next_obs)
        
        # Store data
        observations.append(norm_obs)
        actions.append(action)
        rewards.append(norm_reward)
        raw_rewards.append(raw_reward)
        values.append(value)
        dones.append(torch.tensor(done, dtype=torch.bool, device=device))
        log_probs.append(log_prob)
        sinr_violations.append(torch.tensor(info.get('sinr_violations', 0.0), dtype=torch.float32, device=device))
        qos_violations.append(torch.tensor(info.get('qos_violations', 0.0), dtype=torch.float32, device=device))
        
        # Update for next step
        norm_obs = next_norm_obs
        
        if done:
            break
            
    if len(observations) == 0:
        return create_empty_recurrent_trajectory(policy_net, device)
        
    # Get final value
    with torch.no_grad():
        final_value = value_net(norm_obs).squeeze()
        
    # Stack tensors
    trajectory = RecurrentTrajectory(
        observations=torch.stack(observations),
        actions=torch.stack(actions),
        rewards=torch.stack(rewards),
        raw_rewards=torch.stack(raw_rewards),
        values=torch.stack(values),
        dones=torch.stack(dones),
        log_probs=torch.stack(log_probs),
        sinr_violations=torch.stack(sinr_violations),
        qos_violations=torch.stack(qos_violations),
        initial_hidden_state=initial_hidden,
        final_value=final_value,
        final_hidden_state=hidden_state
    )
    
    return trajectory


def create_empty_recurrent_trajectory(policy_net: RecurrentPolicyNetwork, device: torch.device) -> RecurrentTrajectory:
    """Create empty trajectory for error cases"""
    # =====================================================================
    # FIX: The hidden state must be 3D.
    # Shape: (num_layers, batch_size, hidden_dim)
    # Here, batch_size is 1.
    # =====================================================================
    lstm_hidden_dim = policy_net.lstm_hidden_dim
    empty_hidden = LSTMState(
        torch.zeros(1, 1, lstm_hidden_dim, device=device),
        torch.zeros(1, 1, lstm_hidden_dim, device=device)
    )
    
    num_bs = policy_net.num_bs
    num_bands = policy_net.num_bands
    

    obs_dim = 10*3 + 3*5 + 10*2 + 3*5 + 3*5 + 1
    
    return RecurrentTrajectory(
        observations=torch.empty(0, obs_dim, device=device),
        actions=torch.empty(0, num_bs * num_bands, device=device),
        rewards=torch.empty(0, device=device),
        raw_rewards=torch.empty(0, device=device),
        values=torch.empty(0, device=device),
        dones=torch.empty(0, dtype=torch.bool, device=device),
        log_probs=torch.empty(0, device=device),
        sinr_violations=torch.empty(0, device=device),
        qos_violations=torch.empty(0, device=device),
        initial_hidden_state=empty_hidden,
        final_value=torch.tensor(0.0, device=device),
        final_hidden_state=empty_hidden,
        returns=torch.empty(0, device=device),
        advantages=torch.empty(0, device=device)
    )

# ==============================================================================
# PPO LOSS COMPUTATION
# ==============================================================================

def compute_ppo_loss(
    policy_net: RecurrentPolicyNetwork,
    value_net: ValueNetwork,
    trajectory: RecurrentTrajectory,
    clip_ratio: float = 0.15,
    value_coeff: float = 0.3,
    entropy_coeff: float = 0.05,
    device: torch.device = DEVICE
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Compute PPO loss for recurrent policy"""
    
    if trajectory.observations.shape[0] == 0:
        zero_loss = torch.tensor(0.0, device=device)
        return zero_loss, (zero_loss, zero_loss, zero_loss)
        
    obs = trajectory.observations
    actions = trajectory.actions
    returns = trajectory.returns
    advantages = trajectory.advantages
    old_log_probs = trajectory.log_probs
    dones = trajectory.dones
    
    # Process sequence with recurrent policy
    hidden_state = trajectory.initial_hidden_state
    policy_losses = []
    value_losses = []
    entropy_bonuses = []
    
    for t in range(obs.shape[0]):
        # Get current predictions
        logits, hidden_state = policy_net(obs[t], hidden_state)
        dist = Categorical(logits=logits)
        
        current_log_prob = dist.log_prob(actions[t])
        if current_log_prob.dim() > 1:
            current_log_prob = current_log_prob.sum(dim=-1).squeeze()
            
        entropy = dist.entropy()
        if entropy.dim() > 1:
            entropy = entropy.mean()
            
        value = value_net(obs[t]).squeeze()
        
        # PPO objective
        log_ratio = current_log_prob - old_log_probs[t]
        log_ratio = torch.clamp(log_ratio, -5.0, 5.0)
        ratio = torch.exp(log_ratio)
        
        surr1 = ratio * advantages[t]
        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages[t]
        policy_loss = -torch.min(surr1, surr2)
        
        # Value loss
        value_loss = (returns[t] - value) ** 2
        
        # Entropy bonus
        entropy_bonus = -entropy
        
        policy_losses.append(policy_loss)
        value_losses.append(value_loss)
        entropy_bonuses.append(entropy_bonus)
        
    # Stack and compute valid steps
    policy_losses = torch.stack(policy_losses)
    value_losses = torch.stack(value_losses)
    entropy_bonuses = torch.stack(entropy_bonuses)
    
    # Find valid steps
    if torch.any(dones):
        first_done_idx = torch.where(dones)[0][0].item() + 1
        num_valid_steps = min(first_done_idx, obs.shape[0])
    else:
        num_valid_steps = obs.shape[0]
        
    if num_valid_steps == 0:
        zero_loss = torch.tensor(0.0, device=device)
        return zero_loss, (zero_loss, zero_loss, zero_loss)
        
    # Apply mask and compute means
    valid_mask = torch.arange(obs.shape[0], device=device) < num_valid_steps
    
    mean_policy_loss = (policy_losses * valid_mask).sum() / num_valid_steps
    mean_value_loss = (value_losses * valid_mask).sum() / num_valid_steps
    mean_entropy_bonus = (entropy_bonuses * valid_mask).sum() / num_valid_steps
    
    # Clamp losses
    mean_policy_loss = torch.clamp(mean_policy_loss, -10.0, 10.0)
    mean_value_loss = torch.clamp(mean_value_loss, 0.0, 10.0)
    mean_entropy_bonus = torch.clamp(mean_entropy_bonus, -2.0, 2.0)
    
    # Total loss
    total_loss = mean_policy_loss + value_coeff * mean_value_loss + entropy_coeff * mean_entropy_bonus
    
    # Safety check
    if not torch.isfinite(total_loss):
        total_loss = torch.tensor(0.0, device=device)
        
    return total_loss, (mean_policy_loss, mean_value_loss, mean_entropy_bonus)

# ==============================================================================
# PPO INNER ADAPTATION
# ==============================================================================

def ppo_inner_adaptation(
    policy_net: RecurrentPolicyNetwork,
    value_net: ValueNetwork,
    trajectory: RecurrentTrajectory,
    inner_lr: float,
    inner_steps: int,
    clip_ratio: float = 0.1,
    device: torch.device = DEVICE
) -> Tuple[RecurrentPolicyNetwork, ValueNetwork]:
    """Perform PPO inner loop adaptation"""
    
    if trajectory.observations.shape[0] == 0:
        return policy_net, value_net
        
    # Create copies for adaptation
    adapted_policy = copy.deepcopy(policy_net)
    adapted_value = copy.deepcopy(value_net)
    
    # Inner loop optimization
    for step in range(inner_steps):
        # Zero gradients
        adapted_policy.zero_grad()
        adapted_value.zero_grad()
        
        # Compute loss
        loss, _ = compute_ppo_loss(
            adapted_policy, adapted_value, trajectory, clip_ratio, device=device
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        max_grad_norm = 0.5
        torch.nn.utils.clip_grad_norm_(adapted_policy.parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(adapted_value.parameters(), max_grad_norm)
        
        # Update parameters
        effective_lr = inner_lr * 0.5
        with torch.no_grad():
            for param in adapted_policy.parameters():
                if param.grad is not None:
                    param.data -= effective_lr * param.grad
                    
            for param in adapted_value.parameters():
                if param.grad is not None:
                    param.data -= effective_lr * param.grad
                    
    return adapted_policy, adapted_value

# ==============================================================================
# META OBJECTIVE AND TRAINING
# ==============================================================================

def compute_meta_objective(
    policy_net: RecurrentPolicyNetwork,
    value_net: ValueNetwork,
    train_trajectory: RecurrentTrajectory,
    test_trajectory: RecurrentTrajectory,
    inner_lr: float,
    inner_steps: int,
    clip_ratio: float = 0.1,
    device: torch.device = DEVICE
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Compute meta-objective for recurrent MAML-PPO"""
    
    # Adapt on training trajectory
    adapted_policy, adapted_value = ppo_inner_adaptation(
        policy_net, value_net, train_trajectory, inner_lr, inner_steps, clip_ratio, device
    )
    
    # Evaluate on test trajectory
    test_loss, loss_components = compute_ppo_loss(
        adapted_policy, adapted_value, test_trajectory, clip_ratio, device=device
    )
    
    return test_loss, loss_components


def prepare_recurrent_trajectory(trajectory: RecurrentTrajectory) -> RecurrentTrajectory:
    """Add returns and advantages to recurrent trajectory"""
    if trajectory.observations.shape[0] == 0:
        trajectory.returns = torch.empty(0)
        trajectory.advantages = torch.empty(0)
        return trajectory
        
    returns, advantages = compute_gae(
        trajectory.rewards,
        trajectory.values,
        trajectory.dones,
        trajectory.final_value
    )
    
    trajectory.returns = returns
    trajectory.advantages = advantages
    return trajectory


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    final_value: torch.Tensor,
    gamma: float = 0.99,
    lambda_: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (same as MAML)"""
    
    if rewards.shape[0] == 0:
        return torch.empty(0), torch.empty(0)
        
    device = rewards.device
    episode_length = rewards.shape[0]
    
    advantages = torch.zeros_like(rewards)
    next_value = final_value
    next_advantage = torch.tensor(0.0, device=device)
    
    for t in reversed(range(episode_length)):
        mask = (~dones[t]).float()
        td_error = rewards[t] + gamma * next_value * mask - values[t]
        advantages[t] = td_error + gamma * lambda_ * next_advantage * mask
        next_value = values[t]
        next_advantage = advantages[t]
        
    returns = advantages + values
    
    # Find actual sequence length
    if torch.any(dones):
        first_done_idx = torch.where(dones)[0][0].item() + 1
        num_actual_steps = min(first_done_idx, episode_length)
    else:
        num_actual_steps = episode_length
        
    if num_actual_steps > 0:
        # Normalize advantages
        valid_advantages = advantages[:num_actual_steps]
        mean = valid_advantages.mean()
        std = valid_advantages.std() + 1e-8
        advantages = (advantages - mean) / std
        
    # Safety checks
    advantages = torch.nan_to_num(advantages, nan=0.0, posinf=3.0, neginf=-3.0)
    returns = torch.nan_to_num(returns, nan=0.0, posinf=10.0, neginf=-10.0)
    
    return returns, advantages


def sample_task(base_env: DynamicSpectrumEnv) -> DynamicSpectrumEnv:
    """Sample a task by varying environment parameters (same as MAML)"""
    
    # Sample variations
    interference_var = np.random.uniform(0.8, 1.2)
    fading_var = np.random.uniform(0.9, 1.1)
    latency_var = np.random.uniform(0.9, 1.1)
    sinr_var = np.random.uniform(0.95, 1.05)
    
    # Apply variations
    new_max_interference = base_env.max_interference * interference_var
    new_fading_coherence = np.clip(base_env.fading_coherence * fading_var, 0.5, 0.99)
    new_max_latency = base_env.max_latency * latency_var
    new_min_sinr = base_env.min_sinr * sinr_var
    
    # Create new environment
    try:
        new_env = DynamicSpectrumEnv(
            num_bs=base_env.num_bs,
            num_users=base_env.num_users,
            num_bands=base_env.num_bands,
            max_steps=base_env.max_steps,
            max_latency=new_max_latency,
            max_power=base_env.max_power,
            num_power_levels=base_env.num_power_levels,
            power_levels=base_env.power_levels.cpu().numpy(),
            fading_coherence=new_fading_coherence,
            max_interference=new_max_interference,
            min_sinr=new_min_sinr,
            device=base_env.device.type
        )
        
        # Preserve bandwidth and noise figure
        new_env.bandwidth_hz = base_env.bandwidth_hz
        new_env.noise_figure_db = base_env.noise_figure_db
        new_env.thermal_noise_dbm_hz = base_env.thermal_noise_dbm_hz
        
        return new_env
        
    except Exception as e:
        print(f"Error creating task environment: {e}")
        return base_env


def train_recurrent_maml_ppo(
    env: DynamicSpectrumEnv,
    policy_net: RecurrentPolicyNetwork,
    value_net: ValueNetwork,
    num_tasks: int = 4,
    inner_lr: float = 0.01,
    inner_steps: int = 5,
    meta_lr: float = 0.001,
    num_iterations: int = 1000,
    clip_ratio: float = 0.1,
    eval_interval: int = 10,
    num_eval_tasks: int = 5,
    rollout_length: int = ROLLOUT_LENGTH,
    device: torch.device = DEVICE,
    use_wandb: bool = True,
    wandb_project: str = "recurrent-maml-ppo",
    wandb_name: Optional[str] = None
) -> Tuple[Tuple[RecurrentPolicyNetwork, ValueNetwork], Dict[str, List]]:
    """Train Recurrent MAML with PPO"""
    
    # Move networks to device
    policy_net = policy_net.to(device)
    value_net = value_net.to(device)
    
    # Initialize normalizers
    obs_dim = env.observation_space.shape[0]
    obs_normalizer = ObservationNormalizer(obs_dim)
    reward_normalizer = RewardNormalizer()
    
    # Meta-optimizer with learning rate schedule
    meta_optimizer = optim.Adam(
        list(policy_net.parameters()) + list(value_net.parameters()),
        lr=meta_lr * 0.5,  # Conservative initial LR
        eps=1e-8
    )
    
    scheduler = optim.lr_scheduler.ExponentialLR(meta_optimizer, gamma=0.98)
    
    # History tracking
    history = defaultdict(list)
    
    # Initialize wandb
    if use_wandb:
        config = {
            "num_tasks": num_tasks,
            "inner_lr": inner_lr,
            "inner_steps": inner_steps,
            "meta_lr": meta_lr,
            "num_iterations": num_iterations,
            "clip_ratio": clip_ratio,
            "rollout_length": rollout_length
        }
        wandb.init(project=wandb_project, name=wandb_name, config=config)
        
    successful_iterations = 0
    consecutive_failures = 0
    
    # Main training loop
    for iteration in range(num_iterations):
        # Clear caches periodically
        if iteration > 0 and iteration % 10 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        # Meta-gradient accumulation
        meta_loss_total = 0.0
        num_successful_tasks = 0
        task_failures = []
        
        # Zero meta-gradients
        meta_optimizer.zero_grad()
        
        # Process each task
        for task_idx in range(num_tasks):
            try:
                # Sample task
                task_env = sample_task(env)
                
                # Sample training trajectory
                train_traj = sample_recurrent_trajectory(
                    task_env, policy_net, value_net,
                    obs_normalizer, reward_normalizer,
                    rollout_length, device
                )
                
                if train_traj.observations.shape[0] == 0:
                    task_failures.append(f"Task {task_idx}: Empty training trajectory")
                    continue
                    
                # Check for NaN in observations
                if not torch.all(torch.isfinite(train_traj.observations)):
                    task_failures.append(f"Task {task_idx}: NaN in training observations")
                    continue
                    
                # Prepare trajectory with GAE
                train_traj = prepare_recurrent_trajectory(train_traj)
                
                # Check GAE outputs
                if (train_traj.returns.shape[0] > 0 and 
                    not torch.all(torch.isfinite(train_traj.returns))):
                    task_failures.append(f"Task {task_idx}: NaN in returns")
                    continue
                
                # Sample test trajectory (using current parameters for consistency)
                test_traj = sample_recurrent_trajectory(
                    task_env, policy_net, value_net,
                    obs_normalizer, reward_normalizer,
                    rollout_length, device
                )
                
                if test_traj.observations.shape[0] == 0:
                    task_failures.append(f"Task {task_idx}: Empty test trajectory")
                    continue
                    
                if not torch.all(torch.isfinite(test_traj.observations)):
                    task_failures.append(f"Task {task_idx}: NaN in test observations")
                    continue
                    
                # Prepare test trajectory
                test_traj = prepare_recurrent_trajectory(test_traj)
                
                # Compute meta-objective
                meta_loss, _ = compute_meta_objective(
                    policy_net, value_net, train_traj, test_traj,
                    inner_lr, inner_steps, clip_ratio, device
                )
                
                # Check validity
                if not torch.isfinite(meta_loss):
                    task_failures.append(f"Task {task_idx}: Non-finite meta loss: {meta_loss}")
                    continue
                    
                # Accumulate meta-loss
                meta_loss_total += meta_loss
                num_successful_tasks += 1
                
                # Backward pass for this task
                meta_loss.backward()
                
                # Update normalizers
                if train_traj.observations.shape[0] > 0:
                    obs_normalizer.update(train_traj.observations)
                    if train_traj.raw_rewards.shape[0] > 0:
                        reward_normalizer.update(train_traj.raw_rewards)
                        
                if test_traj.observations.shape[0] > 0:
                    obs_normalizer.update(test_traj.observations)
                    if test_traj.raw_rewards.shape[0] > 0:
                        reward_normalizer.update(test_traj.raw_rewards)
                        
            except Exception as e:
                task_failures.append(f"Task {task_idx}: Exception - {str(e)[:100]}")
                continue
                
        # Debug output
        if task_failures and iteration % 5 == 0:
            print(f"Iteration {iteration} task failures:")
            for failure in task_failures[:3]:
                print(f"  {failure}")
            if len(task_failures) > 3:
                print(f"  ... and {len(task_failures) - 3} more")
                
        # Apply meta-update if we have successful tasks
        if num_successful_tasks > 0:
            # Average meta-loss
            avg_meta_loss = meta_loss_total / num_successful_tasks
            history['meta_losses'].append(avg_meta_loss.item())
            
            # Gradient clipping
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                list(policy_net.parameters()) + list(value_net.parameters()),
                max_norm=0.5
            )
            
            if torch.isfinite(torch.tensor(total_grad_norm)) and total_grad_norm < 100.0:
                # Meta-update
                meta_optimizer.step()
                consecutive_failures = 0
                successful_iterations += 1
            else:
                print(f"Warning: Extreme gradient norm {total_grad_norm} at iteration {iteration}")
                consecutive_failures += 1
        else:
            history['meta_losses'].append(float('nan'))
            consecutive_failures += 1
            print(f"Warning: Iteration {iteration} had no successful training tasks.")
            
        if consecutive_failures > 50:
            print(f"Early stopping due to {consecutive_failures} consecutive failures")
            break
            
        # Logging
        log_data = {
            'meta_loss': history['meta_losses'][-1] if history['meta_losses'] else 0.0,
            'successful_tasks_ratio': num_successful_tasks / num_tasks if num_tasks > 0 else 0.0,
            'iteration': iteration,
            'consecutive_failures': consecutive_failures
        }
        
        # Evaluation
        if iteration % eval_interval == 0:
            eval_metrics = evaluate_recurrent_maml(
                env, policy_net, value_net, obs_normalizer, reward_normalizer,
                inner_lr, inner_steps, clip_ratio, num_eval_tasks, rollout_length, device
            )
            
            # Update history
            for key, value in eval_metrics.items():
                history[key].append(value)
                log_data[key] = value
                
            # Print progress
            pre_r = eval_metrics.get('eval_avg_pre_reward', 0.0)
            post_r = eval_metrics.get('eval_avg_post_reward', 0.0)
            improvement_pct = ((post_r - pre_r) / abs(pre_r) * 100) if pre_r != 0 else 0.0
            
            print(f"[Iter {iteration:3d}] "
                  f"meta_loss={log_data.get('meta_loss', 0.0):6.3f} | "
                  f"success_rate={log_data.get('successful_tasks_ratio', 0.0):.2f} | "
                  f"pre_r={pre_r:6.3f} post_r={post_r:6.3f} | "
                  f"gain={improvement_pct:+5.2f}%")
                  
            # Enhanced logging
            if iteration > 0 and iteration % 20 == 0 and len(history.get('eval_avg_post_reward', [])) >= 2:
                recent_trend = "↗️" if history['eval_avg_post_reward'][-1] > history['eval_avg_post_reward'][-2] else "↘️"
                print(f"\n=== Training Summary (Iteration {iteration}) ===")
                print(f"Recent reward trend: {recent_trend}")
                print(f"Current: pre={pre_r:.3f}, post={post_r:.3f}, gain={improvement_pct:.1f}%")
                print("=" * 50)
        else:
            print(f"[Iter {iteration:3d}] "
                  f"meta_loss={log_data.get('meta_loss', 0.0):6.3f} | "
                  f"success_rate={log_data.get('successful_tasks_ratio', 0.0):.2f} | "
                  f"training...")
                  
        if use_wandb:
            wandb.log(log_data, step=iteration)
            
        # Update learning rate
        if iteration > 0 and iteration % 20 == 0:
            scheduler.step()
            
    if use_wandb:
        wandb.finish()
        
    # Final history
    history['training_stats'] = {
        'successful_iterations': successful_iterations,
        'total_iterations': iteration + 1,
        'final_consecutive_failures': consecutive_failures
    }
    
    return (policy_net, value_net), dict(history)


def evaluate_recurrent_maml(
    env: DynamicSpectrumEnv,
    policy_net: RecurrentPolicyNetwork,
    value_net: ValueNetwork,
    obs_normalizer: ObservationNormalizer,
    reward_normalizer: RewardNormalizer,
    inner_lr: float,
    inner_steps: int,
    clip_ratio: float,
    num_eval_tasks: int,
    rollout_length: int,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate Recurrent MAML performance"""
    
    pre_rewards, post_rewards = [], []
    pre_sinrs, post_sinrs = [], []
    pre_qoss, post_qoss = [], []
    
    for eval_idx in range(num_eval_tasks):
        try:
            # Sample evaluation task
            eval_env = sample_task(env)
            
            # Pre-adaptation trajectory
            pre_traj = sample_recurrent_trajectory(
                eval_env, policy_net, value_net,
                obs_normalizer, reward_normalizer,
                rollout_length, device
            )
            
            if pre_traj.observations.shape[0] > 0:
                pre_rewards.append(pre_traj.raw_rewards.mean().item()) # Using raw rewards for interpretable eval
                pre_sinrs.append(pre_traj.sinr_violations.mean().item())
                pre_qoss.append(pre_traj.qos_violations.mean().item())
                
            # Adapt to evaluation task
            adapted_policy, adapted_value = policy_net, value_net
            if pre_traj.observations.shape[0] > 0:
                pre_traj = prepare_recurrent_trajectory(pre_traj)
                adapted_policy, adapted_value = ppo_inner_adaptation(
                    policy_net, value_net, pre_traj, inner_lr, inner_steps, clip_ratio, device
                )
                
            # Post-adaptation trajectory
            post_traj = sample_recurrent_trajectory(
                eval_env, adapted_policy, adapted_value,
                obs_normalizer, reward_normalizer,
                rollout_length, device
            )
            
            if post_traj.observations.shape[0] > 0:
                post_rewards.append(post_traj.raw_rewards.mean().item()) # Using raw rewards for interpretable eval
                post_sinrs.append(post_traj.sinr_violations.mean().item())
                post_qoss.append(post_traj.qos_violations.mean().item())
                
        except Exception as e:
            print(f"Error in evaluation task {eval_idx}: {e}")
            continue
            
    # Aggregate metrics
    metrics = {}
    
    if pre_rewards:
        avg_pre_r = np.mean(pre_rewards)
        avg_post_r = np.mean(post_rewards) if post_rewards else avg_pre_r
        improvement = avg_post_r - avg_pre_r
        
        metrics.update({
            'eval_avg_pre_reward': avg_pre_r,
            'eval_avg_post_reward': avg_post_r,
            'eval_avg_reward_improvement': improvement
        })
        
    if pre_sinrs:
        avg_pre_s = np.mean(pre_sinrs)
        avg_post_s = np.mean(post_sinrs) if post_sinrs else avg_pre_s
        improvement_s = avg_pre_s - avg_post_s  # Lower is better
        
        metrics.update({
            'eval_avg_pre_sinr_violation': avg_pre_s,
            'eval_avg_post_sinr_violation': avg_post_s,
            'eval_avg_sinr_improvement': improvement_s
        })
        
    if pre_qoss:
        avg_pre_q = np.mean(pre_qoss)
        avg_post_q = np.mean(post_qoss) if post_qoss else avg_pre_q
        improvement_q = avg_pre_q - avg_post_q  # Lower is better
        
        metrics.update({
            'eval_avg_pre_qos_violation': avg_pre_q,
            'eval_avg_post_qos_violation': avg_post_q,
            'eval_avg_qos_improvement': improvement_q
        })
        
    return metrics


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    # Create environment
    env = DynamicSpectrumEnv(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Get dimensions
    obs_dim = env.observation_space.shape[0]
    num_bs = env.num_bs
    num_bands = env.num_bands
    num_power_levels = env.num_power_levels
    
    # Create networks
    policy_net = RecurrentPolicyNetwork(obs_dim, num_bs, num_bands, num_power_levels)
    value_net = ValueNetwork(obs_dim)
    
    # Train Recurrent MAML-PPO
    print("Starting Recurrent MAML-PPO training...")
    (trained_policy, trained_value), history = train_recurrent_maml_ppo(
        env=env,
        policy_net=policy_net,
        value_net=value_net,
        num_tasks=4,
        inner_lr=0.01,
        inner_steps=5,
        meta_lr=0.001,
        num_iterations=100,
        clip_ratio=0.1,
        eval_interval=10,
        num_eval_tasks=5,
        rollout_length=20,
        use_wandb=False  # Set to True to use wandb logging
    )
    
    print("\nTraining completed!")
    print(f"Final meta loss: {history['meta_losses'][-1] if history['meta_losses'] else 'N/A'}")
    
    # Save models
    torch.save(trained_policy.state_dict(), 'recurrent_maml_policy.pth')
    torch.save(trained_value.state_dict(), 'recurrent_maml_value.pth')
    print("Models saved!")
