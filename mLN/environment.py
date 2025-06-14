import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import warnings

# Constants and hyperparameters
MAX_INTERFERENCE = 25.0  
MAX_POWER = 23.0  
MAX_LATENCY = 50.0  
NUM_BS = 3
NUM_USERS = 10
NUM_BANDS = 5
NUM_POWER_LEVELS = 4 
POWER_LEVELS = np.linspace(0, MAX_POWER, NUM_POWER_LEVELS) 
FADING_COHERENCE = 0.9  
MAX_STEPS = 100 
MIN_SINR = 5.0 
# Additional constants for proper wireless calculations
NOISE_FIGURE_DB = 7.0
BANDWIDTH_HZ = 10e6  # 10 MHz per band
THERMAL_NOISE_DBM_HZ = -174.0  # Thermal noise density

@dataclass
class SpectrumState:
    """State representation for the spectrum allocation environment."""
    channel_gains: torch.Tensor      # Shape: (num_users, num_bs)
    interference_map: torch.Tensor   # Shape: (num_bs, num_bands)
    qos_metrics: torch.Tensor        # Shape: (num_users, 2) - [latency, throughput]
    spectrum_alloc: torch.Tensor     # Shape: (num_bs, num_bands)
    tx_power: torch.Tensor           # Shape: (num_bs, num_bands)
    time: int                        # Current timestep
    
    def to_numpy(self) -> np.ndarray:
        """Convert state to numpy array for observation."""
        return np.concatenate([
            self.channel_gains.cpu().numpy().flatten(),
            self.interference_map.cpu().numpy().flatten(),
            self.qos_metrics.cpu().numpy().flatten(),
            self.spectrum_alloc.cpu().numpy().flatten(),
            self.tx_power.cpu().numpy().flatten(),
            np.array([self.time])
        ])
    
    @classmethod
    def from_numpy(cls, obs: np.ndarray, num_users: int, num_bs: int, num_bands: int, device: torch.device):
        """Reconstruct state from numpy observation."""
        idx = 0
        channel_gains = torch.tensor(obs[idx:idx + num_users * num_bs]).reshape(num_users, num_bs).to(device)
        idx += num_users * num_bs
        
        interference_map = torch.tensor(obs[idx:idx + num_bs * num_bands]).reshape(num_bs, num_bands).to(device)
        idx += num_bs * num_bands
        
        qos_metrics = torch.tensor(obs[idx:idx + num_users * 2]).reshape(num_users, 2).to(device)
        idx += num_users * 2
        
        spectrum_alloc = torch.tensor(obs[idx:idx + num_bs * num_bands]).reshape(num_bs, num_bands).to(device)
        idx += num_bs * num_bands
        
        tx_power = torch.tensor(obs[idx:idx + num_bs * num_bands]).reshape(num_bs, num_bands).to(device)
        idx += num_bs * num_bands
        
        time = int(obs[idx])
        
        return cls(channel_gains, interference_map, qos_metrics, spectrum_alloc, tx_power, time)


class DynamicSpectrumEnv(gym.Env):
    """Dynamic Spectrum Allocation Environment using PyTorch and Gymnasium."""
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(self, 
                 num_bs: int = NUM_BS,
                 num_users: int = NUM_USERS,
                 num_bands: int = NUM_BANDS,
                 max_steps: int = MAX_STEPS,
                 max_latency: float = MAX_LATENCY,
                 max_power: float = MAX_POWER,
                 num_power_levels: int = NUM_POWER_LEVELS,
                 power_levels: Optional[np.ndarray] = None,
                 fading_coherence: float = FADING_COHERENCE,
                 max_interference: float = MAX_INTERFERENCE,
                 min_sinr: float = MIN_SINR,
                 device: str = "cpu",
                 render_mode: Optional[str] = None):
        
        super().__init__()
        
        self.num_bs = num_bs
        self.num_users = num_users
        self.num_bands = num_bands
        self.max_steps = max_steps
        self.max_latency = max_latency
        self.max_power = max_power
        self.num_power_levels = num_power_levels
        self.power_levels = torch.tensor(
            power_levels if power_levels is not None else POWER_LEVELS,
            dtype=torch.float32
        )
        self.fading_coherence = fading_coherence
        self.max_interference = max_interference
        self.min_sinr = min_sinr
        self.noise_figure_db = NOISE_FIGURE_DB
        self.bandwidth_hz = BANDWIDTH_HZ
        self.thermal_noise_dbm_hz = THERMAL_NOISE_DBM_HZ
        
        self.device = torch.device(device)
        self.power_levels = self.power_levels.to(self.device)
        
        self.render_mode = render_mode
        
        # Define action and observation spaces
        self.action_space = spaces.MultiDiscrete(
            [self.num_power_levels] * (self.num_bs * self.num_bands)
        )
        
        # Calculate observation space dimensions
        obs_dim = (
            num_users * num_bs +      # channel_gains
            num_bs * num_bands +      # interference_map
            num_users * 2 +           # qos_metrics
            num_bs * num_bands +      # spectrum_alloc
            num_bs * num_bands +      # tx_power
            1                         # time
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.state: Optional[SpectrumState] = None
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Scenario selection
        scenario = torch.randint(0, 3, (1,)).item()
        
        # Distance-dependent path loss (in dB)
        distances = torch.rand(self.num_users, self.num_bs, device=self.device) * 1.9 + 0.1  # 0.1-2.0 km
        
        # Path loss models for different scenarios (Urban, Suburban, Rural)
        if scenario == 0:  # Urban
            path_loss = 128.1 + 37.6 * torch.log10(distances)
        elif scenario == 1:  # Suburban
            path_loss = 98.5 + 23.1 * torch.log10(distances)
        else:  # Rural
            path_loss = 105.3 + 34.2 * torch.log10(distances)
        
        # Shadow fading (log-normal)
        shadow_fading = torch.randn(self.num_users, self.num_bs, device=self.device) * 8.0  # 8 dB std
        
        # Total channel gains
        channel_gains = -(path_loss + shadow_fading)
        
        # External interference
        interference_mw = torch.rand(self.num_bs, self.num_bands, device=self.device) * (self.max_interference * 0.5 - 0.001) + 0.001
        interference_map = 10.0 * torch.log10(interference_mw + 1e-12)
        
        self.state = SpectrumState(
            channel_gains=channel_gains,
            interference_map=interference_map,
            qos_metrics=torch.zeros(self.num_users, 2, dtype=torch.float32, device=self.device),
            spectrum_alloc=torch.zeros(self.num_bs, self.num_bands, dtype=torch.int32, device=self.device),
            tx_power=torch.zeros(self.num_bs, self.num_bands, dtype=torch.float32, device=self.device),
            time=0
        )
        
        return self.state.to_numpy(), {}
    
    def _mask_unsafe_actions(self, state: SpectrumState) -> torch.Tensor:
        """Generate action mask based on interference levels."""
        # Mask based on interference levels
        interference_linear = 10 ** (state.interference_map / 10.0)
        max_interference_linear = 10 ** (torch.log10(torch.tensor(self.max_interference * 1000)) / 10.0)
        interference_mask = interference_linear < max_interference_linear
        
        # Expand mask for all power levels
        interference_mask = interference_mask.unsqueeze(-1)
        interference_mask = interference_mask.expand(-1, -1, self.num_power_levels)
        
        return interference_mask
    
    def _compute_sinr(self, state: SpectrumState) -> torch.Tensor:
        """Compute Signal-to-Interference-plus-Noise Ratio for all users and bands."""
        # Thermal noise power in dBm
        noise_power_dbm = (self.thermal_noise_dbm_hz + self.noise_figure_db + 
                          10 * np.log10(self.bandwidth_hz))
        
        sinr_db = torch.zeros(self.num_users, self.num_bands, device=self.device)
        
        for user_idx in range(self.num_users):
            for band_idx in range(self.num_bands):
                # Received powers from all base stations
                received_powers = state.tx_power[:, band_idx] + state.channel_gains[user_idx, :]
                
                # Find serving base station (highest received power)
                serving_bs = torch.argmax(received_powers)
                signal_power_dbm = received_powers[serving_bs]
                
                # Interference from other BSs
                other_bs_mask = torch.arange(self.num_bs, device=self.device) != serving_bs
                interference_from_bs = torch.where(
                    other_bs_mask,
                    10 ** (received_powers / 10.0),
                    torch.tensor(0.0, device=self.device)
                )
                total_interference_from_bs = torch.sum(interference_from_bs)
                
                # External interference
                external_interference_linear = 10 ** (state.interference_map[serving_bs, band_idx] / 10.0)
                
                # Thermal noise (convert to linear)
                noise_power_linear = 10 ** (noise_power_dbm / 10.0)
                
                # Total interference + noise
                total_interference_linear = (total_interference_from_bs + 
                                           external_interference_linear + 
                                           noise_power_linear)
                
                # Signal power to linear
                signal_power_linear = 10 ** (signal_power_dbm / 10.0)
                
                # Calculate SINR and convert back to dB
                sinr_linear = signal_power_linear / (total_interference_linear + 1e-12)
                sinr_db[user_idx, band_idx] = 10 * torch.log10(sinr_linear + 1e-12)
        
        return sinr_db
    
    def _calculate_reward(self, state: SpectrumState, action: torch.Tensor, previous_tx_power: torch.Tensor) -> float:
        """Calculate reward based on multiple objectives."""
        POWER_COST_COEFF = 0.05
        SWITCHING_COST_COEFF = 0.5
        UTILIZATION_BONUS_COEFF = 1.0
        VIOLATION_PENALTY_COEFF = 15.0
        FAIRNESS_COEFF = 1.0
        THROUGHPUT_COEFF = 2.0
        
        # Calculate SINR for all users
        sinr_db = self._compute_sinr(state)
        
        # Calculate throughput using Shannon capacity
        sinr_linear = 10 ** (sinr_db / 10.0)
        best_sinr_per_user = torch.max(sinr_linear, dim=1)[0]
        
        # Spectral efficiency (bits/s/Hz)
        spectral_efficiency = torch.log2(1 + best_sinr_per_user + 1e-12)
        # Total throughput (normalized)
        throughput = torch.sum(spectral_efficiency) / self.num_users
        
        # Fairness using Jain's fairness index
        throughput_per_user = spectral_efficiency + 1e-12
        fairness = (torch.sum(throughput_per_user)**2) / (self.num_users * torch.sum(throughput_per_user**2))
        
        # SINR violations (users below minimum SINR threshold)
        sinr_violations = torch.sum(torch.max(sinr_db, dim=1)[0] < self.min_sinr)
        
        # QoS violations (latency)
        latency_violations = torch.sum(state.qos_metrics[:, 0] > self.max_latency)
        
        # Total violations penalty
        violation_penalty = VIOLATION_PENALTY_COEFF * (latency_violations + sinr_violations)
        
        # Power consumption cost
        new_tx_power = self.power_levels[action]
        total_power_linear = torch.sum(10 ** (new_tx_power / 10.0))
        power_cost = POWER_COST_COEFF * torch.log10(total_power_linear + 1e-12)
        
        # Switching cost
        power_changes = torch.abs(previous_tx_power - new_tx_power)
        switching_cost = SWITCHING_COST_COEFF * torch.sum(power_changes) / (self.num_bs * self.num_bands)
        
        # Utilization bonus
        active_channels = torch.sum(action > 0)
        utilization_bonus = UTILIZATION_BONUS_COEFF * (active_channels / (self.num_bs * self.num_bands))
        
        # Combined reward
        total_reward = (THROUGHPUT_COEFF * throughput + 
                       FAIRNESS_COEFF * fairness + 
                       utilization_bonus - 
                       violation_penalty - 
                       power_cost - 
                       switching_cost)
        
        return total_reward.item()
    
    def _adaptive_penalty(self, state: SpectrumState) -> float:
        """Calculate adaptive penalty based on network conditions."""
        # High interference penalty
        interference_linear = 10 ** (state.interference_map / 10.0)
        high_interference_threshold = 0.8 * 10 ** (torch.log10(torch.tensor(self.max_interference * 1000)) / 10.0)
        high_interference_penalty = 0.1 * torch.sum(interference_linear > high_interference_threshold)
        
        # Poor coverage penalty
        sinr_db = self._compute_sinr(state)
        poor_coverage_penalty = 0.05 * torch.sum(torch.max(sinr_db, dim=1)[0] < (self.min_sinr - 5.0))
        
        penalty = high_interference_penalty + poor_coverage_penalty
        return penalty.item()
    
    def _step_dynamics(self, state: SpectrumState, action: torch.Tensor) -> SpectrumState:
        """Update environment dynamics."""
        # Correlated fading
        fading_noise = torch.randn_like(state.channel_gains) * 2.0  # 2 dB std
        correlation_factor = np.sqrt(1 - self.fading_coherence**2)
        new_channel = (self.fading_coherence * state.channel_gains + 
                      correlation_factor * fading_noise)
        
        # Update spectrum allocation and power
        new_alloc = action.int()
        new_tx_power = self.power_levels[action]
        
        # Update QoS metrics
        sinr_db = self._compute_sinr(state)
        user_best_sinr = torch.max(sinr_db, dim=1)[0]
        
        # Latency penalty for poor SINR
        latency_penalty = torch.where(
            user_best_sinr < self.min_sinr,
            2.0 * torch.exp(-(user_best_sinr - self.min_sinr) / 5.0),
            torch.tensor(0.1, device=self.device)
        )
        
        new_latency = state.qos_metrics[:, 0] + latency_penalty
        
        # Throughput based on spectral efficiency
        sinr_linear = 10 ** (user_best_sinr / 10.0)
        spectral_efficiency = torch.log2(1 + sinr_linear + 1e-12)
        normalized_throughput = spectral_efficiency / 10.0
        
        new_qos = torch.stack([new_latency, normalized_throughput], dim=1)
        
        # Evolve external interference
        interference_noise = torch.randn_like(state.interference_map) * 0.5
        new_interference_map = state.interference_map + interference_noise
        new_interference_map = torch.clamp(new_interference_map, -20.0, 30.0)
        
        return SpectrumState(
            channel_gains=new_channel,
            interference_map=new_interference_map,
            qos_metrics=new_qos,
            spectrum_alloc=new_alloc,
            tx_power=new_tx_power,
            time=state.time + 1
        )
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one environment step."""
        if self.state is None:
            raise RuntimeError("Must call reset() before step()")
        
        # Convert action to tensor and reshape
        action_tensor = torch.tensor(action, dtype=torch.long, device=self.device)
        action_reshaped = action_tensor.reshape(self.num_bs, self.num_bands)
        
        # Apply safety mask
        action_mask = self._mask_unsafe_actions(self.state)
        mask_indices = torch.arange(self.num_bs, device=self.device)[:, None], \
                      torch.arange(self.num_bands, device=self.device)[None, :], \
                      action_reshaped
        safety = action_mask[mask_indices]
        safe_action = torch.where(safety, action_reshaped, torch.tensor(0, device=self.device))
        
        # Store previous power for reward calculation
        previous_tx_power = self.state.tx_power.clone()
        
        # Update state
        self.state = self._step_dynamics(self.state, safe_action)
        
        # Calculate reward
        reward = self._calculate_reward(self.state, safe_action, previous_tx_power) - self._adaptive_penalty(self.state)
        
        # Check termination conditions
        terminated = torch.any(self.state.qos_metrics[:, 0] > 2 * self.max_latency).item()
        sinr_db = self._compute_sinr(self.state)
        severe_sinr_violations = torch.sum(torch.max(sinr_db, dim=1)[0] < (self.min_sinr - 10.0)) > (0.8 * self.num_users)
        terminated = terminated or severe_sinr_violations.item()
        
        truncated = self.state.time >= self.max_steps
        
        # Additional info
        info = {
            "latency_violations": torch.sum(self.state.qos_metrics[:, 0] > self.max_latency).item(),
            "sinr_violations": torch.sum(torch.max(sinr_db, dim=1)[0] < self.min_sinr).item(),
            "qos_violations": torch.sum(self.state.qos_metrics[:, 0] > self.max_latency).item(),  # Same as latency_violations for compatibility
            "average_sinr": torch.mean(torch.max(sinr_db, dim=1)[0]).item(),
            "average_throughput": torch.mean(self.state.qos_metrics[:, 1]).item(),
            "action_mask": action_mask.cpu().numpy()
        }
        
        return self.state.to_numpy(), reward, terminated, truncated, info
    
    def render(self) -> Optional[str]:
        """Render the environment state."""
        if self.state is None:
            return None
        
        sinr_db = self._compute_sinr(self.state)
        
        output = f"Step {self.state.time}\n"
        output += "Spectrum Allocation:\n"
        output += str(self.state.spectrum_alloc.cpu().numpy()) + "\n"
        output += "Transmit Power (dBm):\n"
        output += str(self.state.tx_power.cpu().numpy()) + "\n"
        output += f"Latency violations: {torch.sum(self.state.qos_metrics[:, 0] > self.max_latency).item()}\n"
        output += f"SINR violations: {torch.sum(torch.max(sinr_db, dim=1)[0] < self.min_sinr).item()}\n"
        output += f"Average user SINR: {torch.mean(torch.max(sinr_db, dim=1)[0]).item():.2f} dB\n"
        output += f"Average throughput: {torch.mean(self.state.qos_metrics[:, 1]).item():.3f}\n"
        
        if self.render_mode == "human":
            print(output)
        
        return output
    
    def close(self):
        """Clean up resources."""
        pass
    
    def get_action_mask(self) -> np.ndarray:
        """Get current valid actions mask for safe exploration."""
        if self.state is None:
            raise RuntimeError("Must call reset() before getting action mask")
        
        action_mask = self._mask_unsafe_actions(self.state)
        return action_mask.cpu().numpy()


# Example usage and testing
if __name__ == "__main__":
    # Create environment
    env = DynamicSpectrumEnv(device="cuda" if torch.cuda.is_available() else "cpu", render_mode="human")
    
    # Test basic functionality
    obs, info = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward:.3f}")
    print(f"Info: {info}")
    
    # Test rendering
    env.render()
    
    # Simple training loop example
    def train_random_policy(num_episodes=10):
        """Simple training loop with random policy."""
        episode_rewards = []
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Get action mask for safe exploration
                action_mask = env.get_action_mask()
                
                # Random valid action
                action = env.action_space.sample()
                
                # Apply mask to ensure safety
                action_reshaped = action.reshape(env.num_bs, env.num_bands)
                for i in range(env.num_bs):
                    for j in range(env.num_bands):
                        if not action_mask[i, j, action_reshaped[i, j]]:
                            # Choose lowest valid power level
                            valid_actions = np.where(action_mask[i, j])[0]
                            if len(valid_actions) > 0:
                                action_reshaped[i, j] = valid_actions[0]
                            else:
                                action_reshaped[i, j] = 0
                
                action = action_reshaped.flatten()
                
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Total reward = {episode_reward:.2f}, Steps = {env.state.time}")
        
        print(f"\nAverage reward over {num_episodes} episodes: {np.mean(episode_rewards):.2f}")
        print(f"Std deviation: {np.std(episode_rewards):.2f}")
    
    # Run training
    train_random_policy(num_episodes=5)
    
    env.close()
