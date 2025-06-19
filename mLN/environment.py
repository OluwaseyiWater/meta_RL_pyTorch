import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import warnings

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
MIN_SINR = 0.0
NOISE_FIGURE_DB = 7.0
BANDWIDTH_HZ = 10e6
THERMAL_NOISE_DBM_HZ = -174.0

@dataclass
class SpectrumState:
    channel_gains: torch.Tensor
    interference_map: torch.Tensor
    qos_metrics: torch.Tensor
    spectrum_alloc: torch.Tensor
    tx_power: torch.Tensor
    time: int

    def to_numpy(self) -> np.ndarray:
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
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, num_bs: int = NUM_BS, num_users: int = NUM_USERS, num_bands: int = NUM_BANDS,
                 max_steps: int = MAX_STEPS, max_latency: float = MAX_LATENCY, max_power: float = MAX_POWER,
                 num_power_levels: int = NUM_POWER_LEVELS, power_levels: Optional[np.ndarray] = None,
                 fading_coherence: float = FADING_COHERENCE, max_interference: float = MAX_INTERFERENCE,
                 min_sinr: float = MIN_SINR, device: str = "cpu", render_mode: Optional[str] = None):

        super().__init__()

        self.num_bs = num_bs
        self.num_users = num_users
        self.num_bands = num_bands
        self.max_steps = max_steps
        self.max_latency = max_latency
        self.max_power = max_power
        self.num_power_levels = num_power_levels
        self.power_levels = torch.tensor(power_levels if power_levels is not None else POWER_LEVELS,
                                         dtype=torch.float32).to(device)
        self.fading_coherence = fading_coherence
        self.max_interference = max_interference
        self.min_sinr = min_sinr
        self.noise_figure_db = NOISE_FIGURE_DB
        self.bandwidth_hz = BANDWIDTH_HZ
        self.thermal_noise_dbm_hz = THERMAL_NOISE_DBM_HZ
        self.device = torch.device(device)
        self.render_mode = render_mode
        self.max_interference_linear = self.max_interference * 1000.0

        self.action_space = spaces.MultiDiscrete(
            [self.num_power_levels] * (self.num_bs * self.num_bands)
        )
        obs_dim = (
            self.num_users * self.num_bs +
            self.num_bs * self.num_bands +       # interference_map
            self.num_users * 2 +                  # qos_metrics
            self.num_bs * self.num_bands +       # spectrum_alloc
            self.num_bs * self.num_bands +       # tx_power
            1
        )
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.state: Optional[SpectrumState] = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        scenario = torch.randint(0, 3, (1,), device=self.device).item()
        distances = torch.rand(self.num_users, self.num_bs, device=self.device) * 1.9 + 0.1
        if scenario == 0:
            path_loss = 128.1 + 37.6 * torch.log10(distances)
        elif scenario == 1:
            path_loss = 98.5 + 23.1 * torch.log10(distances)
        else:
            path_loss = 105.3 + 34.2 * torch.log10(distances)
        path_loss = path_loss * 0.6

        shadow_fading = torch.randn(self.num_users, self.num_bs, device=self.device) * 8.0
        channel_gains = -(path_loss + shadow_fading)
        channel_gains = torch.clamp(channel_gains, min=-40.0)
        interference_mw = torch.rand(self.num_bs, self.num_bands, device=self.device) * (
            self.max_interference * 0.5 - 0.001) + 0.001
        interference_map = 10.0 * torch.log10(interference_mw + 1e-12)

        self.state = SpectrumState(
            channel_gains=channel_gains,
            interference_map=interference_map,
            qos_metrics=torch.zeros(self.num_users, 2, dtype=torch.float32, device=self.device),
            spectrum_alloc=torch.zeros(self.num_bs, self.num_bands, dtype=torch.int32, device=self.device),
            tx_power=torch.zeros(self.num_bs, self.num_bands, dtype=torch.float32, device=self.device),
            time=0
        )

        return self.state.to_numpy(), {"seed": seed}

    def _mask_unsafe_actions(self, state: SpectrumState) -> torch.Tensor:
        il = 10 ** (state.interference_map / 10.0)
        mask = il < self.max_interference_linear
        return mask.unsqueeze(-1).expand(-1, -1, self.num_power_levels)

    def _compute_sinr(self, state: SpectrumState) -> torch.Tensor:
        # Compute thermal noise in dBm and linear
        noise_dbm = (self.thermal_noise_dbm_hz + self.noise_figure_db +
                     10 * torch.log10(torch.tensor(self.bandwidth_hz, device=self.device)))
        noise_lin = 10 ** (noise_dbm / 10.0)

        # Received power per user, bs, band
        tx_p = state.tx_power.unsqueeze(0).expand(self.num_users, -1, -1)
        cg = state.channel_gains.unsqueeze(2).expand(-1, -1, self.num_bands)
        rcv = tx_p + cg

        # Serving BS (max received power) per user-band
        serving_bs = torch.argmax(rcv, dim=1)  # shape: (users, bands)
        one_hot = torch.nn.functional.one_hot(serving_bs, num_classes=self.num_bs) \
                             .permute(0, 2, 1).bool()

        # Signal power
        sig_dbm = torch.where(one_hot, rcv, torch.tensor(-float('inf'), device=self.device)) \
                        .max(dim=1)[0]
        sig_lin = 10 ** (sig_dbm / 10.0)

        # Interference from other BSs
        other_mask = ~one_hot
        int_bs = torch.where(other_mask, 10 ** (rcv / 10.0),
                             torch.tensor(0.0, device=self.device)).sum(dim=1)

        # External interference gathered per serving BS
        im_lin = 10 ** (state.interference_map / 10.0)        # shape: (bs, bands)
        im_exp = im_lin.unsqueeze(0).expand(self.num_users, -1, -1)
        ext_int = torch.gather(im_exp, 1, serving_bs.unsqueeze(-1)).squeeze(-1)

        total_int = int_bs + ext_int + noise_lin
        sinr_lin = sig_lin / (total_int + 1e-12)

        return 10 * torch.log10(sinr_lin + 1e-12)
        
    def _calculate_reward(self, state: SpectrumState, action: torch.Tensor, previous_tx_power: torch.Tensor) -> float:
        POWER_COST_COEFF = 0.05
        SWITCHING_COST_COEFF = 0.0
        UTILIZATION_BONUS_COEFF = 1.0
        VIOLATION_PENALTY_COEFF = 0.1
        FAIRNESS_COEFF = 1.0
        THROUGHPUT_COEFF = 20.0

        sinr_db = self._compute_sinr(state)
        sinr_linear = 10 ** (sinr_db / 10.0)
        best_sinr_per_user = torch.max(sinr_linear, dim=1)[0]
        spectral_efficiency = torch.log2(1 + best_sinr_per_user + 1e-12)
        throughput = torch.sum(spectral_efficiency) / self.num_users

        throughput_per_user = spectral_efficiency + 1e-12
        fairness = (torch.sum(throughput_per_user) ** 2) / (self.num_users * torch.sum(throughput_per_user ** 2))

        sinr_violations = torch.sum(torch.max(sinr_db, dim=1)[0] < self.min_sinr)
        latency_violations = torch.sum(state.qos_metrics[:, 0] > self.max_latency)
        violation_penalty = VIOLATION_PENALTY_COEFF * (latency_violations + sinr_violations)

        new_tx_power = self.power_levels[action]
        total_power_linear = torch.sum(10 ** (new_tx_power / 10.0))
        power_cost = POWER_COST_COEFF * torch.log10(total_power_linear + 1e-12)

        power_changes = torch.abs(previous_tx_power - new_tx_power)
        switching_cost = SWITCHING_COST_COEFF * torch.sum(power_changes) / (self.num_bs * self.num_bands)

        active_channels = torch.sum(action > 0)
        utilization_bonus = UTILIZATION_BONUS_COEFF * (active_channels / (self.num_bs * self.num_bands))

        total_reward = (
            THROUGHPUT_COEFF * throughput +
            FAIRNESS_COEFF * fairness +
            utilization_bonus -
            violation_penalty -
            power_cost -
            switching_cost
        )
        return {
            "throughput_bonus": THROUGHPUT_COEFF * throughput,
            "fairness_bonus": FAIRNESS_COEFF * fairness,
            "utilization_bonus": utilization_bonus,
            "violation_penalty": violation_penalty,
            "power_cost": power_cost,
            "switching_cost": switching_cost,
        }

    def _adaptive_penalty(self, state: SpectrumState) -> float:
        interference_linear = 10 ** (state.interference_map / 10.0)
        high_threshold = 0.8 * self.max_interference_linear
        high_interference_penalty = 0.1 * torch.sum(interference_linear > high_threshold)

        sinr_db = self._compute_sinr(state)
        poor_coverage_penalty = 0.05 * torch.sum(torch.max(sinr_db, dim=1)[0] < (self.min_sinr - 5.0))

        return (high_interference_penalty + poor_coverage_penalty).item()

    def _step_dynamics(self, state: SpectrumState, action: torch.Tensor) -> SpectrumState:
        fading_noise = torch.randn_like(state.channel_gains) * 2.0
        correlation_factor = np.sqrt(1 - self.fading_coherence ** 2)
        new_channel = self.fading_coherence * state.channel_gains + correlation_factor * fading_noise

        new_alloc = action.int()
        new_tx_power = self.power_levels[action]

        sinr_db = self._compute_sinr(state)
        user_best_sinr = torch.max(sinr_db, dim=1)[0]

        latency_penalty = torch.where(
            user_best_sinr < self.min_sinr,
            2.0 * torch.exp(-(user_best_sinr - self.min_sinr) / 5.0),
            torch.tensor(0.1, device=self.device)
        )
        new_latency = state.qos_metrics[:, 0] + latency_penalty

        sinr_linear = 10 ** (user_best_sinr / 10.0)
        spectral_efficiency = torch.log2(1 + sinr_linear + 1e-12)
        normalized_throughput = spectral_efficiency / 10.0

        new_qos = torch.stack([new_latency, normalized_throughput], dim=1)

        interference_noise = torch.randn_like(state.interference_map) * 0.5
        new_interference_map = torch.clamp(state.interference_map + interference_noise, -20.0, 30.0)

        return SpectrumState(
            channel_gains=new_channel,
            interference_map=new_interference_map,
            qos_metrics=new_qos,
            spectrum_alloc=new_alloc,
            tx_power=new_tx_power,
            time=state.time + 1
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.state is None:
            raise RuntimeError("Must call reset() before step()")

        action_tensor = torch.tensor(action, dtype=torch.long, device=self.device)
        action_reshaped = action_tensor.reshape(self.num_bs, self.num_bands)

        mask = self._mask_unsafe_actions(self.state)
        idx = torch.arange(self.num_bs)[:, None], torch.arange(self.num_bands)[None, :], action_reshaped
        safe = mask[idx]
        safe_action = torch.where(safe, action_reshaped, torch.tensor(0, device=self.device))

        previous_tx_power = self.state.tx_power.clone()
        self.state = self._step_dynamics(self.state, safe_action)

        reward_terms = self._calculate_reward(self.state, safe_action, previous_tx_power)
        scalar_reward = (
            reward_terms["throughput_bonus"]
            + reward_terms["fairness_bonus"]
            + reward_terms["utilization_bonus"]
            - reward_terms["violation_penalty"]
            - reward_terms["power_cost"]
            - reward_terms["switching_cost"]
        )

        terminated = torch.any(self.state.qos_metrics[:, 0] > 2 * self.max_latency).item()
        sinr_db = self._compute_sinr(self.state)
        severe_violations = torch.sum(torch.max(sinr_db, dim=1)[0] < (self.min_sinr - 10.0)) > (0.8 * self.num_users)
        terminated = terminated or severe_violations.item()
        truncated = self.state.time >= self.max_steps

        info: Dict[str, Any] = {
            "latency_violations": torch.sum(self.state.qos_metrics[:, 0] > self.max_latency).item(),
            "sinr_violations": torch.sum(torch.max(sinr_db, dim=1)[0] < self.min_sinr).item(),
            "qos_violations": torch.sum(self.state.qos_metrics[:, 0] > self.max_latency).item(),
            "average_sinr": torch.mean(torch.max(sinr_db, dim=1)[0]).item(),
            "average_throughput": torch.mean(self.state.qos_metrics[:, 1]).item(),
            "action_mask": mask.cpu().numpy()
        }
        info.update(reward_terms)

        return self.state.to_numpy(), scalar_reward.item(), terminated, truncated, info

       


    def render(self) -> Optional[str]:
        if self.state is None:
            return None

        sinr_db = self._compute_sinr(self.state)

        output = f"Step {self.state.time}\n"
        output += "Spectrum Allocation:\n" + str(self.state.spectrum_alloc.cpu().numpy()) + "\n"
        output += "Transmit Power (dBm):\n" + str(self.state.tx_power.cpu().numpy()) + "\n"
        output += f"Latency violations: {torch.sum(self.state.qos_metrics[:, 0] > self.max_latency).item()}\n"
        output += f"SINR violations: {torch.sum(torch.max(sinr_db, dim=1)[0] < self.min_sinr).item()}\n"
        output += f"Average user SINR: {torch.mean(torch.max(sinr_db, dim=1)[0]).item():.2f} dB\n"
        output += f"Average throughput: {torch.mean(self.state.qos_metrics[:, 1]).item():.3f}\n"

        if self.render_mode == "human":
            print(output)

        return output

    def get_action_mask(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Must call reset() before getting action mask")
        return self._mask_unsafe_actions(self.state).cpu().numpy()
if __name__ == "__main__":
    # Create environment
    env = DynamicSpectrumEnv(device="cuda" if torch.cuda.is_available() else "cpu", render_mode="human")
    
    # Test basic functionality
    obs, info = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Test step
    action = np.full(env.num_bs * env.num_bands, env.num_power_levels-1, dtype=int)
    obs, info = env.reset(seed=42)
    obs, reward, term, trunc, info = env.step(action)
    print("SINR per user-band (dB):\n", env._compute_sinr(env.state).cpu().numpy())
    print("Reward components:", reward, info)

    
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
