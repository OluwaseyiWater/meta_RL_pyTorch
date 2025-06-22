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
    spectrum_alloc: torch.Tensor # Stores power level INDICES
    tx_power: torch.Tensor      # Stores actual power values in dBm
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
        spectrum_alloc = torch.tensor(obs[idx:idx + num_bs * num_bands], dtype=torch.int32).reshape(num_bs, num_bands).to(device) 
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
        
        if power_levels is not None:
            self.power_levels = torch.tensor(power_levels, dtype=torch.float32).to(device)
        else:
            self.power_levels = torch.tensor(POWER_LEVELS, dtype=torch.float32).to(device)

        self.fading_coherence = fading_coherence
        self.max_interference = max_interference 
        self.min_sinr = min_sinr 
        self.noise_figure_db = NOISE_FIGURE_DB
        self.bandwidth_hz = BANDWIDTH_HZ
        self.thermal_noise_dbm_hz = THERMAL_NOISE_DBM_HZ
        self.device = torch.device(device)
        self.render_mode = render_mode
        
        self.max_interference_linear = self.max_interference 

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
        self._last_reward_terms: Dict[str, Any] = {} 

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        scenario = torch.randint(0, 3, (1,), device=self.device).item()
        distances = torch.rand(self.num_users, self.num_bs, device=self.device) * 0.48 + 0.02 
        
        if scenario == 0:
            path_loss = 128.1 + 37.6 * torch.log10(distances)
        elif scenario == 1:
            path_loss = 135.41 + 37.6 * torch.log10(distances) 
        else:
            path_loss = 105.3 + 34.2 * torch.log10(distances)
        
        path_loss = path_loss * 0.35 

        shadow_fading = torch.randn(self.num_users, self.num_bs, device=self.device) * 8.0
        channel_gains = -(path_loss + shadow_fading)
        channel_gains = torch.clamp(channel_gains, min=-150.0, max=-40.0)

        interference_mw = torch.rand(self.num_bs, self.num_bands, device=self.device) * (
            self.max_interference * 0.005) + 0.00001 
        interference_map = 10.0 * torch.log10(interference_mw + 1e-12)

        initial_spectrum_alloc = torch.zeros(self.num_bs, self.num_bands, dtype=torch.int32, device=self.device)
        initial_tx_power = self.power_levels[initial_spectrum_alloc] 

        self.state = SpectrumState(
            channel_gains=channel_gains,
            interference_map=interference_map, 
            qos_metrics=torch.zeros(self.num_users, 2, dtype=torch.float32, device=self.device),
            spectrum_alloc=initial_spectrum_alloc, 
            tx_power=initial_tx_power,             
            time=0
        )
        self._last_reward_terms = {}
        return self.state.to_numpy(), {"seed": seed}

    def _mask_unsafe_actions(self, state: SpectrumState) -> torch.Tensor:
        il = 10 ** (state.interference_map / 10.0) 
        mask_per_channel = il < self.max_interference_linear 
        final_mask = mask_per_channel.unsqueeze(-1).expand(-1, -1, self.num_power_levels)
        return final_mask

    def _compute_sinr(self, state: SpectrumState) -> torch.Tensor: 
        noise_dbm = (self.thermal_noise_dbm_hz + self.noise_figure_db +
                     10 * torch.log10(torch.tensor(self.bandwidth_hz, device=self.device)))
        noise_lin = 10 ** (noise_dbm / 10.0) 

        tx_p = state.tx_power.unsqueeze(0).expand(self.num_users, -1, -1)      
        cg = state.channel_gains.unsqueeze(2).expand(-1, -1, self.num_bands) 
        rcv = tx_p + cg 

        best_rcv_dbm_from_serving_bs, serving_bs = torch.max(rcv, dim=1) 
        sig_lin = 10 ** (best_rcv_dbm_from_serving_bs / 10.0) 

        one_hot_for_interf = torch.nn.functional.one_hot(serving_bs, num_classes=self.num_bs) 
        one_hot_for_interf = one_hot_for_interf.permute(0, 2, 1).bool() 

        all_rcv_lin = 10**(rcv / 10.0) 
        
        interfering_bs_power_lin = torch.where(~one_hot_for_interf, all_rcv_lin, torch.tensor(0.0, device=self.device))
        int_bs = interfering_bs_power_lin.sum(dim=1) 

        im_lin = 10 ** (state.interference_map / 10.0)        
        band_indices = torch.arange(self.num_bands, device=self.device).unsqueeze(0).expand(self.num_users, -1) 
        ext_int = im_lin[serving_bs, band_indices] 

        total_int = int_bs + ext_int + noise_lin 
        sinr_lin = sig_lin / (total_int + 1e-12)

        return 10 * torch.log10(sinr_lin + 1e-12) 
        
    def _calculate_reward(self, state: SpectrumState, action: torch.Tensor, previous_tx_power: torch.Tensor) -> float:
        
        if torch.any(state.tx_power > 30.0) or torch.any(state.tx_power < -100.0): 
            print(f"!!! ALERT: Suspicious state.tx_power values in _calculate_reward: min={state.tx_power.min().item()}, max={state.tx_power.max().item()}")
            print(f"self.power_levels: {self.power_levels.tolist()}")
            print(f"Action indices passed to _calc_reward (shape {action.shape}): {action.tolist()}")
        
        
        POWER_COST_COEFF         = 0.05
        SWITCHING_COST_COEFF     = 0.001 
        UTILIZATION_BONUS_COEFF  = 1.0
        VIOLATION_PENALTY_COEFF  = 10.0
        FAIRNESS_COEFF           = 1.0
        THROUGHPUT_COEFF         = 20.0 

        sinr_db = self._compute_sinr(state) 
        
        best_sinr_per_user_db, _ = torch.max(sinr_db, dim=1) 
        sinr_linear = 10 ** (best_sinr_per_user_db / 10.0) 

        spectral_efficiency = torch.log2(1 + sinr_linear + 1e-12) 
        throughput = torch.sum(spectral_efficiency) / self.num_users 

        throughput_per_user = spectral_efficiency + 1e-12 
        fairness = (torch.sum(throughput_per_user) ** 2) / (self.num_users * torch.sum(throughput_per_user ** 2) + 1e-12)

        sinr_violations = torch.sum(best_sinr_per_user_db < self.min_sinr) 
        latency_violations = torch.sum(state.qos_metrics[:, 0] > self.max_latency) 

        safe_step = (sinr_violations == 0) and (latency_violations == 0)
        
        throughput_bonus   = THROUGHPUT_COEFF * throughput    if safe_step else 0.0
        fairness_bonus     = FAIRNESS_COEFF  * fairness      if safe_step else 0.0

        all_tx_power_dbm = state.tx_power 
        total_power_linear = torch.tensor(0.0, device=self.device) 
        if all_tx_power_dbm.numel() > 0:
            total_power_linear = torch.sum(10 ** (all_tx_power_dbm / 10.0))
            power_cost = POWER_COST_COEFF * torch.log10(total_power_linear + 1e-12)
        else: 
            power_cost = torch.tensor(0.0, device=self.device)

        calculated_power_cost = power_cost.item() 
        if calculated_power_cost > 1.0: 
            print(f"--- High Power Cost Detected ---")
            print(f"  Action indices (safe_action): {action.tolist()}")
            print(f"  state.tx_power (dBm values from state): {np.round(state.tx_power.cpu().numpy(), 2).tolist()}")
            if all_tx_power_dbm.numel() > 0: 
                print(f"  total_power_linear (mW): {total_power_linear.item():.4e}")
            else:
                print(f"  total_power_linear (mW): N/A (no tx_power elements)")
            print(f"  Calculated power_cost: {calculated_power_cost:.4f}")
            print(f"--------------------------------")

        power_changes = torch.abs(previous_tx_power - state.tx_power) 
        switching_cost = SWITCHING_COST_COEFF * torch.sum(power_changes) / (self.num_bs * self.num_bands)

        active_channels = torch.sum(action > 0) 
        utilization_bonus = UTILIZATION_BONUS_COEFF * (active_channels / (self.num_bs * self.num_bands)) if safe_step else 0.0

        violation_penalty = VIOLATION_PENALTY_COEFF * (latency_violations + sinr_violations)

        total_reward_val = ( 
            throughput_bonus +
            fairness_bonus +
            utilization_bonus -
            violation_penalty -
            power_cost -
            switching_cost
        )
        
        self._last_reward_terms = {
            "throughput_bonus": throughput_bonus.item() if torch.is_tensor(throughput_bonus) else float(throughput_bonus),
            "fairness_bonus": fairness_bonus.item() if torch.is_tensor(fairness_bonus) else float(fairness.item()),
            "utilization_bonus": utilization_bonus.item() if torch.is_tensor(utilization_bonus) else float(utilization_bonus),
            "violation_penalty": violation_penalty.item(),
            "power_cost": power_cost.item(),
            "switching_cost": switching_cost.item(),
            "info_sinr_violations_count": sinr_violations.item(),
            "info_latency_violations_count": latency_violations.item(),
            "info_avg_throughput_raw": throughput.item(), 
            "info_fairness_raw": fairness.item(),
            "info_avg_best_sinr_db_users": torch.mean(best_sinr_per_user_db).item() if best_sinr_per_user_db.numel() > 0 else -torch.inf,
            "info_active_channels_count": active_channels.item(),
            "info_total_power_mw_utilized": total_power_linear.item() if all_tx_power_dbm.numel() > 0 else 0.0,
            "scalar_reward_calculated": total_reward_val.item() 
        }
        return total_reward_val.item() 


    def _adaptive_penalty(self, state: SpectrumState) -> float:
        interference_linear = 10 ** (state.interference_map / 10.0) 
        high_threshold = 0.8 * self.max_interference_linear 
        high_interference_penalty = 0.1 * torch.sum(interference_linear > high_threshold)

        sinr_db_calc = self._compute_sinr(state) 
        poor_coverage_penalty = 0.05 * torch.sum(torch.max(sinr_db_calc, dim=1)[0] < (self.min_sinr - 5.0))

        return (high_interference_penalty + poor_coverage_penalty).item()

    def _step_dynamics(self, state: SpectrumState, action: torch.Tensor) -> SpectrumState:
        fading_noise = torch.randn_like(state.channel_gains) * 2.0 
        correlation_factor = np.sqrt(1 - self.fading_coherence ** 2)
        new_channel = self.fading_coherence * state.channel_gains + correlation_factor * fading_noise
        new_channel = torch.clamp(new_channel, min=-150.0, max=-40.0)

        new_alloc = action.int() 
        new_tx_power = self.power_levels[new_alloc] 

        temp_state_for_sinr = SpectrumState( 
            channel_gains=new_channel, 
            interference_map=state.interference_map, 
            qos_metrics=state.qos_metrics, 
            spectrum_alloc=new_alloc, 
            tx_power=new_tx_power, 
            time=state.time 
        )
        sinr_db_for_qos = self._compute_sinr(temp_state_for_sinr) 
        user_best_sinr = torch.max(sinr_db_for_qos, dim=1)[0] 

        latency_penalty = torch.where(
            user_best_sinr < self.min_sinr,
            2.0 * torch.exp(-(user_best_sinr - self.min_sinr) / 5.0), 
            torch.tensor(0.1, device=self.device)
        )
        latency_penalty = torch.clamp(latency_penalty, max=10.0) 
        new_latency = state.qos_metrics[:, 0] + latency_penalty
        new_latency = torch.clamp(new_latency, 0, 2.5 * self.max_latency)

        sinr_linear_for_se = 10 ** (user_best_sinr / 10.0) 
        spectral_efficiency_for_se = torch.log2(1 + sinr_linear_for_se + 1e-12) 
        normalized_throughput = spectral_efficiency_for_se / 10.0 

        new_qos = torch.stack([new_latency, normalized_throughput], dim=1)

        interference_noise = torch.randn_like(state.interference_map) * 0.5
        new_interference_map = torch.clamp(state.interference_map + interference_noise, -30.0, 15.0) 

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
        idx = torch.arange(self.num_bs, device=self.device)[:, None], \
              torch.arange(self.num_bands, device=self.device)[None, :], \
              action_reshaped 
        
        safe_per_channel = mask[idx] 
        safe_action = torch.where(safe_per_channel, action_reshaped, torch.tensor(0, dtype=torch.long, device=self.device))

        previous_tx_power = self.state.tx_power.clone() 
        self.state = self._step_dynamics(self.state, safe_action) 

        scalar_reward = self._calculate_reward(self.state, safe_action, previous_tx_power)

        sinr_db_term = self._compute_sinr(self.state) 
        terminated_qos = torch.any(self.state.qos_metrics[:, 0] > 2 * self.max_latency).item() 
        severe_violations = (torch.sum(torch.max(sinr_db_term, dim=1)[0] < (self.min_sinr - 15.0)) > (0.95 * self.num_users))
        terminated = terminated_qos or severe_violations.item() 
        truncated = self.state.time >= self.max_steps

        info = {k: v for k, v in self._last_reward_terms.items()} 

        return self.state.to_numpy(), float(scalar_reward), bool(terminated), bool(truncated), info 

    def step_with_terms(self, action: np.ndarray): 
        obs, scalar_reward, term, trunc, _info = self.step(action) 
        terms = {k:v for k,v in self._last_reward_terms.items()} 
        return obs, scalar_reward, term, trunc, terms

    def render(self) -> Optional[str]:
        if self.state is None:
            return None

        sinr_db_render = self._compute_sinr(self.state) 

        output = f"Step {self.state.time}\n"
        output += "Spectrum Allocation:\n" + str(self.state.spectrum_alloc.cpu().numpy()) + "\n"
        output += "Transmit Power (dBm):\n" + str(np.round(self.state.tx_power.cpu().numpy(),1)) + "\n" 
        
        if self._last_reward_terms and "info_sinr_violations_count" in self._last_reward_terms:
            output += f"Latency violations: {self._last_reward_terms.get('info_latency_violations_count', 'N/A')}\n"
            output += f"SINR violations: {self._last_reward_terms.get('info_sinr_violations_count', 'N/A')}\n"
            avg_sinr_val = self._last_reward_terms.get('info_avg_best_sinr_db_users', float('-inf'))
            output += f"Average user SINR: {avg_sinr_val:.2f} dB\n"
            avg_tp_val = self._last_reward_terms.get('info_avg_throughput_raw', 0.0) 
            output += f"Average throughput: {avg_tp_val:.3f}\n" 
        else: 
            output += f"Latency violations: {torch.sum(self.state.qos_metrics[:, 0] > self.max_latency).item()}\n"
            output += f"SINR violations: {torch.sum(torch.max(sinr_db_render, dim=1)[0] < self.min_sinr).item()}\n"
            output += f"Average user SINR: {torch.mean(torch.max(sinr_db_render, dim=1)[0]).item():.2f} dB\n"
            output += f"Average throughput: {torch.mean(self.state.qos_metrics[:, 1] * 10.0).item():.3f}\n" 

        if self.render_mode == "human":
            print(output)
        return output

    def get_action_mask(self) -> np.ndarray: 
        if self.state is None:
            raise RuntimeError("Must call reset() before getting action mask")
        return self._mask_unsafe_actions(self.state).cpu().numpy()

if __name__ == "__main__":
    env = DynamicSpectrumEnv(device="cuda" if torch.cuda.is_available() else "cpu", render_mode="human", min_sinr = -5.0)
    
    obs, info_reset = env.reset(seed=42) 
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Internal self.power_levels (dBm): {env.power_levels.cpu().numpy()}") 
    env.render()
    
    action_main = np.full(env.num_bs * env.num_bands, env.num_power_levels-1, dtype=int) 
    print(f"\n--- Taking Action: All Max Power (Index {env.num_power_levels-1}) ---")
    obs, reward_val, term_val, trunc_val, info_dict = env.step(action_main) 
    print("\n--- After Step 1 ---")
    env.render()
    print(f"Returned Reward: {reward_val:.3f}") 
    print(f"Returned Info Dict from step: { {k: (round(v,2) if isinstance(v,float) else v) for k,v in info_dict.items()} }") 

    def train_random_policy(num_episodes=5):
        print(f"\n--- Training with Random (Masked) Policy for {num_episodes} episodes ---")
        episode_rewards = []
        
        for episode in range(num_episodes):
            obs_train, info_train_reset = env.reset() 
            episode_reward = 0
            done_train = False 
            
            steps_in_ep_train = 0 
            while not done_train:
                action_mask_train = env.get_action_mask() 
                
                action_reshaped_train = np.zeros((env.num_bs, env.num_bands), dtype=int) 
                for i in range(env.num_bs):
                    for j in range(env.num_bands):
                        valid_actions_for_channel = np.where(action_mask_train[i, j])[0]
                        if len(valid_actions_for_channel) > 0:
                            action_reshaped_train[i, j] = np.random.choice(valid_actions_for_channel)
                        else: 
                            action_reshaped_train[i, j] = 0 
                
                action_train_flat = action_reshaped_train.flatten() 
                
                obs_train, reward_train, terminated_train, truncated_train, info_train_step = env.step(action_train_flat) 
                episode_reward += reward_train
                done_train = terminated_train or truncated_train
                steps_in_ep_train +=1
                if steps_in_ep_train >= env.max_steps : break 
            
            episode_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Total reward = {episode_reward:.2f}, Steps = {env.state.time if env.state else steps_in_ep_train}")
        
        print(f"\nAverage reward over {num_episodes} episodes: {np.mean(episode_rewards):.2f}")
        print(f"Std deviation: {np.std(episode_rewards):.2f}")
    
    train_random_policy(num_episodes=3)
    env.close()
