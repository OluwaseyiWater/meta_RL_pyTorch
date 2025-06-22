import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.next_state = None 

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
        self.next_state = None

class PPOAgent(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        num_bs: int,
        num_bands: int,
        num_power_levels: int, 
        lr_actor: float,
        lr_critic: float,
        gamma: float,
        K_epochs: int,
        eps_clip: float,
        value_coeff: float,
        entropy_coeff: float,
        hidden_dim: int = 64, 
        num_blocks: int = 2   
    ):
        super().__init__()
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.num_channels = num_bs * num_bands
        self.num_levels = num_power_levels 

        # Actor network
        layers = [nn.Linear(obs_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_blocks):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, self.num_channels * self.num_levels)) 
        self.actor = nn.Sequential(*layers)

        # Critic network
        vlayers = [nn.Linear(obs_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_blocks):
            vlayers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        vlayers.append(nn.Linear(hidden_dim, 1))
        self.critic = nn.Sequential(*vlayers)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.env = None 

    def select_action(self, state: np.ndarray, buffer: RolloutBuffer, device: torch.device):
        if self.env is None: 
            raise RuntimeError("PPOAgent.env is not set by the trainer.")

        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        
        logits_flat = self.actor(state_t) 
        logits = logits_flat.view(1, self.num_channels, self.num_levels) 
        mask_numpy = self.env.get_action_mask() 
        mask_t = torch.tensor(mask_numpy, dtype=torch.bool, device=device).view(1, self.num_channels, self.num_levels)
        masked_logits = logits.masked_fill(~mask_t, -1e9) 
        dist = Categorical(logits=masked_logits.squeeze(0))
        action = dist.sample()                       
        logprob = dist.log_prob(action).sum()        
        if buffer is not None:
            buffer.states.append(state_t)
            buffer.actions.append(action) 
            buffer.logprobs.append(logprob)
            
        return action.cpu().numpy().reshape(-1) 

    def compute_returns_and_advantages(self, buffer: RolloutBuffer, device: torch.device):
        
        next_value_bootstrap = 0.0 
        if buffer.next_state is not None and not buffer.dones[-1]: 
            with torch.no_grad():
                next_state_t = torch.tensor(buffer.next_state, dtype=torch.float32, device=device).unsqueeze(0)
                next_value_bootstrap = self.critic(next_state_t).item()
        
        returns = []
        G = next_value_bootstrap 
        for r_step, done_step in zip(reversed(buffer.rewards), reversed(buffer.dones)): 
            if done_step: 
                G = 0.0
            G = r_step + self.gamma * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

        states_buffer = torch.cat(buffer.states, dim=0)  
        with torch.no_grad():
            values_buffer = self.critic(states_buffer).squeeze(-1)   
        advantages = returns_t - values_buffer   
        
        # Normalize advantages 
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns_t, advantages

    def update(self, buffer: RolloutBuffer, device: torch.device):
        returns_t, advantages = self.compute_returns_and_advantages(buffer, device)  
        old_states_buffer = torch.cat(buffer.states, dim=0).detach()       
        old_actions_buffer = torch.stack(buffer.actions).detach()           
        old_logprobs_buffer = torch.stack(buffer.logprobs).detach()         

        for _ in range(self.K_epochs):
            logits_update = self.actor(old_states_buffer)                       
            logits_update = logits_update.view(-1, self.num_channels, self.num_levels)  
            dist_update = Categorical(logits=logits_update)                 
            new_logprobs_update = dist_update.log_prob(old_actions_buffer).sum(dim=1)  
            entropy_update = dist_update.entropy().sum(dim=1)                     

            ratios_update = torch.exp(new_logprobs_update - old_logprobs_buffer)  
            surr1 = ratios_update * advantages
            surr2 = torch.clamp(ratios_update, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            
            value_preds_update = self.critic(old_states_buffer).squeeze(-1)               
            value_loss = ((returns_t - value_preds_update) ** 2).mean() # MSE Loss

           
            loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy_update.mean()

            # gradient step
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        buffer.clear()

    def evaluate_policy(self, env, episodes: int, device: torch.device):
        all_ep_rewards = []
        all_ep_sinr_violations = [] 
        all_ep_qos_violations = []  
        all_ep_avg_throughput_raw = []
        all_ep_avg_best_sinr_db = []


        for _ in range(episodes):
            state_eval, _ = env.reset() 
            done_eval = False 
            ep_r = 0.0 
            ep_s_viol_sum = 0.0 
            ep_q_viol_sum = 0.0 
            ep_sum_throughput_raw = 0.0
            ep_sum_best_sinr_db = 0.0
            num_steps_this_episode = 0

            
            while not done_eval:
                action_eval = self.select_action(state_eval, None, device) 
                
               
                next_state_eval, r_eval, terminated_eval, truncated_eval, info_eval = env.step(action_eval) 
                done_eval = terminated_eval or truncated_eval
                
                ep_r += r_eval
                ep_s_viol_sum += info_eval.get('info_sinr_violations_count', 0)
                ep_q_viol_sum += info_eval.get('info_latency_violations_count', 0)
                ep_sum_throughput_raw += info_eval.get('info_avg_throughput_raw', 0)
                ep_sum_best_sinr_db += info_eval.get('info_avg_best_sinr_db_users', 0)

                state_eval = next_state_eval
                num_steps_this_episode += 1
                if num_steps_this_episode >= env.max_steps: # Safety break
                    break
            
            all_ep_rewards.append(ep_r)
            all_ep_sinr_violations.append(ep_s_viol_sum / num_steps_this_episode if num_steps_this_episode > 0 else 0) # Avg per step for this ep
            all_ep_qos_violations.append(ep_q_viol_sum / num_steps_this_episode if num_steps_this_episode > 0 else 0)   # Avg per step for this ep
            all_ep_avg_throughput_raw.append(ep_sum_throughput_raw / num_steps_this_episode if num_steps_this_episode > 0 else 0)
            all_ep_avg_best_sinr_db.append(ep_sum_best_sinr_db / num_steps_this_episode if num_steps_this_episode > 0 else 0)

        # Return dictionary with original keys
        return {
            'avg_reward': float(np.mean(all_ep_rewards)),
            'avg_sinr_violation': float(np.mean(all_ep_sinr_violations)), # This is avg per-step violations
            'avg_qos_violation': float(np.mean(all_ep_qos_violations)),   # This is  avg per-step violations
            'details_avg_throughput_raw': float(np.mean(all_ep_avg_throughput_raw)),
            'details_avg_best_sinr_db': float(np.mean(all_ep_avg_best_sinr_db)),
        }
