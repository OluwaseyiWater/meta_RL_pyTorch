import hydra
from omegaconf import DictConfig, OmegaConf
import pickle
import torch
import numpy as np
import wandb
from hydra.utils import instantiate
import os

from mLN.environment import DynamicSpectrumEnv
from models.ppo_agent import PPOAgent, RolloutBuffer

class PPOTrainer:
    def __init__(
        self,
        env,
        ppo: DictConfig, 
        seed: int,
        device: str = None,
        hydra_cfg: DictConfig = None 
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.use_wandb = ppo.get('use_wandb', False) 
        self.wandb_project_from_cfg = ppo.get('wandb_project', "ppo-spectrum-default") 
        self.wandb_name_from_cfg = ppo.get('wandb_name', None) 
        self.ppo_cfg_for_wandb_log = OmegaConf.to_container(ppo, resolve=True) 

        self.env = env
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Using device: {self.device}")


        obs_dim = self.env.observation_space.shape[0]
        self.agent = PPOAgent(
            obs_dim,
            self.env.num_bs,
            self.env.num_bands,
            self.env.num_power_levels,
            lr_actor=ppo.lr_actor,
            lr_critic=ppo.lr_critic,
            gamma=ppo.gamma,
            K_epochs=ppo.ppo_epochs, 
            eps_clip=ppo.clip_ratio,
            value_coeff=ppo.value_coeff,
            entropy_coeff=ppo.entropy_coeff,
            hidden_dim=ppo.get('hidden_dim', 64), 
            num_blocks=ppo.get('num_blocks', 2)
        ).to(self.device)

        self.agent.env = self.env 

        self.ppo_cfg = ppo 
        self.hydra_cfg_internal = hydra_cfg 


    def train(self):
        if self.use_wandb:
            try: 
                wandb.init(
                    project=self.wandb_project_from_cfg, 
                    name=self.wandb_name_from_cfg,       
                    config=self.ppo_cfg_for_wandb_log    
                )
                wandb.watch(self.agent, log_freq=1000, log='all')
            except Exception as e_wandb:
                print(f"Wandb init failed: {e_wandb}. Continuing without wandb.")
                self.use_wandb = False


        buffer = RolloutBuffer()
        history = {} 
        total_steps = 0 
        ep_num = 0
        ep_rew_sum = 0
        ep_len = 0

        print("Performing initial evaluation...")
        metrics_init = self.agent.evaluate_policy(self.env, self.ppo_cfg.eval_episodes, self.device) 
        print(f"[Initial Eval] Avg Reward: {metrics_init['avg_reward']:.3f}")
        if self.use_wandb:
             wandb.log({f"eval/{k.replace('_', '/')}": v for k, v in metrics_init.items()}, step=total_steps)


        state, _ = self.env.reset()

        print(f"Starting training for {self.ppo_cfg.total_steps} total environment steps...")
        while total_steps < self.ppo_cfg.total_steps:
            for _i_rollout_step in range(self.ppo_cfg.rollout_len): 
                action_train = self.agent.select_action(state, buffer, self.device) 
                next_state, reward_train, terminated, truncated, info_train = self.env.step(action_train) 
                done_train = terminated or truncated 

                buffer.rewards.append(reward_train)
                buffer.dones.append(done_train)
                state = next_state
                ep_rew_sum += reward_train
                ep_len += 1
                total_steps += 1

                if done_train:
                    ep_num +=1
                    if self.use_wandb:
                        log_train_ep = { 
                            "train_ep/reward": ep_rew_sum,
                            "train_ep/length": ep_len,
                            "train_ep_end_info/sinr_violations": info_train.get('info_sinr_violations_count',0), 
                            "train_ep_end_info/latency_violations": info_train.get('info_latency_violations_count',0),
                        }
                        wandb.log(log_train_ep, step=total_steps)
                    state, _ = self.env.reset()
                    ep_rew_sum = 0
                    ep_len = 0
                
                if total_steps >= self.ppo_cfg.total_steps:
                    break 
            
            
            if not done_train: 
                 buffer.next_state = state 
            else:
                 buffer.next_state = None 

            self.agent.update(buffer, self.device) 

            if total_steps > 0 and (total_steps // self.ppo_cfg.eval_interval > (history.get('last_eval_at_step', [-1])[-1] // self.ppo_cfg.eval_interval if history.get('last_eval_at_step') else -1)):
                print(f"\nEvaluating at step {total_steps}...")
                eval_metrics = self.agent.evaluate_policy(self.env, self.ppo_cfg.eval_episodes, self.device)
                
                for k_metric, v_metric in eval_metrics.items(): 
                    history.setdefault(k_metric, []).append(v_metric)
                history.setdefault('step', []).append(total_steps) 
                history.setdefault('last_eval_at_step', []).append(total_steps) 


                if self.use_wandb: 
                    wandb.log({f"eval/{k.replace('_', '/')}": v for k, v in eval_metrics.items()}, step=total_steps) 
                
                
                print(
                    f"[Step {total_steps}/{self.ppo_cfg.total_steps}] R={eval_metrics['avg_reward']:.3f} | "
                    f"SINR={eval_metrics['avg_sinr_violation']:.1f} | " 
                    f"QoS={eval_metrics['avg_qos_violation']:.1f}"      
                )


        
        final_save_dir = "." 
        if self.hydra_cfg_internal and hasattr(self.hydra_cfg_internal, 'hydra') and \
           hasattr(self.hydra_cfg_internal.hydra, 'run') and hasattr(self.hydra_cfg_internal.hydra.run, 'dir'):
             final_save_dir = self.hydra_cfg_internal.hydra.run.dir
        
        history_path_final = os.path.join(final_save_dir, 'ppo_history.pkl') 
        try:
            os.makedirs(final_save_dir, exist_ok=True)
            with open(history_path_final, 'wb') as f_hist: 
                pickle.dump(history, f_hist)
            print(f"Training complete. History saved to {history_path_final}")
        except Exception as e_save: 
            print(f"Error saving history to {history_path_final}: {e_save}. Saving to fallback.")
            with open('ppo_history_fallback.pkl', 'wb') as f_hist_fallback: 
                pickle.dump(history, f_hist_fallback)
            print("Fallback history saved to ppo_history_fallback.pkl")


        if self.use_wandb:
            wandb.finish()
        
        return history 


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    env_instance = instantiate(cfg.env) 
    
    trainer: PPOTrainer = instantiate(cfg.trainer, env=env_instance, ppo=cfg.ppo, hydra_cfg=cfg, _recursive_=False)
    
    trainer.train() 

if __name__ == '__main__':
    main()
