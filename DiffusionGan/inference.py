import numpy as np
import torch
import pickle
import os
import wandb
from MaxEntIRL.extract_features import IRLFeatureExtractor
from MaxEntIRL.run_irl import MaxEntIRL
from tbsim.configs.scene_edit_config import SceneEditingConfig
from MaxEntIRL.irl_config import default_config

class AdversarialIRLDiffusion:
    def __init__(self, config):
        self.config = config
        self.extractor: IRLFeatureExtractor | None = None
        self.env = None
        self.policy = None
        self.policy_model = None
        self.current_theta = None
        self.training_history = []
        self.theta_ema = None # EMA of theta for stability
        self.irl_norm_mean = None
        self.irl_norm_std = None
        
        # Initialize wandb
        if self.config.use_wandb:
            self.init_wandb()

    def init_wandb(self):
        """Initialize Weights & Biases logging"""
        wandb_config = {
            "num_iterations": self.config.num_iterations,
            "theta_ema_beta": self.config.theta_ema_beta,
            "guidance_weight": self.config.guidance_weight,
            "step_time": self.config.step_time,
            "num_scenes_to_evaluate": self.config.num_scenes_to_evaluate,
            "num_rollouts": self.config.num_rollouts,
            "horizon": self.config.horizon,
            "feature_names": self.config.feature_names,
            "policy_ckpt_dir": self.config.policy_ckpt_dir,
            "policy_ckpt_key": self.config.policy_ckpt_key,
            "env": self.config.env,
            "eval_class": self.config.eval_class,
        }
        
        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=self.config.wandb_run_name,
            tags=self.config.wandb_tags,
            config=wandb_config,
            reinit=True
        )
        
        print(f"Initialized wandb project: {self.config.wandb_project}")

    def setup_environment(self):
        """Setup environment and models"""
        # Build SceneEditingConfig similarly to extract_features.__main__
        cfg = SceneEditingConfig(registered_name="trajdata_nusc_diff")
        # Apply CLI-like overrides from default_config
        if hasattr(self.config, "eval_class"):
            cfg.eval_class = self.config.eval_class
        if hasattr(self.config, "env"):
            cfg.env = self.config.env
        # editing source (optional)
        if hasattr(cfg, "edits") and hasattr(cfg.edits, "editing_source"):
            if not isinstance(cfg.edits.editing_source, list):
                cfg.edits.editing_source = [cfg.edits.editing_source]
        # Hoist env-specific entries
        for k in cfg[cfg.env]:
            cfg[k] = cfg[cfg.env][k]
        cfg.pop("nusc", None)
        cfg.pop("trajdata", None)
        # Checkpoint paths and output dir
        cfg.ckpt.policy.ckpt_dir = self.config.policy_ckpt_dir
        cfg.ckpt.policy.ckpt_key = self.config.policy_ckpt_key
        cfg.results_dir = self.config.output_dir

        # Create extractor (holds env/policy/model)
        self.extractor = IRLFeatureExtractor(cfg, self.config)
        self.env = self.extractor.env
        self.policy = self.extractor.policy
        self.policy_model = self.extractor.policy_model
        print("Environment and models setup complete")
    
    def inference_adversarial(self, num_iterations=10):
        """
        Main adversarial inference loop
        Using learned reward to guide the generation of Diffusion models
        """

        # load learned reward
        reward = load ('irl_ouput')
           
        # Step 3: Apply learned reward as guidance
        if iteration < num_iterations - 1:
            self.update_diffusion_model_with_reward()
            


    
    def generate_trajectories_with_current_model(self):
        """Generate trajectories using current diffusion model state"""
        print("Generating trajectories with current diffusion model...")

        if self.extractor is None:
            raise RuntimeError("Extractor is not initialized. Call setup_environment() first.")
        # Extract features using current extractor (env, policy, model inside)
        features = self.extractor.extract_irl_features_from_all_frames()
        return features


    def update_diffusion_model_with_reward(self):
        """Update diffusion model using learned reward as guidance (Generator step)"""
        print("Updating diffusion model with learned reward guidance...")
        
        # Convert learned reward to guidance configuration
        reward_guidance = self.convert_reward_to_guidance()
        
        # Apply reward-based guidance to diffusion model
        self.apply_reward_guidance(reward_guidance)
    
    def convert_reward_to_guidance(self):
        """Convert learned reward weights to diffusion guidance"""
        if self.current_theta is None:
            return None        
        
        # Use EMA weights for stability (fallback to current if ema missing)
        theta = self.theta_ema if self.theta_ema is not None else np.array(self.current_theta, dtype=float)
        
        # Define feature names to match your IRL features
        feature_names = self.config.feature_names
               
        # Create custom guidance based on learned reward
        reward_guidance = {
            'name': 'learned_reward_guidance',
            'weight': self.config.guidance_weight,
            'params': {
                'reward_weights': theta.tolist(), 
                'feature_names': feature_names,
                'dt': self.config.step_time,
                'norm_mean': self.irl_norm_mean.tolist(),
                'norm_std': self.irl_norm_std.tolist(),
            },
            'agents': None  # Apply to all agents
        }

        return reward_guidance


    def apply_reward_guidance(self, reward_guidance):
        """Apply learned reward as guidance to diffusion model"""
        if reward_guidance is None:
            print("No reward guidance to apply")
            return
        
        # Attach to eval_cfg so extractor’s guided_rollout picks it up
        if self.env is not None and hasattr(self.env, "eval_cfg"):
            eval_cfg = self.env.eval_cfg
            if hasattr(eval_cfg, "edits"):
                eval_cfg.apply_guidance = True
                
                if not hasattr(eval_cfg.edits, "guidance_config") or eval_cfg.edits.guidance_config is None:
                    eval_cfg.edits.guidance_config = []
                # Flatten per-scene format if present
                if len(eval_cfg.edits.guidance_config) > 0 and isinstance(eval_cfg.edits.guidance_config[0], list):
                    scenes_cfg = eval_cfg.edits.guidance_config
                else:
                    scenes_cfg = [eval_cfg.edits.guidance_config]
                # Replace any previous learned_reward_guidance in each scene
                new_scenes_cfg = []
                for scene_list in scenes_cfg:
                    filtered = [g for g in scene_list if g.get('name') != 'learned_reward_guidance']
                    filtered.append(reward_guidance)
                    new_scenes_cfg.append(filtered)
                eval_cfg.edits.guidance_config = new_scenes_cfg
                print(f"Applied reward guidance with weights: {reward_guidance['params']['reward_weights']}")

    def save_inference_results(self):
        # saving the inferencing results for subsequently metrics analysis
        pass
    
   
        
        
if __name__ == "__main__":
    # Initialize adversarial trainer
    trainer = AdversarialIRLDiffusion(default_config)
    
    # Setup environment
    trainer.setup_environment()
    
    # Run adversarial training
    final_theta, history = trainer.train_adversarial(num_iterations=default_config.num_iterations)
    
    print(f"Final learned reward weights: {final_theta}")
    
    # Save final results
    with open("adversarial_irl_results.pkl", "wb") as f:
        pickle.dump({
            "final_theta": final_theta,
            "training_history": history
        }, f)