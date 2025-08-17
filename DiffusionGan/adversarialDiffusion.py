import numpy as np
import torch
import pickle
import os
from extract_features import IRLFeatureExtractor
from run_irl import MaxEntIRL
from tbsim.configs.scene_edit_config import SceneEditingConfig
from irl_config import default_config

class AdversarialIRLDiffusion:
    def __init__(self, config):
        self.config = config
        self.extractor: IRLFeatureExtractor | None = None
        self.env = None
        self.policy = None
        self.policy_model = None
        self.current_theta = None
        self.training_history = []
        
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
    
    def train_adversarial(self, num_iterations=10):
        """
        Main adversarial training loop
        Generator (Diffusion): Tries to generate realistic trajectories
        Discriminator (MaxEnt IRL): Tries to distinguish real vs generated trajectories
        """
        for iteration in range(num_iterations):
            print(f"\n=== Adversarial Training Iteration {iteration + 1}/{num_iterations} ===")
            
            # Step 1: Generate trajectories using current diffusion model (Generator)
            generated_features = self.generate_trajectories_with_current_model()
            
            # Step 2: Update reward function using MaxEnt IRL (Discriminator)
            self.current_theta = self.get_reward_via_irl(generated_features)

            # Step 3: Apply learned reward as guidance after n iterations
            if iteration < num_iterations - self.config.skip_num_scenes:
                self.update_diffusion_model_with_reward()
            
            # Step 4: Evaluate and log progress
            self.evaluate_iteration(iteration)
            
        return self.current_theta, self.training_history
    
    def generate_trajectories_with_current_model(self):
        """Generate trajectories using current diffusion model state"""
        print("Generating trajectories with current diffusion model...")

        if self.extractor is None:
            raise RuntimeError("Extractor is not initialized. Call setup_environment() first.")
        # Extract features using current extractor (env, policy, model inside)
        features = self.extractor.extract_irl_features_from_all_frames()
        return features

    def get_reward_via_irl(self, generated_features):
        """Update reward function using MaxEnt IRL (Discriminator step)"""
        print("Updating reward function using MaxEnt IRL...")        
        
        # Run MaxEnt IRL to learn reward
        features_list = list(generated_features.values())
        irl = MaxEntIRL(feature_names=self.config.feature_names, n_iters=self.config.num_iterations)
        learned_theta, training_log = irl.fit(features_list)
        print(f"Learned reward weights: {learned_theta}")
        return learned_theta

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
        
        # Define feature names to match your IRL features
        feature_names = self.config.feature_names
               
        # Create custom guidance based on learned reward
        reward_guidance = {
            'name': 'learned_reward_guidance',
            'weight': 5.0,  # Adjust based on your needs
            'params': {
                'reward_weights': self.current_theta.tolist(), 
                'feature_names': feature_names,
                'dt': self.config.step_time
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
                if getattr(eval_cfg, "apply_guidance", None) is False:
                    eval_cfg.apply_guidance = True
                if not hasattr(eval_cfg.edits, "guidance_config") or eval_cfg.edits.guidance_config is None:
                    eval_cfg.edits.guidance_config = []
                if not isinstance(eval_cfg.edits.guidance_config, list):
                    eval_cfg.edits.guidance_config = [eval_cfg.edits.guidance_config]
                eval_cfg.edits.guidance_config.append(reward_guidance)
                print(f"Applied reward guidance with weights: {reward_guidance['params']['reward_weights']}")
                return

    
    def evaluate_iteration(self, iteration):
        """Evaluate progress and log results"""
        # Compute metrics to track training progress
        metrics = {
            'iteration': iteration,
            'reward_weights': self.current_theta.copy() if self.current_theta is not None else None,
            'reward_magnitude': np.linalg.norm(self.current_theta) if self.current_theta is not None else 0,
        }
        
        # Add trajectory quality metrics if available
        quality_metrics = self.compute_trajectory_quality_metrics()
        metrics.update(quality_metrics)
        
        self.training_history.append(metrics)
        
        # Save checkpoint
        self.save_checkpoint(iteration)
        
        print(f"Iteration {iteration} metrics: {metrics}")
    
    def compute_trajectory_quality_metrics(self):
        """Compute metrics to evaluate trajectory quality"""
        # Implement metrics like:
        # - Collision rate
        # - Off-road rate  
        # - Similarity to expert trajectories
        # - Diversity of generated trajectories
        return {
            'collision_rate': 0.0,  # Placeholder
            'off_road_rate': 0.0,   # Placeholder
            'expert_similarity': 0.0,  # Placeholder
        }
    
    def save_checkpoint(self, iteration):
        """Save training checkpoint"""
        checkpoint = {
            'iteration': iteration,
            'theta': self.current_theta,
            'training_history': self.training_history,
        }
        
        checkpoint_path = os.path.join(self.config.output_dir, f"adversarial_checkpoint_{iteration}.pkl")
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"Checkpoint saved to {checkpoint_path}")
        
        
        
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