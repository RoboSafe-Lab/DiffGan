import numpy as np
import torch
import pickle
import os
from extract_features import extract_irl_features_from_all_frames, setup_from_scene_editor_config
from run_irl import maxent_irl, convert_features_to_array
from tbsim.configs.scene_edit_config import SceneEditingConfig
from irl_config import default_config

class AdversarialIRLDiffusion:
    def __init__(self, config):
        self.config = config
        self.env = None
        self.policy = None
        self.policy_model = None
        self.current_theta = None
        self.training_history = []
        
    def setup_environment(self):
        """Setup environment and models"""
        # Use your existing setup
        cfg = SceneEditingConfig(registered_name=self.config.registered_name)
        # Configure as needed...
        
        self.env, self.policy, self.policy_model = setup_from_scene_editor_config(cfg)
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
        
        # Extract features using current model
        features = extract_irl_features_from_all_frames(
            self.env, self.policy, self.policy_model, self.config
        )
        
        return features
    
    def get_reward_via_irl(self, generated_features):
        """Update reward function using MaxEnt IRL (Discriminator step)"""
        print("Updating reward function using MaxEnt IRL...")        
        
        # Run MaxEnt IRL to learn reward
        features_list = list(generated_features.values())
        learned_theta, training_log = maxent_irl(features_list)

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
        feature_names = ['velocity', 'a_long', 'jerk_long', 'a_lateral', 'thw_front', 'thw_rear']
               
        # Create custom guidance based on learned reward
        reward_guidance = {
            'name': 'learned_reward_guidance',
            'weight': 5.0,  # Adjust based on your needs
            'params': {
                'reward_weights': self.current_theta.tolist(), 
                'feature_names': feature_names,
                'dt': 0.1
            },
            'agents': None  # Apply to all agents
        }

        return reward_guidance


    def apply_reward_guidance(self, reward_guidance):
        """Apply learned reward as guidance to diffusion model"""
        if reward_guidance is None:
            print("No reward guidance to apply")
            return
            
        # Update the policy's guidance configuration
        if hasattr(self.policy, 'guidance_config'):
            if not isinstance(self.policy.guidance_config, list):
                self.policy.guidance_config = [self.policy.guidance_config]
            
            # Add or update the learned reward guidance
            self.policy.guidance_config.append(reward_guidance)
        else:
            self.policy.guidance_config = [reward_guidance]
        
        print(f"Applied reward guidance with weights: {reward_guidance['params']['reward_weights']}")

    
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