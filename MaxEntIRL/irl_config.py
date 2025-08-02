import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class FeatureExtractionConfig:
    """Configuration for IRL feature extraction"""
    
    # Model and environment - Match the working scene_editor.py format
    policy_ckpt_dir: str = "../CTG/diffuser_trained_models/test/run0"  # Relative path like scene_editor
    policy_ckpt_key: str = "iter100000.ckpt"        # Full path with checkpoints/ prefix
    
    # Output settings
    output_dir: str = "irl_features_output"
    
    # Scene selection
    scene_indices: List[int] = field(default_factory=lambda: [0]) # Default to first scene
    start_frames: List[int] = field(default_factory=lambda: [31]) # Default start frame for each scene
    num_scenes_to_evaluate: int = 50
    num_scenes_per_batch: int = 1
    num_sim_per_scene: int = 1
    
    debug: bool = True  # Debug mode for additional logging
    
    filter_dynamic: bool = True  # Filter for dynamic agents only    
    min_velocity_threshold: float = 2.0 # Minimum velocity threshold for filtering agents
    
    # Feature extraction parameters
    num_rollouts: int = 8
    horizon: int = 50 # Fixed horizon for generating rollouts

    # Environment settings
    env: str = "trajdata"
    eval_class: str = "Diffuser"
    
    # Trajdata specific
    trajdata_source_test: List[str] = field(default_factory=lambda: ["nusc_trainval"])
    trajdata_data_dirs: Dict[str, str] = field(default_factory=dict)
    
    # History and future settings
    history_sec: float = 3.0
    future_sec: float = 5.2
    history_num_frames: int = 20
    
    # Step settings
    step_time: float = 0.1
    n_step_action: int = 5
    
    # Random seed
    seed: int = 42
    
    # minimum remaining steps for rollouts
    min_remaining_steps: int = 50
    
    def __post_init__(self):
        """Initialize default values after creation"""
        print(f"✓ Policy checkpoint dir: {self.policy_ckpt_dir}")
        print(f"✓ Policy checkpoint key: {self.policy_ckpt_key}")
        
        # Set default scenes if none provided
        if not self.scene_indices:
            self.scene_indices = list(range(num_scenes_to_evaluate))
            
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

# Default configuration instance
default_config = FeatureExtractionConfig()