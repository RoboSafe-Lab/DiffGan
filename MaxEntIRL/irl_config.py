import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class FeatureExtractionConfig:
    """Configuration for IRL feature extraction"""
    
    # Model and environment
    model_path: str = ""
    env_config: str = ""
    
    # Output settings
    output_dir: str = "irl_features_output"
    
    # Scene selection
    scene_indices: List[int] = field(default_factory=list)
    start_frames: List[int] = field(default_factory=list)
    num_scenes_to_evaluate: int = 50
    num_scenes_per_batch: int = 1
    num_sim_per_scene: int = 1  # Number of simulations per scene from different start frames
    
    # Feature extraction parameters
    frame_step: int = 5  # Step size between frames for extraction
    min_horizon: int = 20  # Minimum remaining horizon for rollouts
    num_rollouts: int = 10  # Number of rollouts per frame
    horizon: int = 50  # Rollout horizon
    
    # Environment settings
    env: str = "trajdata"  # or "nusc"
    eval_class: str = "Diffuser"
    
    # Trajdata specific
    trajdata_source_test: List[str] = field(default_factory=lambda: ["nusc_trainval"])
    trajdata_data_dirs: Dict[str, str] = field(default_factory=dict)
    
    # History and future settings
    history_sec: float = 2.0
    future_sec: float = 8.0
    history_num_frames: int = 20
    
    # Step settings
    step_time: float = 0.1
    n_step_action: int = 5
    
    # Random seed
    seed: int = 42
    
    # Auto-determine start frames like scene_editor
    auto_determine_start_frames: bool = True
    min_remaining_steps: int = 50  # Minimum steps remaining after start frame
    
    def __post_init__(self):
        """Initialize default values after creation"""
        if not self.scene_indices and not self.start_frames:
            # Set default scenes if none provided
            self.scene_indices = list(range(10))  # First 10 scenes as default
            
        if self.auto_determine_start_frames:
            # Start frames will be determined automatically like in scene_editor
            pass
        elif not self.start_frames and self.scene_indices:
            # Set default start frames if not provided and not auto-determining
            self.start_frames = [self.history_num_frames + 1] * len(self.scene_indices)
            
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

# Default configuration instance
default_config = FeatureExtractionConfig()