import numpy as np
import os
import torch
import random
import importlib
from irl_config import default_config
from tbsim.configs.scene_edit_config import SceneEditingConfig
from tbsim.evaluation.env_builders import EnvNuscBuilder, EnvUnifiedBuilder, EnvL5Builder
from tbsim.policies.wrappers import RolloutWrapper
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env, set_global_trajdata_batch_raster_cfg

def compute_irl_features(trajectories, dt=0.1):
    """
    Compute IRL features for trajectories
    Args:
        trajectories: numpy array of shape [num_agents, time_steps, 2] (x, y positions)
        dt: time step in seconds
    Returns:
        dict with features: velocity, a_long, jerk_long, thw_front, thw_rear, a_lateral
    """
    num_agents, num_steps, _ = trajectories.shape
    features = {}
    
    # Initialize feature arrays
    features['velocity'] = np.zeros((num_agents, num_steps-1))
    features['a_long'] = np.zeros((num_agents, num_steps-2))
    features['jerk_long'] = np.zeros((num_agents, num_steps-3))
    features['a_lateral'] = np.zeros((num_agents, num_steps-2))
    features['thw_front'] = np.zeros((num_agents, num_steps-1))
    features['thw_rear'] = np.zeros((num_agents, num_steps-1))
    
    for agent_id in range(num_agents):
        traj = trajectories[agent_id]  # [time_steps, 2]
        
        # Compute velocity (magnitude of velocity vector)
        vel_vectors = np.diff(traj, axis=0) / dt  # [time_steps-1, 2]
        velocities = np.linalg.norm(vel_vectors, axis=1)  # [time_steps-1]
        features['velocity'][agent_id] = velocities
        
        # Compute heading angles from velocity vectors
        headings = np.arctan2(vel_vectors[:, 1], vel_vectors[:, 0])  # [time_steps-1]
        
        # Compute longitudinal acceleration
        if len(velocities) > 1:
            a_long = np.diff(velocities) / dt  # [time_steps-2]
            features['a_long'][agent_id] = a_long
            
            # Compute longitudinal jerk
            if len(a_long) > 1:
                jerk_long = np.diff(a_long) / dt  # [time_steps-3]
                features['jerk_long'][agent_id] = jerk_long
        
        # Compute lateral acceleration (change in heading * velocity)
        if len(headings) > 1 and len(velocities) > 1:
            heading_rates = np.diff(headings) / dt  # [time_steps-2]
            # Use velocity at midpoint for lateral acceleration calculation
            mid_velocities = (velocities[:-1] + velocities[1:]) / 2  # [time_steps-2]
            a_lateral = heading_rates * mid_velocities
            features['a_lateral'][agent_id] = a_lateral
        
        # Compute Time Headway (THW) - distance to front/rear vehicles divided by velocity
        for t in range(len(velocities)):
            if velocities[t] > 0.1:  # Only compute if vehicle is moving
                # Find distances to other agents at this timestep
                current_pos = traj[t]
                other_positions = trajectories[:, t, :]  # [num_agents, 2]
                
                # Calculate distances to all other agents
                distances = np.linalg.norm(other_positions - current_pos[None, :], axis=1)
                distances[agent_id] = np.inf  # Ignore self
                
                # Determine which agents are in front/behind based on heading
                if t < len(headings):
                    heading = headings[t]
                    heading_vector = np.array([np.cos(heading), np.sin(heading)])
                    
                    # Vector from current agent to other agents
                    relative_vectors = other_positions - current_pos[None, :]
                    
                    # Dot product to determine front/rear (positive = front, negative = rear)
                    along_heading = np.dot(relative_vectors, heading_vector)
                    
                    # Find closest agent in front and behind
                    front_agents = (along_heading > 0) & (distances < 50.0)  # Within 50m
                    rear_agents = (along_heading < 0) & (distances < 50.0)   # Within 50m
                    
                    if np.any(front_agents):
                        closest_front_dist = np.min(distances[front_agents])
                        features['thw_front'][agent_id, t] = closest_front_dist / velocities[t]
                    else:
                        features['thw_front'][agent_id, t] = 10.0  # Large value if no front vehicle
                    
                    if np.any(rear_agents):
                        closest_rear_dist = np.min(distances[rear_agents])
                        features['thw_rear'][agent_id, t] = closest_rear_dist / velocities[t]
                    else:
                        features['thw_rear'][agent_id, t] = 10.0  # Large value if no rear vehicle
                else:
                    features['thw_front'][agent_id, t] = 10.0
                    features['thw_rear'][agent_id, t] = 10.0
            else:
                features['thw_front'][agent_id, t] = 10.0
                features['thw_rear'][agent_id, t] = 10.0
    
    return features

def extract_irl_features_from_all_frames(env, policy, policy_model, scene_indices, start_frames, output_dir, 
                                       frame_step=5, min_horizon=20, num_sim_per_scene=1):
    """
    Extract IRL features by generating rollouts from ALL frames, not just start frame
    Args:
        env: Environment instance
        policy: Policy for rollouts
        policy_model: Policy model
        scene_indices: List of scene indices
        start_frames: List of starting frame indices (can be None for automatic determination)
        output_dir: Output directory
        frame_step: Step size between frames (e.g., every 5 frames)
        min_horizon: Minimum remaining horizon to generate rollout
        num_sim_per_scene: Number of simulations per scene
    """
    print("Extracting IRL features from ALL frames...")
    
    all_features = {}
    
    # Reset environment to get scene information
    scenes_valid = env.reset(scene_indices=scene_indices, start_frame_index=None)
    scene_indices = [si for si, sval in zip(scene_indices, scenes_valid) if sval]
    
    if len(scene_indices) == 0:
        print('No valid scenes, skipping...')
        return all_features
    
    print(f"Valid scenes after reset: {scene_indices}")

    # Automatically determine start frames
    if not start_frames or num_sim_per_scene > 1:
        start_frame_index = []
        
        for si_idx, scene_idx in enumerate(scene_indices):
            print(f"Processing scene {scene_idx} (index {si_idx})")
            if hasattr(env, '_current_scenes') and len(env._current_scenes) > si_idx:
                current_scene = env._current_scenes[si_idx].scene
                
                # Get history frames from environment config
                history_frames = getattr(env, '_history_num_frames', 10)  # default fallback
                if hasattr(env, 'exp_config') and hasattr(env.exp_config, 'algo'):
                    history_frames = env.exp_config.algo.history_num_frames
                
                sframe = history_frames + 1
                # Ensure there's enough horizon for rollout
                eframe = current_scene.length_timesteps - min_horizon
                
                if eframe <= sframe:
                    print(f"Scene {scene_idx}: insufficient length (length={current_scene.length_timesteps}, need at least {sframe + min_horizon})")
                    continue
                
                if num_sim_per_scene > 1:
                    # Multiple simulations per scene - spread them across the scene
                    scene_frame_inds = np.linspace(sframe, eframe, num=num_sim_per_scene, dtype=int).tolist()
                else:
                    # Single simulation - use default start frame
                    scene_frame_inds = [sframe]
                    
                start_frame_index.append(scene_frame_inds)
                print(f"Scene {scene_idx}: frames {sframe} to {eframe}, selected starts: {scene_frame_inds}")
            else:
                # Fallback if scene info not available
                print(f"Scene {scene_idx}: using fallback start frame")
                start_frame_index.append([min_horizon])
    else:
        # Use provided start_frames, but format as nested list for consistency
        print(f"Using provided start frames")
        start_frame_index = [[sf] for sf in start_frames]
    
    print(f'Automatically determined starting frames: {start_frame_index}')
    
    # Process each scene with its determined start frames
    for si_idx, scene_idx in enumerate(scene_indices):
        scene_start_frames = start_frame_index[si_idx]
        
        for start_frame in scene_start_frames:
            print(f"\nProcessing scene {scene_idx}, start frame {start_frame}")
            
            # Reset to this specific scene and start frame
            scenes_valid = env.reset(scene_indices=[scene_idx], start_frame_index=[start_frame])
            if not scenes_valid[0]:
                print(f"Scene {scene_idx} invalid at start frame {start_frame}, skipping...")
                continue
            
            # Determine scene length from current start frame
            remaining_length = get_remaining_scene_length(env, start_frame)
            max_frame = start_frame + remaining_length - min_horizon
            
            if max_frame <= start_frame:
                print(f"Insufficient remaining length for scene {scene_idx} from frame {start_frame}")
                continue
                
            print(f"Scene remaining length: {remaining_length}, processing frames {start_frame} to {max_frame}")
            
            scene_features = []
            
            # Generate rollouts from frames within this scene
            for current_frame in range(start_frame, max_frame, frame_step):
                remaining_horizon = remaining_length - (current_frame - start_frame)
                
                if remaining_horizon < min_horizon:
                    print(f"  Skipping frame {current_frame}: insufficient horizon ({remaining_horizon})")
                    continue
                    
                print(f"  Processing frame {current_frame}, horizon: {remaining_horizon}")
                
                # Generate rollouts starting from this specific frame
                frame_rollouts, frame_gt = generate_rollouts_from_specific_frame(
                    env, policy, policy_model, scene_idx, current_frame, 
                    num_rollouts=10, horizon=remaining_horizon
                )
                
                if not frame_rollouts or frame_gt is None:
                    print(f"    Warning: No data generated for frame {current_frame}")
                    continue
                
                # Extract and process trajectories
                rollout_trajectories = extract_trajectories_from_rollouts(frame_rollouts)
                
                if not rollout_trajectories:
                    print(f"    Warning: No rollout trajectories for frame {current_frame}")
                    continue
                
                frame_features_data = process_frame_trajectories(
                    scene_idx, current_frame, remaining_horizon,
                    rollout_trajectories, frame_gt
                )
                
                if frame_features_data:
                    scene_features.append(frame_features_data)
            
            # Save features for this scene-start_frame combination
            if scene_features:
                scene_summary = {
                    'scene_idx': scene_idx,
                    'start_frame': start_frame,
                    'num_frames_processed': len(scene_features),
                    'frame_features': scene_features
                }
                
                output_path = os.path.join(output_dir, f"scene_{scene_idx}_start_{start_frame}_all_frames_irl_features.pkl")
                import pickle
                with open(output_path, 'wb') as f:
                    pickle.dump(scene_summary, f)
                
                print(f"Saved {len(scene_features)} frame features to {output_path}")
                all_features[f"{scene_idx}_{start_frame}"] = scene_summary
    
    return all_features

def get_remaining_scene_length(env, start_frame):
    """
    Get the remaining length of the scene from the start frame
    """
    if hasattr(env, '_current_scenes') and len(env._current_scenes) > 0:
        current_scene = env._current_scenes[0].scene
        return current_scene.length_timesteps - start_frame
    else:
        # Fallback: estimate by stepping through
        original_frame = env._frame_index if hasattr(env, '_frame_index') else 0
        step_count = 0
        while step_count < 1000:  # Safety limit
            try:
                obs, _, done, info = env.step(None)
                step_count += 1
                if done:
                    break
            except:
                break
        return step_count

def generate_rollouts_from_specific_frame(env, policy, policy_model, scene_idx, start_frame, 
                                        num_rollouts=10, horizon=50):
    """
    Generate multiple rollouts starting from a SPECIFIC frame
    """
    rollouts = []
    ground_truth = None
    
    # Generate multiple rollouts from this frame
    print(f"    Generating {num_rollouts} rollouts from frame {start_frame}")
    
    for rollout_idx in range(num_rollouts):
        # Reset environment to this specific frame
        env.reset(scene_indices=[scene_idx], start_frame_index=[start_frame])
        
        # Add variation for different rollouts
        if rollout_idx > 0:
            np.random.seed(42 + rollout_idx + start_frame)  # Include frame in seed
            torch.manual_seed(42 + rollout_idx + start_frame)
        
        # Generate single rollout from this frame
        rollout_data = generate_single_rollout_from_frame(
            env, policy, policy_model, horizon, rollout_idx
        )
        
        if rollout_data is not None:
            rollouts.append(rollout_data)
            
            # Extract ground truth from first rollout (should be same for all)
            if ground_truth is None and rollout_data.get('ground_truth') is not None:
                ground_truth = rollout_data['ground_truth']
                print(f"    Extracted GT shape: {ground_truth.shape if hasattr(ground_truth, 'shape') else type(ground_truth)}")
    
    
    return rollouts, ground_truth

def generate_single_rollout_from_frame(env, policy, policy_model, horizon, rollout_idx):
    """
    Generate a single rollout starting from current environment state
    """
    try:
        # Use the guided_rollout function from scene_edit_utils
        from tbsim.utils.scene_edit_utils import guided_rollout
        
        stats, info, renderings = guided_rollout(
            env=env,
            policy=policy,
            policy_model=policy_model,
            n_step_action=5,
            guidance_config=None,  # No guidance for base rollouts
            constraint_config=None,
            render=False,
            scene_indices=None,  # Already set by reset
            device="cuda" if torch.cuda.is_available() else "cpu",
            obs_to_torch=True,
            horizon=horizon,
            start_frames=None,  # Already set by reset
            eval_class='Diffuser',
            apply_guidance=False,
            apply_constraints=False,
        )
        
        # Extract ground truth from the buffer
        ground_truth = None
        if 'buffer' in info and len(info['buffer']) > 0:
            # The buffer contains ground truth trajectories
            buffer = info['buffer'][0]  # First scene's buffer
            
            # Look for ground truth keys in the buffer
            gt_keys = ['target_positions', 'gt_future_positions', 'centroid']
            for key in gt_keys:
                if key in buffer:
                    ground_truth = buffer[key]
                    break
        
        return {
            'rollout_idx': rollout_idx,
            'stats': stats,
            'info': info,
            'buffer': info.get('buffer', None),
            'ground_truth': ground_truth  # Add extracted GT
        }
        
    except Exception as e:
        print(f"      Error in rollout {rollout_idx}: {e}")
        return None

def process_frame_trajectories(scene_idx, frame_number, horizon, rollout_trajectories, ground_truth):
    """
    Process trajectories from a specific frame and compute features
    """
    # Extract trajectory data from rollouts
    scene_rollout_trajectories = []
    for traj_data in rollout_trajectories:
        if traj_data.get("centroid") is not None:
            scene_rollout_trajectories.append(traj_data["centroid"])
    
    if not scene_rollout_trajectories:
        return None
    
    print(f"    Processing {len(scene_rollout_trajectories)} rollout trajectories")
    if ground_truth is not None:
        print(f"    Ground truth shape: {ground_truth.shape}")
    
    # Compute features for all rollouts
    rollout_features_list = []
    for rollout_traj in scene_rollout_trajectories:
        features = compute_irl_features(rollout_traj, dt=0.1)
        rollout_features_list.append(features)
    
    # Compute features for ground truth
    gt_features = None
    if ground_truth is not None:
        gt_features = compute_irl_features(ground_truth, dt=0.1)
    
    # Compute feature expectations across rollouts
    rollout_feature_expectations = compute_feature_expectations(rollout_features_list)
    
    # Compute feature comparison if we have ground truth
    feature_comparison = {}
    if gt_features is not None:
        feature_comparison = compare_rollout_vs_gt_features(
            rollout_feature_expectations, gt_features
        )
    
    return {
        'scene_idx': scene_idx,
        'frame_number': frame_number,
        'horizon': horizon,
        'num_rollouts': len(scene_rollout_trajectories),
        'rollout_features': rollout_features_list,
        'rollout_feature_expectations': rollout_feature_expectations,
        'ground_truth_features': gt_features,
        'feature_comparison': feature_comparison,
        'rollout_trajectories': scene_rollout_trajectories,
        'ground_truth_trajectories': ground_truth
    }

def get_scene_length(env):
    """
    Get the total length of the current scene
    """
    if hasattr(env, '_current_scenes'):
        current_scene = env._current_scenes[0].scene
        return current_scene.length_timesteps
    else:
        # Fallback: try to step through until done
        step_count = 0
        while step_count < 1000:  # Safety limit
            obs, _, done, info = env.step(None)
            step_count += 1
            if done:
                break
        return step_count

def compare_rollout_vs_gt_features(rollout_expectations, gt_features):
    """
    Compare rollout feature expectations with ground truth features
    """
    comparison = {}
    
    for feature_name in rollout_expectations.keys():
        if feature_name.endswith('_std'):
            continue  # Skip std features
            
        if feature_name in gt_features:
            rollout_vals = rollout_expectations[feature_name]
            gt_vals = gt_features[feature_name]
            
            # Ensure compatible shapes
            min_agents = min(rollout_vals.shape[0], gt_vals.shape[0])
            min_timesteps = min(rollout_vals.shape[1], gt_vals.shape[1])
            
            rollout_crop = rollout_vals[:min_agents, :min_timesteps]
            gt_crop = gt_vals[:min_agents, :min_timesteps]
            
            # Compute comparison metrics
            comparison[feature_name] = {
                'mse': np.mean((rollout_crop - gt_crop) ** 2),
                'mae': np.mean(np.abs(rollout_crop - gt_crop)),
                'rollout_mean': np.mean(rollout_crop),
                'rollout_std': np.std(rollout_crop),
                'gt_mean': np.mean(gt_crop),
                'gt_std': np.std(gt_crop)
            }
    
    return comparison

def compute_feature_expectations(feature_list):
    """
    Compute expected feature values across multiple rollouts
    """
    if not feature_list:
        return {}
        
    feature_expectations = {}
    feature_names = feature_list[0].keys()
    
    for feature_name in feature_names:
        # Stack features from all rollouts
        feature_stack = np.stack([f[feature_name] for f in feature_list])
        # Compute expectation (mean across rollouts)
        feature_expectations[feature_name] = np.mean(feature_stack, axis=0)
        # Also compute standard deviation
        feature_expectations[feature_name + '_std'] = np.std(feature_stack, axis=0)
    
    return feature_expectations

def extract_trajectories_from_rollouts(rollouts):
    """
    Extract trajectory data from rollout results
    """
    extracted_trajectories = []
    
    for rollout in rollouts:
        if rollout is None:
            continue
            
        buffer = rollout.get('buffer')
        if buffer is not None:
            for scene_buffer in buffer:
                trajectory_data = {
                    "centroid": scene_buffer.get("centroid", None),
                    "gt_future_positions": scene_buffer.get("gt_future_positions", None),
                    "target_positions": scene_buffer.get("target_positions", None),
                }
                extracted_trajectories.append(trajectory_data)
    
    return extracted_trajectories


def setup_from_scene_editor_config(eval_cfg):
    """
    Setup environment, policy, and model from scene editor configuration
    Enhanced to properly extract algorithm config for history frames
    """
    assert eval_cfg.env in ["nusc", "trajdata"], "Currently only nusc and trajdata environments are supported"
        
    # Set global batch type
    set_global_batch_type("trajdata")
    if eval_cfg.env == "nusc":
        set_global_trajdata_batch_env("nusc_trainval")
    elif eval_cfg.env == "trajdata":
        # assumes all used trajdata datasets use share same map layers
        set_global_trajdata_batch_env(eval_cfg.trajdata_source_test[0])
    
    # Set random seeds for reproducibility
    np.random.seed(eval_cfg.seed)
    random.seed(eval_cfg.seed)
    torch.manual_seed(eval_cfg.seed)
    torch.cuda.manual_seed(eval_cfg.seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create policy and rollout wrapper
    policy_composers = importlib.import_module("tbsim.evaluation.policy_composers")
    composer_class = getattr(policy_composers, eval_cfg.eval_class)
    composer = composer_class(eval_cfg, device)
    policy, exp_config = composer.get_policy()
    
    # Set global trajdata batch raster config
    set_global_trajdata_batch_raster_cfg(exp_config.env.rasterizer)
    
    # Create environment
    if eval_cfg.env == "nusc":
        env_builder = EnvNuscBuilder(eval_cfg, exp_config, device)
    elif eval_cfg.env == "trajdata":
        env_builder = EnvUnifiedBuilder(eval_cfg, exp_config, device)
    else:
        raise ValueError(f"Unknown environment: {eval_cfg.env}")
    
    env = env_builder.get_env()
    
    # Store exp_config in env for access to algorithm parameters
    env.exp_config = exp_config
    
    # Get the underlying policy model
    policy_model = policy.policy if hasattr(policy, 'policy') else policy
    
    return env, policy, policy_model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # Use the SAME arguments as scene_editor.py
    parser.add_argument("--dataset_path", type=str, default="../behavior-generation-dataset/nuscenes",
                       help="Path to dataset")
    parser.add_argument("--registered_name", type=str, default="trajdata_nusc_diff",
                       help="Registered config name")
    parser.add_argument("--env", type=str, default="trajdata", choices=["nusc", "trajdata"],
                       help="Which env to run")
    parser.add_argument("--eval_class", type=str, default="Diffuser",
                       help="Evaluation class")
    parser.add_argument("--policy_ckpt_dir", type=str, default=default_config.policy_ckpt_dir,
                       help=f"Directory for policy checkpoint (default: {default_config.policy_ckpt_dir})")
    parser.add_argument("--policy_ckpt_key", type=str, default=default_config.policy_ckpt_key,
                       help=f"Key for policy checkpoint (default: {default_config.policy_ckpt_key})")
    parser.add_argument("--output_dir", type=str, default=default_config.output_dir,
                       help=f"Output directory (default: {default_config.output_dir})")
    parser.add_argument("--scene_indices", nargs='+', type=int, default=default_config.scene_indices,
                       help=f"Scene indices (default: {default_config.scene_indices})")
    parser.add_argument("--start_frames", nargs='+', type=int, default=default_config.start_frames,
                       help="Optional start frames")
    parser.add_argument("--frame_step", type=int, default=default_config.frame_step,
                       help=f"Step size between frames (default: {default_config.frame_step})")
    parser.add_argument("--min_horizon", type=int, default=default_config.min_horizon,
                       help=f"Minimum horizon for rollouts (default: {default_config.min_horizon})")
    parser.add_argument("--num_sim_per_scene", type=int, default=default_config.num_sim_per_scene,
                       help=f"Number of simulations per scene (default: {default_config.num_sim_per_scene})")
    

    args = parser.parse_args()    
    cfg = SceneEditingConfig(registered_name=args.registered_name)
    
    # Set evaluation class
    if args.eval_class is not None:
        cfg.eval_class = args.eval_class
    # Set dataset path
    if args.dataset_path is not None:
        cfg.dataset_path = args.dataset_path
    
    # Set checkpoint paths 
    if args.policy_ckpt_dir is not None:
        assert args.policy_ckpt_key is not None, "Please specify a key to look for the checkpoint, e.g., 'iter50000'"
        cfg.ckpt.policy.ckpt_dir = args.policy_ckpt_dir
        cfg.ckpt.policy.ckpt_key = args.policy_ckpt_key
    
    # Set environment
    if args.env is not None:
        cfg.env = args.env
    
    # Copy env-specific config to global level (like scene_editor.py does)
    for k in cfg[cfg.env]:
        cfg[k] = cfg[cfg.env][k]
    
    # Remove env-specific sections (like scene_editor.py does)
    cfg.pop("nusc", None)
    cfg.pop("trajdata", None)
    
    # Set results directory to match your output directory
    cfg.results_dir = args.output_dir
    
    try:
        # Setup environment and model
        print("Setting up environment and model...")
        env, policy, policy_model = setup_from_scene_editor_config(cfg)
        
        # Extract features from all frames with automatic start frame determination
        print("Starting feature extraction...")
        features = extract_irl_features_from_all_frames(
            env, policy, policy_model, 
            args.scene_indices, args.start_frames, 
            args.output_dir, args.frame_step, args.min_horizon,
            args.num_sim_per_scene
        )
        
        print("Feature extraction from all frames complete!")
        print(f"Processed {len(features)} scene-start_frame combinations")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()