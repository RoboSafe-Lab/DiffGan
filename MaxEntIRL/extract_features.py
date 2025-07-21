import numpy as np
import os
import torch
import random
import importlib
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
                                       frame_step=5, min_horizon=20):
    """
    Extract IRL features by generating rollouts from ALL frames, not just start frame
    Args:
        env: Environment instance
        policy: Policy for rollouts
        policy_model: Policy model
        scene_indices: List of scene indices
        start_frames: List of starting frame indices
        output_dir: Output directory
        frame_step: Step size between frames (e.g., every 5 frames)
        min_horizon: Minimum remaining horizon to generate rollout
    """
    print("Extracting IRL features from ALL frames...")
    
    all_features = {}
    
    for scene_idx, initial_start_frame in zip(scene_indices, start_frames):
        print(f"\nProcessing scene {scene_idx}, initial start frame {initial_start_frame}")
        
        # First, get the scene length to know all available frames
        env.reset(scene_indices=[scene_idx], start_frame_index=[initial_start_frame])
        
        # Determine scene length
        scene_length = get_scene_length(env)
        max_frame = initial_start_frame + scene_length - min_horizon
        
        print(f"Scene length: {scene_length}, processing frames {initial_start_frame} to {max_frame}")
        
        scene_features = []
        
        # Generate rollouts from EVERY frame (or every frame_step frames)
        for current_frame in range(initial_start_frame, max_frame, frame_step):
            remaining_horizon = scene_length - (current_frame - initial_start_frame)
            
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
            
            # Extract trajectories from rollouts
            rollout_trajectories = extract_trajectories_from_rollouts(frame_rollouts)
            
            if not rollout_trajectories:
                print(f"    Warning: No rollout trajectories for frame {current_frame}")
                continue
            
            # Process trajectories for this frame
            frame_features_data = process_frame_trajectories(
                scene_idx, current_frame, remaining_horizon,
                rollout_trajectories, frame_gt
            )
            
            scene_features.append(frame_features_data)
        
        # Save all features for this scene
        if scene_features:
            scene_summary = {
                'scene_idx': scene_idx,
                'initial_start_frame': initial_start_frame,
                'num_frames_processed': len(scene_features),
                'frame_features': scene_features
            }
            
            output_path = os.path.join(output_dir, f"scene_{scene_idx}_all_frames_irl_features.pkl")
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(scene_summary, f)
            
            print(f"Saved {len(scene_features)} frame features to {output_path}")
            all_features[scene_idx] = scene_summary
        
    return all_features

def generate_rollouts_from_specific_frame(env, policy, policy_model, scene_idx, start_frame, 
                                        num_rollouts=10, horizon=50):
    """
    Generate multiple rollouts starting from a SPECIFIC frame
    """
    rollouts = []
    ground_truth = None
    
    # Extract ground truth from this specific frame
    print(f"    Extracting ground truth from frame {start_frame}")
    gt_data = extract_ground_truth_from_env(env, [scene_idx], [start_frame], horizon)
    ground_truth = gt_data.get(f"{scene_idx}_{start_frame}")
    
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
        
        return {
            'rollout_idx': rollout_idx,
            'stats': stats,
            'info': info,
            'buffer': info.get('buffer', None)
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


def setup_from_scene_editor_config(model_path, env_config_path):
    """
    Setup environment, policy, and model from scene editor configuration
    Based on the scene_editor.py setup
    """
    # Load configuration
    eval_cfg = SceneEditingConfig()
    eval_cfg.load_config(env_config_path)
    
    # Set global batch type
    set_global_batch_type("trajdata")
    
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
    
    # Get the underlying policy model
    policy_model = policy.policy if hasattr(policy, 'policy') else policy
    
    return env, policy, policy_model

def extract_ground_truth_from_env(env, scene_indices, start_frames, horizon):
    """
    Extract ground truth trajectories directly from the environment
    """
    ground_truth_data = {}
    
    for scene_idx, start_frame in zip(scene_indices, start_frames):
        print(f"    Extracting GT for scene {scene_idx}, frame {start_frame}")
        
        # Reset environment to the specific scene and frame
        env.reset(scene_indices=[scene_idx], start_frame_index=[start_frame])
        
        gt_trajectories = []
        
        # Extract ground truth by accessing scene data
        if hasattr(env, '_current_scenes') and len(env._current_scenes) > 0:
            current_scene = env._current_scenes[0].scene
            
            # Get all agents in this scene
            agent_ids = list(current_scene.agents.keys())
            agent_trajectories = []
            
            for agent_id in agent_ids:
                agent = current_scene.agents[agent_id]
                agent_traj = []
                
                # Extract trajectory for this agent across time horizon
                for t in range(horizon):
                    frame_idx = start_frame + t
                    if frame_idx < len(agent.states):
                        state = agent.states[frame_idx]
                        pos = [state.position[0], state.position[1]]
                        agent_traj.append(pos)
                    else:
                        break
                
                if len(agent_traj) > 0:
                    agent_trajectories.append(agent_traj)
            
            if agent_trajectories:
                # Convert to [num_agents, time_steps, 2]
                max_len = min(horizon, max(len(traj) for traj in agent_trajectories))
                gt_array = np.zeros((len(agent_trajectories), max_len, 2))
                
                for i, traj in enumerate(agent_trajectories):
                    for j, pos in enumerate(traj[:max_len]):
                        gt_array[i, j] = pos
                
                ground_truth_data[f"{scene_idx}_{start_frame}"] = gt_array
                print(f"      Ground truth shape: {gt_array.shape}")
        
        # Fallback: step through environment to get ground truth
        else:
            try:
                for step in range(horizon):
                    obs = env.get_observation()
                    
                    # Extract agent positions from observation
                    if hasattr(obs, 'agents') and 'centroid' in obs.agents:
                        positions = obs.agents['centroid'][:, :2]  # [num_agents, 2]
                        gt_trajectories.append(positions.cpu().numpy() if torch.is_tensor(positions) else positions)
                    
                    # Step environment without action (just get next state)
                    _, _, done, _ = env.step(None)
                    if done:
                        break
                
                if gt_trajectories:
                    # Convert to [num_agents, time_steps, 2]
                    gt_array = np.array(gt_trajectories).transpose(1, 0, 2)
                    ground_truth_data[f"{scene_idx}_{start_frame}"] = gt_array
                    print(f"      Ground truth shape: {gt_array.shape}")
                    
            except Exception as e:
                print(f"      Error extracting ground truth: {e}")
    
    return ground_truth_data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--env_config", type=str, required=True, help="Environment config")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--scene_indices", nargs='+', type=int, required=True)
    parser.add_argument("--start_frames", nargs='+', type=int, required=True)
    parser.add_argument("--frame_step", type=int, default=5, help="Step size between frames")
    parser.add_argument("--min_horizon", type=int, default=20, help="Minimum horizon for rollouts")
    
    args = parser.parse_args()
    
    try:
        # Setup environment and model
        print("Setting up environment and model...")
        env, policy, policy_model = setup_from_scene_editor_config(args.model_path, args.env_config)
        
        # Extract features from all frames
        print("Starting feature extraction...")
        features = extract_irl_features_from_all_frames(
            env, policy, policy_model, 
            args.scene_indices, args.start_frames, 
            args.output_dir, args.frame_step, args.min_horizon
        )
        
        print("Feature extraction from all frames complete!")
        print(f"Processed {len(features)} scenes")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()