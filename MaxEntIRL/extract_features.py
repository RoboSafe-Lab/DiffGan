import numpy as np
import os
import torch
import random
import importlib
import pickle
from irl_config import default_config
from tbsim.configs.scene_edit_config import SceneEditingConfig
from tbsim.evaluation.env_builders import EnvNuscBuilder, EnvUnifiedBuilder, EnvL5Builder
from tbsim.policies.wrappers import RolloutWrapper
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env, set_global_trajdata_batch_raster_cfg


def extract_irl_features_from_all_frames(env, policy, policy_model, scene_indices, start_frames, output_dir, 
                                        horizon=50, num_sim_per_scene=1, num_rollouts=10, filter_dynamic=False,
                                        min_velocity_threshold=2.0, debug=False):
    """
    Extract IRL features by generating rollouts from ALL frames, not just start frame
    Args:
        env: Environment instance
        policy: Policy for rollouts
        policy_model: Policy model
        scene_indices: List of scene indices
        start_frames: List of starting frame indices (can be None for automatic determination)
        output_dir: Output directory
        horizon:  fixed horizon to generate rollout
        num_sim_per_scene: Number of simulations per scene
        num_rollouts: Number of rollouts to generate per frame
        filter_dynamic: Filter for dynamic agents only
        min_velocity_threshold: Minimum velocity threshold for dynamic agents
        debug: Enable debug mode for plotting rollouts and ground truth
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
                eframe = current_scene.length_timesteps - horizon
                
                if eframe <= sframe:
                    print(f"Scene {scene_idx}: insufficient length (length={current_scene.length_timesteps}, need at least {sframe + horizon})")
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
                start_frame_index.append([horizon])
                
        print(f'Automatically determined starting frames: {start_frame_index}')        
    else:
        # Use provided start_frames, but format as nested list for consistency
        print(f"Using provided start frames: {start_frames}")
        start_frame_index = [[sf] for sf in start_frames]

    # Process each scene with its determined start frames
    for si_idx, scene_idx in enumerate(scene_indices):
        scene_features = []    
        scene_start_frames = start_frame_index[si_idx]
        scene_name = env._current_scenes[0].scene.name 
        
        for start_frame in scene_start_frames:
            print(f"\nProcessing scene {scene_idx}, start frame {start_frame}")
            
            # Reset to this specific scene and start frame
            scenes_valid = env.reset(scene_indices=[scene_idx], start_frame_index=[start_frame])
            if not scenes_valid[0]:
                print(f"Scene {scene_idx} invalid at start frame {start_frame}, skipping...")
                continue

            # Generate rollouts starting from this specific frame
            rollout_trajectories, gt_trajectories = generate_rollouts_from_specific_frame(
                env, policy, policy_model, scene_idx, start_frame, 
                num_rollouts=num_rollouts, horizon=horizon
            )

            if not rollout_trajectories or gt_trajectories is None:
                print(f"    Warning: No data generated for frame {start_frame}")
                continue
                        
            frame_features_data = process_frame_trajectories(
                scene_idx, scene_name, start_frame, rollout_trajectories, gt_trajectories, 
                filter_dynamic=filter_dynamic, min_velocity_threshold=2.0, debug=debug
            )
            
            if frame_features_data:
                scene_features.append({
                    "start_frame": start_frame,
                    "frame_features": frame_features_data
                })

        # Save features for current scene
        if scene_features:            
            output_path = os.path.join(output_dir, f"scene_{scene_idx}_irl_features.pkl")
            with open(output_path, 'wb') as f:
                pickle.dump(scene_features, f)

            print(f"Saved {len(scene_features)} frame features to {output_path}")
            all_features[f"{scene_idx}"] = scene_features

    return all_features


def generate_rollouts_from_specific_frame(env, policy, policy_model, scene_idx, start_frame, 
                                        num_rollouts=10, horizon=50, 
                                        filter_dynamic=False, min_velocity_threshold=2.0):
    """
    Generate multiple rollouts starting from a SPECIFIC frame
    """
    # Step 1: Extract ground truth trajectory
    print(f"    Step 1: Extracting ground truth trajectory for scene {scene_idx}, frame {start_frame}")
    ground_truth = extract_ground_truth_trajectory(env, scene_idx, start_frame, horizon)

    if ground_truth is None or len(ground_truth) == 0:
        print(f"    No dynamic agents found in ground truth, skipping rollouts")
        return None, None
    
    # Step 2: Generate rollouts
    rollouts = []
    # Generate multiple rollouts from this frame
    print(f"    Generating {num_rollouts} rollouts from frame {start_frame}")
    
    for rollout_idx in range(num_rollouts):
        # Reset environment to this specific frame
        scenes_valid = env.reset(scene_indices=[scene_idx], start_frame_index=[start_frame])
        if not scenes_valid[0]:
            print(f"      Scene {scene_idx} invalid at frame {start_frame} for rollout {rollout_idx}")
            continue
        
        # Add variation for different rollouts
        if rollout_idx > 0:
            np.random.seed(42 + rollout_idx + start_frame)
            torch.manual_seed(42 + rollout_idx + start_frame)
        
        # Generate single rollout from this frame
        rollout_data = generate_single_rollout_from_frame(
            env, policy, policy_model, scene_idx, start_frame, horizon, rollout_idx
        )

        if rollout_data is not None:
            rollouts.append(rollout_data)
        else:
            print(f"      Failed to generate rollout {rollout_idx}")
    
    print(f"    Generated {len(rollouts)} valid rollouts out of {num_rollouts} attempts")    
    
    # Step 3: Extract trajectories from rollouts
    rollout_trajectories = extract_trajectories_from_rollouts(rollouts)

    if not rollout_trajectories:
        print(f"    Warning: No dynamic agent trajectories found for frame {start_frame}")
        return [], None
           
    return rollout_trajectories, ground_truth

def generate_single_rollout_from_frame(env, policy, policy_model, scene_idx, start_frame, horizon, rollout_idx):
    """
    Generate a single rollout starting from current environment state with guidance like scene_editor.py
    """
    try:
        # Use the guided_rollout function from scene_edit_utils
        from tbsim.utils.scene_edit_utils import guided_rollout, compute_heuristic_guidance, merge_guidance_configs
        from tbsim.policies.wrappers import RolloutWrapper
        import tbsim.utils.tensor_utils as TensorUtils
        
        print(f"      Rollout {rollout_idx}: Starting guided_rollout with horizon={horizon}")
        
        # Wrap policy like scene_editor.py does
        rollout_policy = RolloutWrapper(agents_policy=policy)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Initialize guidance and constraint configs like scene_editor.py (lines 189-235)
        guidance_config = None
        constraint_config = None
        
        # Get the eval_cfg to determine guidance settings
        # You'll need to pass this from your main function or get it from env
        eval_cfg = getattr(env, 'eval_cfg', None)  # Add this to your setup function
        
        if eval_cfg is not None:
            # Determine guidance based on editing source like scene_editor.py
            heuristic_config = None
            
            if hasattr(eval_cfg, 'edits') and hasattr(eval_cfg.edits, 'editing_source'):
                if "heuristic" in eval_cfg.edits.editing_source:
                    # Use heuristic config
                    if eval_cfg.edits.heuristic_config is not None:
                        heuristic_config = eval_cfg.edits.heuristic_config
                    else:
                        heuristic_config = []
                
                # Getting edits from either config file or heuristics like scene_editor.py lines 189-235
                if "config" in eval_cfg.edits.editing_source:
                    guidance_config = eval_cfg.edits.guidance_config
                    constraint_config = eval_cfg.edits.constraint_config  
                
                if "heuristic" in eval_cfg.edits.editing_source and heuristic_config is not None:
                    # Get observation for heuristic guidance computation
                    ex_obs = env.get_observation()
                    obs_to_torch = eval_cfg.eval_class not in ["GroundTruth", "ReplayAction"]
                    
                    if obs_to_torch:
                        ex_obs = TensorUtils.to_torch(ex_obs, device=device, ignore_if_unspecified=True)
                    
                    # Compute heuristic guidance 
                    heuristic_guidance_cfg = compute_heuristic_guidance(
                        heuristic_config,
                        env,
                        [scene_idx],  # Single scene
                        [start_frame],  # Single start frame
                        example_batch=ex_obs['agents']
                    )
                    
                    # Check if heuristic determined valid guidance
                    if len(heuristic_config) > 0:
                        valid_scene_inds = []
                        for sci, sc_cfg in enumerate(heuristic_guidance_cfg):
                            if len(sc_cfg) > 0:
                                valid_scene_inds.append(sci)
                        
                        if len(valid_scene_inds) == 0:
                            print(f"      No valid heuristic configs for scene {scene_idx}, using no guidance")
                            heuristic_guidance_cfg = [[]]
                        else:
                            heuristic_guidance_cfg = [heuristic_guidance_cfg[vi] for vi in valid_scene_inds]
                    
                    # Merge guidance configs
                    guidance_config = merge_guidance_configs(guidance_config, heuristic_guidance_cfg)

        print(f"      Rollout {rollout_idx}: guidance_config type: {type(guidance_config)}, length: {len(guidance_config) if guidance_config else 'None'}")
        print(f"      Rollout {rollout_idx}: constraint_config type: {type(constraint_config)}, length: {len(constraint_config) if constraint_config else 'None'}")
        # Call guided_rollout
        stats, info, renderings = guided_rollout(
            env=env,
            policy=rollout_policy,
            policy_model=policy_model,
            n_step_action=5,  # You may want to get this from eval_cfg.n_step_action
            guidance_config=guidance_config,  # Pass computed guidance
            constraint_config=constraint_config,  # Pass computed constraints
            render=False,
            scene_indices=[scene_idx],
            device=device,
            obs_to_torch=True,  # You may want to get this from eval_cfg
            horizon=horizon,
            start_frames=[start_frame],
            eval_class='Diffuser',  # You may want to get this from eval_cfg.eval_class
        )
        
        # Check if info is None before accessing it
        if info is None:
            print(f"      Rollout {rollout_idx}: WARNING - info is None!")
            return None
        
        print(f"      Rollout {rollout_idx}: guided_rollout completed successfully")
        
        # Extract buffer with proper error handling
        buffer = None
        if isinstance(info, dict) and 'buffer' in info:
            if info['buffer'] is not None and len(info['buffer']) > 0:
                buffer = info['buffer'][0]  # First scene's buffer
                print(f"      Rollout {rollout_idx}: Successfully extracted buffer")
            else:
                print(f"      Rollout {rollout_idx}: Buffer is empty or None")
        else:
            print(f"      Rollout {rollout_idx}: No buffer key in info")
        
        return {
            'rollout_idx': rollout_idx,
            'stats': stats,
            'info': info,
            'buffer': buffer,
        }
        
    except Exception as e:
        print(f"      Error in rollout {rollout_idx}: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_trajectories_from_rollouts(rollouts):
    """
    Extract trajectory data from each rollout results
    """
    agents_trajectories_all_rollouts = []
    
    for rollout_idx, rollout in enumerate(rollouts):
        if rollout is None:
            continue
            
        buffer = rollout.get('buffer')
        if buffer is not None and isinstance(buffer, dict):
            # Extract agents trajectories from each rollout buffer
            agents_per_rollout = {}
            
            # Get agent IDs
            agent_ids = None
            if 'track_id' in buffer:
                track_ids = buffer['track_id']
                if hasattr(track_ids, 'cpu'):
                    agent_ids = track_ids.cpu().numpy()
                elif hasattr(track_ids, 'detach'):
                    agent_ids = track_ids.detach().numpy()
                else:
                    agent_ids = np.array(track_ids)
                    
            unique_agent_ids = agent_ids[:, 0]  # [num_agents, timesteps], get first column for unique IDs
            
            # Get centroid data
            if 'centroid' in buffer and agent_ids is not None:
                centroids = buffer['centroid']
                if hasattr(centroids, 'cpu'):
                    centroids = centroids.cpu().numpy()
                elif hasattr(centroids, 'detach'):
                    centroids = centroids.detach().numpy()
                else:
                    centroids = np.array(centroids)
                
                # Create dictionary with agent_id as key for a rollout
                unique_agent_ids = unique_agent_ids.tolist()
                for i, unique_agent_id in enumerate(unique_agent_ids):
                    if unique_agent_id not in agents_per_rollout:
                        agents_per_rollout[unique_agent_id] = centroids[i]

            if agents_per_rollout:
                agents_trajectories_all_rollouts.append(agents_per_rollout)
                print(f"      Extracted trajectories for {len(agents_per_rollout)} agents from rollout {rollout_idx}")

    return agents_trajectories_all_rollouts

def extract_ground_truth_trajectory(env, scene_idx, start_frame, horizon):
    """
    Extract ground truth for the current scene and start frame
    
    Args:
        env: Environment instance
        scene_idx: Scene index
        start_frame: Starting frame
        horizon: Number of frames
    """
    try:      
        if not (hasattr(env, '_current_scenes') and len(env._current_scenes) > 0):
            return None
        
        current_scene = env._current_scenes[0]
        
        # Get agent names and agent objects
        if hasattr(current_scene, 'agents'):
            agents = current_scene.agents
            agent_names = [agent.name for agent in agents]
        else:
            print("      Cannot access agent names")
            return None
        
        gt_trajectories = {}
        
        # Step 1: Extract gt trajectories for each agent
        for agent_idx, (agent, agent_name) in enumerate(zip(agents, agent_names)):           
            agent_positions = []
            
            for frame_offset in range(horizon):
                frame_idx = start_frame + frame_offset

                # Check if agent is active in this frame using the agent object
                if not (agent.first_timestep <= frame_idx < agent.last_timestep):
                    # Agent is not active in this frame, so we can safely skip it
                    continue
                    
                # Use the dataset's method to get agent states
                if hasattr(current_scene, 'cache'):
                    states = current_scene.cache.get_states([agent_name], frame_idx)
                else:
                    # Fallback: try dataset directly
                    states = current_scene.dataset.get_states([agent_name], frame_idx)
                
                # Process valid states
                if states is not None and hasattr(states, '__len__') and len(states) > 0:
                    state = states[0]
                    # Handle tensor conversion
                    if hasattr(state, 'cpu'):
                        state_np = state.cpu().numpy()
                    elif hasattr(state, 'detach'):
                        state_np = state.detach().numpy()
                    else:
                        state_np = np.array(state)
                    
                    # Extract x, y position
                    agent_positions.append([float(state_np[0]), float(state_np[1])])
                else:
                    # No data returned, skip frame
                    continue

            # Only include agents with sufficient trajectory data
            if len(agent_positions) >= 2:
                gt_trajectories[agent_idx] = np.array(agent_positions)                 
            else:
                print(f"      Agent {agent_name} (ID {agent_idx}): insufficient data ({len(agent_positions)} positions)")
                
        return gt_trajectories
        
    except Exception as e:
        print(f"      Error using dataset get_states: {e}")
        return None

def process_frame_trajectories(scene_idx, scene_name, frame_number, rollout_trajectories, ground_truth, 
                               filter_dynamic=False, min_velocity_threshold=2.0, debug=False):
    """
    Process trajectories from a specific frame and compute features with agent matching
    """
    if not rollout_trajectories or ground_truth is None:
        return None
    
    # Step 1: Filter for dynamic agents for feature computation
    dynamic_agent_ids = []
    for agent_idx, trajectory in ground_truth.items():
        if is_trajectory_dynamic(trajectory, min_velocity_threshold=min_velocity_threshold):
            dynamic_agent_ids.append(agent_idx)

    print(f"    Filtered to {len(dynamic_agent_ids)} dynamic agents out of {len(ground_truth)} total")

    # Compute features for all rollouts (per dynamic agent)
    dynamic_agents_features = {}
    for one_rollout in rollout_trajectories:
        agent_features = compute_irl_features(one_rollout, dynamic_agent_ids, dt=0.1)
        for agent_id, features in agent_features.items():
            if agent_id not in dynamic_agents_features:
                dynamic_agents_features[agent_id] = []
            dynamic_agents_features[agent_id].append(features)
    print(f"    Computed features for {len(dynamic_agents_features)} dynamic agents across {len(rollout_trajectories)} rollouts")
    
    # Compute features for ground truth (per dynamic agent)
    dynamic_gt_features = compute_irl_features(ground_truth, dynamic_agent_ids, dt=0.1)
        
    # Add visualization at the end if debug is enabled
    if debug:
        from visualize_rollout_gt import visualize_guided_rollout_with_gt
        plot_output_dir = os.path.join("irl_features_output", "visualization")
        
        try:
            if not filter_dynamic:
                dynamic_agent_ids = None  # No filtering, use all agents
                
            visualize_guided_rollout_with_gt(
                rollout_trajectories=rollout_trajectories,
                ground_truth=ground_truth,
                scene_idx=scene_idx,
                start_frame=frame_number,
                scene_name=scene_name,
                output_dir=plot_output_dir,
                dynamic_agent_ids=dynamic_agent_ids,
            )

        except Exception as e:
            print(f"    Visualization warning: {e}") 
       
    return {
        'agent_rollout_features': dynamic_agents_features,
        'agent_ground_truth_features': dynamic_gt_features
    }

def compute_irl_features(agent_trajectories_dict, dynamic_agent_ids, dt=0.1):
    """
    Compute IRL features for agent dictionary format
    Args:
        agent_trajectories_dict: Dict with agent_id as key, trajectory [time_steps, 2] as value
        dynamic_agent_ids: List of dynamic agent IDs to compute features for
        dt: time step in seconds
    Returns:
        dict with agent_id as key, features dict as value
    """
    agent_features = {}
    
    # Get all agent positions for relative computations (THW)
    all_agent_data = {}
    for agent_id, traj in agent_trajectories_dict.items():
        all_agent_data[agent_id] = traj
    
    for agent_id, traj in agent_trajectories_dict.items():
        if agent_id not in dynamic_agent_ids:
            continue

        num_steps = len(traj)
        features = {}
        
        # Initialize feature arrays
        features['velocity'] = np.zeros(num_steps-1)
        features['a_long'] = np.zeros(max(0, num_steps-2))
        features['jerk_long'] = np.zeros(max(0, num_steps-3))
        features['a_lateral'] = np.zeros(max(0, num_steps-2))
        features['thw_front'] = np.zeros(num_steps-1)
        features['thw_rear'] = np.zeros(num_steps-1)
        
        # Compute velocity (magnitude of velocity vector)
        vel_vectors = np.diff(traj, axis=0) / dt  # [time_steps-1, 2]
        velocities = np.linalg.norm(vel_vectors, axis=1)  # [time_steps-1]
        features['velocity'] = velocities
        
        # Compute heading angles from velocity vectors
        headings = np.arctan2(vel_vectors[:, 1], vel_vectors[:, 0])  # [time_steps-1]
        
        # Compute longitudinal acceleration
        if len(velocities) > 1:
            a_long = np.diff(velocities) / dt  # [time_steps-2]
            features['a_long'] = a_long
            
            # Compute longitudinal jerk
            if len(a_long) > 1:
                jerk_long = np.diff(a_long) / dt  # [time_steps-3]
                features['jerk_long'] = jerk_long
        
        # Compute lateral acceleration (change in heading * velocity)
        if len(headings) > 1 and len(velocities) > 1:
            heading_rates = np.diff(headings) / dt  # [time_steps-2]
            # Use velocity at midpoint for lateral acceleration calculation
            mid_velocities = (velocities[:-1] + velocities[1:]) / 2  # [time_steps-2]
            a_lateral = heading_rates * mid_velocities
            features['a_lateral'] = a_lateral
        
        # Compute Time Headway (THW) - distance to front/rear vehicles divided by velocity
        for t in range(len(velocities)):
            if velocities[t] > 0.1:  # Only compute if vehicle is moving
                current_pos = traj[t]
                
                # Calculate distances to all other agents at this timestep
                distances = {}
                for other_agent_id, other_traj in all_agent_data.items():
                    if other_agent_id != agent_id and t < len(other_traj):
                        other_pos = other_traj[t]
                        dist = np.linalg.norm(other_pos - current_pos)
                        distances[other_agent_id] = dist
                
                if distances and t < len(headings):
                    heading = headings[t]
                    heading_vector = np.array([np.cos(heading), np.sin(heading)])
                    
                    front_distances = []
                    rear_distances = []
                    
                    for other_agent_id, dist in distances.items():
                        if dist < 50.0:  # Within 50m
                            other_pos = all_agent_data[other_agent_id][t]
                            relative_vector = other_pos - current_pos
                            along_heading = np.dot(relative_vector, heading_vector)
                            
                            if along_heading > 0:  # In front
                                front_distances.append(dist)
                            else:  # Behind
                                rear_distances.append(dist)
                    
                    # Find closest in front and behind
                    if front_distances:
                        closest_front_dist = min(front_distances)
                        features['thw_front'][t] = closest_front_dist / velocities[t]
                    else:
                        features['thw_front'][t] = 10.0  # Large value if no front vehicle
                        
                    if rear_distances:
                        closest_rear_dist = min(rear_distances)
                        features['thw_rear'][t] = closest_rear_dist / velocities[t]
                    else:
                        features['thw_rear'][t] = 10.0  # Large value if no rear vehicle
                else:
                    features['thw_front'][t] = 10.0
                    features['thw_rear'][t] = 10.0
            else:
                features['thw_front'][t] = 10.0
                features['thw_rear'][t] = 10.0
        
        agent_features[agent_id] = features
    
    return agent_features

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
    
    # Extract policy_model
    policy_model = None
    print('policy', policy)
    if hasattr(policy, 'model'):
        policy_model = policy.model
    
    # Set evaluation time sampling/optimization parameters - like scene_editor.py
    if eval_cfg.apply_guidance:
        if eval_cfg.eval_class in ['SceneDiffuser', 'Diffuser', 'TrafficSim', 'BC', 'HierarchicalSampleNew']:
            if policy_model is not None:
                policy_model.set_guidance_optimization_params(eval_cfg.guidance_optimization_params)
        if eval_cfg.eval_class in ['SceneDiffuser', 'Diffuser']:
            if policy_model is not None:
                policy_model.set_diffusion_specific_params(eval_cfg.diffusion_specific_params)
    
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
    env.eval_cfg = eval_cfg
    
    return env, policy, policy_model


def is_trajectory_dynamic(trajectory, min_velocity_threshold=2.0, dt=0.1, min_distance_threshold=5.0):
    """
    Check if a trajectory represents dynamic behavior using multiple criteria
    """
    if len(trajectory) < 2:
        return False
    
    # Compute velocities
    vel_vectors = np.diff(trajectory, axis=0) / dt
    velocities = np.linalg.norm(vel_vectors, axis=1)
    
    # Criterion 1: Maximum velocity
    max_velocity = np.max(velocities)
    if max_velocity > min_velocity_threshold:
        return True
    
    # Criterion 2: Average velocity
    avg_velocity = np.mean(velocities)
    if avg_velocity > min_velocity_threshold * 0.5:
        return True
    
    # Criterion 3: Total distance traveled
    total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
    if total_distance > min_distance_threshold:
        return True
    
    # Criterion 4: End-to-end displacement
    displacement = np.linalg.norm(trajectory[-1] - trajectory[0])
    if displacement > min_distance_threshold * 0.5:
        return True
    
    return False


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

    parser.add_argument(
        "--editing_source",
        type=str,
        choices=["config", "heuristic", "none"],
        default=["config", "heuristic"],
        nargs="+",
        help="Which edits to use. config is directly from the configuration file. heuristic will \
              set edits automatically based on heuristics. If none, does not use edits."
    )

    args = parser.parse_args()    
    cfg = SceneEditingConfig(registered_name=args.registered_name)
    
    # Set evaluation class
    if args.eval_class is not None:
        cfg.eval_class = args.eval_class
    # Set dataset path
    if args.dataset_path is not None:
        cfg.dataset_path = args.dataset_path   
    
    # Set environment
    if args.env is not None:
        cfg.env = args.env
    
    if args.editing_source is not None:
        cfg.edits.editing_source = args.editing_source
    if not isinstance(cfg.edits.editing_source, list):
        cfg.edits.editing_source = [cfg.edits.editing_source]      
        
    # Copy env-specific config to global level 
    for k in cfg[cfg.env]:
        cfg[k] = cfg[cfg.env][k]
    
    # Remove env-specific sections
    cfg.pop("nusc", None)
    cfg.pop("trajdata", None)
    
    # Set checkpoint paths   
    cfg.ckpt.policy.ckpt_dir = default_config.policy_ckpt_dir
    cfg.ckpt.policy.ckpt_key = default_config.policy_ckpt_key
    # Set results directory to match your output directory
    cfg.results_dir = default_config.output_dir
    
    try:
        # Setup environment and model
        print("Setting up environment and model...")
        env, policy, policy_model = setup_from_scene_editor_config(cfg)
        
        # Extract features from all frames with automatic start frame determination
        print("Starting feature extraction...")
        features = extract_irl_features_from_all_frames(
            env, policy, policy_model, 
            default_config.scene_indices, 
            default_config.start_frames, 
            default_config.output_dir, 
            default_config.horizon,
            default_config.num_sim_per_scene,
            default_config.num_rollouts,
            default_config.filter_dynamic,
            default_config.min_velocity_threshold,
            default_config.debug
        )
        
        print("Feature extraction from all frames complete!")
        print(f"Processed {len(features)} scene-start_frame combinations")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

