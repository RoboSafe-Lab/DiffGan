import numpy as np
import os
import torch
import random
import importlib
import pickle
from .irl_config import default_config
from .visualize_rollout_gt import visualize_guided_rollout_with_gt, visualize_trajectories_simple
from tbsim.configs.scene_edit_config import SceneEditingConfig
from tbsim.evaluation.env_builders import EnvNuscBuilder, EnvUnifiedBuilder
from tbsim.policies.wrappers import RolloutWrapper
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env, set_global_trajdata_batch_raster_cfg
from tbsim.utils.scene_edit_utils import guided_rollout, compute_heuristic_guidance, merge_guidance_configs
import tbsim.utils.tensor_utils as TensorUtils
            
class IRLFeatureExtractor:
    def __init__(self, eval_cfg, config=default_config):
        self.config = config
        self.env = None
        self.policy = None
        self.policy_model = None
        self._setup_from_scene_editor_config(eval_cfg)
        
    def _setup_from_scene_editor_config(self, eval_cfg):
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
        
        # Set evaluation time sampling/optimization parameters
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
        
        self.env = env
        self.policy = policy
        self.policy_model = policy_model


    def extract_irl_features_from_all_frames(self):
        """
        Extract IRL features by generating rollouts from ALL frames, not just start frame
        """
        print("Extracting IRL features from ALL frames...")
        
        all_features = {}
        scene_i = 0
        eval_scenes = self.config.eval_scenes

        while scene_i < self.config.num_scenes_to_evaluate:
            scene_indices = eval_scenes[scene_i: scene_i + self.config.num_scenes_per_batch]
            scene_i += self.config.num_scenes_per_batch
            print(f'Processing scene_indices: {scene_indices}')
            
            # Reset environment to get scene information
            scenes_valid = self.env.reset(scene_indices=scene_indices, start_frame_index=None)
            scene_indices = [si for si, sval in zip(scene_indices, scenes_valid) if sval]
            
            if len(scene_indices) == 0:
                print('No valid scenes in this batch, skipping...')
                torch.cuda.empty_cache()
                continue
        
            # Determine start frames for each scene
            start_frame_index: list[list[int]] = []
            valid_scene_indices = []
            valid_scene_wrappers = [] 
            
            # History frames needed by policy
            history_frames = getattr(self.env.exp_config.algo, "history_num_frames", 0)

            # Multiple sims per scene: spread starts across valid range
            for si_idx, scene_idx in enumerate(scene_indices):
                print(f"Processing scene {scene_idx} (index {si_idx})")
                
                current_scene = self.env._current_scenes[si_idx].scene
                sframe = history_frames + 1
                # Ensure there's enough horizon for rollout
                eframe = current_scene.length_timesteps - self.config.horizon
                if eframe <= sframe:
                    print(f"Scene {scene_idx}: insufficient length (length={current_scene.length_timesteps}, need at least {sframe + self.config.horizon})")
                    continue
                valid_scene_indices.append(scene_idx)
                valid_scene_wrappers.append(self.env._current_scenes[si_idx]) 

                if self.config.num_sim_per_scene > 1:
                    # Multiple simulations per scene - spread them across the scene
                    scene_frame_inds = np.linspace(sframe, eframe, num=self.config.num_sim_per_scene, dtype=int).tolist()
                    start_frame_index.append(scene_frame_inds)
                    print(f"Scene {scene_idx}: frames {sframe} to {eframe}, selected starts: {scene_frame_inds}")          
                else:
                    # Single sim per scene: default start
                    start_frame_index.append([sframe])
                    print(f"Scene {scene_idx}: using start frame {sframe}")
                    
            # Update scene_indices to only include valid scenes
            scene_indices = valid_scene_indices
            self.env._current_scenes = valid_scene_wrappers
            
            # Now scene_indices and start_frame_index have the same length
            assert len(scene_indices) == len(start_frame_index), f"Mismatch: {len(scene_indices)} scenes vs {len(start_frame_index)} start_frame entries"
                    
            # Process each scene with its determined start frames
            for si_idx, scene_idx in enumerate(scene_indices):
                scene_features = []    
                scene_start_frames = start_frame_index[si_idx]
                scene_name = self.env._current_scenes[si_idx].scene.name 

                for start_frame in scene_start_frames:
                    print(f"\nProcessing scene {scene_idx}, start frame {start_frame}")
                    
                    # Reset to this specific scene and start frame
                    scenes_valid = self.env.reset(scene_indices=[scene_idx], start_frame_index=[start_frame])
                    if not scenes_valid[0]:
                        print(f"Scene {scene_idx} invalid at start frame {start_frame}, skipping...")
                        torch.cuda.empty_cache()
                        continue

                    # Generate rollouts starting from this specific frame
                    rollout_trajectories, gt_trajectories = self._generate_rollouts_from_specific_frame(
                        scene_idx, start_frame)                    

                    if not rollout_trajectories or gt_trajectories is None:
                        print(f"    Warning: No data generated for frame {start_frame}")
                        torch.cuda.empty_cache()
                        continue
                                
                    frame_features_data = self._process_frame_trajectories(
                        scene_idx, scene_name, start_frame, rollout_trajectories, gt_trajectories)
                    
                    if frame_features_data:
                        scene_features.append({
                            "start_frame": start_frame,
                            "frame_features": frame_features_data
                        })

                # return features for current scene, and save if necessary
                if scene_features:
                    all_features[f"{scene_idx}"] = scene_features
                    
                    if self.config.save_features:
                        features_output_dir = os.path.join(self.config.output_dir, "features")
                        if not os.path.exists(features_output_dir):
                            os.makedirs(features_output_dir)
                        output_path = os.path.join(features_output_dir, f"{scene_name}_irl_features.pkl")
                        with open(output_path, 'wb') as f:
                            pickle.dump(scene_features, f)

                        print(f"Saved {len(scene_features)} frame features to {output_path}")                    
                    
            torch.cuda.empty_cache()
        return all_features


    def _generate_rollouts_from_specific_frame(self, scene_idx, start_frame):
        """
        Generate multiple rollouts starting from a SPECIFIC frame
        """
        # Step 1: Extract ground truth trajectory
        print(f"    Step 1: Extracting ground truth trajectory for scene {scene_idx}, frame {start_frame}")
        ground_truth = self._extract_ground_truth_trajectory(start_frame)

        if ground_truth is None or len(ground_truth) == 0:
            print(f"    No dynamic agents found in ground truth, skipping rollouts")
            return None, None
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
        # Initialize guidance and constraint configs like scene_editor.py (lines 189-235)
        guidance_config = None
        constraint_config = None
        
        # Get the eval_cfg to determine guidance settings
        eval_cfg = getattr(self.env, 'eval_cfg', None)  
        obs_to_torch = eval_cfg.eval_class not in ["GroundTruth", "ReplayAction"]
        if eval_cfg is not None:
                # Determine guidance based on editing source
                heuristic_config = []
                
                if hasattr(eval_cfg, 'edits') and hasattr(eval_cfg.edits, 'editing_source'):
                    if "heuristic" in eval_cfg.edits.editing_source:
                        # Use heuristic config
                        if eval_cfg.edits.heuristic_config is not None:
                            heuristic_config = eval_cfg.edits.heuristic_config
                    
                    # Getting edits from either config file or heuristics
                    if "config" in eval_cfg.edits.editing_source:
                        guidance_config = eval_cfg.edits.guidance_config
                        constraint_config = eval_cfg.edits.constraint_config  
                    
                    if "heuristic" in eval_cfg.edits.editing_source and heuristic_config is not None:
                        # Get observation for heuristic guidance computation
                        ex_obs = self.env.get_observation()                        

                        if obs_to_torch:
                            ex_obs = TensorUtils.to_torch(ex_obs, device=device, ignore_if_unspecified=True)
                        
                        # Compute heuristic guidance 
                        heuristic_guidance_cfg = compute_heuristic_guidance(
                            heuristic_config,
                            self.env,
                            [scene_idx],  # Single scene
                            [start_frame],  # Single start frame
                            example_batch=ex_obs['agents']
                        )
                        # Keep only scenes with at least one guidance
                        valid_scene_inds = [i for i, sc in enumerate(heuristic_guidance_cfg) if len(sc) > 0]
                        heuristic_guidance_cfg = [heuristic_guidance_cfg[i] for i in valid_scene_inds] if len(valid_scene_inds) > 0 else [[]]
                        guidance_config = merge_guidance_configs(guidance_config, heuristic_guidance_cfg)
        
        print(f"      Rollouts: guidance_config type: {type(guidance_config)}, length: {len(guidance_config) if guidance_config else 'None'}")
        print(f"      Rollouts: constraint_config type: {type(constraint_config)}, length: {len(constraint_config) if constraint_config else 'None'}")

        # Step 2: Generate rollouts
        rollouts = []
        # Generate multiple rollouts from this frame
        print(f"    Generating {self.config.num_rollouts} rollouts from frame {start_frame}")

        for rollout_idx in range(self.config.num_rollouts):
            # Reset environment to this specific frame
            scenes_valid = self.env.reset(scene_indices=[scene_idx], start_frame_index=[start_frame])
            if not scenes_valid[0]:
                print(f"      Scene {scene_idx} invalid at frame {start_frame} for rollout {rollout_idx}")
                continue
            
            # Add variation for different rollouts
            if rollout_idx > 0:
                np.random.seed(42 + rollout_idx + start_frame)
                torch.manual_seed(42 + rollout_idx + start_frame)
            
            # Generate single rollout from this frame
            rollout_data = self._generate_single_rollout_from_frame(
                scene_idx, start_frame, rollout_idx,
                guidance_config=guidance_config,
                constraint_config=constraint_config,
                obs_to_torch=obs_to_torch,
                device=device
            )

            if rollout_data is not None:
                rollouts.append(rollout_data)
            else:
                print(f"      Failed to generate rollout {rollout_idx}")

        print(f"    Generated {len(rollouts)} valid rollouts out of {self.config.num_rollouts} attempts")
        # Step 3: Extract trajectories from rollouts
        rollout_trajectories = self._extract_trajectories_from_rollouts(rollouts)

        if not rollout_trajectories:
            print(f"    Warning: No dynamic agent trajectories found for frame {start_frame}")
            return [], None
            
        return rollout_trajectories, ground_truth

    def _generate_single_rollout_from_frame(self, scene_idx, start_frame, rollout_idx,
                                            guidance_config=None, constraint_config=None,
                                            obs_to_torch=None, device=None):
        """
        Generate a single rollout starting from current environment state with guidance like scene_editor.py
        """
        try:
            # Use the guided_rollout function from scene_edit_utils            
            print(f"      Rollout {rollout_idx}: Starting guided_rollout with horizon={self.config.horizon}")
            
            # Wrap policy like scene_editor.py does
            rollout_policy = RolloutWrapper(agents_policy=self.policy)            
            eval_cfg = getattr(self.env, 'eval_cfg', None)  
            
            # Call guided_rollout
            stats, info, renderings = guided_rollout(
                env=self.env,
                policy=rollout_policy,
                policy_model=self.policy_model,
                n_step_action=eval_cfg.n_step_action, 
                guidance_config=guidance_config,  # Pass computed guidance
                constraint_config=constraint_config,  # Pass computed constraints
                render=False,
                scene_indices=[scene_idx],
                device=device,
                obs_to_torch=obs_to_torch, 
                horizon=self.config.horizon,
                start_frames=[start_frame],
                eval_class='Diffuser',  # You may want to get this from eval_cfg.eval_class
                apply_guidance=eval_cfg.apply_guidance
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

    def _extract_trajectories_from_rollouts(self, rollouts):
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
                agent_ids = np.asarray(buffer['track_id'])
                        
                unique_agent_ids = agent_ids[:, 0]  # [num_agents, timesteps], get first column for unique IDs
                
                # Get centroid data
                if 'centroid' in buffer:
                    centroids = np.asarray(buffer['centroid'])
                    
                    # Create dictionary with agent_id as key for a rollout
                    unique_agent_ids = unique_agent_ids.tolist()
                    for i, unique_agent_id in enumerate(unique_agent_ids):
                        if unique_agent_id not in agents_per_rollout:
                            agents_per_rollout[unique_agent_id] = centroids[i]

                if agents_per_rollout:
                    agents_trajectories_all_rollouts.append(agents_per_rollout)
                    print(f"      Extracted trajectories for {len(agents_per_rollout)} agents from rollout {rollout_idx}")

        return agents_trajectories_all_rollouts

    def _extract_ground_truth_trajectory(self, start_frame):
        """
        Extract ground truth from cached trajdata
        """
        try:      
            if not (hasattr(self.env, '_current_scenes') and len(self.env._current_scenes) > 0):
                return None

            current_scene = self.env._current_scenes[0]

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

                for frame_offset in range(self.config.horizon):
                    frame_idx = start_frame + frame_offset

                    # Check if agent is active in this frame using the agent object
                    if not (agent.first_timestep <= frame_idx < agent.last_timestep):
                        # Agent is not active in this frame, so we can safely skip it
                        continue
                        
                    # Use the dataset's method to get agent states
                    states = current_scene.cache.get_states([agent_name], frame_idx)
                    
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
            print(f"      Error extracting GT from cache: {e}")
            return None

    def _process_frame_trajectories(self, scene_idx, scene_name, frame_number, rollout_trajectories, ground_truth):
        """
        Process trajectories from a specific frame and compute features with agent matching
        """
        if not rollout_trajectories or ground_truth is None:
            return None
        
        # Step 1: Filter for dynamic agents for feature computation
        dynamic_agent_ids = []
        for agent_idx, trajectory in ground_truth.items():
            if self.is_trajectory_dynamic(trajectory):
                dynamic_agent_ids.append(agent_idx)

        print(f"    Filtered to {len(dynamic_agent_ids)} dynamic agents out of {len(ground_truth)} total")

        # Compute features for all rollouts (per dynamic agent)
        dynamic_agents_features = {}
        for one_rollout in rollout_trajectories:
            agent_features = self.compute_irl_features(one_rollout, dynamic_agent_ids)
            for agent_id, features in agent_features.items():
                if agent_id not in dynamic_agents_features:
                    dynamic_agents_features[agent_id] = []
                dynamic_agents_features[agent_id].append(features)
        print(f"    Computed features for {len(dynamic_agents_features)} dynamic agents across {len(rollout_trajectories)} rollouts")

        # Compute features for ground truth (per dynamic agent)
        dynamic_gt_features = self.compute_irl_features(ground_truth, dynamic_agent_ids)

        # Add visualization at the end if debug is enabled
        if self.config.debug:        
            plot_output_dir = os.path.join(self.config.output_dir, "visualization")
            
            try:
                if not self.config.filter_dynamic:
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

    def compute_irl_features(self, agent_trajectories_dict, dynamic_agent_ids):
        """
        Compute IRL features for agent dictionary format
        Args:
            agent_trajectories_dict: Dict with agent_id as key, trajectory [time_steps, 2] as value
            dynamic_agent_ids: List of dynamic agent IDs to compute features for
        Returns:
            dict with agent_id as key, features dict as value
        """
        feature_names = self.config.feature_names
        dt = self.config.step_time
        
        agent_features = {}
        
        # Get all agent positions for relative computations (THW)
        all_agent_data = {}
        for agent_id, traj in agent_trajectories_dict.items():
            all_agent_data[agent_id] = traj
        
        for agent_id, traj in agent_trajectories_dict.items():
            if agent_id not in dynamic_agent_ids:
                continue

            features = {}
            vel_vectors = np.diff(traj, axis=0) / dt
            velocities = np.linalg.norm(vel_vectors, axis=1) if len(vel_vectors) > 0 else np.zeros(0)
            headings = np.arctan2(vel_vectors[:, 1], vel_vectors[:, 0]) if len(vel_vectors) > 0 else np.zeros(0)

            if 'velocity' in feature_names:
                features['velocity'] = velocities

            a_long = None
            if 'a_long' in feature_names:
                a_long = np.diff(velocities) / dt if len(velocities) > 1 else np.zeros(0)
                features['a_long'] = a_long

            if 'jerk_long' in feature_names:
                if a_long is None:
                    a_long = np.diff(velocities) / dt if len(velocities) > 1 else np.zeros(0)
                jerk_long = np.diff(a_long) / dt if len(a_long) > 1 else np.zeros(0)
                features['jerk_long'] = jerk_long

            if 'a_lateral' in feature_names:
                if len(headings) > 1 and len(velocities) > 1:
                    heading_rates = np.diff(headings) / dt
                    mid_velocities = (velocities[:-1] + velocities[1:]) / 2
                    a_lateral = heading_rates * mid_velocities
                else:
                    a_lateral = np.zeros(max(0, len(velocities) - 1))
                features['a_lateral'] = a_lateral

            if 'min_dis' in feature_names:
                min_dis = np.zeros(len(velocities))
                for t in range(len(velocities)):
                    current_pos = traj[t]
                    md = float('inf')
                    for other_id, other_traj in all_agent_data.items():
                        if other_id == agent_id or t >= len(other_traj):
                            continue
                        dist = np.linalg.norm(other_traj[t] - current_pos)
                        if dist < md:
                            md = dist
                    min_dis[t] = md if md != float('inf') else 50.0
                features['min_dis'] = min_dis
            
            agent_features[agent_id] = features
        
        return agent_features    

    def is_trajectory_dynamic(self, trajectory):
        """
        Check if a trajectory represents dynamic behavior using multiple criteria
        """
        if len(trajectory) < 2:
            return False
        dt = self.config.step_time
        
        # Compute velocities
        vel_vectors = np.diff(trajectory, axis=0) / dt
        velocities = np.linalg.norm(vel_vectors, axis=1)
        
        # Criterion 1: Maximum velocity
        max_velocity = np.max(velocities)
        if max_velocity > self.config.min_velocity_threshold:
            return True
        
        # Criterion 2: Average velocity
        avg_velocity = np.mean(velocities)
        if avg_velocity > self.config.min_velocity_threshold * 0.5:
            return True
        
        # Criterion 3: Total distance traveled
        total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        if total_distance > self.config.min_distance_threshold:
            return True
        
        # Criterion 4: End-to-end displacement
        displacement = np.linalg.norm(trajectory[-1] - trajectory[0])
        if displacement > self.config.min_distance_threshold * 0.5:
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
        extractor = IRLFeatureExtractor(cfg, default_config)
        
        # Extract features from all frames with automatic start frame determination
        print("Starting feature extraction...")
        features = extractor.extract_irl_features_from_all_frames()        
        print("Feature extraction from all frames complete!")
        print(f"Processed {len(features)} scene-start_frame combinations")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

