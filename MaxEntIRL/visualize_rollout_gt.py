def transform_points_to_raster(points, raster_from_world):
    """Transform world coordinates to raster coordinates"""
    if len(points.shape) == 2:  # [N, 2] or [N, features]
        # Ensure we only use x,y coordinates
        if points.shape[1] >= 2:
            xy_points = points[:, :2]  # Take only x,y
        else:
            print(f"Warning: points shape {points.shape} has less than 2 coordinates")
            return points
            
        # Add homogeneous coordinate
        points_h = np.concatenate([xy_points, np.ones((xy_points.shape[0], 1))], axis=1)
        # Transform points  
        raster_points = points_h @ raster_from_world.T
        return raster_points[:, :2]
    else:
        # Handle batch of trajectories
        return np.array([transform_points_to_raster(traj, raster_from_world) for traj in points])

# Custom visualization functions to overlay ground truth
def draw_ground_truth_trajectories(ax, scene_data, starting_frame, raster_from_world, traj_len=200, render_rasterizer=None, scene_name=None):
    """Draw ground truth trajectories if available in scene data"""
    t = starting_frame
    
    # First check if we have ground truth data saved directly in the scene buffer (preferred method)
    gt_trajectories = None
    gt_label = "Ground Truth"
    
    # Check for ground truth future trajectories (saved directly in HDF5)
    if "gt_future_positions" in scene_data:
        gt_trajectories = scene_data["gt_future_positions"]
        gt_label = "Ground Truth Future"
        print(f"Found saved ground truth future positions with shape: {gt_trajectories.shape}")
        print(f"Current frame: {t}, Trajectory starting from current agent position")
    # Check for ground truth history trajectories
    elif "gt_history_positions" in scene_data:
        gt_trajectories = scene_data["gt_history_positions"] 
        gt_label = "Ground Truth History"
        print(f"Found saved ground truth history positions with shape: {gt_trajectories.shape}")
    else:
        # Fallback: Try different possible fields for ground truth data from the scene buffer
        possible_gt_fields = ["target_positions", "history_positions", "all_other_agents_future_positions", "agent_fut_positions"]
        
        for field_name in possible_gt_fields:
            if field_name in scene_data and scene_data[field_name] is not None:
                if field_name == "history_positions":
                    # For history, we want to show past trajectory leading up to current frame
                    if t > 0:
                        gt_trajectories = scene_data[field_name][:, max(0, t-traj_len):t+1]
                        gt_label = "History"
                        break
                else:
                    # For future positions, show trajectory from current frame forward
                    gt_trajectories = scene_data[field_name][:, t:t+traj_len]
                    gt_label = "Ground Truth Future"
                    break
    
    if gt_trajectories is not None and len(gt_trajectories) > 0:
        try:
            # Get current agent positions for validation
            current_positions = scene_data["centroid"][:, t]  # [num_agents, 2]
            print(f"Current agent positions at frame {t}: {current_positions[:2]}")  # Show first 2 agents
            
            # Transform ground truth to raster coordinates  
            raster_trajs = []
            
            # Handle different trajectory shapes
            if len(gt_trajectories.shape) == 3:  # [num_agents, time_steps, 2]
                for i in range(gt_trajectories.shape[0]):
                    traj = gt_trajectories[i]  # [time_steps, 2]
                    if traj.shape[1] >= 2:  # Ensure we have x,y coordinates
                        print(f"Agent {i} GT trajectory starts at: {traj[0, :2]}")
                        raster_traj = transform_points_to_raster(traj[:, :2], raster_from_world)
                        raster_trajs.append(raster_traj)
            elif len(gt_trajectories.shape) == 4:  # [batch, num_agents, time_steps, 2]  
                # Take first batch for now
                batch_trajs = gt_trajectories[0]
                for i in range(batch_trajs.shape[0]):
                    traj = batch_trajs[i]
                    if traj.shape[1] >= 2:
                        print(f"Agent {i} GT trajectory starts at: {traj[0, :2]}")
                        raster_traj = transform_points_to_raster(traj[:, :2], raster_from_world)
                        raster_trajs.append(raster_traj)
            
            # Draw ground truth trajectories in dashed lines with different color
            gt_drawn = False
            for i, traj in enumerate(raster_trajs):
                if len(traj) > 0 and not np.all(np.isnan(traj)):
                    # Filter out NaN values
                    valid_mask = ~np.any(np.isnan(traj), axis=1)
                    if np.any(valid_mask):
                        valid_traj = traj[valid_mask]
                        if len(valid_traj) > 1:  # Only draw if we have multiple points
                            ax.plot(valid_traj[:, 0], valid_traj[:, 1], 
                                   linestyle='--', 
                                   linewidth=2.0, 
                                   alpha=0.9, 
                                   color='red',
                                   label=gt_label if i == 0 else '')
                            gt_drawn = True
            return gt_drawn
        except Exception as e:
            print(f"Error drawing ground truth trajectories: {e}")
            import traceback
            print(traceback.format_exc())
            return False
    
        
    return False

def visualize_guided_rollout_with_gt(output_dir, rasterizer, si, scene_data,
                            guidance_config=None,
                            constraint_config=None,
                            filter_yaw=False,
                            fps=10,
                            n_step_action=5,
                            viz_diffusion_steps=False,
                            first_frame_only=False,
                            sim_num=0,
                            save_every_n_frames=1,
                            draw_mode='action',
                            draw_ground_truth=False):
    """
    Custom version of visualize_guided_rollout that can overlay ground truth trajectories
    """
    from tbsim.utils.scene_edit_utils import preprocess, scene_diffusion_video
    
    if viz_diffusion_steps:
        print('Visualizing diffusion for %s...' % (si))
        scene_diffusion_video(rasterizer, scene_data, si, output_dir,
                                n_step_action=n_step_action,
                                viz_prof=False,
                                viz_traj=True)
    print('Visualizing rollout for %s...' % (si))
    
    # Use custom scene_to_video function that supports ground truth
    scene_to_video_with_gt(rasterizer, scene_data, si, output_dir,
                    guidance_config=guidance_config,
                    constraint_config=constraint_config,
                    filter_yaw=filter_yaw,
                    fps=fps,
                    n_step_action=n_step_action,
                    first_frame_only=first_frame_only,
                    sim_num=sim_num,
                    save_every_n_frames=save_every_n_frames,
                    draw_mode=draw_mode,
                    draw_ground_truth=draw_ground_truth)

def scene_to_video_with_gt(rasterizer, scene_data, scene_name, output_dir,
                    guidance_config=None,
                    constraint_config=None,
                    filter_yaw=False,
                    fps=10,
                    n_step_action=5,
                    first_frame_only=False,
                    sim_num=0,
                    save_every_n_frames=1,
                    draw_mode='action',
                    draw_ground_truth=False):
    """
    Custom version of scene_to_video that can overlay ground truth trajectories
    """
    from tbsim.utils.scene_edit_utils import preprocess, draw_scene_data, create_video
    import matplotlib.pyplot as plt
    
    scene_data = preprocess(scene_data, filter_yaw)
    frames = [0] if first_frame_only else range(0, scene_data["centroid"].shape[1], save_every_n_frames)
    
    for i, frame_i in enumerate(frames):
        fig, ax = plt.subplots()
        
        # Set up drawing parameters based on draw_mode
        if draw_mode == 'action':
            draw_agents = True
            draw_action = True
            draw_trajectory = False
            traj_len = 200
            draw_action_sample = True
            linewidth = 2.0
            use_agt_color = False
            marker_size = 32
            traj_alpha = 1.0
        elif draw_mode in ['entire_traj', 'entire_traj_attn']:
            draw_agents = True
            draw_action = False
            draw_trajectory = True
            draw_action_sample = False
            traj_len = 200
            linewidth = 2.0
            use_agt_color = True
            marker_size = 32
            traj_alpha = 0.6
            if draw_mode == 'entire_traj_attn':
                traj_alpha = 1.0
        elif draw_mode == 'map':
            draw_agents = False
            draw_action = False
            draw_trajectory = False
            draw_action_sample = False
            traj_len = 200
            linewidth = 2.0
            use_agt_color = True
            marker_size = 800
            traj_alpha = 1.0
        else:
            raise NotImplementedError

        # Draw the main scene data (rollout trajectories)
        draw_scene_data(
            ax,
            scene_name,
            scene_data,
            frame_i,
            rasterizer,
            guidance_config=guidance_config,
            constraint_config=constraint_config,
            draw_agents=draw_agents,
            draw_trajectory=draw_trajectory,
            draw_action=draw_action,
            draw_action_sample=draw_action_sample,
            n_step_action=n_step_action,
            traj_len=traj_len,
            traj_alpha=traj_alpha,
            use_agt_color=use_agt_color,
            marker_size=marker_size,
            linewidth=linewidth,
            ras_pos=np.mean(scene_data["centroid"][:, 0], axis=0),
            draw_mode=draw_mode,
        )
        
        # Overlay ground truth trajectories if requested
        if draw_ground_truth:
            # Get raster_from_world transformation matrix
            ras_pos = np.mean(scene_data["centroid"][:, 0], axis=0)
            state_im, raster_from_world = rasterizer.render(
                ras_pos=ras_pos,
                ras_yaw=0,
                scene_name=scene_name
            )
            gt_drawn = draw_ground_truth_trajectories(ax, scene_data, frame_i, raster_from_world, traj_len, rasterizer, scene_name)
            
            # Add legend to distinguish rollout vs ground truth only if GT was actually drawn
            if gt_drawn:
                ax.legend(loc='upper right', fontsize=8, framealpha=0.8)

        if first_frame_only:
            ffn = os.path.join(output_dir, "{sname}_{simnum:04d}_{framei:03d}.png").format(sname=scene_name, simnum=sim_num, framei=frame_i)
        else:
            video_dir = os.path.join(output_dir, scene_name + '_%04d' % (sim_num))
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            ffn = os.path.join(video_dir, "{:03d}.png").format(i)
        
        plt.savefig(ffn, dpi=200, bbox_inches="tight", pad_inches=0)
        print("Figure written to {}".format(ffn))
        fig.clf()
        plt.close(fig)

    if not first_frame_only:
        create_video(os.path.join(video_dir, "%03d.png"), video_dir + ".mp4", fps=fps)
        



def rasterize_rendering(render_cfg):
    trajdata_data_dirs = {"nusc_trainval" : "../behavior-generation-dataset/nuscenes"} # "nusc_mini"
    trajdata_source_test=["nusc_trainval-val"] # "nusc_mini"
    from tbsim.utils.scene_edit_utils import get_trajdata_renderer
    # initialize rasterizer once for all scenes
    render_rasterizer = get_trajdata_renderer(trajdata_source_test,
                                                trajdata_data_dirs,
                                                future_sec=5.2,
                                                history_sec=3.0,
                                                raster_size=render_cfg['size'],
                                                px_per_m=render_cfg['px_per_m'],
                                                rebuild_maps=False,
                                                cache_location='~/.unified_data_cache')
    return render_rasterizer

def render_cfg(h5_path, render_cfg):        
    from tbsim.utils.scene_edit_utils import visualize_guided_rollout
    
    results_dir = os.path.dirname(h5_path)
    render_rasterizer = rasterize_rendering(render_cfg)
    viz_dir = os.path.join(results_dir, "viz/")
    render_to_img = True
    
    # Use custom visualization function that supports ground truth overlay
    if render_cfg.get('draw_ground_truth', False):

        visualize_guided_rollout_with_gt(
            viz_dir,
            render_rasterizer,
            si,
            scene_buffer,
            guidance_config=None,
            constraint_config=None,
            fps=(1.0 / 0.1),
            n_step_action=5,
            viz_diffusion_steps=False,
            first_frame_only=render_to_img,
            sim_num=int(sim_start_frames),
            save_every_n_frames=render_cfg['save_every_n_frames'],
            draw_mode=render_cfg['draw_mode'],
            draw_ground_truth=True
        )
    else:
        visualize_guided_rollout(
            viz_dir,
            render_rasterizer,
            si,
            scene_buffer,
            guidance_config=None,
            constraint_config=None,
            fps=(1.0 / 0.1),
            n_step_action=5,
            viz_diffusion_steps=False,
            first_frame_only=render_to_img,
            sim_num=int(sim_start_frames),
            save_every_n_frames=render_cfg['save_every_n_frames'],
            draw_mode=render_cfg['draw_mode']
        )