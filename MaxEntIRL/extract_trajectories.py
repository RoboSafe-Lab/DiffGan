import h5py
import numpy as np
import os

def extract_trajectories(h5_path, render_cfg):
    """
    Extracts all agent trajectories from the episode buffer HDF5 file.
    Each trajectory is saved as a numpy file for easy use in IRL algorithms.
    Args:
        h5_path (str): Path to the HDF5 buffer file (e.g., data.hdf5)
        out_dir (str): Directory to save extracted trajectories. If None, saves next to h5_path.
    Returns:
        List of file paths to saved trajectory numpy files.
    """
    from tbsim.utils.scene_edit_utils import visualize_guided_rollout
    results_dir = os.path.dirname(h5_path)
    viz_dir = os.path.join(results_dir, "viz/")
    render_rasterizer = rasterize_rendering(render_cfg)
    render_to_img = True
    
    with h5py.File(h5_path, "r") as h5_file:
        scene_cnt = 0
        for key in h5_file.keys():
            if "_" not in key:
                continue
            parts = key.split("_")
            if len(parts) < 2:
                continue
            si = parts[0]
            sim_start_frames = parts[1]
            scene_buffer = h5_file[key]
            
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
            scene_cnt += 1
        

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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", type=str, required=True, help="Path to buffer HDF5 file")
    parser.add_argument(
        "--render_size",
        type=int,
        default=400,
        help="width and height of the rendered image size in pixels"
    )

    parser.add_argument(
        "--render_px_per_m",
        type=float,
        default=2.0,
        help="resolution of rendering"
    )

    parser.add_argument(
        "--save_every_n_frames",
        type=int,
        default=5,
        help="saving videos while skipping every n frames"
    )

    parser.add_argument(
        "--draw_mode",
        type=str,
        default='action',
        help="['action', 'entire_traj', 'map']"
    )
    
    args = parser.parse_args()
    
    
    render_cfg = {
        'size' : args.render_size,
        'px_per_m' : args.render_px_per_m,
        'save_every_n_frames': args.save_every_n_frames,
        'draw_mode': args.draw_mode,
    }
    
    extract_trajectories(args.h5_path, render_cfg)
