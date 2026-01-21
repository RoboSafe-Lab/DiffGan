import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
import tempfile
import shutil

from trajdata import MapAPI
from trajdata.maps.vec_map_elements import RoadLane
from trajdata.caching.env_cache import EnvCache

from PIL import Image


@dataclass
class DynamicCheckConfig:
    min_velocity_threshold: float = 0.5  # m/s
    min_distance_threshold: float = 2.0  # meters

default_config = DynamicCheckConfig()

def is_trajectory_dynamic(trajectory: dict, config: DynamicCheckConfig = default_config) -> bool:
    max_velocity = np.max(trajectory['speeds'])
    if max_velocity > config.min_velocity_threshold:
        return True

    avg_velocity = np.mean(trajectory['speeds'])
    if avg_velocity > config.min_velocity_threshold * 0.5:
        return True

    total_distance = np.sum(np.linalg.norm(np.diff(trajectory['positions'], axis=0), axis=1))
    if total_distance > config.min_distance_threshold:
        return True

    displacement = np.linalg.norm(trajectory['positions'][-1] - trajectory['positions'][0])
    if displacement > config.min_distance_threshold * 0.5:
        return True

    return False


def plot_lane(ax, lane: RoadLane, color: str = 'gray', alpha: float = 0.3,
              linewidth: float = 2.0, rotation_matrix: Optional[np.ndarray] = None):

    if lane.left_edge is not None and lane.right_edge is not None:
        left_pts = lane.left_edge.points[:, :2].copy()
        right_pts = lane.right_edge.points[:, :2].copy()

        if rotation_matrix is not None:
            left_pts = (rotation_matrix @ left_pts.T).T
            right_pts = (rotation_matrix @ right_pts.T).T

        lane_polygon = np.vstack([left_pts, right_pts[::-1]])
        ax.fill(lane_polygon[:, 0], lane_polygon[:, 1],
                color=color, alpha=alpha, edgecolor=None)

    center_pts = lane.center.points[:, :2].copy()

    if rotation_matrix is not None:
        center_pts = (rotation_matrix @ center_pts.T).T

    ax.plot(center_pts[:, 0], center_pts[:, 1],
            color='dimgray', linewidth=1.2, linestyle=':', alpha=0.8)


def plot_vehicle_box(ax, position: np.ndarray, yaw: float, extent: np.ndarray,
                     color: str = 'black', alpha: float = 1.0, linewidth: float = 1.5):

    length, width = extent[0], extent[1]

    rect_points = np.array([
        [-length/2, -width/2],
        [length/2, -width/2],
        [length/2, width/2],
        [-length/2, width/2],
        [-length/2, -width/2]
    ])

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw, cos_yaw]
    ])

    rotated_points = rect_points @ rotation_matrix.T
    translated_points = rotated_points + position

    ax.fill(translated_points[:, 0], translated_points[:, 1],
            color=color, alpha=1.0, zorder=10, edgecolor=None, linewidth=1.0)

    arrow_length = length * 0.3
    arrow_width = width * 0.3

    offset = length * 0.15
    chevron_points = np.array([
        [offset + arrow_length * 0.2, -arrow_width],
        [offset + arrow_length * 0.6, 0],
        [offset + arrow_length * 0.2, arrow_width],
    ])

    rotated_chevron = chevron_points @ rotation_matrix.T
    translated_chevron = rotated_chevron + position

    ax.plot(translated_chevron[:, 0], translated_chevron[:, 1],
            color='white', linewidth=1, alpha=1.0, zorder=11)


def get_scene_location(scene_name: str, cache_path: Path, env_name: str = 'nusc_trainval') -> Optional[str]:

    try:
        env_cache = EnvCache(cache_path)
        scenes_list = env_cache.load_env_scenes_list(env_name)

        for scene_info in scenes_list:
            if scene_name in scene_info.name:
                scene = env_cache.load_scene(env_name, scene_info.name, scene_dt=0.1)
                return scene.location

        print(f"Warning: Could not find scene {scene_name} in {env_name}")
        return None
    except Exception as e:
        print(f"Error loading scene info: {e}")
        return None

def generate_gif_from_hdf5(
    hdf5_path: str,
    scene_idx: int = 0,
    output_path: str = "output.gif",
    cache_path: str = "~/.unified_data_cache",
    env_name: str = "nnusc_trainval",
    figsize: Tuple[int, int] = (16, 4),
    frame_interval: int = 5,
    fps: int = 25,
    show_predictions: bool = True,
    show_only_first_prediction: bool = False,
    prediction_alpha: float = 0.8,
    map_extent_size: float = 50.0,
    ego_aligned: bool = True,
    center_agent_idx: int = 0,
    start_timestep: int = 0,
    end_timestep: Optional[int] = None,
    loop: int = 0,
):
    cache_path = Path(cache_path).expanduser()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading simulation data from {hdf5_path}...")

    temp_dir = Path(tempfile.mkdtemp())
    print(f"Using temporary directory: {temp_dir}")

    try:
        with h5py.File(hdf5_path, 'r') as f:
            scene_names = list(f.keys())

            if scene_idx >= len(scene_names):
                print(f"Error: scene_idx {scene_idx} out of range (max: {len(scene_names)-1})")
                return

            scene_name_with_episode = scene_names[scene_idx]
            base_scene_name = scene_name_with_episode.rsplit('_', 1)[0]

            print(f"Processing scene: {scene_name_with_episode}")
            print(f"Base scene name: {base_scene_name}")

            sim_group = f[scene_name_with_episode]
            sim_pos = sim_group["centroid"][:]  # (agent, T, 2)
            sim_yaw = sim_group['yaw'][:]  # (agent, T)
            sim_extent = sim_group['extent'][:]  # (agent, T, 3)
            agent_ids = sim_group["agent_name"][:][:, 0, 0]  # (agent,)

            if show_predictions and "action_sample_positions" in sim_group:
                all_agent_predictions = sim_group["action_sample_positions"][:]  # (agent, T, N, M, 2)
            else:
                all_agent_predictions = None

            num_agents = sim_pos.shape[0]
            num_timesteps = sim_pos.shape[1]

            print(f"Number of agents: {num_agents}")
            print(f"Number of timesteps: {num_timesteps}")

            if center_agent_idx >= num_agents or center_agent_idx < 0:
                print(f"Error: center_agent_idx {center_agent_idx} out of range (must be 0-{num_agents-1})")
                return

            if end_timestep is None:
                end_timestep = num_timesteps
            else:
                end_timestep = min(end_timestep, num_timesteps)

            timesteps_to_render = list(range(start_timestep, end_timestep, frame_interval))
            print(f"Will render {len(timesteps_to_render)} frames from timestep {start_timestep} to {end_timestep} with interval {frame_interval}")

            all_positions_original = sim_pos.reshape(-1, 2)
            center_x_orig = (all_positions_original[:, 0].min() + all_positions_original[:, 0].max()) / 2
            center_y_orig = (all_positions_original[:, 1].min() + all_positions_original[:, 1].max()) / 2
            original_map_center = np.array([center_x_orig, center_y_orig])

            rotation_angle = None
            if ego_aligned:
                center_agent_initial_yaw = sim_yaw[center_agent_idx, 0]
                rotation_angle = -center_agent_initial_yaw
                print(f"Agent-aligned mode (agent {center_agent_idx}): rotation angle = {rotation_angle:.3f} rad ({np.degrees(rotation_angle):.1f} deg)")

                cos_rot = np.cos(rotation_angle)
                sin_rot = np.sin(rotation_angle)
                rotation_matrix = np.array([
                    [cos_rot, -sin_rot],
                    [sin_rot, cos_rot]
                ])

                original_shape = sim_pos.shape
                sim_pos_flat = sim_pos.reshape(-1, 2)
                sim_pos_rotated = (rotation_matrix @ sim_pos_flat.T).T
                sim_pos = sim_pos_rotated.reshape(original_shape)
                sim_yaw = sim_yaw + rotation_angle

            all_positions = sim_pos.reshape(-1, 2)
            center_x = (all_positions[:, 0].min() + all_positions[:, 0].max()) / 2
            center_y = (all_positions[:, 1].min() + all_positions[:, 1].max()) / 2

            x_range = all_positions[:, 0].max() - all_positions[:, 0].min()
            y_range = all_positions[:, 1].max() - all_positions[:, 1].min()

            max_range = max(x_range, y_range) + map_extent_size

            if ego_aligned and figsize[0] != figsize[1]:
                fig_aspect = figsize[1] / figsize[0]  # height / width
                half_x = max_range / 2.0
                half_y = max_range * fig_aspect / 2.0
                fixed_bounds = (
                    center_x - half_x,
                    center_x + half_x,
                    center_y - half_y,
                    center_y + half_y
                )
            else:
                half_size = max_range / 2.0
                fixed_bounds = (
                    center_x - half_size,
                    center_x + half_size,
                    center_y - half_size,
                    center_y + half_size
                )

            print(f"Fixed bounds: x=[{fixed_bounds[0]:.1f}, {fixed_bounds[1]:.1f}], y=[{fixed_bounds[2]:.1f}, {fixed_bounds[3]:.1f}]")

            print("Calculating dynamic status for agents...")
            is_dynamic = np.zeros(num_agents, dtype=bool)

            for i in range(num_agents):
                positions = sim_pos[i]  # (T, 2)
                velocities = np.diff(positions, axis=0)  # (T-1, 2)
                speeds = np.linalg.norm(velocities, axis=1)  # (T-1,)
                speeds = np.concatenate([[0], speeds])  # (T,)

                trajectory = {
                    'positions': positions,
                    'speeds': speeds
                }
                is_dynamic[i] = is_trajectory_dynamic(trajectory)

            print(f"Dynamic agents: {np.sum(is_dynamic)}/{num_agents}")

        print("Loading scene metadata and map...")
        location = get_scene_location(base_scene_name, cache_path, env_name)

        if location is None:
            print("Could not determine scene location. Trying common locations...")
            for loc in ['boston-seaport', 'singapore-onenorth', 'singapore-hollandvillage',
                        'singapore-queenstown', 'boston', 'singapore', 'pittsburgh', 'las_vegas']:
                try:
                    map_api = MapAPI(cache_path)
                    vec_map = map_api.get_map(f"{env_name}:{loc}")
                    location = loc
                    print(f"Successfully loaded map for location: {location}")
                    break
                except:
                    continue

            if location is None:
                print("Error: Could not load any map. Please check your cache path and dataset.")
                return
        else:
            print(f"Scene location: {location}")
            map_api = MapAPI(cache_path)
            vec_map = map_api.get_map(f"{env_name}:{location}")

        print(f"Map loaded: {vec_map.env_name}:{vec_map.map_name}")

        print(f"Generating {len(timesteps_to_render)} frames...")
        frame_paths = []

        map_rotation_matrix = None
        if ego_aligned and rotation_angle is not None:
            cos_rot = np.cos(rotation_angle)
            sin_rot = np.sin(rotation_angle)
            map_rotation_matrix = np.array([
                [cos_rot, -sin_rot],
                [sin_rot, cos_rot]
            ])

        for frame_idx, timestep in enumerate(timesteps_to_render):
            print(f"Rendering frame {frame_idx+1}/{len(timesteps_to_render)} (timestep {timestep})...")

            fig, ax = plt.subplots(figsize=figsize, dpi=100)

            min_x, max_x, min_y, max_y = fixed_bounds
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            max_range = max(max_x - min_x, max_y - min_y)

            if ego_aligned:
                query_center = original_map_center
            else:
                query_center = np.array([center_x, center_y])

            center_point = np.array([query_center[0], query_center[1], 0])
            radius = max_range * 0.6

            nearby_lanes = vec_map.get_lanes_within(center_point, radius)

            for lane in nearby_lanes:
                plot_lane(ax, lane, color='lightgray', alpha=0.5, rotation_matrix=map_rotation_matrix)

            if all_agent_predictions is not None and show_predictions:
                for i in range(num_agents):
                    if not is_dynamic[i]:
                        continue

                    predictions = all_agent_predictions[i, timestep]  # (N, M, 2)

                    if predictions.shape[0] == 0:
                        continue

                    num_traj_to_show = 1 if show_only_first_prediction else predictions.shape[0]

                    agent_pos = sim_pos[i, timestep]
                    agent_yaw = sim_yaw[i, timestep]

                    for n in range(num_traj_to_show):
                        traj_relative = predictions[n]

                        if np.all(np.isnan(traj_relative)) or np.all(traj_relative == 0):
                            continue

                        cos_yaw = np.cos(agent_yaw)
                        sin_yaw = np.sin(agent_yaw)
                        rotation_mat = np.array([
                            [cos_yaw, -sin_yaw],
                            [sin_yaw, cos_yaw]
                        ])

                        traj_world = (rotation_mat @ traj_relative.T).T + agent_pos

                        if n == 0:
                            ax.plot(traj_world[:, 0], traj_world[:, 1],
                                   color='#6495ED', linewidth=2, alpha=0.8, zorder=2)
                        else:
                            ax.plot(traj_world[:, 0], traj_world[:, 1],
                                   color='#6495ED', linewidth=1, alpha=0.3, zorder=1)

            for i in range(num_agents):
                pos = sim_pos[i, timestep]
                yaw = sim_yaw[i, timestep]
                extent = sim_extent[i, timestep]

                if i == center_agent_idx:
                    color = "#6495ED"
                    alpha = 1.0
                elif is_dynamic[i]:
                    color = "#6495ED"
                    alpha = 1.0
                else:
                    color = 'gray'
                    alpha = 0.7

                plot_vehicle_box(ax, pos, yaw, extent, color=color, alpha=alpha)

            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)

            if ego_aligned and figsize[0] != figsize[1]:
                x_range = max_x - min_x
                y_range = max_y - min_y
                data_aspect = y_range / x_range
                fig_aspect = figsize[1] / figsize[0]
                ax.set_aspect(data_aspect / fig_aspect, adjustable='box')
            else:
                ax.set_aspect('equal', adjustable='box')

            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_facecolor('white')
            fig.patch.set_facecolor('white')

            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

            frame_path = temp_dir / f"frame_{frame_idx:05d}.png"
            plt.savefig(frame_path, dpi=100, bbox_inches='tight', pad_inches=0, facecolor='white')
            plt.close(fig)

            frame_paths.append(frame_path)

        print(f"Creating GIF with {len(frame_paths)} frames at {fps} fps...")
        images = []
        for frame_path in frame_paths:
            images.append(Image.open(frame_path))

        duration = int(1000 / fps)

        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop,
            optimize=False
        )

        print(f"GIF saved to: {output_path}")
        print(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")

    finally:
        print("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
        print("Done!")

if __name__ == "__main__":
    """
    Tools to generate a GIF from hdf5.
    python generate_gif.py --hdf5 <path-to-hdf5> --output <path-to-output> --scene-idx <idx-in-hdf5>
    python generate_gif.py --hdf5 infer_results/boston/best/data.hdf5 --output out.gif --scene-idx 0
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate animated GIF from autonomous driving simulation results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--hdf5', type=str, required=True,
                        help='Path to HDF5 file with simulation results')
    parser.add_argument('--output', type=str, default='output.gif',
                        help='Output GIF file path')
    parser.add_argument('--scene-idx', type=int, default=0,
                        help='Scene index to plot')

    # Data source settings
    parser.add_argument('--cache-path', type=str, default='~/.unified_data_cache',
                        help='Path to trajdata cache')
    parser.add_argument('--env-name', type=str, default='nusc_trainval',
                        help='Dataset name')

    # Figure settings
    parser.add_argument('--fig-width', type=int, default=16,
                        help='Figure width')
    parser.add_argument('--fig-height', type=int, default=4,
                        help='Figure height')

    # Animation settings
    parser.add_argument('--frame-interval', type=int, default=5,
                        help='Interval between frames (plot every Nth frame)')
    parser.add_argument('--fps', type=int, default=25,
                        help='Frames per second in output GIF')
    parser.add_argument('--loop', type=int, default=0,
                        help='Number of times to loop GIF (0 = forever)')

    # Timestep settings
    parser.add_argument('--start-timestep', type=int, default=0,
                        help='First timestep to include')
    parser.add_argument('--end-timestep', type=int, default=None,
                        help='Last timestep to include (None = all)')

    # Prediction settings
    parser.add_argument('--no-predictions', action='store_true',
                        help='Do not show prediction trajectories')
    parser.add_argument('--show-all-predictions', action='store_true',
                        help='Show all prediction trajectories (default: only first)')
    parser.add_argument('--prediction-alpha', type=float, default=0.3,
                        help='Transparency for prediction lines')

    # View settings
    parser.add_argument('--map-size', type=float, default=20.0,
                        help='Map extent size in meters')
    parser.add_argument('--no-ego-aligned', action='store_true',
                        help='Do not rotate map to align with ego vehicle')
    parser.add_argument('--center-agent', type=int, default=0,
                        help='Index of the agent to use as center for rotation')

    args = parser.parse_args()

    generate_gif_from_hdf5(
        hdf5_path=args.hdf5,
        scene_idx=args.scene_idx,
        output_path=args.output,
        cache_path=args.cache_path,
        env_name=args.env_name,
        figsize=(args.fig_width, args.fig_height),
        frame_interval=args.frame_interval,
        fps=args.fps,
        show_predictions=not args.no_predictions,
        show_only_first_prediction=not args.show_all_predictions,
        prediction_alpha=args.prediction_alpha,
        map_extent_size=args.map_size,
        ego_aligned=not args.no_ego_aligned,
        center_agent_idx=args.center_agent,
        start_timestep=args.start_timestep,
        end_timestep=args.end_timestep,
        loop=args.loop
    )
