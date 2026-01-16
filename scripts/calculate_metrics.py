import argparse
import itertools

import h5py
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity

from trajdata import UnifiedDataset, AgentType
import os

# the key names corresponding to each data in HDF5
hdf5_info = {
    "vel": "curr_speed",
    "yaw": "yaw",
    "pos": "centroid",
    "extent": "extent",
    "raster": "raster_from_world",
    "map": "drivable_map",
    "sample_positions": "action_sample_positions",
    "agent_ids": "agent_name"
}

RSS_PARAMS = {
    "rho": 0.5, "a_max_accel": 2.0, "a_min_brake": 4.0, "a_max_brake": 4.0,
    "mu": 0.2, "a_lat_max_accel": 1.5, "a_lat_min_brake": 2.5,
    "lateral_check_range_m": 5
}

# Dynamic trajectory detection parameters
DYNAMIC_PARAMS = {
    "min_velocity_threshold": 2.0,  # m/s
    "min_distance_threshold": 5.0,  # meters
}


def is_trajectory_dynamic(trajectory):
    min_vel_threshold = DYNAMIC_PARAMS['min_velocity_threshold']
    min_dist_threshold = DYNAMIC_PARAMS['min_distance_threshold']

    # Criterion 1: Maximum velocity
    max_velocity = np.max(trajectory['speeds'])
    if max_velocity > min_vel_threshold:
        return True

    # Criterion 2: Average velocity
    avg_velocity = np.mean(trajectory['speeds'])
    if avg_velocity > min_vel_threshold * 0.5:
        return True

    # Criterion 3: Total distance traveled
    if len(trajectory['positions']) > 1:
        total_distance = np.sum(np.linalg.norm(np.diff(trajectory['positions'], axis=0), axis=1))
        if total_distance > min_dist_threshold:
            return True

    # Criterion 4: End-to-end displacement
    displacement = np.linalg.norm(trajectory['positions'][-1] - trajectory['positions'][0])
    if displacement > min_dist_threshold * 0.5:
        return True

    return False


def get_dynamic_agent_indices(agent_ids, agent_df, dynamic_only=False):
    if not dynamic_only:
        return list(range(len(agent_ids)))

    dynamic_indices = []

    for idx, agent_id in enumerate(agent_ids):
        agent_name = agent_id
        if isinstance(agent_name, bytes):
            agent_name = agent_name.decode('utf-8')

        gt_df = agent_df[agent_df['agent_id'] == agent_name].sort_values('scene_ts')

        if not gt_df.empty:
            # Build trajectory dict
            traj = {
                "speeds": np.linalg.norm(gt_df[['vx', 'vy']].to_numpy(), axis=1),
                "positions": gt_df[['x', 'y']].to_numpy()
            }

            if is_trajectory_dynamic(traj):
                dynamic_indices.append(idx)

    return dynamic_indices


def calculate_jsd_kde(gt_data: np.ndarray, sim_data: np.ndarray, n_points: int = 1000,
                      const_tol: float = 0.01) -> float:
    gt_std, sim_std = np.std(gt_data), np.std(sim_data)
    is_gt_const, is_sim_const = gt_std < const_tol, sim_std < const_tol

    if is_gt_const and is_sim_const:
        print("jsd: all const")
        return 0.0 if np.isclose(np.mean(gt_data), np.mean(sim_data), atol=const_tol) else 1.0

    if is_gt_const or is_sim_const:
        print("jsd: one const")
        const_data = gt_data if is_gt_const else sim_data
        var_data = sim_data if is_gt_const else gt_data

        const_mean = np.mean(const_data)
        var_mean = np.mean(var_data)
        var_std = np.std(var_data)
        n = len(var_data)
        bw = 1.06 * var_std * (n ** (-0.2))

        const_std = max(var_std * 0.01, 1e-6)

        combined_data = np.concatenate([const_data, var_data])
        min_val, max_val = np.min(combined_data), np.max(combined_data)
        margin = max((max_val - min_val) * 0.2, bw * 4.0)
        x_grid = np.linspace(min_val - margin, max_val + margin, n_points).reshape(-1, 1)

        kde_const = KernelDensity(kernel='gaussian', bandwidth=bw)
        kde_const.fit(const_data.reshape(-1, 1))
        p_const = np.exp(kde_const.score_samples(x_grid))

        kde_var = KernelDensity(kernel='gaussian', bandwidth=bw)
        kde_var.fit(var_data.reshape(-1, 1))
        p_var = np.exp(kde_var.score_samples(x_grid))

        if is_gt_const:
            P, Q = p_const, p_var
        else:
            P, Q = p_var, p_const

        epsilon = 1e-8
        P = (P + epsilon) / (P.sum() + epsilon * len(P))
        Q = (Q + epsilon) / (Q.sum() + epsilon * len(Q))
        M = 0.5 * (P + Q)

        jsd_val = 0.5 * (entropy(P, M, base=2) + entropy(Q, M, base=2))
        return np.clip(jsd_val, 0, 1)

    combined_data = np.concatenate([gt_data, sim_data])
    min_val, max_val = np.min(combined_data), np.max(combined_data)
    margin = (max_val - min_val) * 0.1
    x_grid = np.linspace(min_val - margin, max_val + margin, n_points).reshape(-1, 1)
    data_std = np.std(combined_data)
    n = len(combined_data)
    bw = 1.06 * data_std * (n ** (-0.2))

    kde = KernelDensity(kernel='gaussian', bandwidth=bw)
    kde.fit(gt_data.reshape(-1, 1))
    p_vals = np.exp(kde.score_samples(x_grid))

    kde.fit(sim_data.reshape(-1, 1))
    q_vals = np.exp(kde.score_samples(x_grid))

    epsilon = 1e-8
    P = (p_vals + epsilon) / (p_vals.sum() + epsilon * len(p_vals))
    Q = (q_vals + epsilon) / (q_vals.sum() + epsilon * len(q_vals))
    M = 0.5 * (P + Q)

    jsd_val = 0.5 * (entropy(P, M, base=2) + entropy(Q, M, base=2))
    return np.clip(jsd_val, 0, 1)


def get_vehicle_corners(pos, yaw, extent):
    length, width = extent[0], extent[1]
    half_length, half_width = length / 2, width / 2

    local_corners = np.array([
        [half_length, half_width],  # front right
        [half_length, -half_width],  # front left
        [-half_length, -half_width],  # rear left
        [-half_length, half_width]  # rear right
    ])

    # rotation matrix
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw, cos_yaw]
    ])

    # Transform corners to global coordinate system
    global_corners = np.dot(local_corners, rotation_matrix.T) + pos

    return global_corners


def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def line_segments_intersect(p1, q1, p2, q2):
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-10:
            return 0
        return 1 if val > 0 else 2

    def on_segment(p, q, r):
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special cases
    if (o1 == 0 and on_segment(p1, p2, q1)) or \
            (o2 == 0 and on_segment(p1, q2, q1)) or \
            (o3 == 0 and on_segment(p2, p1, q2)) or \
            (o4 == 0 and on_segment(p2, q1, q2)):
        return True

    return False


def polygons_intersect(poly1, poly2):
    # check if any vertex of poly1 is inside poly2
    for point in poly1:
        if point_in_polygon(point, poly2):
            return True

    # check if any vertex of poly2 is inside poly1
    for point in poly2:
        if point_in_polygon(point, poly1):
            return True

    # Check if any edges intersect
    n1, n2 = len(poly1), len(poly2)
    for i in range(n1):
        for j in range(n2):
            if line_segments_intersect(poly1[i], poly1[(i + 1) % n1],
                                       poly2[j], poly2[(j + 1) % n2]):
                return True
    return False


def check_collision(pos_i, pos_j, yaw_i, yaw_j, extent_i, extent_j):
    # get corners of both vehicles
    corners_i = get_vehicle_corners(pos_i, yaw_i, extent_i)
    corners_j = get_vehicle_corners(pos_j, yaw_j, extent_j)

    # check if the oriented bounding boxes intersect
    return polygons_intersect(corners_i, corners_j)


def calculate_ttc(pos_i, vel_i, pos_j, vel_j,
                            heading_i, heading_j,
                            angle_threshold=30.0,
                            lateral_threshold=2.0,
                            min_speed_diff=0.1):

    heading_diff = np.abs(heading_i - heading_j)
    heading_diff = np.min([heading_diff, 2 * np.pi - heading_diff])
    heading_diff_deg = np.degrees(heading_diff)

    if heading_diff_deg > angle_threshold:
        return np.inf

    relative_pos = pos_j - pos_i
    direction_i = np.array([np.cos(heading_i), np.sin(heading_i)])
    longitudinal_distance = np.dot(relative_pos, direction_i)
    lateral_vector = relative_pos - longitudinal_distance * direction_i
    lateral_distance = np.linalg.norm(lateral_vector)

    if lateral_distance > lateral_threshold:
        return np.inf

    if longitudinal_distance <= 0:
        return np.inf

    longitudinal_speed_i = np.dot(vel_i, direction_i)
    longitudinal_speed_j = np.dot(vel_j, direction_i)
    relative_speed = longitudinal_speed_i - longitudinal_speed_j

    if relative_speed < min_speed_diff:
        return np.inf

    ttc = longitudinal_distance / relative_speed
    return ttc

def calculate_lon_dmin(v_rear, v_front, params):
    rho, a_max_accel, a_min_brake, a_max_brake = params['rho'], params['a_max_accel'], params['a_min_brake'], params[
        'a_max_brake']
    dist_reaction = v_rear * rho + 0.5 * a_max_accel * rho ** 2
    v_rear_after_reaction = v_rear + rho * a_max_accel
    dist_brake_rear = (v_rear_after_reaction ** 2) / (2 * a_min_brake)
    dist_brake_front = (v_front ** 2) / (2 * a_max_brake)
    d_min = dist_reaction + dist_brake_rear - dist_brake_front
    return max(0, d_min)


def calculate_lat_dmin(v1_lat, v2_lat, params):
    rho, mu, a_lat_max_accel, a_lat_min_brake = params['rho'], params['mu'], params['a_lat_max_accel'], params[
        'a_lat_min_brake']

    def get_lateral_stop_dist(v_lat):
        dist_reaction = abs(v_lat) * rho + 0.5 * a_lat_max_accel * rho ** 2
        v_lat_after_reaction = abs(v_lat) + rho * a_lat_max_accel
        dist_brake = (v_lat_after_reaction ** 2) / (2 * a_lat_min_brake)
        return dist_reaction + dist_brake

    dist1 = get_lateral_stop_dist(v1_lat)
    dist2 = get_lateral_stop_dist(v2_lat)
    return mu + dist1 + dist2


def check_rss_pair(ego_pos, ego_yaw, ego_vel, ego_extent, other_pos, other_yaw, other_vel, other_extent, params):
    LATERAL_CHECK_RANGE_M = params["lateral_check_range_m"]
    lon_violation = 0
    lat_violation = 0

    cos_yaw, sin_yaw = np.cos(ego_yaw), np.sin(ego_yaw)
    ego_half_len, ego_half_width = ego_extent[0] / 2, ego_extent[1] / 2
    other_half_len, other_half_width = other_extent[0] / 2, other_extent[1] / 2

    # calculate relative position
    rel_pos = other_pos - ego_pos
    x_ego_frame = rel_pos[0] * cos_yaw + rel_pos[1] * sin_yaw
    y_ego_frame = -rel_pos[0] * sin_yaw + rel_pos[1] * cos_yaw

    # lon
    if x_ego_frame > 0 and abs(y_ego_frame) < (ego_half_width + other_half_width):
        actual_gap = x_ego_frame - ego_half_len - other_half_len
        if actual_gap < 0:
            lon_violation = 1
        else:
            ego_v_lon = np.hypot(ego_vel[0], ego_vel[1])
            other_v_lon = other_vel[0] * cos_yaw + other_vel[1] * sin_yaw

            d_min_lon = calculate_lon_dmin(v_rear=ego_v_lon, v_front=other_v_lon, params=params)

            if actual_gap < d_min_lon:
                lon_violation = 1

    # lat
    is_laterally_aligned = abs(x_ego_frame) < (ego_half_len + other_half_len)
    is_laterally_close = abs(y_ego_frame) < (ego_half_width + other_half_width + LATERAL_CHECK_RANGE_M)

    if is_laterally_aligned and is_laterally_close:
        actual_lateral_gap = abs(y_ego_frame) - ego_half_width - other_half_width
        if actual_lateral_gap < 0:
            lat_violation = 1
        else:
            other_v_lat = -other_vel[0] * sin_yaw + other_vel[1] * cos_yaw

            d_min_lat = calculate_lat_dmin(v1_lat=0, v2_lat=other_v_lat, params=params)

            if actual_lateral_gap < d_min_lat:
                lat_violation = 1

    return lon_violation, lat_violation


def get_vehicle_pixel_corners(world_pos, world_yaw, world_extent, raster_from_world):
    world_corners = get_vehicle_corners(world_pos, world_yaw, world_extent)
    pixel_corners = np.zeros_like(world_corners)
    for i, corner in enumerate(world_corners):
        world_homogeneous = np.array([corner[0], corner[1], 1.0])
        pixel_homogeneous = np.dot(raster_from_world, world_homogeneous)
        pixel_corners[i] = pixel_homogeneous[:2] / pixel_homogeneous[2]
    return pixel_corners


def check_offroad_status(pixel_corners, drivable_map):
    map_height, map_width = drivable_map.shape

    min_x = max(0, int(np.floor(np.min(pixel_corners[:, 0]))))
    max_x = min(map_width - 1, int(np.ceil(np.max(pixel_corners[:, 0]))))
    min_y = max(0, int(np.floor(np.min(pixel_corners[:, 1]))))
    max_y = min(map_height - 1, int(np.ceil(np.max(pixel_corners[:, 1]))))

    if min_x >= map_width or max_x < 0 or min_y >= map_height or max_y < 0:
        return True, 1.0

    total_pixels = 0
    offroad_pixels = 0

    for py in range(min_y, max_y + 1):
        for px in range(min_x, max_x + 1):
            if point_in_polygon([px, py], pixel_corners):
                total_pixels += 1
                if drivable_map[py, px] == 0:
                    offroad_pixels += 1

    if offroad_pixels == 0:
        return False
    else:
        return True


def calculate_fdd(endpoints):
    N = endpoints.shape[0]
    if N < 2:
        return 0.0
    pair_indices = itertools.combinations(range(N), 2)
    distances = [
        np.linalg.norm(endpoints[u] - endpoints[v])
        for u, v in pair_indices
    ]
    fdd = np.sum(distances) * (2 / (N * (N - 1)))
    return fdd


def extract_kinematics_data(sim_vel_agent, agent_gt_df, dt, sframe):
    sim_acc_agent = np.diff(sim_vel_agent) / dt if len(sim_vel_agent) >= 2 else np.array([])
    sim_jerk_agent = np.diff(sim_acc_agent) / dt if len(sim_acc_agent) >= 2 else np.array([])

    gt_vel_vector_xy = agent_gt_df[['vx', 'vy']].to_numpy()
    gt_acc_vector_xy = agent_gt_df[['ax', 'ay']].to_numpy()

    gt_vel = np.linalg.norm(gt_vel_vector_xy, axis=1)
    gt_acc = np.linalg.norm(gt_acc_vector_xy, axis=1)
    gt_jerk = np.diff(gt_acc) / dt if len(gt_acc) >= 2 else np.array([])

    if len(gt_vel) > sframe and len(gt_acc) > sframe and len(gt_jerk) > sframe:
        gt_vel = gt_vel[sframe:]
        gt_acc = gt_acc[sframe:]
        gt_jerk = gt_jerk[sframe:]
    else:
        return None, None, None, None, None, None

    len_vel = min(len(gt_vel), len(sim_vel_agent))
    gt_vel_aligned = gt_vel[:len_vel]
    sim_vel_aligned = sim_vel_agent[:len_vel]

    len_acc = min(len(gt_acc), len(sim_acc_agent))
    gt_acc_aligned = gt_acc[:len_acc]
    sim_acc_aligned = sim_acc_agent[:len_acc]

    len_jerk = min(len(gt_jerk), len(sim_jerk_agent))
    gt_jerk_aligned = gt_jerk[:len_jerk]
    sim_jerk_aligned = sim_jerk_agent[:len_jerk]

    return (gt_vel_aligned, sim_vel_aligned,
            gt_acc_aligned, sim_acc_aligned,
            gt_jerk_aligned, sim_jerk_aligned)


def main(args):
    results_list = []

    jsd_scene_data = {}

    # get dataset
    desired_data = ['val' if args.dataset == 'nusc' else 'nuplan_mini']
    data_dirs = {'nusc_trainval' if args.dataset == 'nusc' else 'nuplan_mini': args.dataset_dir}

    dataset = UnifiedDataset(
        desired_data=desired_data,
        data_dirs=data_dirs,
        only_types=[AgentType.VEHICLE],
        num_workers=os.cpu_count(),
        desired_dt=args.desired_dt,
        obs_format="x,y,z,xd,yd,xdd,ydd,s,c",
    )

    cache_path, CacheClass = dataset.cache_path, dataset.cache_class
    scene_name_to_gt_map = {scene.name: scene for scene in dataset.scenes()}

    with h5py.File(args.hdf5_dir, 'r') as f:
        scene_num = len(f.keys())
        for i, scene_name_with_episode in enumerate(f.keys()):
            # get base scene name
            if 'ego' in scene_name_with_episode:
                # fix safe-sim style
                base_scene_name = scene_name_with_episode.split('_')[0]
            else:
                base_scene_name = scene_name_with_episode.rsplit('_', 1)[0]

            jsd_scene_data[base_scene_name] = {
                'gt_vel': [], 'sim_vel': [],
                'gt_acc': [], 'sim_acc': [],
                'gt_jerk': [], 'sim_jerk': []
            }

            # get basic data
            sim_group = f[scene_name_with_episode]
            sim_vel = sim_group[hdf5_info['vel']]  # Shape: (agent, T)
            sim_yaw = sim_group[hdf5_info['yaw']]  # Shape: (agent, T)
            sim_pos = sim_group[hdf5_info['pos']]  # Shape: (agent, T, 2)
            sim_extent = sim_group[hdf5_info['extent']]  # Shape: (agent, T, 3)
            raster_from_world = sim_group[hdf5_info['raster']][:]  # Shape: (agent, T, 3, 3)
            drivable_map = sim_group[hdf5_info['map']][:]  # Shape: (agent, T, 224, 224)
            all_agent_predictions = sim_group[hdf5_info['sample_positions']][:]  # Shape: (agent, T, N, M, 2)
            try:
                agent_ids = sim_group[hdf5_info['agent_ids']][:][:, 0, 0]  # Shape: (agent, 1)
            except IndexError:
                # fix safe-sim style
                agent_ids = sim_group[hdf5_info['agent_ids']][:][:, 0]  # Shape: (agent, 1)

            scene_num = len(f.keys())
            print(f"Process {i + 1}/{scene_num}: {base_scene_name}")

            num_agents, num_timesteps = sim_pos.shape[:2]

            # Get ground truth data
            scene_metadata = scene_name_to_gt_map.get(base_scene_name)
            if scene_metadata is None:
                print(f"Warning: Skipping {base_scene_name}, scene not found in dataset.")
                continue

            scene_cache_instance = CacheClass(cache_path, scene_metadata, augmentations=None)
            agent_df = scene_cache_instance.scene_data_df.reset_index()
            dt = scene_metadata.dt

            # Filter for dynamic agents if requested
            dynamic_agent_indices = get_dynamic_agent_indices(agent_ids, agent_df, args.dynamic_only)

            if args.dynamic_only:
                print(f"  Dynamic agents: {len(dynamic_agent_indices)}/{num_agents}")
                if len(dynamic_agent_indices) == 0:
                    print(f"  Skipping scene - no dynamic agents found")
                    continue

            # Use filtered indices for processing
            active_agent_indices = dynamic_agent_indices

            # init metrics
            scene_metrics = {
                'min_ttc': [],
                'rss_lon': 0,
                'rss_lat': 0,
                'fdd_list': [],
                'off-road_counts': 0,
                'collision_counts': 0,
            }

            if 'jsd' in args.metrics:
                ### calculate JSD ###
                # Data Collection for JSD
                for agent_idx in active_agent_indices:
                    agent_name = agent_ids[agent_idx]
                    if isinstance(agent_name, bytes):
                        agent_name = agent_name.decode('utf-8')

                    gt_df = agent_df[agent_df['agent_id'] == agent_name].sort_values('scene_ts')

                    if not gt_df.empty:
                        vals = extract_kinematics_data(sim_vel[agent_idx, :], gt_df, dt, args.sframe)
                        g_v, s_v, g_a, s_a, g_j, s_j = vals

                        if g_v is not None:
                            jsd_scene_data[base_scene_name]['gt_vel'].append(g_v)
                            jsd_scene_data[base_scene_name]['sim_vel'].append(s_v)
                        if g_a is not None and len(g_a) > 0:
                            jsd_scene_data[base_scene_name]['gt_acc'].append(g_a)
                            jsd_scene_data[base_scene_name]['sim_acc'].append(s_a)
                        if g_j is not None and len(g_j) > 0:
                            jsd_scene_data[base_scene_name]['gt_jerk'].append(g_j)
                            jsd_scene_data[base_scene_name]['sim_jerk'].append(s_j)

            ### min-ttc
            for idx_i, agent_i in enumerate(active_agent_indices):
                agent_min_ttc = float('inf')
                vel_i = np.diff(sim_pos[agent_i, :], axis=0) / dt
                vel_i = np.vstack([vel_i[0:1], vel_i])
                if not is_trajectory_dynamic({"speeds": vel_i,"positions": sim_pos[agent_i, :]}):
                    continue
                for idx_j, agent_j in enumerate(active_agent_indices):
                    vel_j = np.diff(sim_pos[agent_j, :], axis=0) / dt
                    vel_j = np.vstack([vel_j[0:1], vel_j])
                    if not is_trajectory_dynamic({"speeds": vel_j,"positions": sim_pos[agent_j, :]}):
                        continue
                    if idx_i == idx_j:
                        continue
                    for t in range(num_timesteps):

                        pos_i = sim_pos[agent_i, t]
                        pos_j = sim_pos[agent_j, t]
                        heading_i = sim_yaw[agent_i, t]
                        heading_j = sim_yaw[agent_j, t]
                        ttc = calculate_ttc(pos_i, vel_i[t], pos_j, vel_j[t], heading_i, heading_j)
                        if 0 < ttc < agent_min_ttc:
                            agent_min_ttc = ttc

                if agent_min_ttc != float('inf'):
                    scene_metrics['min_ttc'].append(agent_min_ttc)

            # traverse timestep
            for t in range(num_timesteps):
                velocities = {}
                fdd_for_agent = []

                # calculate vel vector for active agents only
                for agent_idx in active_agent_indices:
                    if t > 0:
                        vel_x = (sim_pos[agent_idx, t, 0] - sim_pos[agent_idx, t - 1, 0]) / dt
                        vel_y = (sim_pos[agent_idx, t, 1] - sim_pos[agent_idx, t - 1, 1]) / dt
                        velocities[agent_idx] = np.array([vel_x, vel_y])
                    else:
                        velocities[agent_idx] = np.array([0.0, 0.0])

                    if 'offroad' in args.metrics:
                        ### check off-road ###
                        if agent_idx < raster_from_world.shape[0] and agent_idx < drivable_map.shape[0]:
                            agent_pos = sim_pos[agent_idx, t]
                            agent_yaw = sim_yaw[agent_idx, t]
                            agent_extent = sim_extent[agent_idx, t]
                            agent_transform = raster_from_world[agent_idx, t]
                            agent_map = drivable_map[agent_idx, t]

                            pixel_corners = get_vehicle_pixel_corners(agent_pos, agent_yaw, agent_extent,
                                                                      agent_transform)
                            if check_offroad_status(pixel_corners, agent_map):
                                scene_metrics['off-road_counts'] += 1

                    if 'fdd' in args.metrics:
                        ### calculate FDD ###
                        trajectories = all_agent_predictions[agent_idx, t]
                        endpoints = trajectories[:, -1, :]  # shape: (N, 2)
                        fdd_t = calculate_fdd(endpoints)
                        fdd_for_agent.append(fdd_t)

                if len(fdd_for_agent) > 0:
                    scene_metrics['fdd_list'].append(np.mean(fdd_for_agent))

                # Check interactions between active agents only
                for idx_i, agent_i in enumerate(active_agent_indices):
                    for idx_j in range(idx_i + 1, len(active_agent_indices)):
                        agent_j = active_agent_indices[idx_j]

                        pos_i = sim_pos[agent_i, t]
                        pos_j = sim_pos[agent_j, t]
                        yaw_i = sim_yaw[agent_i, t]
                        yaw_j = sim_yaw[agent_j, t]
                        extent_i = sim_extent[agent_i, t]
                        extent_j = sim_extent[agent_j, t]
                        vel_i = velocities[agent_i]
                        vel_j = velocities[agent_j]

                        if 'collision' in args.metrics:
                            ### check collision ###
                            if check_collision(pos_i, pos_j, yaw_i, yaw_j, extent_i, extent_j):
                                scene_metrics['collision_counts'] += 1

                        if 'rss' in args.metrics:
                            ### calculate RSS ###
                            lon_viol_i, lat_viol_i = check_rss_pair(pos_i, yaw_i, vel_i, extent_i,
                                                                    pos_j, yaw_j, vel_j, extent_j, RSS_PARAMS)
                            scene_metrics['rss_lon'] += lon_viol_i
                            scene_metrics['rss_lat'] += lat_viol_i

                            lon_viol_j, lat_viol_j = check_rss_pair(pos_j, yaw_j, vel_j, extent_j,
                                                                    pos_i, yaw_i, vel_i, extent_i, RSS_PARAMS)
                            scene_metrics['rss_lon'] += lon_viol_j
                            scene_metrics['rss_lat'] += lat_viol_j




            # Calculate rates based on active agents count
            num_active_agents = len(active_agent_indices)

            final_metrics = {
                'scene_name': base_scene_name,
                'num_agents': num_agents,
                'num_dynamic_agents': num_active_agents,
                'min_ttc': np.min(scene_metrics['min_ttc']) if scene_metrics['min_ttc'] else 0,
                'off-road_rate': scene_metrics['off-road_counts'] / (
                            num_timesteps * num_active_agents) if num_active_agents > 0 else 0,
                'collision_rate': scene_metrics['collision_counts'] / (
                            num_timesteps * num_active_agents) if num_active_agents > 0 else 0,
                'rss_lon': scene_metrics['rss_lon'],
                'rss_lat': scene_metrics['rss_lat'],
                'fdd': np.mean(scene_metrics['fdd_list']) if len(scene_metrics['fdd_list']) > 0 else 0
            }

            results_list.append(final_metrics)

        if 'jsd' in args.metrics:
            jsd_save_path = os.path.join(args.output_dir, 'jsd_scene_data.npy')
            np.save(jsd_save_path, jsd_scene_data)
            print(f"JSD scene-wise data saved to: {jsd_save_path}")

            agg_data = {
                'gt_vel': [], 'sim_vel': [],
                'gt_acc': [], 'sim_acc': [],
                'gt_jerk': [], 'sim_jerk': []
            }
            for scene_name in jsd_scene_data:
                for key in agg_data:
                    if key in jsd_scene_data[scene_name]:
                        agg_data[key].extend(jsd_scene_data[scene_name][key])

            gt_v = np.concatenate(agg_data['gt_vel'])
            sim_v = np.concatenate(agg_data['sim_vel'])
            gt_a = np.concatenate(agg_data['gt_acc'])
            sim_a = np.concatenate(agg_data['sim_acc'])
            gt_j = np.concatenate(agg_data['gt_jerk'])
            sim_j = np.concatenate(agg_data['sim_jerk'])

            jsd_vel = calculate_jsd_kde(gt_v, sim_v)
            jsd_acc = calculate_jsd_kde(gt_a, sim_a)
            jsd_jerk = calculate_jsd_kde(gt_j, sim_j)
            print(f"JSD results:\n\tjsd vel: {jsd_vel}\n\tjsd acc: {jsd_acc}\n\tjsd jerk: {jsd_jerk}")

    if results_list:
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(f'{args.output_dir}/results.csv', index=False)
        print(f"\nEvaluation results saved to: {args.output_dir}\\results.csv")
    else:
        print("No results to save.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hdf5_dir",
        type=str,
        required=True,
        help="A directory of the hdf5 file",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=['nusc', 'nuplan'],
        default='nusc',
        help="Type of dataset"
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default='.',
        help="A directory of the dataset"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=r".",
        help="A output folder"
    )

    parser.add_argument(
        "--desired_dt",
        type=float,
        default=0.1,
        help="A desired timestep for nusc dataset",
    )

    parser.add_argument(
        "--sframe",
        type=int,
        default=31,
        help="Start frame",
    )

    parser.add_argument(
        "--metrics",
        type=str,
        default=['ttc','rss','fdd','offroad','collision','jsd'], # 'ttc','rss','fdd','offroad','collision','jsd'
        action='append',
        help="A list of metrics to calculate",
    )

    parser.add_argument(
        "--dynamic_only",
        action="store_true",
        default=False,
        help="If set, only evaluate dynamic agents based on ground truth trajectories"
    )

    args = parser.parse_args()
    main(args)