import numpy as np
from numba import njit


def point_to_segment_distance_batch(points, seg_starts, seg_ends):
    """
    Args:
        points: (M, 2)
        seg_starts: (N, 2)
        seg_ends: (N, 2)

    Returns:
        distances: (M, N)
    """
    segment_vecs = seg_ends - seg_starts
    segment_len_sq = np.sum(segment_vecs ** 2, axis=1, keepdims=True)  # (N, 1)

    # (M, 1, 2) - (1, N, 2) = (M, N, 2)
    point_vecs = points[:, np.newaxis, :] - seg_starts[np.newaxis, :, :]

    # (M, N, 2) * (1, N, 2) -> (M, N)
    t = np.sum(point_vecs * segment_vecs[np.newaxis, :, :], axis=2) / (segment_len_sq.squeeze() + 1e-10)
    t = np.clip(t, 0.0, 1.0)  # (M, N)

    # (M, N, 2)
    closest_points = seg_starts[np.newaxis, :, :] + t[:, :, np.newaxis] * segment_vecs[np.newaxis, :, :]

    # (M, N)
    distances = np.linalg.norm(points[:, np.newaxis, :] - closest_points, axis=2)

    return distances


def calculate_lane_distance(pos, lanes):
    """
    Args:
        pos: traj (T,2)
        lanes: lane (T,15,80,3)
    Returns:
        distance (T,)
    """
    T = pos.shape[0]
    num_lanes = lanes.shape[1]
    distances = np.full((T,), np.inf)
    lane_xy = lanes[:, :, :, :2]

    for t in range(T):
        vehicle_pos = pos[t]  # (2,)
        time_lanes = lane_xy[t]  # (15, 80, 2)
        min_distance = np.inf

        for lane_idx in range(num_lanes):
            lane_points = time_lanes[lane_idx]  # (80, 2)

            valid_mask = ~(np.isnan(lane_points[:, 0]) | np.isnan(lane_points[:, 1]) |
                          ((np.abs(lane_points[:, 0]) < 1e-6) & (np.abs(lane_points[:, 1]) < 1e-6)))

            if valid_mask.sum() == 0:
                continue

            valid_lane_points = lane_points[valid_mask]  # (num_valid, 2)
            num_valid = valid_lane_points.shape[0]

            if num_valid == 0:
                continue

            # (1, 2) - (num_valid, 2) = (num_valid, 2)
            diffs = vehicle_pos[np.newaxis, :] - valid_lane_points
            point_distances = np.linalg.norm(diffs, axis=1)  # (num_valid,)
            min_point_dist = np.min(point_distances)

            if num_valid > 1:
                seg_starts = valid_lane_points[:-1]  # (num_valid-1, 2)
                seg_ends = valid_lane_points[1:]     # (num_valid-1, 2)

                segment_distances = point_to_segment_distance_batch(
                    vehicle_pos[np.newaxis, :], seg_starts, seg_ends
                )  # (1, num_valid-1)

                min_segment_dist = np.min(segment_distances)

                lane_distance = min(min_point_dist, min_segment_dist)
            else:
                lane_distance = min_point_dist

            min_distance = min(min_distance, lane_distance)

        if np.isnan(vehicle_pos[0]) or np.isnan(vehicle_pos[1]):
            distances[t] = 0.0
        elif np.isinf(min_distance):
            distances[t] = 0.0
        else:
            distances[t] = min_distance

    return distances


@njit(cache=True, fastmath=True)
def _compute_max_deviation_numba(traj_padded, T, window_size):
    """
    Numba-accelerated function to compute maximum geometric deviation.

    Args:
        traj_padded: Padded trajectory array
        T: Original trajectory length
        window_size: Window size for analysis

    Returns:
        max_deviation: Array of maximum deviations for each timestep
    """
    max_deviation = np.zeros(T)

    for i in range(T):
        # Extract window from padded array (always same size)
        window_traj = traj_padded[i:i + window_size]

        # Line from first to last point in window
        start_point = window_traj[0]
        end_point = window_traj[-1]

        # Compute line vector and length
        line_vec_x = end_point[0] - start_point[0]
        line_vec_y = end_point[1] - start_point[1]
        line_length = np.sqrt(line_vec_x**2 + line_vec_y**2)

        if line_length < 1e-6:
            max_deviation[i] = 0.0
        else:
            # Normalize line vector
            line_vec_norm_x = line_vec_x / line_length
            line_vec_norm_y = line_vec_y / line_length

            max_dev = 0.0
            # Compute perpendicular distances for all points in window
            for j in range(window_size):
                # Vector from start to current point
                vec_x = window_traj[j, 0] - start_point[0]
                vec_y = window_traj[j, 1] - start_point[1]

                # Project onto line direction
                projection = vec_x * line_vec_norm_x + vec_y * line_vec_norm_y

                # Clamp to line segment
                if projection < 0:
                    projection = 0.0
                elif projection > line_length:
                    projection = line_length

                # Compute closest point on line
                closest_x = start_point[0] + projection * line_vec_norm_x
                closest_y = start_point[1] + projection * line_vec_norm_y

                # Compute perpendicular distance
                deviation = np.sqrt((window_traj[j, 0] - closest_x)**2 +
                                  (window_traj[j, 1] - closest_y)**2)

                if deviation > max_dev:
                    max_dev = deviation

            max_deviation[i] = max_dev

    return max_deviation


def detect_line_curve(traj: np.ndarray, yaw: np.ndarray,
                      window_size: int = 15,
                      curvature_threshold: float = 0.5,
                      deviation_threshold: float = 0.5,
                      yaw_range_threshold: float = 0.25) -> np.ndarray:
    """
    Detect whether each timestep in a trajectory is in a straight line or curve region.
    Highly optimized using vectorized operations and Numba JIT compilation.

    Args:
        traj: Trajectory array of shape (T, 2) containing (x, y) positions
        yaw: Yaw angle array of shape (T,) in radians
        window_size: Size of the sliding window for analysis. Default: 15
        curvature_threshold: Maximum curvature for straight line (1/meter). Default: 0.5
        deviation_threshold: Maximum perpendicular deviation from line segment (meter). Default: 0.5
        yaw_range_threshold: Maximum yaw range within window (radians). Default: 0.25

    Returns:
        is_straight: Boolean array of shape (T,) where True indicates straight line region
    """
    T = len(yaw)

    # Handle edge cases
    if T <= 2:
        return np.ones(T, dtype=bool)

    # Step 1: Compute point-wise curvature efficiently
    displacements = np.diff(traj, axis=0)  # (T-1, 2)
    distances = np.linalg.norm(displacements, axis=1)  # (T-1,)
    distances = np.maximum(distances, 1e-6)  # Avoid division by zero

    # Calculate yaw changes (handle angle wrapping)
    yaw_changes = np.diff(yaw)
    yaw_changes = np.arctan2(np.sin(yaw_changes), np.cos(yaw_changes))

    # Curvature = |change in heading| / distance traveled
    curvature = np.abs(yaw_changes) / distances  # (T-1,)
    curvature = np.concatenate([[curvature[0]], curvature])  # Pad to (T,)

    # Step 2: Vectorized sliding window maximum curvature
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    max_curvature = maximum_filter1d(curvature, size=window_size, mode='nearest')

    # Step 3: Vectorized yaw range calculation
    yaw_unwrapped = np.unwrap(yaw)
    yaw_max = maximum_filter1d(yaw_unwrapped, size=window_size, mode='nearest')
    yaw_min = minimum_filter1d(yaw_unwrapped, size=window_size, mode='nearest')
    yaw_range = yaw_max - yaw_min

    # Step 4: Geometric deviation using Numba acceleration
    half_win = window_size // 2
    traj_padded = np.pad(traj, ((half_win, half_win), (0, 0)), mode='edge')
    max_deviation = _compute_max_deviation_numba(traj_padded, T, window_size)

    # Step 5: Combine all criteria
    is_straight = (
        (max_curvature < curvature_threshold) &
        (max_deviation < deviation_threshold) &
        (yaw_range < yaw_range_threshold)
    )

    return is_straight
