import time

import h5py
import numpy as np
import matplotlib.pyplot as plt
from numba import njit


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



def visualize_trajectory(traj: np.ndarray, is_straight: np.ndarray,
                         title: str = "Trajectory Classification",
                         save_path: str = None,
                         plot_size: float = 80.0) -> None:
    """
    Visualize trajectory with different colors for straight and curve regions.
    All plots use the same fixed scale for consistent comparison.

    Args:
        traj: Trajectory array of shape (T, 2)
        is_straight: Boolean array of shape (T,) indicating straight line regions
        title: Plot title
        save_path: Path to save the figure (optional)
        plot_size: Size of the plot area in meters (plot_size x plot_size). Default: 80.0
    """
    # Use square figure to match square plot area
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot trajectory segments with different colors
    for i in range(len(traj) - 1):
        if is_straight[i]:
            color = 'blue'
            label = 'Straight' if i == 0 or not is_straight[i-1] else None
        else:
            color = 'red'
            label = 'Curve' if i == 0 or is_straight[i-1] else None

        ax.plot(traj[i:i+2, 0], traj[i:i+2, 1],
                color=color, linewidth=2, label=label, alpha=0.7)

    # Mark start and end points
    ax.scatter(traj[0, 0], traj[0, 1], c='green', s=100,
               marker='o', label='Start', zorder=5)
    ax.scatter(traj[-1, 0], traj[-1, 1], c='purple', s=100,
               marker='s', label='End', zorder=5)

    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    # Calculate trajectory center
    center_x = np.mean(traj[:, 0])
    center_y = np.mean(traj[:, 1])

    # Set fixed plot limits centered on trajectory
    half_size = plot_size / 2.0
    ax.set_xlim(center_x - half_size, center_x + half_size)
    ax.set_ylim(center_y - half_size, center_y + half_size)

    # Ensure equal aspect ratio (1:1 scale for x and y)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()
    plt.close()




if __name__ == "__main__":
    hdf5_path = r'../nusc_results/safe-sim/safe-sim.hdf5'
    with h5py.File(hdf5_path, 'r') as f:
        scene_num = len(f.keys())
        st = time.time()
        for i, scene_name_with_episode in enumerate(f.keys()):
            base_scene_name = scene_name_with_episode.split('_')[0]
            sim_group = f[scene_name_with_episode]
            sim_pos = sim_group['centroid']  # Shape: (agent, T, 2)
            sim_yaw = sim_group['yaw']

            for i, (agent_traj, agent_yaw) in enumerate(zip(sim_pos, sim_yaw)):
                is_straight = detect_line_curve(agent_traj, agent_yaw)
                # visualize_trajectory(agent_traj, is_straight, save_path=f'viz/{base_scene_name}_{i}.png')

        print(time.time() - st)


