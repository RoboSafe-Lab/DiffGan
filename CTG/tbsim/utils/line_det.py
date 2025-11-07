"""
Line/Curve detection for trajectory analysis on GPU tensors.

This module provides functions to classify trajectory segments as straight lines
or curves based on curvature, geometric deviation, and heading stability.
Optimized for batch processing on GPU using PyTorch.
"""
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import time

def detect_line_curve(
    trajectory_data: torch.Tensor,
    window_size: int = 15,
    curvature_threshold: float = 0.5,
    deviation_threshold: float = 0.5,
    yaw_range_threshold: float = 0.25
) -> torch.Tensor:
    """
    Detect whether each timestep in trajectories is in a straight line or curve region.

    Optimized for GPU batch processing using PyTorch operations.
    Uses a combination of geometric curvature, trajectory deviation, and yaw stability.

    Args:
        trajectory_data: Tensor of shape (B, N, T, 6) where:
                        - B: batch size
                        - N: number of agents
                        - T: time steps
                        - 6: [x, y, vel, yaw, acc, yawvel]
        window_size: Size of the sliding window for analysis. Default: 15
        curvature_threshold: Maximum curvature for straight line (1/meter). Default: 0.5
        deviation_threshold: Maximum perpendicular deviation from line segment (meter). Default: 0.5
        yaw_range_threshold: Maximum yaw range within window (radians). Default: 0.25 (~14.3°)

    Returns:
        is_straight: Boolean tensor of shape (B, N, T) where True indicates straight line region

    Example:
        >>> data = torch.randn(4, 10, 50, 6).cuda()  # 4 batches, 10 agents, 50 timesteps
        >>> result = detect_line_curve(data)
        >>> print(result.shape)  # torch.Size([4, 10, 50])
    """
    B, N, T, _ = trajectory_data.shape
    device = trajectory_data.device

    # Handle edge cases
    if T <= 2:
        return torch.ones(B, N, T, dtype=torch.bool, device=device)

    # Extract position and yaw
    traj = trajectory_data[..., :2]  # (B, N, T, 2) - x, y coordinates
    yaw = trajectory_data[..., 3]    # (B, N, T) - yaw angle

    # Reshape to (B*N, T, 2) and (B*N, T) for batch processing
    traj_flat = traj.reshape(B * N, T, 2)
    yaw_flat = yaw.reshape(B * N, T)

    # Step 1: Compute point-wise curvature
    # Calculate distances between consecutive points
    displacements = traj_flat[:, 1:, :] - traj_flat[:, :-1, :]  # (B*N, T-1, 2)
    distances = torch.norm(displacements, dim=-1)  # (B*N, T-1)
    distances = torch.clamp(distances, min=1e-6)  # Avoid division by zero

    # Calculate yaw changes (handle angle wrapping)
    yaw_changes = yaw_flat[:, 1:] - yaw_flat[:, :-1]  # (B*N, T-1)
    yaw_changes = torch.atan2(torch.sin(yaw_changes), torch.cos(yaw_changes))

    # Curvature = |change in heading| / distance traveled
    curvature = torch.abs(yaw_changes) / distances  # (B*N, T-1)

    # Pad to match original length
    curvature = F.pad(curvature, (1, 0), mode='replicate')  # (B*N, T)

    # Step 2: Sliding window maximum curvature using unfold
    max_curvature = _sliding_window_max(curvature, window_size)  # (B*N, T)

    # Step 3: Sliding window yaw range
    # Unwrap yaw angles to handle discontinuities
    yaw_unwrapped = _unwrap_angle(yaw_flat)  # (B*N, T)

    yaw_max = _sliding_window_max(yaw_unwrapped, window_size)
    yaw_min = _sliding_window_min(yaw_unwrapped, window_size)
    yaw_range = yaw_max - yaw_min  # (B*N, T)

    # Step 4: Geometric deviation
    max_deviation = _compute_geometric_deviation(traj_flat, window_size)  # (B*N, T)

    # Step 5: Combine all criteria
    is_straight_flat = (
        (max_curvature < curvature_threshold) &
        (max_deviation < deviation_threshold) &
        (yaw_range < yaw_range_threshold)
    )

    # Reshape back to (B, N, T)
    is_straight = is_straight_flat.reshape(B, N, T)

    return is_straight


def _sliding_window_max(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Compute sliding window maximum using unfold operation.

    Args:
        x: Input tensor of shape (B, T)
        window_size: Window size

    Returns:
        max_values: Tensor of shape (B, T) with sliding window max
    """
    B, T = x.shape
    half_win = window_size // 2

    # Pad the input
    x_padded = F.pad(x.unsqueeze(1), (half_win, half_win), mode='replicate')  # (B, 1, T+pad)

    # Use max_pool1d with stride=1 to get sliding window max
    max_values = F.max_pool1d(x_padded, kernel_size=window_size, stride=1, padding=0)

    return max_values.squeeze(1)  # (B, T)


def _sliding_window_min(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Compute sliding window minimum using unfold operation.

    Args:
        x: Input tensor of shape (B, T)
        window_size: Window size

    Returns:
        min_values: Tensor of shape (B, T) with sliding window min
    """
    # Min is -max(-x)
    return -_sliding_window_max(-x, window_size)


def _unwrap_angle(angles: torch.Tensor) -> torch.Tensor:
    """
    Unwrap angles to handle discontinuities at ±π.

    Args:
        angles: Tensor of shape (B, T) with angles in radians

    Returns:
        unwrapped: Unwrapped angles of shape (B, T)
    """
    # Compute differences
    diff = angles[:, 1:] - angles[:, :-1]

    # Wrap differences to [-π, π]
    diff_wrapped = torch.atan2(torch.sin(diff), torch.cos(diff))

    # Compute cumulative sum to unwrap
    unwrapped = torch.cat([
        angles[:, :1],
        angles[:, :1] + torch.cumsum(diff_wrapped, dim=1)
    ], dim=1)

    return unwrapped


def _compute_geometric_deviation(traj: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Compute maximum perpendicular deviation from line segment in sliding windows.

    For each timestep, computes the maximum perpendicular distance of trajectory points
    within the window to the line segment connecting the window's first and last points.

    Args:
        traj: Trajectory tensor of shape (B, T, 2)
        window_size: Window size for analysis

    Returns:
        max_deviation: Tensor of shape (B, T) with maximum deviations
    """
    B, T, _ = traj.shape
    device = traj.device
    half_win = window_size // 2

    # Pad trajectory for consistent window extraction
    traj_padded = F.pad(
        traj.permute(0, 2, 1),  # (B, 2, T)
        (half_win, half_win),
        mode='replicate'
    ).permute(0, 2, 1)  # (B, T+2*half_win, 2)

    # Pre-allocate output
    max_deviation = torch.zeros(B, T, device=device)

    # Extract all windows using unfold
    # This creates (B, T, window_size, 2)
    windows = traj_padded.unfold(1, window_size, 1)  # (B, T, 2, window_size)
    windows = windows.permute(0, 1, 3, 2)  # (B, T, window_size, 2)

    # Get start and end points for each window
    start_points = windows[:, :, 0, :]  # (B, T, 2)
    end_points = windows[:, :, -1, :]   # (B, T, 2)

    # Compute line vectors
    line_vecs = end_points - start_points  # (B, T, 2)
    line_lengths = torch.norm(line_vecs, dim=-1, keepdim=True)  # (B, T, 1)

    # Handle zero-length lines
    valid_lines = line_lengths.squeeze(-1) > 1e-6  # (B, T)

    # Normalize line vectors
    line_vecs_normalized = line_vecs / torch.clamp(line_lengths, min=1e-6)  # (B, T, 2)

    # Compute deviations for all points in all windows
    # windows: (B, T, window_size, 2)
    # start_points: (B, T, 1, 2)
    vectors_from_start = windows - start_points.unsqueeze(2)  # (B, T, window_size, 2)

    # Project onto line direction
    # (B, T, window_size, 2) @ (B, T, 2, 1) -> (B, T, window_size, 1)
    projections = torch.matmul(
        vectors_from_start,
        line_vecs_normalized.unsqueeze(-1)
    ).squeeze(-1)  # (B, T, window_size)

    # Clamp projections to line segment [0, line_length]
    # Use torch.clamp_min and torch.minimum for tensor max
    projections = torch.clamp_min(projections, 0.0)  # Lower bound at 0
    projections = torch.minimum(
        projections,
        line_lengths.squeeze(-1).unsqueeze(-1)  # (B, T, 1) -> broadcasts to (B, T, window_size)
    )  # (B, T, window_size)

    # Compute closest points on line segments
    # (B, T, window_size, 1) * (B, T, 1, 2) -> (B, T, window_size, 2)
    closest_points = start_points.unsqueeze(2) + \
                     projections.unsqueeze(-1) * line_vecs_normalized.unsqueeze(2)

    # Compute perpendicular distances
    deviations = torch.norm(windows - closest_points, dim=-1)  # (B, T, window_size)

    # Get maximum deviation in each window
    max_deviation_all = torch.max(deviations, dim=-1)[0]  # (B, T)

    # Set deviation to 0 for invalid lines
    max_deviation = torch.where(valid_lines, max_deviation_all, torch.zeros_like(max_deviation_all))

    return max_deviation


def visualize_trajectory_batch(
    trajectory_data: torch.Tensor,
    is_straight: torch.Tensor,
    batch_idx: int = 0,
    agent_idx: int = 0,
    save_path: str = None,
    plot_size: float = 80.0
):
    """
    Visualize a single trajectory from the batch with straight/curve classification.

    Args:
        trajectory_data: Tensor of shape (B, N, T, 6)
        is_straight: Boolean tensor of shape (B, N, T)
        batch_idx: Batch index to visualize
        agent_idx: Agent index to visualize
        save_path: Path to save figure (optional)
        plot_size: Size of plot area in meters
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Extract single trajectory
    traj = trajectory_data[batch_idx, agent_idx, :, :2].cpu().numpy()  # (T, 2)
    straight = is_straight[batch_idx, agent_idx].cpu().numpy()  # (T,)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot trajectory segments
    for i in range(len(traj) - 1):
        color = 'blue' if straight[i] else 'red'
        label = 'Straight' if i == 0 or straight[i] != straight[i-1] and straight[i] else None
        if not straight[i] and i > 0 and straight[i-1]:
            label = 'Curve'

        ax.plot(traj[i:i+2, 0], traj[i:i+2, 1],
                color=color, linewidth=2, label=label, alpha=0.7)

    # Mark start and end
    ax.scatter(traj[0, 0], traj[0, 1], c='green', s=100,
               marker='o', label='Start', zorder=5)
    ax.scatter(traj[-1, 0], traj[-1, 1], c='purple', s=100,
               marker='s', label='End', zorder=5)

    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    # Set fixed plot limits
    center_x = np.mean(traj[:, 0])
    center_y = np.mean(traj[:, 1])
    half_size = plot_size / 2.0

    ax.set_xlim(center_x - half_size, center_x + half_size)
    ax.set_ylim(center_y - half_size, center_y + half_size)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title(f'Trajectory Classification (Batch {batch_idx}, Agent {agent_idx})', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()
    plt.close()


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")



    hdf5_path = r'C:\myAppData\CTGTest\nusc_results\safe-sim\safe-sim.hdf5'
    with h5py.File(hdf5_path, 'r') as f:
        scene_num = len(f.keys())
        st = time.time()
        for i, scene_name_with_episode in enumerate(f.keys()):
            if i < 2:
                continue

            base_scene_name = scene_name_with_episode.split('_')[0]
            sim_group = f[scene_name_with_episode]
            sim_pos = sim_group['centroid']  # Shape: (agent, T, 2)
            sim_yaw = sim_group['yaw']

            N,T,_ = sim_pos.shape

            data = np.zeros((1, N, T, 6))
            data[0,:,:,:2] = sim_pos
            data[0,:,:,3] = sim_yaw
            break
    print(1)

    data = torch.from_numpy(data).float().to(device)

    # Warm-up run
    _ = detect_line_curve(data)
    B,N,T,_= data.shape

    # Performance test
    import time
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()

    num_runs = 100
    for _ in range(num_runs):
        result = detect_line_curve(data)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start

    print(f"\nPerformance:")
    print(f"  Shape: {data.shape}")
    print(f"  Runs: {num_runs}")
    print(f"  Total time: {elapsed:.4f}s")
    print(f"  Time per run: {elapsed/num_runs*1000:.2f}ms")


    print(f"\nResult shape: {result.shape}")
    print(f"Result dtype: {result.dtype}")

    # Count straight line segments
    straight_ratio = result.float().mean().item()
    print(f"Straight line ratio: {straight_ratio:.2%}")

    # Visualize one example if matplotlib is available
    try:
        for b in range(B):
            for n in range(N):
                visualize_trajectory_batch(data, result, batch_idx=b, agent_idx=n)
    except Exception as e:
        print(f"\nVisualization skipped: {e}")
