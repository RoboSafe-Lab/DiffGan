import torch
import numpy as np
import cv2
from typing import Dict, Tuple, List
from multiprocessing import Pool, cpu_count
from functools import partial
from collections import deque
import traceback


class GPUAwareLaneDetector:
    def __init__(self, n_workers=None, return_tensor=True, debug=False):
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        self.return_tensor = return_tensor
        self.debug = debug

        # Algorithm parameters (aligned with optimized version)
        self.search_box_l_mult = 4.0
        self.search_box_w_mult = 4.0
        self.color_tolerance = 25
        self.bfs_max_points = 50
        self.min_points_for_direction = 15
        self.march_steps = 50
        self.march_step_size = 2
        self.march_probe_range = 5
        self.min_points_for_final_fit = 25

        # Target colors (RGB)
        self.color_dividers = np.array([164, 184, 196])
        self.color_crosswalks = np.array([96, 117, 138])

        # Debug info storage (only used when debug=True)
        self.debug_info = None

    def compute_distances(
            self,
            x: torch.Tensor,  # (B, N, T, 6) on GPU
            data_batch: Dict[str, torch.Tensor],  # All on GPU
            use_multiprocessing: bool = True
    ) -> torch.Tensor:
        # Detach tensors that require grad to avoid gradient computation
        if x.requires_grad:
            x = x.detach()

        # Detach all tensors in data_batch that require grad
        data_batch_detached = {}
        for key, value in data_batch.items():
            if isinstance(value, torch.Tensor) and value.requires_grad:
                data_batch_detached[key] = value.detach()
            else:
                data_batch_detached[key] = value
        data_batch = data_batch_detached

        device = x.device
        B, N, T, _ = x.shape

        with torch.no_grad():
            # Prepare boundary masks (OPTIMIZED: vectorized on GPU)
            # boundary_masks: 1 = boundary (dividers/crosswalks), 0 = lane area
            boundary_masks_gpu = self._prepare_boundary_masks_gpu(data_batch['maps'])

            positions_px_gpu = self._transform_to_pixel_gpu(
                x[..., :2],
                data_batch['raster_from_agent']
            )
            search_limits_gpu = self._compute_search_limits_gpu(
                data_batch['extent'],
                data_batch['raster_from_agent']
            )

            yaws_gpu = x[..., 3]

        # Transfer to CPU for CV operations
        boundary_masks_cpu = boundary_masks_gpu.cpu().numpy().astype(np.uint8)  # (B, H, W)
        positions_px_cpu = positions_px_gpu.cpu().numpy()  # (B, N, T, 2)
        yaws_cpu = yaws_gpu.cpu().numpy()  # (B, N, T)
        search_limits_cpu = search_limits_gpu.cpu().numpy()  # (B, 2)

        if use_multiprocessing and self.n_workers > 1:
            distances_cpu, debug_results = self._process_parallel(
                B, N, T, boundary_masks_cpu, positions_px_cpu, yaws_cpu, search_limits_cpu
            )
        else:
            distances_cpu, debug_results = self._process_sequential(
                B, N, T, boundary_masks_cpu, positions_px_cpu, yaws_cpu, search_limits_cpu
            )

        if self.debug:
            self.debug_info = {
                'B': B, 'N': N, 'T': T,
                'boundary_masks': boundary_masks_cpu,
                'positions_px': positions_px_cpu,
                'yaws': yaws_cpu,
                'search_limits': search_limits_cpu,
                'results': debug_results,
                'distances': distances_cpu
            }

        if self.return_tensor:
            distances = torch.from_numpy(distances_cpu).to(device=device, dtype=torch.float32)
        else:
            distances = distances_cpu
        return distances

    def visualize_results(self):

        if not self.debug:
            print("Error: Visualization only available in debug mode. Set debug=True when creating detector.")
            return

        if self.debug_info is None:
            print("Error: No results to visualize. Run compute_distances() first.")
            return

        print("Launching interactive visualizer...")
        visualizer = LaneDetectionVisualizer(self.debug_info, self)
        visualizer.show()

    def _prepare_boundary_masks_gpu(self, maps: torch.Tensor) -> torch.Tensor:
        """
        OPTIMIZED: Prepare boundary masks directly on GPU using vectorized operations.
        This replaces the old _prepare_bgr_images_gpu + per-pixel checking approach.

        Args:
            maps: (B, C, H, W) with channels [drivable, dividers, crosswalks]

        Returns:
            boundary_masks: (B, H, W) binary mask where:
                1 = boundary pixel (dividers/crosswalks) - NOT on lane
                0 = lane area - ON lane
        """
        B, _, H, W = maps.shape
        device = maps.device

        dividers = maps[:, 1]  # (B, H, W)
        crosswalks = maps[:, 2]  # (B, H, W)

        # Create binary boundary mask: any pixel that is divider OR crosswalk
        # This is much faster than checking colors pixel-by-pixel
        boundary_masks = ((dividers > 0) | (crosswalks > 0)).float()

        return boundary_masks

    def _transform_to_pixel_gpu(
            self,
            points_world: torch.Tensor,  # (B, N, T, 2)
            matrices: torch.Tensor  # (B, 3, 3)
    ) -> torch.Tensor:
        B, N, T, _ = points_world.shape
        device = points_world.device

        # Flatten spatial dimensions
        points_flat = points_world.reshape(B, -1, 2)  # (B, N*T, 2)
        num_points = points_flat.shape[1]

        # Add homogeneous coordinate
        ones = torch.ones(B, num_points, 1, device=device)
        points_h = torch.cat([points_flat, ones], dim=-1)  # (B, N*T, 3)

        # Batch matrix multiply
        points_transformed = torch.bmm(points_h, matrices.transpose(1, 2))  # (B, N*T, 3)

        # Normalize by homogeneous coordinate
        points_px = points_transformed[..., :2] / points_transformed[..., 2:3]

        return points_px.reshape(B, N, T, 2)

    def _compute_search_limits_gpu(
            self,
            extents: torch.Tensor,  # (B, 3)
            matrices: torch.Tensor  # (B, 3, 3)
    ) -> torch.Tensor:
        B = extents.shape[0]
        device = extents.device

        # Compute pixel scales using unit vectors
        origin = torch.zeros(B, 2, device=device)
        unit_x = torch.zeros(B, 2, device=device)
        unit_x[:, 0] = 1.0
        unit_y = torch.zeros(B, 2, device=device)
        unit_y[:, 1] = 1.0

        origin_px = self._transform_points_single_gpu(origin, matrices)
        unit_x_px = self._transform_points_single_gpu(unit_x, matrices)
        unit_y_px = self._transform_points_single_gpu(unit_y, matrices)

        scale_x = torch.norm(unit_x_px - origin_px, dim=1)
        scale_y = torch.norm(unit_y_px - origin_px, dim=1)

        length_px = extents[:, 0] * scale_x
        width_px = extents[:, 1] * scale_y

        limits = torch.stack([
            length_px * self.search_box_l_mult,
            width_px * self.search_box_w_mult
        ], dim=1)

        return limits

    def _transform_points_single_gpu(
            self,
            points: torch.Tensor,  # (B, 2)
            matrices: torch.Tensor  # (B, 3, 3)
    ) -> torch.Tensor:
        B = points.shape[0]
        device = points.device

        ones = torch.ones(B, 1, device=device)
        points_h = torch.cat([points, ones], dim=1)  # (B, 3)

        # (B, 3) @ (B, 3, 3).T -> (B, 3)
        points_transformed = torch.bmm(
            points_h.unsqueeze(1),
            matrices.transpose(1, 2)
        ).squeeze(1)

        points_px = points_transformed[:, :2] / points_transformed[:, 2:3]
        return points_px

    def _process_parallel(
            self,
            B: int, N: int, T: int,
            boundary_masks: np.ndarray,
            positions_px: np.ndarray,
            yaws: np.ndarray,
            search_limits: np.ndarray
    ) -> Tuple[np.ndarray, List]:
        """
        OPTIMIZED: Process in parallel at batch level instead of individual samples.
        This reduces the overhead of creating tasks and transferring data.
        """
        # Create tasks at batch level (B tasks instead of B*N*T tasks)
        batch_tasks = []
        for b in range(B):
            batch_tasks.append((
                b,
                boundary_masks[b],
                positions_px[b],  # (N, T, 2)
                yaws[b],  # (N, T)
                search_limits[b]
            ))

        # Worker function with parameters
        worker_fn = partial(
            self._compute_batch_worker,
            bfs_max_points=self.bfs_max_points,
            min_points_for_direction=self.min_points_for_direction,
            march_steps=self.march_steps,
            march_step_size=self.march_step_size,
            march_probe_range=self.march_probe_range,
            min_points_for_final_fit=self.min_points_for_final_fit,
            debug=self.debug
        )

        distances = np.full((B, N, T), np.nan)
        debug_results = [[[None for _ in range(T)] for _ in range(N)] for _ in range(B)]

        with Pool(processes=self.n_workers) as pool:
            batch_results = pool.map(worker_fn, batch_tasks)

        # Unpack results
        for b, (batch_distances, batch_debug) in enumerate(batch_results):
            distances[b] = batch_distances
            if self.debug:
                debug_results[b] = batch_debug

        return distances, debug_results

    def _process_sequential(
            self,
            B: int, N: int, T: int,
            boundary_masks: np.ndarray,
            positions_px: np.ndarray,
            yaws: np.ndarray,
            search_limits: np.ndarray
    ) -> Tuple[np.ndarray, List]:
        distances = np.full((B, N, T), np.nan)
        debug_results = [[[None for _ in range(T)] for _ in range(N)] for _ in range(B)]

        total = B * N * T
        for b in range(B):
            for n in range(N):
                for t in range(T):
                    idx = b * N * T + n * T + t
                    if idx % 100 == 0:
                        print(f"  Progress: {idx}/{total} ({100 * idx / total:.1f}%)")

                    result = self._compute_single_sample(
                        boundary_masks[b],
                        positions_px[b, n, t],
                        yaws[b, n, t],
                        search_limits[b]
                    )

                    if self.debug:
                        distances[b, n, t] = result['distance']
                        debug_results[b][n][t] = result
                    else:
                        distances[b, n, t] = result

        return distances, debug_results

    @staticmethod
    def _compute_batch_worker(task, **params):
        """
        OPTIMIZED: Process entire batch (N, T) samples with shared boundary mask.
        This eliminates the overhead of creating detector instances per sample.
        """
        b, boundary_mask, positions_px_batch, yaws_batch, search_limits = task
        N, T = positions_px_batch.shape[:2]

        distances = np.full((N, T), np.nan)
        debug_results = [[None for _ in range(T)] for _ in range(N)]

        for n in range(N):
            for t in range(T):
                result = GPUAwareLaneDetector._compute_single_sample_static(
                    boundary_mask,
                    positions_px_batch[n, t],
                    yaws_batch[n, t],
                    search_limits,
                    **params
                )

                if params.get('debug', False):
                    distances[n, t] = result['distance']
                    debug_results[n][t] = result
                else:
                    distances[n, t] = result

        return distances, debug_results

    def _compute_single_sample(
            self,
            boundary_mask: np.ndarray,
            pos_px: np.ndarray,
            yaw: float,
            limit: np.ndarray
    ):
        """Instance method wrapper for compatibility."""
        return self._compute_single_sample_static(
            boundary_mask, pos_px, yaw, limit,
            bfs_max_points=self.bfs_max_points,
            min_points_for_direction=self.min_points_for_direction,
            march_steps=self.march_steps,
            march_step_size=self.march_step_size,
            march_probe_range=self.march_probe_range,
            min_points_for_final_fit=self.min_points_for_final_fit,
            debug=self.debug
        )

    @staticmethod
    def _compute_single_sample_static(
            boundary_mask: np.ndarray,
            pos_px: np.ndarray,
            yaw: float,
            limit: np.ndarray,
            bfs_max_points: int = 50,
            min_points_for_direction: int = 15,
            march_steps: int = 50,
            march_step_size: int = 2,
            march_probe_range: int = 5,
            min_points_for_final_fit: int = 25,
            debug: bool = False
    ):
        """
        OPTIMIZED: Static method for computing distance using pre-computed boundary mask.
        Based on the efficient algorithm from lane_distance.py.

        NEW: Checks if vehicle is on lane before computing distance.
        boundary_mask: 1 = boundary (dividers/crosswalks), 0 = lane area
        If vehicle is NOT on lane (boundary_mask == 1), returns distance = 0.0.
        """
        try:
            if not np.isfinite(pos_px).all() or not np.isfinite(yaw) or not np.isfinite(limit).all():
                return GPUAwareLaneDetector._create_result_static(
                    np.nan, "Invalid input values (NaN or Inf)", debug
                )

            h, w = boundary_mask.shape
            if pos_px[0] < 0 or pos_px[0] >= w or pos_px[1] < 0 or pos_px[1] >= h:
                return GPUAwareLaneDetector._create_result_static(
                    np.nan, "Vehicle position outside image bounds", debug
                )

            # NEW: Check if vehicle is on lane
            # boundary_mask: 1 = boundary, 0 = lane
            # Check both vehicle center and front position

            # Vehicle center
            px_center = int(round(pos_px[0]))
            py_center = int(round(pos_px[1]))

            # Check center is in bounds and on lane
            if boundary_mask[py_center, px_center] != 0:
                # Vehicle center is on boundary (NOT on lane area)
                return GPUAwareLaneDetector._create_result_static(
                    0.0, "Vehicle center not on lane (on boundary/off-road)", debug
                )

            # Calculate vehicle front position
            # Vehicle front = center + (vehicle_length/2) * heading_direction
            # We use limit[0] as an approximation of vehicle length in pixels
            vehicle_length_px = limit[0] / 8.0  # limit is search_box_l_mult * length, so divide by mult
            # Use standard direction (front is where the vehicle is heading)
            heading_vec = np.array([np.cos(yaw), np.sin(yaw)])
            front_offset = heading_vec * vehicle_length_px / 2.0

            pos_front = pos_px + front_offset
            px_front = int(round(pos_front[0]))
            py_front = int(round(pos_front[1]))

            # Check front is in bounds
            if not (0 <= px_front < w and 0 <= py_front < h):
                # Front position is outside image bounds
                return GPUAwareLaneDetector._create_result_static(
                    0.0, "Vehicle front outside image bounds", debug
                )

            # Check front is on lane
            if boundary_mask[py_front, px_front] != 0:
                # Vehicle front is on boundary (NOT on lane area)
                return GPUAwareLaneDetector._create_result_static(
                    0.0, "Vehicle front not on lane (on boundary/off-road)", debug
                )

            # Precompute vehicle state for fast lookup
            cos_yaw = np.cos(-yaw)
            sin_yaw = np.sin(-yaw)
            vehicle_state = {
                'vehicle_pos_px': pos_px,
                'cos_sin_yaw': (cos_yaw, sin_yaw),
                'search_limits_px': limit
            }

            heading = np.array([np.cos(yaw), np.sin(yaw)])
            perp = np.array([-heading[1], heading[0]])

            # Find initial boundaries (optimized)
            b1 = GPUAwareLaneDetector._find_initial_boundary_optimized(
                pos_px, perp, boundary_mask, vehicle_state
            )
            b2 = GPUAwareLaneDetector._find_initial_boundary_optimized(
                pos_px, -perp, boundary_mask, vehicle_state
            )

            if b1 is None or b2 is None:
                return GPUAwareLaneDetector._create_result_static(
                    np.nan, "Could not find initial boundary points", debug
                )

            # BFS search for local points
            pts1 = GPUAwareLaneDetector._find_local_points_bfs_optimized(
                b1, boundary_mask, vehicle_state, bfs_max_points
            )
            pts2 = GPUAwareLaneDetector._find_local_points_bfs_optimized(
                b2, boundary_mask, vehicle_state, bfs_max_points
            )

            if len(pts1) < min_points_for_direction or len(pts2) < min_points_for_direction:
                return GPUAwareLaneDetector._create_result_static(
                    np.nan, f"Not enough local points ({len(pts1)}, {len(pts2)})", debug
                )

            # Fit lines to determine lane direction
            dir1 = cv2.fitLine(pts1, cv2.DIST_L2, 0, 0.01, 0.01)[:2].flatten()
            dir2 = cv2.fitLine(pts2, cv2.DIST_L2, 0, 0.01, 0.01)[:2].flatten()

            if np.dot(dir1, dir2) < 0:
                dir2 = -dir2

            final_dir = (dir1 + dir2) / 2.0
            final_dir /= np.linalg.norm(final_dir)

            # March search for global points
            global_pts1 = GPUAwareLaneDetector._find_global_points_marching_optimized(
                b1, final_dir, boundary_mask, vehicle_state, march_steps, march_step_size, march_probe_range
            )
            global_pts2 = GPUAwareLaneDetector._find_global_points_marching_optimized(
                b2, final_dir, boundary_mask, vehicle_state, march_steps, march_step_size, march_probe_range
            )

            if len(global_pts1) < min_points_for_final_fit or len(global_pts2) < min_points_for_final_fit:
                return GPUAwareLaneDetector._create_result_static(
                    np.nan, f"Not enough global points ({len(global_pts1)}, {len(global_pts2)})", debug
                )

            # Fit final lines
            line1 = cv2.fitLine(global_pts1, cv2.DIST_L2, 0, 0.01, 0.01)
            line2 = cv2.fitLine(global_pts2, cv2.DIST_L2, 0, 0.01, 0.01)

            proj1 = GPUAwareLaneDetector._project_point(pos_px, line1)
            proj2 = GPUAwareLaneDetector._project_point(pos_px, line2)
            center = (proj1 + proj2) / 2.0

            distance = np.linalg.norm(pos_px - center)

            if not np.isfinite(distance):
                return GPUAwareLaneDetector._create_result_static(
                    np.nan, "Computed distance is invalid", debug
                )

            return GPUAwareLaneDetector._create_result_static(
                float(distance),
                "Success",
                debug,
                global_points1=global_pts1,
                global_points2=global_pts2,
                line1_params=line1,
                line2_params=line2,
                center_point=center,
                lane_direction=final_dir,
                proj_p1=proj1,
                proj_p2=proj2
            )

        except Exception as e:
            error_msg = f"Exception: {type(e).__name__}: {str(e)}"
            if debug:
                error_msg += f"\n{traceback.format_exc()}"
            print(f"[ERROR] {error_msg}")
            return GPUAwareLaneDetector._create_result_static(np.nan, error_msg, debug)

    def _create_result(self, distance, status, **kwargs):
        """Instance method wrapper."""
        return self._create_result_static(distance, status, self.debug, **kwargs)

    @staticmethod
    def _create_result_static(distance, status, debug, **kwargs):
        """Static method for creating results."""
        if debug:
            result = {
                'distance': distance,
                'status': 'Success' if np.isfinite(distance) else 'Failure',
                'status_detail': status
            }
            result.update(kwargs)
            return result
        else:
            return distance

    @staticmethod
    def _is_within_search_box_vectorized(points, vehicle_pos_px, cos_yaw, sin_yaw, max_along, max_perp):
        """OPTIMIZED: Vectorized search box check."""
        vec_to_points = points - vehicle_pos_px
        local_x = cos_yaw * vec_to_points[:, 0] - sin_yaw * vec_to_points[:, 1]
        local_y = sin_yaw * vec_to_points[:, 0] + cos_yaw * vec_to_points[:, 1]
        return (np.abs(local_x) < max_along) & (np.abs(local_y) < max_perp)

    @staticmethod
    def _find_initial_boundary_optimized(start_pos, direction, boundary_mask, vehicle_state, max_search_dist=500):
        """OPTIMIZED: Vectorized initial boundary search."""
        height, width = boundary_mask.shape
        vx, vy = vehicle_state["vehicle_pos_px"]
        cos_yaw, sin_yaw = vehicle_state["cos_sin_yaw"]
        max_along, max_perp = vehicle_state["search_limits_px"]

        steps = np.arange(1, min(max_search_dist, 500))
        search_points = start_pos + direction * steps[:, np.newaxis]
        search_points_int = np.round(search_points).astype(np.int32)

        valid_mask = ((search_points_int[:, 0] >= 0) & (search_points_int[:, 0] < width) &
                      (search_points_int[:, 1] >= 0) & (search_points_int[:, 1] < height))

        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return None

        valid_points = search_points_int[valid_indices]

        in_box = GPUAwareLaneDetector._is_within_search_box_vectorized(
            valid_points.astype(float),
            np.array([vx, vy]),
            cos_yaw, sin_yaw,
            max_along, max_perp
        )

        first_outside = np.where(~in_box)[0]
        if len(first_outside) > 0:
            valid_points = valid_points[:first_outside[0]]
            if len(valid_points) == 0:
                return None

        for pt in valid_points:
            if boundary_mask[pt[1], pt[0]] > 0:
                return tuple(pt)

        return None

    @staticmethod
    def _find_local_points_bfs_optimized(start_node, boundary_mask, vehicle_state, bfs_max_points):
        """OPTIMIZED: BFS search with boundary mask."""
        if start_node is None:
            return np.array([])

        height, width = boundary_mask.shape
        vx, vy = vehicle_state["vehicle_pos_px"]
        cos_yaw, sin_yaw = vehicle_state["cos_sin_yaw"]
        max_along, max_perp = vehicle_state["search_limits_px"]

        line_points = []
        q = deque([start_node])
        visited = np.zeros((height, width), dtype=bool)
        visited[start_node[1], start_node[0]] = True

        directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int32)

        while q and len(line_points) < bfs_max_points:
            x, y = q.popleft()
            line_points.append([x, y])

            neighbors = np.array([[x, y]], dtype=np.int32) + directions

            valid = ((neighbors[:, 0] >= 0) & (neighbors[:, 0] < width) &
                     (neighbors[:, 1] >= 0) & (neighbors[:, 1] < height))

            for i, (nx, ny) in enumerate(neighbors):
                if valid[i] and not visited[ny, nx] and boundary_mask[ny, nx] > 0:
                    in_box = GPUAwareLaneDetector._is_within_search_box_vectorized(
                        np.array([[nx, ny]], dtype=float),
                        np.array([vx, vy]),
                        cos_yaw, sin_yaw,
                        max_along, max_perp
                    )[0]

                    if in_box:
                        visited[ny, nx] = True
                        q.append((nx, ny))

        return np.array(line_points, dtype=np.int32)

    @staticmethod
    def _find_global_points_marching_optimized(start_node, line_direction, boundary_mask, vehicle_state,
                                                march_steps, march_step_size, march_probe_range):
        """OPTIMIZED: Marching search with boundary mask."""
        if start_node is None:
            return np.array([])

        height, width = boundary_mask.shape
        vx, vy = vehicle_state["vehicle_pos_px"]
        cos_yaw, sin_yaw = vehicle_state["cos_sin_yaw"]
        max_along, max_perp = vehicle_state["search_limits_px"]

        probe_direction = np.array([-line_direction[1], line_direction[0]])
        line_points = [start_node]

        for direction_multiplier in [1, -1]:
            current_pos = np.array(start_node, dtype=float)
            step_vec = line_direction * march_step_size * direction_multiplier

            for _ in range(march_steps):
                current_pos += step_vec

                cx, cy = int(round(current_pos[0])), int(round(current_pos[1]))
                if not (0 <= cx < width and 0 <= cy < height):
                    break

                in_box = GPUAwareLaneDetector._is_within_search_box_vectorized(
                    current_pos.reshape(1, 2),
                    np.array([vx, vy]),
                    cos_yaw, sin_yaw,
                    max_along, max_perp
                )[0]

                if not in_box:
                    break

                best_point_found = None

                for probe_dist in range(march_probe_range + 1):
                    if probe_dist == 0:
                        probe_points = current_pos.reshape(1, 2)
                    else:
                        probe_points = np.vstack([
                            current_pos + probe_direction * probe_dist,
                            current_pos - probe_direction * probe_dist
                        ])

                    probe_points_int = np.round(probe_points).astype(np.int32)

                    for px, py in probe_points_int:
                        if (0 <= px < width and 0 <= py < height and boundary_mask[py, px] > 0):
                            in_probe_box = GPUAwareLaneDetector._is_within_search_box_vectorized(
                                np.array([[px, py]], dtype=float),
                                np.array([vx, vy]),
                                cos_yaw, sin_yaw,
                                max_along, max_perp
                            )[0]

                            if in_probe_box:
                                best_point_found = (px, py)
                                break

                    if best_point_found:
                        break

                if best_point_found:
                    line_points.append(best_point_found)
                    current_pos = np.array(best_point_found, dtype=float)
                else:
                    break

        return np.array(line_points, dtype=np.int32)

    @staticmethod
    def _project_point(point, line_params):
        """Project point onto line."""
        vx, vy, x0, y0 = line_params.flatten()
        v = np.array([vx, vy])
        p0 = np.array([x0, y0])
        return p0 + v * np.dot(point - p0, v)



class LaneDetectionVisualizer:

    def __init__(self, debug_info: Dict, detector: GPUAwareLaneDetector):
        self.debug_info = debug_info
        self.detector = detector

        self.B = debug_info['B']
        self.N = debug_info['N']
        self.T = debug_info['T']

        self.boundary_masks = debug_info['boundary_masks']
        self.positions_px = debug_info['positions_px']
        self.yaws = debug_info['yaws']
        self.search_limits = debug_info['search_limits']
        self.results = debug_info['results']
        self.distances = debug_info['distances']

        # Convert boundary masks to RGB images for visualization
        self.rgb_images = []
        for mask in self.boundary_masks:
            # Create a grayscale image where boundaries are white
            img = np.zeros((*mask.shape, 3), dtype=np.uint8)
            img[mask > 0] = [255, 255, 255]  # White for boundaries
            self.rgb_images.append(img)

    def show(self):
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, Button

        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(bottom=0.25)

        ax_b = plt.axes([0.25, 0.15, 0.65, 0.03])
        ax_n = plt.axes([0.25, 0.10, 0.65, 0.03])
        ax_t = plt.axes([0.25, 0.05, 0.65, 0.03])

        self.slider_b = Slider(ax_b, 'Batch (B)', 0, self.B - 1, valinit=0, valstep=1)
        self.slider_n = Slider(ax_n, 'Sample (N)', 0, self.N - 1, valinit=0, valstep=1)
        self.slider_t = Slider(ax_t, 'Timestep (T)', 0, self.T - 1, valinit=0, valstep=1)

        self.slider_b.on_changed(self._update)
        self.slider_n.on_changed(self._update)
        self.slider_t.on_changed(self._update)

        self._update(None)

        plt.show()

    def _update(self, val):
        b = int(self.slider_b.val)
        n = int(self.slider_n.val)
        t = int(self.slider_t.val)

        self.ax.clear()

        self.ax.imshow(self.rgb_images[b])
        self.ax.grid(False)

        pos_px = self.positions_px[b, n, t]
        yaw = self.yaws[b, n, t]
        result = self.results[b][n][t]
        distance = self.distances[b, n, t]

        self.ax.plot(pos_px[0], pos_px[1], 'o', color='red', markersize=10, label='Vehicle')

        # Visualization arrow (should match front calculation direction)
        heading_vec = np.array([np.cos(yaw), np.sin(yaw)])
        arrow_len = 30
        self.ax.arrow(
            pos_px[0], pos_px[1],
            heading_vec[0] * arrow_len, heading_vec[1] * arrow_len,
            head_width=8, head_length=8, fc='red', ec='red', lw=2
        )

        if result and result.get('status') == 'Success':
            if 'global_points1' in result:
                pts1 = result['global_points1']
                if len(pts1) > 0:
                    self.ax.scatter(pts1[:, 0], pts1[:, 1], c='yellow', s=5, alpha=0.6, label='Boundary 1')

            if 'global_points2' in result:
                pts2 = result['global_points2']
                if len(pts2) > 0:
                    self.ax.scatter(pts2[:, 0], pts2[:, 1], c='cyan', s=5, alpha=0.6, label='Boundary 2')

            if 'line1_params' in result and 'proj_p1' in result:
                self._draw_line(result['line1_params'], result['proj_p1'], 'blue', 2)

            if 'line2_params' in result and 'proj_p2' in result:
                self._draw_line(result['line2_params'], result['proj_p2'], 'blue', 2)

            if 'lane_direction' in result and 'center_point' in result:
                center_params = np.array([
                    result['lane_direction'][0],
                    result['lane_direction'][1],
                    result['center_point'][0],
                    result['center_point'][1]
                ])
                self._draw_line(center_params, result['center_point'], 'green', 2, label='Lane Center')

            if 'center_point' in result:
                center = result['center_point']
                self.ax.plot([pos_px[0], center[0]], [pos_px[1], center[1]],
                             'r--', lw=2, label=f'Distance: {distance:.2f}px')
                self.ax.plot(center[0], center[1], 'o', color='green', markersize=8)

            title = f"B={b}, N={n}, T={t} | Status: Success | Distance: {distance:.2f} px"
        else:
            status_detail = result.get('status_detail', 'Unknown') if result else 'No result'
            title = f"B={b}, N={n}, T={t} | Status: Failure | Reason: {status_detail}"

        self.ax.set_title(title, fontsize=10)
        self.ax.legend(loc='upper right', fontsize=8)
        self.fig.canvas.draw_idle()

    def _draw_line(self, line_params, center, color, lw, label=None):
        vx, vy, x0, y0 = line_params.flatten()
        length = 20

        p1 = center - np.array([vx, vy]) * length
        p2 = center + np.array([vx, vy]) * length

        self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw, label=label)


def compute_lane_distances(
        x: torch.Tensor,
        data_batch: Dict[str, torch.Tensor],
        n_workers: int = None,
        return_tensor: bool = True,
        use_multiprocessing: bool = True,
        debug: bool = False
) -> torch.Tensor:
    """
    Args:
        x: (B, N, T, 6) GPU tensor
        data_batch: dict with GPU tensors ('maps', 'raster_from_agent', 'extent')
        n_workers: the num of workers
        return_tensor: weather use GPU tensor
        use_multiprocessing: weather use multiprocessing
        debug: debug mode

    Returns:
        distances: (B, N, T) - GPU tensor if return_tensor=True, else numpy array
    """
    detector = GPUAwareLaneDetector(n_workers=n_workers, return_tensor=return_tensor, debug=debug)
    return detector.compute_distances(x, data_batch, use_multiprocessing=use_multiprocessing)


if __name__ == '__main__':
    import time

    d = np.load(r"C:\myAppData\CTGTest\CTG\scripts\map_layer_plots\data.npy", allow_pickle=True).item()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    x_tensor = torch.from_numpy(d['x']).float().to(device)
    data_batch_tensor = {
        'maps': torch.from_numpy(d['data_batch']['maps']).float().to(device),
        'raster_from_agent': torch.from_numpy(d['data_batch']['raster_from_agent']).float().to(device),
        'extent': torch.from_numpy(d['data_batch']['extent']).float().to(device)
    }

    print(f"\nData shape: {x_tensor.shape}")
    print(f"Total samples: {np.prod(x_tensor.shape[:3])}")


    start = time.time()
    distances = compute_lane_distances(
        x_tensor,
        data_batch_tensor,
        n_workers=4,
        return_tensor=True,
        use_multiprocessing=True
    )
    elapsed = time.time() - start

    print(f"\n结果:")
    print(f"  输出设备: {distances.device}")
    print(f"  输出形状: {distances.shape}")
    print(f"  总耗时: {elapsed:.2f}s")
    print(f"  速度: {distances.numel() / elapsed:.1f} samples/s")

    valid = ~torch.isnan(distances)
    print(f"  成功率: {100 * valid.sum().item() / distances.numel():.1f}%")

    if valid.any():
        print(f"  平均距离: {distances[valid].mean().item():.2f} pixels")
        print(f"  最小距离: {distances[valid].min().item():.2f} pixels")
        print(f"  最大距离: {distances[valid].max().item():.2f} pixels")


    print(f"\n{'=' * 60}")
    print("测试Debug模式")
    print(f"{'=' * 60}")

    # 创建debug模式的detector
    detector_debug = GPUAwareLaneDetector(
        n_workers=4,
        return_tensor=False,
        debug=True
    )

    start = time.time()
    distances_debug = detector_debug.compute_distances(
        x_tensor,
        data_batch_tensor,
        use_multiprocessing=True
    )
    elapsed = time.time() - start

    detector_debug.visualize_results()

