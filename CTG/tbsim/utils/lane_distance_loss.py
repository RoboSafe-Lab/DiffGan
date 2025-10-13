import torch
import numpy as np
import cv2
from typing import Dict, Tuple, List
from multiprocessing import Pool, cpu_count
from functools import partial
import traceback


class GPUAwareLaneDetector:
    def __init__(self, n_workers=None, return_tensor=True, debug=False):
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        self.return_tensor = return_tensor
        self.debug = debug

        # Algorithm parameters
        self.search_box_l_mult = 4.0
        self.search_box_w_mult = 4.0
        self.color_tolerance = 25
        self.bfs_max_points = 100
        self.min_points_for_direction = 15
        self.march_steps = 100
        self.march_step_size = 2
        self.march_probe_range = 5
        self.min_points_for_final_fit = 25

        # Target colors
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

        device = x.device
        B, N, T, _ = x.shape

        with torch.no_grad():
            bgr_images_gpu = self._prepare_bgr_images_gpu(data_batch['maps'])
            positions_px_gpu = self._transform_to_pixel_gpu(
                x[..., :2],
                data_batch['raster_from_agent']
            )
            search_limits_gpu = self._compute_search_limits_gpu(
                data_batch['extent'],
                data_batch['raster_from_agent']
            )

            yaws_gpu = x[..., 3]

        bgr_images_cpu = bgr_images_gpu.cpu().numpy()  # (B, H, W, 3)
        positions_px_cpu = positions_px_gpu.cpu().numpy()  # (B, N, T, 2)
        yaws_cpu = yaws_gpu.cpu().numpy()  # (B, N, T)
        search_limits_cpu = search_limits_gpu.cpu().numpy()  # (B, 2)

        if use_multiprocessing and self.n_workers > 1:
            distances_cpu, debug_results = self._process_parallel(
                B, N, T, bgr_images_cpu, positions_px_cpu, yaws_cpu, search_limits_cpu
            )
        else:
            distances_cpu, debug_results = self._process_sequential(
                B, N, T, bgr_images_cpu, positions_px_cpu, yaws_cpu, search_limits_cpu
            )

        if self.debug:
            self.debug_info = {
                'B': B, 'N': N, 'T': T,
                'bgr_images': bgr_images_cpu,
                'positions_px': positions_px_cpu,
                'yaws': yaws_cpu,
                'search_limits': search_limits_cpu,
                'results': debug_results,
                'distances': distances_cpu
            }

        if self.return_tensor:
            distances = torch.from_numpy(distances_cpu).to(device)
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

    def _prepare_bgr_images_gpu(self, maps: torch.Tensor) -> torch.Tensor:

        B, _, H, W = maps.shape
        device = maps.device

        drivable = maps[:, 0]  # (B, H, W)
        dividers = maps[:, 1]
        crosswalks = maps[:, 2]

        color_drivable = torch.tensor([213, 211, 200], device=device, dtype=torch.uint8)
        color_dividers = torch.tensor([196, 184, 164], device=device, dtype=torch.uint8)
        color_crosswalks = torch.tensor([138, 117, 96], device=device, dtype=torch.uint8)

        bgr = torch.zeros((B, H, W, 3), device=device, dtype=torch.uint8)

        for b in range(B):
            mask_drivable = drivable[b] > 0
            mask_dividers = dividers[b] > 0
            mask_crosswalks = crosswalks[b] > 0

            bgr[b, mask_drivable] = color_drivable
            bgr[b, mask_dividers] = color_dividers
            bgr[b, mask_crosswalks] = color_crosswalks

        return bgr

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
            bgr_images: np.ndarray,
            positions_px: np.ndarray,
            yaws: np.ndarray,
            search_limits: np.ndarray
    ) -> Tuple[np.ndarray, List]:
        tasks = []
        for b in range(B):
            for n in range(N):
                for t in range(T):
                    tasks.append((
                        b, n, t,
                        bgr_images[b],
                        positions_px[b, n, t],
                        yaws[b, n, t],
                        search_limits[b]
                    ))

        worker_fn = partial(
            self._compute_single_sample_worker,
            color_dividers=self.color_dividers,
            color_crosswalks=self.color_crosswalks,
            color_tolerance=self.color_tolerance,
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
            results = pool.map(worker_fn, tasks, chunksize=32)

        for (b, n, t, _, _, _, _), result in zip(tasks, results):
            if self.debug:
                distances[b, n, t] = result['distance']
                debug_results[b][n][t] = result
            else:
                distances[b, n, t] = result

        return distances, debug_results

    def _process_sequential(
            self,
            B: int, N: int, T: int,
            bgr_images: np.ndarray,
            positions_px: np.ndarray,
            yaws: np.ndarray,
            search_limits: np.ndarray
    ) -> Tuple[np.ndarray, List]:
        distances = np.full((B, N, T), np.nan)
        debug_results = [[[None for _ in range(T)] for _ in range(N)] for _ in range(B)]

        total = B * N * T
        for idx, (b, n, t) in enumerate(np.ndindex(B, N, T)):
            if idx % 100 == 0:
                print(f"  Progress: {idx}/{total} ({100 * idx / total:.1f}%)")

            result = self._compute_single_sample(
                bgr_images[b],
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
    def _compute_single_sample_worker(task, **params):
        b, n, t, bgr_image, pos_px, yaw, limit = task

        debug = params.pop('debug', False)
        detector = GPUAwareLaneDetector(n_workers=1, return_tensor=False, debug=debug)

        for key, value in params.items():
            setattr(detector, key, value)

        return detector._compute_single_sample(bgr_image, pos_px, yaw, limit)

    def _compute_single_sample(
            self,
            bgr_image: np.ndarray,
            pos_px: np.ndarray,
            yaw: float,
            limit: np.ndarray
    ):
        try:
            if not np.isfinite(pos_px).all() or not np.isfinite(yaw) or not np.isfinite(limit).all():
                return self._create_result(np.nan, "Invalid input values (NaN or Inf)")

            h, w = bgr_image.shape[:2]
            if pos_px[0] < 0 or pos_px[0] >= w or pos_px[1] < 0 or pos_px[1] >= h:
                return self._create_result(np.nan, "Vehicle position outside image bounds")

            heading = np.array([np.cos(yaw), np.sin(yaw)])
            perp = np.array([-heading[1], heading[0]])

            b1 = self._find_boundary(pos_px, perp, bgr_image, pos_px, yaw, limit)
            b2 = self._find_boundary(pos_px, -perp, bgr_image, pos_px, yaw, limit)

            if b1 is None or b2 is None:
                return self._create_result(np.nan, "Could not find initial boundary points")

            pts1 = self._bfs_search(b1, bgr_image, pos_px, yaw, limit)
            pts2 = self._bfs_search(b2, bgr_image, pos_px, yaw, limit)

            if len(pts1) < self.min_points_for_direction or len(pts2) < self.min_points_for_direction:
                return self._create_result(
                    np.nan,
                    f"Not enough local points ({len(pts1)}, {len(pts2)})"
                )

            dir1 = cv2.fitLine(pts1, cv2.DIST_L2, 0, 0.01, 0.01)[:2].flatten()
            dir2 = cv2.fitLine(pts2, cv2.DIST_L2, 0, 0.01, 0.01)[:2].flatten()

            if np.dot(dir1, dir2) < 0:
                dir2 = -dir2

            final_dir = (dir1 + dir2) / 2.0
            final_dir /= np.linalg.norm(final_dir)

            global_pts1 = self._march_search(b1, final_dir, bgr_image, pos_px, yaw, limit)
            global_pts2 = self._march_search(b2, final_dir, bgr_image, pos_px, yaw, limit)

            if len(global_pts1) < self.min_points_for_final_fit or \
                    len(global_pts2) < self.min_points_for_final_fit:
                return self._create_result(
                    np.nan,
                    f"Not enough global points ({len(global_pts1)}, {len(global_pts2)})"
                )

            line1 = cv2.fitLine(global_pts1, cv2.DIST_L2, 0, 0.01, 0.01)
            line2 = cv2.fitLine(global_pts2, cv2.DIST_L2, 0, 0.01, 0.01)

            proj1 = self._project_point(pos_px, line1)
            proj2 = self._project_point(pos_px, line2)
            center = (proj1 + proj2) / 2.0

            distance = np.linalg.norm(pos_px - center)

            if not np.isfinite(distance):
                return self._create_result(np.nan, "Computed distance is invalid")

            return self._create_result(
                float(distance),
                "Success",
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
            if self.debug:
                error_msg += f"\n{traceback.format_exc()}"
            print(f"[ERROR] {error_msg}")
            return self._create_result(np.nan, error_msg)

    def _create_result(self, distance, status, **kwargs):
        if self.debug:
            result = {
                'distance': distance,
                'status': 'Success' if np.isfinite(distance) else 'Failure',
                'status_detail': status
            }
            result.update(kwargs)
            return result
        else:
            return distance

    def _find_boundary(self, start, direction, img, veh_pos, veh_yaw, limit):
        h, w = img.shape[:2]
        for i in range(1, 500):
            point = start + direction * i
            px, py = point.round().astype(int)

            if px < 0 or px >= w or py < 0 or py >= h:
                return None

            if not self._in_search_box((px, py), veh_pos, veh_yaw, limit):
                return None

            if self._is_boundary(img[py, px]):
                return np.array([px, py], dtype=np.float64)
        return None

    def _is_boundary(self, pixel_bgr):
        if pixel_bgr is None or len(pixel_bgr) != 3:
            return False

        pixel_rgb = pixel_bgr[::-1].astype(np.float64)

        dist_divider = np.linalg.norm(pixel_rgb - self.color_dividers)
        dist_crosswalk = np.linalg.norm(pixel_rgb - self.color_crosswalks)

        return dist_divider < self.color_tolerance or dist_crosswalk < self.color_tolerance

    def _in_search_box(self, point, veh_pos, veh_yaw, limit):
        vec = np.array(point) - veh_pos
        cos_yaw, sin_yaw = np.cos(-veh_yaw), np.sin(-veh_yaw)
        local = np.array([
            cos_yaw * vec[0] - sin_yaw * vec[1],
            sin_yaw * vec[0] + cos_yaw * vec[1]
        ])
        return abs(local[0]) < limit[0] and abs(local[1]) < limit[1]

    def _bfs_search(self, start, img, veh_pos, veh_yaw, limit):
        from collections import deque

        h, w = img.shape[:2]
        points = []
        start_tuple = tuple(start.round().astype(int))
        queue = deque([start_tuple])
        visited = {start_tuple}

        while queue and len(points) < self.bfs_max_points:
            x, y = queue.popleft()
            points.append([x, y])

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy

                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    continue

                if (nx, ny) in visited:
                    continue

                if not self._in_search_box((nx, ny), veh_pos, veh_yaw, limit):
                    continue

                if self._is_boundary(img[ny, nx]):
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        return np.array(points, dtype=np.int32) if points else np.array([], dtype=np.int32).reshape(0, 2)

    def _march_search(self, start, direction, img, veh_pos, veh_yaw, limit):
        h, w = img.shape[:2]
        probe_dir = np.array([-direction[1], direction[0]])
        points = [start]

        for mult in [1, -1]:
            pos = start.copy().astype(np.float64)

            for _ in range(self.march_steps):
                pos += direction * self.march_step_size * mult

                if not self._in_search_box(pos, veh_pos, veh_yaw, limit):
                    break

                found = None
                for dist in range(self.march_probe_range + 1):
                    for pm in ([1, -1] if dist > 0 else [1]):
                        probe = pos + probe_dir * dist * pm
                        px, py = probe.round().astype(int)

                        if px < 0 or px >= w or py < 0 or py >= h:
                            continue

                        if self._in_search_box((px, py), veh_pos, veh_yaw, limit) and \
                                self._is_boundary(img[py, px]):
                            found = np.array([px, py], dtype=np.float64)
                            break
                    if found is not None:
                        break

                if found is not None:
                    points.append(found)
                    pos = found.copy()
                else:
                    break

        return np.array(points, dtype=np.int32) if len(points) > 1 else np.array([], dtype=np.int32).reshape(0, 2)

    def _project_point(self, point, line_params):
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

        self.bgr_images = debug_info['bgr_images']
        self.positions_px = debug_info['positions_px']
        self.yaws = debug_info['yaws']
        self.search_limits = debug_info['search_limits']
        self.results = debug_info['results']
        self.distances = debug_info['distances']

        self.rgb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in self.bgr_images]

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

    d = np.load("C:\myAppData\pythonProject\CTG-Cloud\CTG\scripts\map_layer_plots\data.npy", allow_pickle=True).item()
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
        n_workers=12,
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

