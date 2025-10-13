import time

import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

# --- Algorithm Parameters ---
SEARCH_BOX_L_MULTIPLIER = 8.0
SEARCH_BOX_W_MULTIPLIER = 8.0
LINE_DRAW_LENGTH_MULTIPLIER = 8.0
BFS_SEARCH_RADIUS = 30
BFS_MAX_POINTS = 50
MIN_POINTS_FOR_DIRECTION_EST = 15
MARCH_NUM_STEPS = 50
MARCH_STEP_SIZE = 2
MARCH_PROBE_RANGE = 5
MIN_POINTS_FOR_FINAL_FIT = 25

# --- Fixed Vehicle Dimensions (meters) ---
VEHICLE_LENGTH = 2.5
VEHICLE_WIDTH = 1.5

# --- Color Definitions (RGB, 0-255) ---
COLOR_DIVIDERS_RGB = np.array([164, 184, 196])
COLOR_CROSSWALKS_RGB = np.array([96, 117, 138])
COLOR_TOLERANCE = 25


# ==============================================================================
# OPTIMIZED CORE FUNCTIONS (Pure NumPy)
# ==============================================================================
def _is_within_search_box_vectorized(points, vehicle_pos_px, cos_yaw, sin_yaw, max_along, max_perp):
    vec_to_points = points - vehicle_pos_px
    local_x = cos_yaw * vec_to_points[:, 0] - sin_yaw * vec_to_points[:, 1]
    local_y = sin_yaw * vec_to_points[:, 0] + cos_yaw * vec_to_points[:, 1]
    return (np.abs(local_x) < max_along) & (np.abs(local_y) < max_perp)


def _prepare_boundary_mask(cv_image, tolerance=25):
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    diff_dividers = np.sqrt(np.sum((rgb_image - COLOR_DIVIDERS_RGB[np.newaxis, np.newaxis, :]) ** 2, axis=2))
    diff_crosswalks = np.sqrt(np.sum((rgb_image - COLOR_CROSSWALKS_RGB[np.newaxis, np.newaxis, :]) ** 2, axis=2))

    mask = (diff_dividers < tolerance) | (diff_crosswalks < tolerance)
    return mask.astype(np.uint8)


def _find_initial_boundary_optimized(start_pos, direction, boundary_mask, vehicle_state, max_search_dist=500):
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

    in_box = _is_within_search_box_vectorized(valid_points.astype(float),
                                              np.array([vx, vy]),
                                              cos_yaw, sin_yaw,
                                              max_along, max_perp)

    first_outside = np.where(~in_box)[0]
    if len(first_outside) > 0:
        valid_points = valid_points[:first_outside[0]]
        if len(valid_points) == 0:
            return None

    for pt in valid_points:
        if boundary_mask[pt[1], pt[0]] > 0:
            return tuple(pt)

    return None


def _find_local_points_bfs_optimized(start_node, boundary_mask, vehicle_state):
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

    while q and len(line_points) < BFS_MAX_POINTS:
        x, y = q.popleft()
        line_points.append([x, y])

        neighbors = np.array([[x, y]], dtype=np.int32) + directions

        valid = ((neighbors[:, 0] >= 0) & (neighbors[:, 0] < width) &
                 (neighbors[:, 1] >= 0) & (neighbors[:, 1] < height))

        for i, (nx, ny) in enumerate(neighbors):
            if (valid[i] and not visited[ny, nx] and boundary_mask[ny, nx] > 0):
                in_box = _is_within_search_box_vectorized(
                    np.array([[nx, ny]], dtype=float),
                    np.array([vx, vy]),
                    cos_yaw, sin_yaw,
                    max_along, max_perp
                )[0]

                if in_box:
                    visited[ny, nx] = True
                    q.append((nx, ny))

    return np.array(line_points, dtype=np.int32)


def _find_global_points_marching_optimized(start_node, line_direction, boundary_mask, vehicle_state):
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
        step_vec = line_direction * MARCH_STEP_SIZE * direction_multiplier

        for _ in range(MARCH_NUM_STEPS):
            current_pos += step_vec

            cx, cy = int(round(current_pos[0])), int(round(current_pos[1]))
            if not (0 <= cx < width and 0 <= cy < height):
                break

            in_box = _is_within_search_box_vectorized(
                current_pos.reshape(1, 2),
                np.array([vx, vy]),
                cos_yaw, sin_yaw,
                max_along, max_perp
            )[0]

            if not in_box:
                break

            best_point_found = None

            for probe_dist in range(MARCH_PROBE_RANGE + 1):
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
                        in_probe_box = _is_within_search_box_vectorized(
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


def _project_point_onto_line(point, line_params):
    vx, vy, x0, y0 = line_params.flatten()
    p0, v = np.array([x0, y0]), np.array([vx, vy])
    return p0 + np.dot(np.array(point) - p0, v) * v


def _transform_world_to_pixel_batch(points, matrix):
    points = np.asarray(points)
    if points.ndim == 1:
        points = points[np.newaxis, :]

    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    points_transformed_h = points_h @ matrix.T
    points_transformed = points_transformed_h[:, :2] / points_transformed_h[:, 2, np.newaxis]

    return points_transformed


def _get_pixel_extent(raster_from_agent):
    origin = np.array([[0, 0], [1, 0], [0, 1]])
    transformed = _transform_world_to_pixel_batch(origin, raster_from_agent)

    scale_x = np.linalg.norm(transformed[1] - transformed[0])
    scale_y = np.linalg.norm(transformed[2] - transformed[0])

    return VEHICLE_LENGTH * scale_x, VEHICLE_WIDTH * scale_y


def _calculate_single_frame_distance(boundary_mask, vehicle_pos_px, vehicle_yaw_rad, search_limits_px):
    results = {"status": "Failure", "distance": np.nan}

    cos_yaw = np.cos(-vehicle_yaw_rad)
    sin_yaw = np.sin(-vehicle_yaw_rad)

    vehicle_state = {
        "vehicle_pos_px": vehicle_pos_px,
        "cos_sin_yaw": (cos_yaw, sin_yaw),
        "search_limits_px": search_limits_px
    }

    heading_vec = np.array([np.cos(vehicle_yaw_rad), np.sin(vehicle_yaw_rad)])
    perp_vec = np.array([-heading_vec[1], heading_vec[0]])

    initial_boundary1 = _find_initial_boundary_optimized(vehicle_pos_px, perp_vec, boundary_mask, vehicle_state)
    initial_boundary2 = _find_initial_boundary_optimized(vehicle_pos_px, -perp_vec, boundary_mask, vehicle_state)

    if not initial_boundary1 or not initial_boundary2:
        results["status_detail"] = "Could not find initial boundary points"
        return results

    local_points1 = _find_local_points_bfs_optimized(initial_boundary1, boundary_mask, vehicle_state)
    local_points2 = _find_local_points_bfs_optimized(initial_boundary2, boundary_mask, vehicle_state)

    if len(local_points1) < MIN_POINTS_FOR_DIRECTION_EST or len(local_points2) < MIN_POINTS_FOR_DIRECTION_EST:
        results["status_detail"] = "Not enough local points"
        return results

    dir_params1 = cv2.fitLine(local_points1, cv2.DIST_L2, 0, 0.01, 0.01)
    dir_params2 = cv2.fitLine(local_points2, cv2.DIST_L2, 0, 0.01, 0.01)
    dir_vec1, dir_vec2 = dir_params1[:2].flatten(), dir_params2[:2].flatten()

    if np.dot(dir_vec1, dir_vec2) < 0:
        dir_vec2 = -dir_vec2

    final_lane_direction = (dir_vec1 + dir_vec2) / 2.0
    final_lane_direction /= np.linalg.norm(final_lane_direction)

    global_points1 = _find_global_points_marching_optimized(initial_boundary1, final_lane_direction, boundary_mask,
                                                            vehicle_state)
    global_points2 = _find_global_points_marching_optimized(initial_boundary2, final_lane_direction, boundary_mask,
                                                            vehicle_state)

    if len(global_points1) < MIN_POINTS_FOR_FINAL_FIT or len(global_points2) < MIN_POINTS_FOR_FINAL_FIT:
        results["status_detail"] = "Not enough global points"
        return results

    final_line1_params = cv2.fitLine(global_points1, cv2.DIST_L2, 0, 0.01, 0.01)
    final_line2_params = cv2.fitLine(global_points2, cv2.DIST_L2, 0, 0.01, 0.01)

    proj_p1 = _project_point_onto_line(vehicle_pos_px, final_line1_params)
    proj_p2 = _project_point_onto_line(vehicle_pos_px, final_line2_params)
    stable_center_point = (proj_p1 + proj_p2) / 2.0

    distance = np.linalg.norm(vehicle_pos_px - stable_center_point)

    results.update({
        "status": "Success",
        "distance": distance,
        "global_points1": global_points1,
        "global_points2": global_points2,
        "line1_params": final_line1_params,
        "line2_params": final_line2_params,
        "center_point": stable_center_point,
        "lane_direction": final_lane_direction,
        "vehicle_pos_px": vehicle_pos_px,
        "proj_p1": proj_p1,
        "proj_p2": proj_p2
    })

    return results


# ==============================================================================
# BATCH PROCESSING FUNCTION (OPTIMIZED)
# ==============================================================================
def calculate_lane_distances(pos, yaw, maps, raster_from_agent, debug=False):
    T = pos.shape[0]
    distances = np.full(T, np.nan)

    cv_images = []
    boundary_masks = []

    for t in range(T):
        drivable, dividers, crosswalks = maps[t, 0], maps[t, 1], maps[t, 2]
        h, w = drivable.shape
        rgba = np.zeros((h, w, 4), dtype=np.float32)

        color_drivable = np.array([200, 211, 213]) / 255.0
        color_dividers = np.array([164, 184, 196]) / 255.0
        color_crosswalks = np.array([96, 117, 138]) / 255.0

        rgba[crosswalks > 0, :3] = color_crosswalks
        rgba[drivable > 0, :3] = color_drivable
        rgba[dividers > 0, :3] = color_dividers
        rgba[drivable > 0, 3] = 1.0

        cv_image = cv2.cvtColor((rgba[:, :, :3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv_images.append(cv_image)

        boundary_mask = _prepare_boundary_mask(cv_image)
        boundary_masks.append(boundary_mask)

    pos_px = np.zeros((T, 2))
    for t in range(T):
        pos_px[t] = _transform_world_to_pixel_batch(pos[t], raster_from_agent[t])[0]

    results_list = []
    for t in range(T):
        length_px, width_px = _get_pixel_extent(raster_from_agent[t])
        search_limits_px = (length_px * SEARCH_BOX_L_MULTIPLIER, width_px * SEARCH_BOX_W_MULTIPLIER)

        result = _calculate_single_frame_distance(
            boundary_masks[t],
            pos_px[t],
            yaw[t],
            search_limits_px
        )

        results_list.append(result)
        if result["status"] == "Success":
            distances[t] = result["distance"]


    if debug:
        _show_debug_visualization(T, cv_images, pos_px, yaw, raster_from_agent, results_list)

    return distances


def _show_debug_visualization(T, cv_images, pos_px, yaw, raster_from_agent, results_list):
    root = tk.Tk()
    root.title("Lane Distance Debug Viewer - Use ←→ keys or buttons to navigate")

    fig, ax = plt.subplots(figsize=(10, 10))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    control_frame = tk.Frame(root)
    control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

    button_frame = tk.Frame(control_frame)
    button_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

    slider_frame = tk.Frame(control_frame)
    slider_frame.pack(side=tk.TOP, fill=tk.X)

    slider_var = tk.IntVar(value=0)

    def set_frame(t):
        t = max(0, min(T - 1, t))
        slider_var.set(t)
        update_plot(t)

    btn_first = tk.Button(button_frame, text="⏮ First", command=lambda: set_frame(0), width=10)
    btn_first.pack(side=tk.LEFT, padx=2)

    btn_prev = tk.Button(button_frame, text="◀ Prev", command=lambda: set_frame(slider_var.get() - 1), width=10)
    btn_prev.pack(side=tk.LEFT, padx=2)

    frame_label = tk.Label(button_frame, text=f"Frame: 0 / {T - 1}", font=("Arial", 12, "bold"))
    frame_label.pack(side=tk.LEFT, padx=20)

    btn_next = tk.Button(button_frame, text="Next ▶", command=lambda: set_frame(slider_var.get() + 1), width=10)
    btn_next.pack(side=tk.LEFT, padx=2)

    btn_last = tk.Button(button_frame, text="Last ⏭", command=lambda: set_frame(T - 1), width=10)
    btn_last.pack(side=tk.LEFT, padx=2)

    tk.Label(slider_frame, text="Time Step:").pack(side=tk.LEFT, padx=5)
    slider = tk.Scale(
        slider_frame,
        from_=0,
        to=T - 1,
        orient=tk.HORIZONTAL,
        variable=slider_var,
        length=600
    )
    slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    def on_key_press(event):
        current = slider_var.get()
        if event.keysym == 'Left':
            set_frame(current - 1)
        elif event.keysym == 'Right':
            set_frame(current + 1)
        elif event.keysym == 'Home':
            set_frame(0)
        elif event.keysym == 'End':
            set_frame(T - 1)

    root.bind('<Left>', on_key_press)
    root.bind('<Right>', on_key_press)
    root.bind('<Home>', on_key_press)
    root.bind('<End>', on_key_press)

    def update_plot(t):
        t = int(t)
        ax.clear()

        frame_label.config(text=f"Frame: {t} / {T - 1}")

        rgba = cv2.cvtColor(cv_images[t], cv2.COLOR_BGR2RGB)
        ax.imshow(rgba)
        ax.grid(False)

        vehicle_pos = pos_px[t]
        yaw_rad = yaw[t]
        length_px, width_px = _get_pixel_extent(raster_from_agent[t])

        rect = ((vehicle_pos[0], vehicle_pos[1]), (length_px, width_px), -np.rad2deg(yaw_rad))
        box = np.int0(cv2.boxPoints(rect))
        ax.plot(np.append(box[:, 0], box[0, 0]), np.append(box[:, 1], box[0, 1]),
                color='purple', lw=2, zorder=10)

        heading_vec = np.array([np.cos(yaw_rad), -np.sin(yaw_rad)])
        ax.arrow(vehicle_pos[0], vehicle_pos[1], heading_vec[0] * 20, heading_vec[1] * 20,
                 head_width=5, head_length=5, fc='red', ec='red', lw=2, zorder=11)

        result = results_list[t]
        if result["status"] == "Success":
            for pt in result["global_points1"]:
                ax.plot(pt[0], pt[1], 'o', color='yellow', markersize=1)
            for pt in result["global_points2"]:
                ax.plot(pt[0], pt[1], 'o', color='yellow', markersize=1)

            draw_length = length_px * LINE_DRAW_LENGTH_MULTIPLIER

            def draw_line(line_params, center, color, lw):
                vx, vy = line_params[:2].flatten()
                p1 = center - np.array([vx, vy]) * draw_length / 2
                p2 = center + np.array([vx, vy]) * draw_length / 2
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw)

            draw_line(result["line1_params"], result["proj_p1"], 'blue', 2)
            draw_line(result["line2_params"], result["proj_p2"], 'blue', 2)

            center_line_params = np.array([*result["lane_direction"], *result["center_point"]])
            draw_line(center_line_params, result["center_point"], 'green', 2)

            ax.plot([vehicle_pos[0], result["center_point"][0]],
                    [vehicle_pos[1], result["center_point"][1]],
                    color='red', lw=2)

            title = f"T={t} | Status: Success | Distance: {result['distance']:.2f} px"
        else:
            title = f"T={t} | Status: {result['status']} | {result.get('status_detail', 'N/A')}"

        ax.set_title(title)
        canvas.draw()

    slider.config(command=lambda v: update_plot(v))
    update_plot(0)

    root.mainloop()


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================
if __name__ == '__main__':
    d = np.load("data2.npy", allow_pickle=True).item()
    ids = d['ids']
    traj = d['traj']

    data = traj[ids[0]]
    pos = data['positions']
    yaw = data['yaw']
    maps = data['map']
    raster = data['raster']
    T = yaw.shape[0]

    st = time.time()
    distances = calculate_lane_distances(pos, yaw, maps, raster, debug=False)
    print(f"Time: {time.time() - st}")

    distances = calculate_lane_distances(pos, yaw, maps, raster, debug=True)
    distances_filled = np.nan_to_num(distances, nan=0.0)
