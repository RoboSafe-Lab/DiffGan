import numpy as np


def point_to_segment_distance_batch(points, seg_starts, seg_ends):
    """
    批量计算点到线段的最短距离（NumPy版本）

    Args:
        points: (M, 2) 点坐标 [x, y]
        seg_starts: (N, 2) 线段起点 [x, y]
        seg_ends: (N, 2) 线段终点 [x, y]

    Returns:
        distances: (M, N) 每个点到每条线段的距离
    """
    # 线段向量 (N, 2)
    segment_vecs = seg_ends - seg_starts
    segment_len_sq = np.sum(segment_vecs ** 2, axis=1, keepdims=True)  # (N, 1)

    # 处理退化线段（长度为0）
    valid_segments = segment_len_sq.squeeze() > 1e-10

    # 点到起点的向量 (M, 1, 2) - (1, N, 2) = (M, N, 2)
    point_vecs = points[:, np.newaxis, :] - seg_starts[np.newaxis, :, :]

    # 计算投影参数 t
    # (M, N, 2) * (1, N, 2) -> (M, N)
    t = np.sum(point_vecs * segment_vecs[np.newaxis, :, :], axis=2) / (segment_len_sq.squeeze() + 1e-10)
    t = np.clip(t, 0.0, 1.0)  # (M, N)

    # 计算最近点 (M, N, 2)
    closest_points = seg_starts[np.newaxis, :, :] + t[:, :, np.newaxis] * segment_vecs[np.newaxis, :, :]

    # 计算距离 (M, N)
    distances = np.linalg.norm(points[:, np.newaxis, :] - closest_points, axis=2)

    return distances


def calculate_lane_distance(pos, lanes):
    """
    :param pos: 车辆轨迹 (T,2)
    :param lanes: 车道线 (T,15,80,3) 对每时刻，都会获取最近的15条车道线，每条车道线包含80个点，每个点有x,y,z三个坐标表示
    :return: 车辆距离车道线的距离 (T,)
    """
    T = pos.shape[0]
    num_lanes = lanes.shape[1]
    num_points = lanes.shape[2]

    # 初始化距离数组
    distances = np.full((T,), np.inf)

    # 提取车道线xy坐标 (T, 15, 80, 2)
    lane_xy = lanes[:, :, :, :2]

    # 对每个时间步进行处理
    for t in range(T):
        # 当前时间步的车辆位置 (2,)
        vehicle_pos = pos[t]  # (2,)

        # 当前时间步的车道线 (15, 80, 2)
        time_lanes = lane_xy[t]  # (15, 80, 2)

        # 存储当前时间步的最小距离
        min_distance = np.inf

        # 对每条车道
        for lane_idx in range(num_lanes):
            lane_points = time_lanes[lane_idx]  # (80, 2)

            # 过滤有效点
            # 检查是否为NaN或接近0
            valid_mask = ~(np.isnan(lane_points[:, 0]) | np.isnan(lane_points[:, 1]) |
                          ((np.abs(lane_points[:, 0]) < 1e-6) & (np.abs(lane_points[:, 1]) < 1e-6)))

            if valid_mask.sum() == 0:
                continue

            valid_lane_points = lane_points[valid_mask]  # (num_valid, 2)
            num_valid = valid_lane_points.shape[0]

            if num_valid == 0:
                continue

            # 1. 计算到车道点的距离
            # (1, 2) - (num_valid, 2) = (num_valid, 2)
            diffs = vehicle_pos[np.newaxis, :] - valid_lane_points
            point_distances = np.linalg.norm(diffs, axis=1)  # (num_valid,)
            min_point_dist = np.min(point_distances)

            # 2. 计算到线段的距离（如果有多个点）
            if num_valid > 1:
                seg_starts = valid_lane_points[:-1]  # (num_valid-1, 2)
                seg_ends = valid_lane_points[1:]     # (num_valid-1, 2)

                # 批量计算车辆位置到所有线段的距离
                segment_distances = point_to_segment_distance_batch(
                    vehicle_pos[np.newaxis, :], seg_starts, seg_ends
                )  # (1, num_valid-1)

                min_segment_dist = np.min(segment_distances)

                # 取点距离和线段距离的最小值
                lane_distance = min(min_point_dist, min_segment_dist)
            else:
                lane_distance = min_point_dist

            # 更新最小距离
            min_distance = min(min_distance, lane_distance)

        # 处理无效的车辆位置（NaN）或没有找到有效车道
        # Use 0.0 instead of NaN to avoid issues in feature extraction
        if np.isnan(vehicle_pos[0]) or np.isnan(vehicle_pos[1]):
            distances[t] = 0.0
        # 如果没有找到有效车道，设为0
        elif np.isinf(min_distance):
            distances[t] = 0.0
        else:
            distances[t] = min_distance

    return distances
