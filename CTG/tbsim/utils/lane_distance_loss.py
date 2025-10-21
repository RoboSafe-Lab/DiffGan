import torch


def point_to_segment_distance_batch(points, seg_starts, seg_ends):
    """
    批量计算点到线段的最短距离（GPU加速）

    Args:
        points: (M, 2) 点坐标 [x, y]
        seg_starts: (N, 2) 线段起点 [x, y]
        seg_ends: (N, 2) 线段终点 [x, y]

    Returns:
        distances: (M, N) 每个点到每条线段的距离
    """
    # 线段向量 (N, 2)
    segment_vecs = seg_ends - seg_starts
    segment_len_sq = torch.sum(segment_vecs ** 2, dim=1, keepdim=True)  # (N, 1)

    # 处理退化线段（长度为0）
    valid_segments = segment_len_sq.squeeze() > 1e-10

    # 点到起点的向量 (M, 1, 2) - (1, N, 2) = (M, N, 2)
    point_vecs = points.unsqueeze(1) - seg_starts.unsqueeze(0)

    # 计算投影参数 t
    # (M, N, 2) * (1, N, 2) -> (M, N)
    t = torch.sum(point_vecs * segment_vecs.unsqueeze(0), dim=2) / (segment_len_sq.squeeze() + 1e-10)
    t = torch.clamp(t, 0.0, 1.0)  # (M, N)

    # 计算最近点 (M, N, 2)
    closest_points = seg_starts.unsqueeze(0) + t.unsqueeze(2) * segment_vecs.unsqueeze(0)

    # 计算距离 (M, N)
    distances = torch.norm(points.unsqueeze(1) - closest_points, dim=2)

    return distances


def calculate_lane_distance(x, data_batch):
    """
    计算车辆到最近车道线的距离（GPU加速版本）

    Args:
        x: (B, N, T, 6) 车辆轨迹数据，前两位为x, y坐标（GPU tensor）
        data_batch: 包含车道线数据的字典
            - data_batch['extras']['closest_lane_point']: (B, 15, 80, 3) 车道线数据（GPU tensor）

    Returns:
        distances: (B, N, T) 每个时间步车辆到最近车道线的距离（GPU tensor）
    """
    device = x.device
    dtype = x.dtype

    lanes = data_batch['extras']['closest_lane_point']  # (B, 15, 80, 3)

    B, N, T, _ = x.shape
    _, num_lanes, num_points, _ = lanes.shape

    # 初始化距离数组
    distances = torch.full((B, N, T), float('inf'), device=device, dtype=dtype)

    # 提取车辆位置 (B, N, T, 2)
    vehicle_positions = x[:, :, :, :2]

    # 提取车道线xy坐标 (B, 15, 80, 2)
    lane_xy = lanes[:, :, :, :2]

    # 对每个batch进行处理
    for b in range(B):
        # 当前batch的车辆位置 (N, T, 2)
        batch_vehicles = vehicle_positions[b]  # (N, T, 2)

        # 当前batch的车道线 (15, 80, 2)
        batch_lanes = lane_xy[b]  # (15, 80, 2)

        # 重塑车辆位置为 (N*T, 2) 以便批量处理
        vehicles_flat = batch_vehicles.reshape(-1, 2)  # (N*T, 2)

        # 存储当前batch的最小距离
        batch_min_distances = torch.full((N * T,), float('inf'), device=device, dtype=dtype)

        # 对每条车道
        for lane_idx in range(num_lanes):
            lane_points = batch_lanes[lane_idx]  # (80, 2)

            # 过滤有效点
            # 检查是否为NaN或接近0
            valid_mask = ~(torch.isnan(lane_points[:, 0]) | torch.isnan(lane_points[:, 1]) |
                          ((torch.abs(lane_points[:, 0]) < 1e-6) & (torch.abs(lane_points[:, 1]) < 1e-6)))

            if valid_mask.sum() == 0:
                continue

            valid_lane_points = lane_points[valid_mask]  # (num_valid, 2)
            num_valid = valid_lane_points.shape[0]

            if num_valid == 0:
                continue

            # 1. 计算到车道点的距离
            # (N*T, 1, 2) - (1, num_valid, 2) = (N*T, num_valid, 2)
            diffs = vehicles_flat.unsqueeze(1) - valid_lane_points.unsqueeze(0)
            point_distances = torch.norm(diffs, dim=2)  # (N*T, num_valid)
            min_point_dist = torch.min(point_distances, dim=1)[0]  # (N*T,)

            # 2. 计算到线段的距离（如果有多个点）
            if num_valid > 1:
                seg_starts = valid_lane_points[:-1]  # (num_valid-1, 2)
                seg_ends = valid_lane_points[1:]     # (num_valid-1, 2)

                # 批量计算所有车辆到所有线段的距离
                segment_distances = point_to_segment_distance_batch(
                    vehicles_flat, seg_starts, seg_ends
                )  # (N*T, num_valid-1)

                min_segment_dist = torch.min(segment_distances, dim=1)[0]  # (N*T,)

                # 取点距离和线段距离的最小值
                lane_distances = torch.min(min_point_dist, min_segment_dist)
            else:
                lane_distances = min_point_dist

            # 更新最小距离
            batch_min_distances = torch.min(batch_min_distances, lane_distances)

        # 处理无效的车辆位置（NaN）
        invalid_vehicles = torch.isnan(vehicles_flat[:, 0]) | torch.isnan(vehicles_flat[:, 1])
        batch_min_distances[invalid_vehicles] = float('nan')

        # 如果没有找到有效车道，设为NaN
        batch_min_distances[torch.isinf(batch_min_distances)] = float('nan')

        # 重塑回 (N, T) 并存储
        distances[b] = batch_min_distances.reshape(N, T)

    return distances


