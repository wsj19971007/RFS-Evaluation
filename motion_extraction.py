#!/usr/bin/env python3
"""
运动参数提取模块 v3.0
通过pose差分计算速度，不依赖twist.linear数据
支持命令行参数输入
"""

import json
import math
import sys
import argparse
import os
from collections import OrderedDict


def load_odometry_data(file_path):
    """
    加载Odometry JSON文件并按时间戳排序
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    # 按时间戳排序数据
    sorted_data = OrderedDict()
    for key in sorted(data.keys(), key=float):
        sorted_data[key] = data[key]

    return sorted_data


def extract_motion_data(sorted_data):
    """
    提取时间戳、位置和四元数数据
    """
    timestamps = []
    positions = []
    quaternions = []

    for key, frame_data in sorted_data.items():
        # 提取时间戳
        timestamp = frame_data['header']['time_stamp']
        timestamps.append(timestamp)

        # 提取位置 (x, y, z)
        pos = frame_data['pose']['position']
        positions.append([pos['x'], pos['y'], pos['z']])

        # 提取四元数 (x, y, z, w)
        quat = frame_data['pose']['quaternion']
        quaternions.append([quat['x'], quat['y'], quat['z'], quat['w']])

    return timestamps, positions, quaternions


def calculate_velocities_from_pose(timestamps, positions, quaternions):
    """
    通过位置差分计算世界坐标系下的速度
    """
    n_frames = len(timestamps)
    world_velocities = [[0.0, 0.0, 0.0] for _ in range(n_frames)]

    for i in range(n_frames):
        if i == 0:
            # 第一帧：使用下一帧计算
            dt = timestamps[i+1] - timestamps[i]
            if dt > 0:
                velocity = vector_divide(vector_subtract(positions[i+1], positions[i]), dt)
                world_velocities[i] = velocity
        elif i == n_frames - 1:
            # 最后一帧：使用上一帧计算
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                velocity = vector_divide(vector_subtract(positions[i], positions[i-1]), dt)
                world_velocities[i] = velocity
        else:
            # 中间帧：使用中心差分
            dt1 = timestamps[i] - timestamps[i-1]
            dt2 = timestamps[i+1] - timestamps[i]
            if dt1 > 0 and dt2 > 0:
                v1 = vector_divide(vector_subtract(positions[i], positions[i-1]), dt1)
                v2 = vector_divide(vector_subtract(positions[i+1], positions[i]), dt2)
                # 中心差分平均
                world_velocities[i] = vector_multiply([v1[j] + v2[j] for j in range(3)], 0.5)

    return world_velocities


def quaternion_to_rotation_matrix(q):
    """
    将四元数转换为旋转矩阵
    q = [x, y, z, w]
    """
    x, y, z, w = q

    # 归一化四元数
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if norm > 0:
        x, y, z, w = x/norm, y/norm, z/norm, w/norm

    # 计算旋转矩阵
    R = [
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ]

    return R


def matrix_multiply_vector(matrix, vector):
    """
    矩阵向量乘法
    """
    result = [0.0, 0.0, 0.0]
    for i in range(3):
        for j in range(3):
            result[i] += matrix[i][j] * vector[j]
    return result


def vector_subtract(v1, v2):
    """
    向量减法
    """
    return [v1[i] - v2[i] for i in range(len(v1))]


def vector_divide(v, scalar):
    """
    向量除法（处理除零）
    """
    if abs(scalar) < 1e-10:  # 防止除零
        return [0.0, 0.0, 0.0]
    return [v[i] / scalar for i in range(len(v))]


def vector_multiply(v, scalar):
    """
    向量乘法
    """
    return [v[i] * scalar for i in range(len(v))]


def quaternion_to_angular_velocity(quat_current, quat_next, dt):
    """
    将四元数变化转换为角速度（在世界坐标系下）
    改进的算法，更稳定的数值计算
    """
    # 归一化四元数
    quat_current = list(quat_current)
    quat_next = list(quat_next)

    norm_current = math.sqrt(sum(x*x for x in quat_current))
    norm_next = math.sqrt(sum(x*x for x in quat_next))

    if norm_current > 0:
        quat_current = [x/norm_current for x in quat_current]
    if norm_next > 0:
        quat_next = [x/norm_next for x in quat_next]

    # 计算相对旋转
    qc = [-quat_current[0], -quat_current[1], -quat_current[2], quat_current[3]]

    q_rel = [
        quat_next[3]*qc[0] + quat_next[0]*qc[3] + quat_next[1]*qc[2] - quat_next[2]*qc[1],
        quat_next[3]*qc[1] + quat_next[1]*qc[3] + quat_next[2]*qc[0] - quat_next[0]*qc[2],
        quat_next[3]*qc[2] + quat_next[2]*qc[3] + quat_next[0]*qc[1] - quat_next[1]*qc[0],
        quat_next[3]*qc[3] - quat_next[0]*qc[0] - quat_next[1]*qc[1] - quat_next[2]*qc[2]
    ]

    # 从相对四元数计算角速度向量
    if abs(q_rel[3]) < 1e-10:  # 防止除零
        return [0.0, 0.0, 0.0]

    angle = 2 * math.acos(abs(q_rel[3]))
    if abs(angle) < 1e-10:  # 小角度近似
        return [2.0 * q_rel[0] / dt, 2.0 * q_rel[1] / dt, 2.0 * q_rel[2] / dt]

    sin_half_angle = math.sin(angle / 2)
    if abs(sin_half_angle) < 1e-10:  # 防止除零
        return [0.0, 0.0, 0.0]

    axis = [q_rel[0] / sin_half_angle, q_rel[1] / sin_half_angle, q_rel[2] / sin_half_angle]

    return [axis[i] * angle / dt for i in range(3)]


def transform_to_vehicle_frame(world_vector, quaternion):
    """
    将世界坐标系下的向量转换到车辆坐标系
    """
    # 获取旋转矩阵
    R = quaternion_to_rotation_matrix(quaternion)

    # 计算旋转矩阵的转置（逆矩阵）
    R_T = [[R[j][i] for j in range(3)] for i in range(3)]

    # 进行坐标变换
    vehicle_vector = matrix_multiply_vector(R_T, world_vector)

    return vehicle_vector


def low_pass_filter(data, alpha=0.1):
    """
    低通滤波器，平滑数据
    alpha: 滤波系数 (0-1)，越小越平滑
    """
    if not data:
        return data

    filtered_data = [data[0]]  # 初始值

    for i in range(1, len(data)):
        filtered_value = alpha * data[i] + (1 - alpha) * filtered_data[i-1]
        filtered_data.append(filtered_value)

    return filtered_data


def enhanced_low_pass_filter(data, alpha=0.1, iterations=2):
    """
    增强型低通滤波器，多次滤波以获得更好的平滑效果
    alpha: 滤波系数 (0-1)，越小越平滑
    iterations: 滤波迭代次数
    """
    if not data:
        return data

    filtered_data = data[:]
    for _ in range(iterations):
        filtered_data = low_pass_filter(filtered_data, alpha)

    return filtered_data


def robust_jerk_calculation(timestamps, accelerations):
    """
    更稳健的急动度计算方法
    """
    n_frames = len(timestamps)
    jerks = [[0.0, 0.0, 0.0] for _ in range(n_frames)]

    # 首先对加速度进行强滤波
    filtered_accelerations = [[0.0, 0.0, 0.0] for _ in range(n_frames)]
    for i in range(3):
        acc_i = [a[i] for a in accelerations]
        # 使用更强的滤波
        acc_i_filtered = enhanced_low_pass_filter(acc_i, 0.15, 3)  # α=0.15, 3次迭代
        for j, value in enumerate(acc_i_filtered):
            filtered_accelerations[j][i] = value

    # 使用改进的差分方法计算急动度
    for i in range(n_frames):
        if i == 0:
            # 第一帧：使用前三点的最小二乘法
            if n_frames >= 3:
                dt1 = timestamps[1] - timestamps[0]
                dt2 = timestamps[2] - timestamps[0]
                if dt1 > 0 and dt2 > 0:
                    # 最小二乘法拟合
                    a1 = filtered_accelerations[0]
                    a2 = filtered_accelerations[1]
                    a3 = filtered_accelerations[2]
                    for j in range(3):
                        # 使用二次多项式拟合的导数
                        jerks[i][j] = (3*a1[j] - 4*a2[j] + a3[j]) / dt1
                else:
                    # 回退到简单差分
                    dt = timestamps[1] - timestamps[0]
                    if dt > 0:
                        jerks[i] = vector_divide(vector_subtract(filtered_accelerations[1], filtered_accelerations[0]), dt)
        elif i == n_frames - 1:
            # 最后一帧：使用前两帧的差分
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                jerks[i] = vector_divide(vector_subtract(filtered_accelerations[i], filtered_accelerations[i-1]), dt)
        elif i == 1:
            # 第二帧：使用前三点
            dt1 = timestamps[i] - timestamps[i-1]
            dt2 = timestamps[i+1] - timestamps[i]
            if dt1 > 0 and dt2 > 0:
                j1 = vector_divide(vector_subtract(filtered_accelerations[i], filtered_accelerations[i-1]), dt1)
                j2 = vector_divide(vector_subtract(filtered_accelerations[i+1], filtered_accelerations[i]), dt2)
                jerks[i] = vector_multiply([j1[j] + j2[j] for j in range(3)], 0.5)
        else:
            # 中间帧：使用五点中心差分（如果可用）
            if i >= 2 and i <= n_frames - 3:
                # 五点中心差分公式
                dt = timestamps[i+1] - timestamps[i]
                if dt > 0:
                    for j in range(3):
                        jerk_val = (
                            filtered_accelerations[i-2][j] - 8*filtered_accelerations[i-1][j] +
                            8*filtered_accelerations[i+1][j] - filtered_accelerations[i+2][j]
                        ) / (12 * dt)
                        jerks[i][j] = jerk_val
            else:
                # 回退到三点中心差分
                dt1 = timestamps[i] - timestamps[i-1]
                dt2 = timestamps[i+1] - timestamps[i]
                if dt1 > 0 and dt2 > 0:
                    j1 = vector_divide(vector_subtract(filtered_accelerations[i], filtered_accelerations[i-1]), dt1)
                    j2 = vector_divide(vector_subtract(filtered_accelerations[i+1], filtered_accelerations[i]), dt2)
                    jerks[i] = vector_multiply([j1[j] + j2[j] for j in range(3)], 0.5)

    # 对急动度进行最后的滤波
    for i in range(3):
        jerk_i = [j[i] for j in jerks]
        jerk_i_filtered = enhanced_low_pass_filter(jerk_i, 0.2, 2)  # 强滤波
        for j, value in enumerate(jerk_i_filtered):
            jerks[j][i] = value

    # 限制急动度的最大值以避免异常值
    max_jerk = 10.0  # m/s³，合理的急动度上限
    for i in range(n_frames):
        for j in range(3):
            if abs(jerks[i][j]) > max_jerk:
                jerks[i][j] = max_jerk if jerks[i][j] > 0 else -max_jerk

    return jerks


def calculate_derivatives_from_pose(timestamps, positions, quaternions):
    """
    通过pose计算车辆坐标系下的运动参数
    """
    n_frames = len(timestamps)

    # 初始化结果数组
    vehicle_velocities = [[0.0, 0.0, 0.0] for _ in range(n_frames)]
    vehicle_accelerations = [[0.0, 0.0, 0.0] for _ in range(n_frames)]
    vehicle_jerks = [[0.0, 0.0, 0.0] for _ in range(n_frames)]
    angular_velocities = [[0.0, 0.0, 0.0] for _ in range(n_frames)]
    angular_accelerations = [[0.0, 0.0, 0.0] for _ in range(n_frames)]

    # 通过位置差分计算世界坐标系下的速度
    world_velocities = calculate_velocities_from_pose(timestamps, positions, quaternions)

    # 将世界坐标系速度转换到车辆坐标系
    for i in range(n_frames):
        vehicle_velocities[i] = transform_to_vehicle_frame(world_velocities[i], quaternions[i])

    # 对速度数据进行低通滤波以减少噪声
    for i in range(3):
        vel_i = [v[i] for v in vehicle_velocities]
        vel_i_filtered = low_pass_filter(vel_i, 0.2)  # 较强的滤波
        for j, value in enumerate(vel_i_filtered):
            vehicle_velocities[j][i] = value

    # 计算车辆坐标系下的加速度（通过滤波后速度差分）
    for i in range(n_frames):
        if i == 0:
            dt = timestamps[i+1] - timestamps[i]
            if dt > 0:
                vehicle_accelerations[i] = vector_divide(vector_subtract(vehicle_velocities[i+1], vehicle_velocities[i]), dt)
        elif i == n_frames - 1:
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                vehicle_accelerations[i] = vector_divide(vector_subtract(vehicle_velocities[i], vehicle_velocities[i-1]), dt)
        else:
            # 使用中心差分，更稳定
            dt1 = timestamps[i] - timestamps[i-1]
            dt2 = timestamps[i+1] - timestamps[i]
            if dt1 > 0 and dt2 > 0:
                v1 = vector_divide(vector_subtract(vehicle_velocities[i], vehicle_velocities[i-1]), dt1)
                v2 = vector_divide(vector_subtract(vehicle_velocities[i+1], vehicle_velocities[i]), dt2)
                vehicle_accelerations[i] = vector_multiply([v1[j] + v2[j] for j in range(3)], 0.5)

    # 对加速度数据进行增强低通滤波
    for i in range(3):
        acc_i = [a[i] for a in vehicle_accelerations]
        acc_i_filtered = enhanced_low_pass_filter(acc_i, 0.2, 2)  # 更强的滤波
        for j, value in enumerate(acc_i_filtered):
            vehicle_accelerations[j][i] = value

    # 使用改进的急动度计算方法
    vehicle_jerks = robust_jerk_calculation(timestamps, vehicle_accelerations)

    # 计算角速度（改进的四元数方法）
    for i in range(n_frames):
        if i == 0:
            dt = timestamps[i+1] - timestamps[i]
            if dt > 0:
                angular_velocities[i] = quaternion_to_angular_velocity(quaternions[i], quaternions[i+1], dt)
        elif i == n_frames - 1:
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                angular_velocities[i] = quaternion_to_angular_velocity(quaternions[i-1], quaternions[i], dt)
        else:
            dt1 = timestamps[i] - timestamps[i-1]
            dt2 = timestamps[i+1] - timestamps[i]
            if dt1 > 0 and dt2 > 0:
                w1 = quaternion_to_angular_velocity(quaternions[i-1], quaternions[i], dt1)
                w2 = quaternion_to_angular_velocity(quaternions[i], quaternions[i+1], dt2)
                angular_velocities[i] = vector_multiply([w1[j] + w2[j] for j in range(3)], 0.5)

    # 将角速度转换到车辆坐标系
    for i in range(n_frames):
        angular_velocities[i] = transform_to_vehicle_frame(angular_velocities[i], quaternions[i])

    # 对角速度进行滤波
    for i in range(3):
        ang_vel_i = [w[i] for w in angular_velocities]
        ang_vel_i_filtered = low_pass_filter(ang_vel_i, 0.3)
        for j, value in enumerate(ang_vel_i_filtered):
            angular_velocities[j][i] = value

    # 计算角加速度
    for i in range(n_frames):
        if i == 0:
            dt = timestamps[i+1] - timestamps[i]
            if dt > 0:
                angular_accelerations[i] = vector_divide(vector_subtract(angular_velocities[i+1], angular_velocities[i]), dt)
        elif i == n_frames - 1:
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                angular_accelerations[i] = vector_divide(vector_subtract(angular_velocities[i], angular_velocities[i-1]), dt)
        else:
            dt1 = timestamps[i] - timestamps[i-1]
            dt2 = timestamps[i+1] - timestamps[i]
            if dt1 > 0 and dt2 > 0:
                a1 = vector_divide(vector_subtract(angular_velocities[i], angular_velocities[i-1]), dt1)
                a2 = vector_divide(vector_subtract(angular_velocities[i+1], angular_velocities[i]), dt2)
                angular_accelerations[i] = vector_multiply([a1[j] + a2[j] for j in range(3)], 0.5)

    # 对角加速度进行滤波
    for i in range(3):
        ang_acc_i = [a[i] for a in angular_accelerations]
        ang_acc_i_filtered = low_pass_filter(ang_acc_i, 0.3)
        for j, value in enumerate(ang_acc_i_filtered):
            angular_accelerations[j][i] = value

    return vehicle_velocities, vehicle_accelerations, vehicle_jerks, angular_velocities, angular_accelerations


def save_results_to_csv(timestamps, positions, vehicle_velocities, vehicle_accelerations,
                       vehicle_jerks, angular_velocities, angular_accelerations, output_file):
    """
    保存结果到CSV文件
    """
    with open(output_file, 'w') as f:
        # 写入标题
        header = ['timestamp', 'pos_x', 'pos_y', 'pos_z',
                 'vehicle_vel_x', 'vehicle_vel_y', 'vehicle_vel_z',
                 'vehicle_acc_x', 'vehicle_acc_y', 'vehicle_acc_z',
                 'vehicle_jerk_x', 'vehicle_jerk_y', 'vehicle_jerk_z',
                 'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
                 'angular_acc_x', 'angular_acc_y', 'angular_acc_z']
        f.write(','.join(header) + '\n')

        # 写入数据
        for i in range(len(timestamps)):
            row = [
                str(timestamps[i]),
                str(positions[i][0]), str(positions[i][1]), str(positions[i][2]),
                str(vehicle_velocities[i][0]), str(vehicle_velocities[i][1]), str(vehicle_velocities[i][2]),
                str(vehicle_accelerations[i][0]), str(vehicle_accelerations[i][1]), str(vehicle_accelerations[i][2]),
                str(vehicle_jerks[i][0]), str(vehicle_jerks[i][1]), str(vehicle_jerks[i][2]),
                str(angular_velocities[i][0]), str(angular_velocities[i][1]), str(angular_velocities[i][2]),
                str(angular_accelerations[i][0]), str(angular_accelerations[i][1]), str(angular_accelerations[i][2])
            ]
            f.write(','.join(row) + '\n')


def calculate_statistics(data):
    """
    计算数据的统计信息
    """
    if not data:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'rms': 0}

    # 计算均值
    mean = sum(data) / len(data)

    # 计算标准差
    variance = sum([(x - mean) ** 2 for x in data]) / len(data)
    std = math.sqrt(variance)

    # 计算最小值和最大值
    min_val = min(data)
    max_val = max(data)

    # 计算RMS
    rms = math.sqrt(sum([x*x for x in data]) / len(data))

    return {'mean': mean, 'std': std, 'min': min_val, 'max': max_val, 'rms': rms}


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description='运动参数提取模块 v3.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python3 motion_extraction.py input.json
  python3 motion_extraction.py input.json -o output.csv
  python3 motion_extraction.py input.json --verbose
        """
    )

    parser.add_argument(
        'json_file',
        help='Odometry JSON文件路径'
    )

    parser.add_argument(
        '-o', '--output',
        default='motion_parameters_pose.csv',
        help='输出CSV文件路径 (默认: motion_parameters_pose.csv)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细输出信息'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='运动参数提取模块 v3.0.0'
    )

    return parser.parse_args()


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_arguments()

    # 获取参数
    json_file_path = args.json_file
    output_file = args.output
    verbose = args.verbose

    if verbose:
        print(f"输入文件: {json_file_path}")
        print(f"输出文件: {output_file}")

    # 检查输入文件是否存在
    if not os.path.exists(json_file_path):
        print(f"错误: 文件不存在: {json_file_path}")
        sys.exit(1)

    print("正在加载Odometry数据...")
    try:
        sorted_data = load_odometry_data(json_file_path)
    except Exception as e:
        print(f"错误: 加载JSON文件失败: {e}")
        sys.exit(1)

    print("正在提取运动数据...")
    try:
        timestamps, positions, quaternions = extract_motion_data(sorted_data)
    except Exception as e:
        print(f"错误: 提取运动数据失败: {e}")
        sys.exit(1)

    if len(timestamps) == 0:
        print("错误: 没有找到有效的数据")
        sys.exit(1)

    print(f"总共 {len(timestamps)} 帧数据")
    print(f"时间范围: {timestamps[0]:.3f} - {timestamps[-1]:.3f} 秒")
    print(f"使用pose差分计算速度: 是")

    if verbose:
        print("数据样本:")
        print(f"  时间戳范围: {min(timestamps):.6f} - {max(timestamps):.6f}")
        print(f"  位置范围: X[{min(p[0] for p in positions):.3f}, {max(p[0] for p in positions):.3f}]")
        print(f"             Y[{min(p[1] for p in positions):.3f}, {max(p[1] for p in positions):.3f}]")
        print(f"             Z[{min(p[2] for p in positions):.3f}, {max(p[2] for p in positions):.3f}]")

    print("正在计算运动参数（基于pose差分算法）...")
    try:
        vehicle_velocities, vehicle_accelerations, vehicle_jerks, angular_velocities, angular_accelerations = calculate_derivatives_from_pose(
            timestamps, positions, quaternions)
    except Exception as e:
        print(f"错误: 计算运动参数失败: {e}")
        sys.exit(1)

    print("正在保存结果...")
    try:
        save_results_to_csv(timestamps, positions, vehicle_velocities, vehicle_accelerations, vehicle_jerks,
                        angular_velocities, angular_accelerations, output_file)
        print(f"结果已保存到: {output_file}")
    except Exception as e:
        print(f"错误: 保存结果失败: {e}")
        sys.exit(1)

    # 显示统计信息
    print("\n=== 运动参数统计信息（基于pose差分） ===")

    print("\n车辆坐标系线速度 (m/s):")
    print("  X轴(纵向): 前进方向")
    print("  Y轴(横向): 左侧方向")
    print("  Z轴(垂直): 上方方向")

    for i, axis_name in enumerate(['X(纵向)', 'Y(横向)', 'Z(垂直)']):
        vel_i = [v[i] for v in vehicle_velocities]
        stats = calculate_statistics(vel_i)
        print(f"  {axis_name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, rms={stats['rms']:.4f}, range=[{stats['min']:.4f}, {stats['max']:.4f}]")

    print("\n车辆坐标系线加速度 (m/s²):")
    for i, axis_name in enumerate(['X(纵向)', 'Y(横向)', 'Z(垂直)']):
        acc_i = [a[i] for a in vehicle_accelerations]
        stats = calculate_statistics(acc_i)
        print(f"  {axis_name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, rms={stats['rms']:.4f}, range=[{stats['min']:.4f}, {stats['max']:.4f}]")

    print("\n车辆坐标系急动度 (m/s³):")
    for i, axis_name in enumerate(['X(纵向)', 'Y(横向)', 'Z(垂直)']):
        jerk_i = [j[i] for j in vehicle_jerks]
        stats = calculate_statistics(jerk_i)
        print(f"  {axis_name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, rms={stats['rms']:.4f}, range=[{stats['min']:.4f}, {stats['max']:.4f}]")

    print("\n车辆坐标系角速度 (rad/s):")
    for i, axis_name in enumerate(['X(横滚)', 'Y(俯仰)', 'Z(偏航)']):
        angular_vel_i = [w[i] for w in angular_velocities]
        stats = calculate_statistics(angular_vel_i)
        print(f"  {axis_name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, rms={stats['rms']:.4f}, range=[{stats['min']:.4f}, {stats['max']:.4f}]")

    print("\n车辆坐标系角加速度 (rad/s²):")
    for i, axis_name in enumerate(['X(横滚)', 'Y(俯仰)', 'Z(偏航)']):
        angular_acc_i = [a[i] for a in angular_accelerations]
        stats = calculate_statistics(angular_acc_i)
        print(f"  {axis_name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, rms={stats['rms']:.4f}, range=[{stats['min']:.4f}, {stats['max']:.4f}]")

    # 显示前几行结果
    print("\n=== 前5行结果示例 ===")
    print("timestamp,pos_x,pos_y,pos_z,vehicle_vel_x,vehicle_vel_y,vehicle_vel_z,vehicle_acc_x,vehicle_acc_y,vehicle_acc_z,vehicle_jerk_x,vehicle_jerk_y,vehicle_jerk_z,angular_vel_x,angular_vel_y,angular_vel_z,angular_acc_x,angular_acc_y,angular_acc_z")
    for i in range(min(5, len(timestamps))):
        row = [
            f"{timestamps[i]:.6f}",
            f"{positions[i][0]:.6f}", f"{positions[i][1]:.6f}", f"{positions[i][2]:.6f}",
            f"{vehicle_velocities[i][0]:.6f}", f"{vehicle_velocities[i][1]:.6f}", f"{vehicle_velocities[i][2]:.6f}",
            f"{vehicle_accelerations[i][0]:.6f}", f"{vehicle_accelerations[i][1]:.6f}", f"{vehicle_accelerations[i][2]:.6f}",
            f"{vehicle_jerks[i][0]:.6f}", f"{vehicle_jerks[i][1]:.6f}", f"{vehicle_jerks[i][2]:.6f}",
            f"{angular_velocities[i][0]:.6f}", f"{angular_velocities[i][1]:.6f}", f"{angular_velocities[i][2]:.6f}",
            f"{angular_accelerations[i][0]:.6f}", f"{angular_accelerations[i][1]:.6f}", f"{angular_accelerations[i][2]:.6f}"
        ]
        print(','.join(row))

    print("\n✅ 运动参数提取完成（基于pose差分）!")

    return {
        'vehicle_velocities': vehicle_velocities,
        'vehicle_accelerations': vehicle_accelerations,
        'vehicle_jerks': vehicle_jerks,
        'angular_velocities': angular_velocities,
        'angular_accelerations': angular_accelerations
    }


if __name__ == "__main__":
    main()