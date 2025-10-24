#!/usr/bin/env python3
"""
RFS舒适度评价Markdown报告生成器
包含车辆轨迹图、运动参数图表和详细的评价结果
"""

import json
import os
import math
import sys
import argparse
from datetime import datetime

# 检查是否安装了matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib import font_manager
    import platform
    matplotlib.use('Agg')  # 使用非交互式后端
    MATPLOTLIB_AVAILABLE = True

    # 配置中文字体支持
    def configure_chinese_font():
        """配置matplotlib支持中文显示"""
        system = platform.system()

        if system == "Windows":
            # Windows系统字体
            chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
        elif system == "Darwin":  # macOS
            chinese_fonts = ['PingFang SC', 'Helvetica Neue', 'STHeiti', 'Arial Unicode MS']
        else:  # Linux
            chinese_fonts = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'WenQuanYi Micro Hei']

        # 添加通用的后备字体
        chinese_fonts.extend(['DejaVu Sans', 'Arial', 'Liberation Sans'])

        # 设置字体
        plt.rcParams['font.sans-serif'] = chinese_fonts
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.family'] = 'sans-serif'

        # 尝试设置更具体的字体参数
        try:
            # 获取系统字体
            available_fonts = [f.name for f in font_manager.fontManager.ttflist]

            # 查找可用的中文字体
            for font in chinese_fonts:
                if font in available_fonts:
                    plt.rcParams['font.sans-serif'] = [font] + chinese_fonts[1:]
                    print(f"✅ 使用字体: {font}")
                    break
            else:
                print("⚠️  未找到合适的中文字体，使用默认字体")

        except Exception as e:
            print(f"⚠️  字体配置警告: {e}")

        # 设置其他字体参数
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 11
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10

    # 配置字体
    configure_chinese_font()

except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib未安装，将跳过图表生成")


def load_motion_data_from_csv(csv_file):
    """
    从CSV文件加载运动数据
    """
    motion_data = {
        'vehicle_velocities': [],
        'vehicle_accelerations': [],
        'vehicle_jerks': [],
        'angular_velocities': [],
        'angular_accelerations': [],
        'positions': [],
        'timestamps': []
    }

    try:
        with open(csv_file, 'r') as f:
            lines = f.readlines()

            # 跳过标题行
            for line in lines[1:]:
                if line.strip():
                    parts = line.strip().split(',')
                    if len(parts) >= 19:
                        motion_data['timestamps'].append(float(parts[0]))

                        # 提取位置 (列2-4)
                        position = [float(parts[1]), float(parts[2]), float(parts[3])]
                        motion_data['positions'].append(position)

                        # 提取车辆速度 (列5-7)
                        vehicle_vel = [float(parts[4]), float(parts[5]), float(parts[6])]
                        motion_data['vehicle_velocities'].append(vehicle_vel)

                        # 提取车辆加速度 (列8-10)
                        vehicle_acc = [float(parts[7]), float(parts[8]), float(parts[9])]
                        motion_data['vehicle_accelerations'].append(vehicle_acc)

                        # 提取急动度 (列11-13)
                        vehicle_jerk = [float(parts[10]), float(parts[11]), float(parts[12])]
                        motion_data['vehicle_jerks'].append(vehicle_jerk)

                        # 提取角速度 (列14-16)
                        angular_vel = [float(parts[13]), float(parts[14]), float(parts[15])]
                        motion_data['angular_velocities'].append(angular_vel)

                        # 提取角加速度 (列17-19)
                        angular_acc = [float(parts[16]), float(parts[17]), float(parts[18])]
                        motion_data['angular_accelerations'].append(angular_acc)

        print(f"成功加载 {len(motion_data['vehicle_velocities'])} 帧运动数据")
        return motion_data

    except Exception as e:
        print(f"加载CSV文件时出错: {e}")
        return None


def load_comfort_results_from_json(json_file):
    """
    从JSON文件加载舒适度评价结果
    """
    try:
        with open(json_file, 'r') as f:
            results = json.load(f)
        print(f"成功加载舒适度评价结果")
        return results
    except Exception as e:
        print(f"加载JSON文件时出错: {e}")
        return None


def create_trajectory_plot(positions, timestamps, output_path):
    """
    创建车辆轨迹图
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    try:
        # 重新配置字体，确保中文显示正常
        configure_chinese_font()
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]

        # 创建图表
        plt.figure(figsize=(12, 8))

        # 绘制轨迹线
        plt.plot(x_coords, y_coords, 'b-', linewidth=2.5, alpha=0.8, label='Vehicle Path')

        # 标记起点和终点
        plt.plot(x_coords[0], y_coords[0], 'go', markersize=12, label='Start Point',
                markeredgecolor='darkgreen', markeredgewidth=2)
        plt.plot(x_coords[-1], y_coords[-1], 'ro', markersize=12, label='End Point',
                markeredgecolor='darkred', markeredgewidth=2)

        # 添加方向箭头
        n_arrows = min(15, len(positions) // 30)
        if n_arrows > 0:
            arrow_indices = [i * len(positions) // (n_arrows + 1) for i in range(1, n_arrows + 1)]
            for idx in arrow_indices:
                if idx < len(positions) - 1:
                    dx = x_coords[idx + 1] - x_coords[idx]
                    dy = y_coords[idx + 1] - y_coords[idx]
                    # 归一化方向向量
                    length = math.sqrt(dx**2 + dy**2)
                    if length > 0:
                        dx_norm = dx / length * 2  # 箭头长度
                        dy_norm = dy / length * 2
                        plt.arrow(x_coords[idx], y_coords[idx], dx_norm, dy_norm,
                                 head_width=0.8, head_length=0.5, fc='red', ec='red',
                                 alpha=0.7, linewidth=1)

        # 设置坐标轴标签和标题
        plt.xlabel('X Coordinate (m) - Longitudinal Position', fontsize=12, fontweight='bold')
        plt.ylabel('Y Coordinate (m) - Lateral Position', fontsize=12, fontweight='bold')
        plt.title('Vehicle Trajectory', fontsize=16, fontweight='bold', pad=20)

        # 设置网格和坐标轴
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.axis('equal')

        # 设置图例
        plt.legend(loc='upper right', fontsize=11, framealpha=0.9, title='Legend')

        # 计算轨迹统计信息
        total_distance = 0
        max_x, min_x = max(x_coords), min(x_coords)
        max_y, min_y = max(y_coords), min(y_coords)

        for i in range(1, len(positions)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            total_distance += math.sqrt(dx**2 + dy**2)

        # 计算总时间和平均速度
        total_time = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        avg_speed = total_distance / total_time if total_time > 0 else 0

        # 添加统计信息框 - 使用简单中文文本
        stats_text = f"""Track Statistics:
Total Distance: {total_distance:.2f} m
Total Time: {total_time:.2f} s
Average Speed: {avg_speed:.2f} m/s
Sample Points: {len(positions)}
X Range: [{min_x:.2f}, {max_x:.2f}] m
Y Range: [{min_y:.2f}, {max_y:.2f}] m"""

        plt.text(0.02, 0.98, stats_text,
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                fontsize=10, family='monospace')

        # 添加比例尺
        if max_x - min_x > 0:
            scale_length = min(50, (max_x - min_x) * 0.1)  # 比例尺长度
            scale_x = min_x + (max_x - min_x) * 0.05
            scale_y = min_y + (max_y - min_y) * 0.05
            plt.plot([scale_x, scale_x + scale_length], [scale_y, scale_y],
                    'k-', linewidth=3)
            plt.text(scale_x + scale_length/2, scale_y - (max_y - min_y) * 0.02,
                    f'{scale_length:.0f} m', ha='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        return f"车辆轨迹图已保存至: {output_path}"

    except Exception as e:
        print(f"创建轨迹图时出错: {e}")
        return None


def create_motion_plots(motion_data, output_dir):
    """
    创建运动参数图表
    """
    if not MATPLOTLIB_AVAILABLE:
        return []

    plots_info = []
    timestamps = motion_data['timestamps']

    # 时间转换为秒
    time_seconds = [t - timestamps[0] for t in timestamps]

    try:
        # 1. 速度图
        plt.figure(figsize=(12, 8))
        vel_x = [v[0] for v in motion_data['vehicle_velocities']]
        vel_y = [v[1] for v in motion_data['vehicle_velocities']]
        vel_z = [v[2] for v in motion_data['vehicle_velocities']]

        plt.plot(time_seconds, vel_x, 'r-', label='Longitudinal Velocity (X)', linewidth=2, alpha=0.8)
        plt.plot(time_seconds, vel_y, 'g-', label='Lateral Velocity (Y)', linewidth=2, alpha=0.8)
        plt.plot(time_seconds, vel_z, 'b-', label='Vertical Velocity (Z)', linewidth=2, alpha=0.8)

        plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
        plt.ylabel('Velocity (m/s)', fontsize=12, fontweight='bold')
        plt.title('Vehicle Velocity - Time Series', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=11, title='Velocity Components', loc='upper right')
        plt.tight_layout()

        vel_plot_path = os.path.join(output_dir, 'vehicle_velocity.png')
        plt.savefig(vel_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_info.append(("Velocity Time Series", vel_plot_path))

        # 2. 加速度图
        plt.figure(figsize=(12, 8))
        acc_x = [a[0] for a in motion_data['vehicle_accelerations']]
        acc_y = [a[1] for a in motion_data['vehicle_accelerations']]
        acc_z = [a[2] for a in motion_data['vehicle_accelerations']]

        plt.plot(time_seconds, acc_x, 'r-', label='Longitudinal Acceleration (X)', linewidth=2, alpha=0.8)
        plt.plot(time_seconds, acc_y, 'g-', label='Lateral Acceleration (Y)', linewidth=2, alpha=0.8)
        plt.plot(time_seconds, acc_z, 'b-', label='Vertical Acceleration (Z)', linewidth=2, alpha=0.8)

        plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
        plt.ylabel('Acceleration (m/s²)', fontsize=12, fontweight='bold')
        plt.title('Vehicle Acceleration - Time Series', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=11, title='Acceleration Components', loc='upper right')
        plt.tight_layout()

        acc_plot_path = os.path.join(output_dir, 'vehicle_acceleration.png')
        plt.savefig(acc_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_info.append(("Acceleration Time Series", acc_plot_path))

        # 3. 急动度图
        plt.figure(figsize=(12, 8))
        jerk_x = [j[0] for j in motion_data['vehicle_jerks']]
        jerk_y = [j[1] for j in motion_data['vehicle_jerks']]
        jerk_z = [j[2] for j in motion_data['vehicle_jerks']]

        plt.plot(time_seconds, jerk_x, 'r-', label='Longitudinal Jerk (X)', linewidth=2, alpha=0.8)
        plt.plot(time_seconds, jerk_y, 'g-', label='Lateral Jerk (Y)', linewidth=2, alpha=0.8)
        plt.plot(time_seconds, jerk_z, 'b-', label='Vertical Jerk (Z)', linewidth=2, alpha=0.8)

        plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
        plt.ylabel('Jerk (m/s³)', fontsize=12, fontweight='bold')
        plt.title('Vehicle Jerk - Time Series', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=11, title='Jerk Components', loc='upper right')
        plt.tight_layout()

        jerk_plot_path = os.path.join(output_dir, 'vehicle_jerk.png')
        plt.savefig(jerk_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_info.append(("Jerk Time Series", jerk_plot_path))

        # 4. 合成加速度幅值图
        plt.figure(figsize=(12, 6))
        acc_magnitude = [math.sqrt(a[0]**2 + a[1]**2 + a[2]**2) for a in motion_data['vehicle_accelerations']]

        plt.plot(time_seconds, acc_magnitude, color='purple', linewidth=2.5, alpha=0.8, label='Resultant Acceleration')
        plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
        plt.ylabel('Resultant Acceleration (m/s²)', fontsize=12, fontweight='bold')
        plt.title('Resultant Acceleration - Time Series', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')

        # 添加统计线
        avg_acc = sum(acc_magnitude) / len(acc_magnitude)
        max_acc = max(acc_magnitude)
        plt.axhline(y=avg_acc, color='red', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Average: {avg_acc:.3f} m/s²')
        plt.axhline(y=max_acc, color='orange', linestyle=':', linewidth=2, alpha=0.7,
                   label=f'Maximum: {max_acc:.3f} m/s²')
        plt.legend(fontsize=11, title='Legend', loc='upper right')
        plt.tight_layout()

        acc_mag_path = os.path.join(output_dir, 'acceleration_magnitude.png')
        plt.savefig(acc_mag_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_info.append(("Resultant Acceleration", acc_mag_path))

        # 5. 合成急动度幅值图
        plt.figure(figsize=(12, 6))
        jerk_magnitude = [math.sqrt(j[0]**2 + j[1]**2 + j[2]**2) for j in motion_data['vehicle_jerks']]

        plt.plot(time_seconds, jerk_magnitude, color='darkgreen', linewidth=2.5, alpha=0.8, label='Resultant Jerk')
        plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
        plt.ylabel('Resultant Jerk (m/s³)', fontsize=12, fontweight='bold')
        plt.title('Resultant Jerk - Time Series', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')

        # 添加统计线
        avg_jerk = sum(jerk_magnitude) / len(jerk_magnitude)
        max_jerk = max(jerk_magnitude)
        plt.axhline(y=avg_jerk, color='red', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Average: {avg_jerk:.3f} m/s³')
        plt.axhline(y=max_jerk, color='orange', linestyle=':', linewidth=2, alpha=0.7,
                   label=f'Maximum: {max_jerk:.3f} m/s³')

        # 添加急动度阈值线（2.0 m/s³为常用阈值）
        plt.axhline(y=2.0, color='red', linestyle='-.', linewidth=1.5, alpha=0.6,
                   label='Jerk Threshold: 2.0 m/s³')

        plt.legend(fontsize=11, title='Legend', loc='upper right')
        plt.tight_layout()

        jerk_mag_path = os.path.join(output_dir, 'jerk_magnitude.png')
        plt.savefig(jerk_mag_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_info.append(("Resultant Jerk", jerk_mag_path))

        # 6. 角速度图
        plt.figure(figsize=(12, 8))
        omega_x = [w[0] for w in motion_data['angular_velocities']]
        omega_y = [w[1] for w in motion_data['angular_velocities']]
        omega_z = [w[2] for w in motion_data['angular_velocities']]

        plt.plot(time_seconds, omega_x, 'r-', label='Roll Angular Velocity (X)', linewidth=2, alpha=0.8)
        plt.plot(time_seconds, omega_y, 'g-', label='Pitch Angular Velocity (Y)', linewidth=2, alpha=0.8)
        plt.plot(time_seconds, omega_z, 'b-', label='Yaw Angular Velocity (Z)', linewidth=2, alpha=0.8)

        plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
        plt.ylabel('Angular Velocity (rad/s)', fontsize=12, fontweight='bold')
        plt.title('Vehicle Angular Velocity - Time Series', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=11, title='Angular Velocity Components', loc='upper right')
        plt.tight_layout()

        omega_plot_path = os.path.join(output_dir, 'angular_velocity.png')
        plt.savefig(omega_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_info.append(("Angular Velocity Time Series", omega_plot_path))

        # 7. 角加速度图
        plt.figure(figsize=(12, 8))
        alpha_x = [a[0] for a in motion_data['angular_accelerations']]
        alpha_y = [a[1] for a in motion_data['angular_accelerations']]
        alpha_z = [a[2] for a in motion_data['angular_accelerations']]

        plt.plot(time_seconds, alpha_x, 'r-', label='Roll Angular Acceleration (X)', linewidth=2, alpha=0.8)
        plt.plot(time_seconds, alpha_y, 'g-', label='Pitch Angular Acceleration (Y)', linewidth=2, alpha=0.8)
        plt.plot(time_seconds, alpha_z, 'b-', label='Yaw Angular Acceleration (Z)', linewidth=2, alpha=0.8)

        plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
        plt.ylabel('Angular Acceleration (rad/s²)', fontsize=12, fontweight='bold')
        plt.title('Vehicle Angular Acceleration - Time Series', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=11, title='Angular Acceleration Components', loc='upper right')
        plt.tight_layout()

        alpha_plot_path = os.path.join(output_dir, 'angular_acceleration.png')
        plt.savefig(alpha_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_info.append(("Angular Acceleration Time Series", alpha_plot_path))

    except Exception as e:
        print(f"创建运动参数图表时出错: {e}")

    return plots_info


def calculate_motion_statistics(motion_data):
    """
    计算运动参数统计信息
    """
    stats = {}

    # 速度统计
    vel_x = [v[0] for v in motion_data['vehicle_velocities']]
    vel_y = [v[1] for v in motion_data['vehicle_velocities']]
    vel_z = [v[2] for v in motion_data['vehicle_velocities']]

    stats['velocity'] = {
        'x': {'mean': sum(vel_x)/len(vel_x), 'max': max(vel_x), 'min': min(vel_x), 'rms': math.sqrt(sum(v*v for v in vel_x)/len(vel_x))},
        'y': {'mean': sum(vel_y)/len(vel_y), 'max': max(vel_y), 'min': min(vel_y), 'rms': math.sqrt(sum(v*v for v in vel_y)/len(vel_y))},
        'z': {'mean': sum(vel_z)/len(vel_z), 'max': max(vel_z), 'min': min(vel_z), 'rms': math.sqrt(sum(v*v for v in vel_z)/len(vel_z))}
    }

    # 加速度统计
    acc_x = [a[0] for a in motion_data['vehicle_accelerations']]
    acc_y = [a[1] for a in motion_data['vehicle_accelerations']]
    acc_z = [a[2] for a in motion_data['vehicle_accelerations']]

    stats['acceleration'] = {
        'x': {'mean': sum(acc_x)/len(acc_x), 'max': max(acc_x), 'min': min(acc_x), 'rms': math.sqrt(sum(a*a for a in acc_x)/len(acc_x))},
        'y': {'mean': sum(acc_y)/len(acc_y), 'max': max(acc_y), 'min': min(acc_y), 'rms': math.sqrt(sum(a*a for a in acc_y)/len(acc_y))},
        'z': {'mean': sum(acc_z)/len(acc_z), 'max': max(acc_z), 'min': min(acc_z), 'rms': math.sqrt(sum(a*a for a in acc_z)/len(acc_z))}
    }

    # 急动度统计
    jerk_x = [j[0] for j in motion_data['vehicle_jerks']]
    jerk_y = [j[1] for j in motion_data['vehicle_jerks']]
    jerk_z = [j[2] for j in motion_data['vehicle_jerks']]

    stats['jerk'] = {
        'x': {'mean': sum(jerk_x)/len(jerk_x), 'max': max(jerk_x), 'min': min(jerk_x), 'rms': math.sqrt(sum(j*j for j in jerk_x)/len(jerk_x))},
        'y': {'mean': sum(jerk_y)/len(jerk_y), 'max': max(jerk_y), 'min': min(jerk_y), 'rms': math.sqrt(sum(j*j for j in jerk_y)/len(jerk_y))},
        'z': {'mean': sum(jerk_z)/len(jerk_z), 'max': max(jerk_z), 'min': min(jerk_z), 'rms': math.sqrt(sum(j*j for j in jerk_z)/len(jerk_z))}
    }

    # 角速度统计
    omega_x = [w[0] for w in motion_data['angular_velocities']]
    omega_y = [w[1] for w in motion_data['angular_velocities']]
    omega_z = [w[2] for w in motion_data['angular_velocities']]

    stats['angular_velocity'] = {
        'x': {'mean': sum(omega_x)/len(omega_x), 'max': max(omega_x), 'min': min(omega_x), 'rms': math.sqrt(sum(w*w for w in omega_x)/len(omega_x))},
        'y': {'mean': sum(omega_y)/len(omega_y), 'max': max(omega_y), 'min': min(omega_y), 'rms': math.sqrt(sum(w*w for w in omega_y)/len(omega_y))},
        'z': {'mean': sum(omega_z)/len(omega_z), 'max': max(omega_z), 'min': min(omega_z), 'rms': math.sqrt(sum(w*w for w in omega_z)/len(omega_z))}
    }

    # 角加速度统计
    alpha_x = [a[0] for a in motion_data['angular_accelerations']]
    alpha_y = [a[1] for a in motion_data['angular_accelerations']]
    alpha_z = [a[2] for a in motion_data['angular_accelerations']]

    stats['angular_acceleration'] = {
        'x': {'mean': sum(alpha_x)/len(alpha_x), 'max': max(alpha_x), 'min': min(alpha_x), 'rms': math.sqrt(sum(a*a for a in alpha_x)/len(alpha_x))},
        'y': {'mean': sum(alpha_y)/len(alpha_y), 'max': max(alpha_y), 'min': min(alpha_y), 'rms': math.sqrt(sum(a*a for a in alpha_y)/len(alpha_y))},
        'z': {'mean': sum(alpha_z)/len(alpha_z), 'max': max(alpha_z), 'min': min(alpha_z), 'rms': math.sqrt(sum(a*a for a in alpha_z)/len(alpha_z))}
    }

    # 轨迹统计
    positions = motion_data['positions']
    if len(positions) > 1:
        total_distance = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            dz = positions[i][2] - positions[i-1][2]
            total_distance += math.sqrt(dx**2 + dy**2 + dz**2)

        stats['trajectory'] = {
            'total_distance': total_distance,
            'start_pos': positions[0],
            'end_pos': positions[-1],
            'duration': motion_data['timestamps'][-1] - motion_data['timestamps'][0]
        }

    return stats


def generate_comfort_improvement_suggestions(comfort_results, motion_stats):
    """
    生成舒适度提升建议
    """
    suggestions = []
    results = comfort_results['results']
    dimensions = results['dimensions']
    overall_score = results['overall']['rfs_score']

    # 总体建议
    if overall_score >= 90:
        suggestions.append({
            "level": "优秀",
            "title": "乘坐舒适度表现卓越",
            "content": "当前车辆的乘坐舒适度表现卓越，各项指标均达到优秀水平。建议保持当前的调校参数，并建立定期监测机制以持续维持高水平表现。"
        })
    elif overall_score >= 80:
        suggestions.append({
            "level": "良好",
            "title": "乘坐舒适度表现良好",
            "content": "当前车辆的乘坐舒适度表现良好，仍有优化空间。建议关注个别指标的改进，以进一步提升整体乘坐体验。"
        })
    elif overall_score >= 70:
        suggestions.append({
            "level": "一般",
            "title": "乘坐舒适度有待提升",
            "content": "当前车辆的乘坐舒适度表现一般，存在明显的改进需求。建议重点优化主要问题指标，改善乘客体验。"
        })
    elif overall_score >= 60:
        suggestions.append({
            "level": "较差",
            "title": "乘坐舒适度需要大幅改进",
            "content": "当前车辆的乘坐舒适度表现较差，多项指标未达标准。建议制定系统性的优化方案，优先解决关键问题。"
        })
    else:
        suggestions.append({
            "level": "很差",
            "title": "乘坐舒适度急需改进",
            "content": "当前车辆的乘坐舒适度表现很差，需要立即采取改进措施。建议进行全面的系统调校和硬件优化。"
        })

    # 分维度建议
    dimension_suggestions = []

    # 持续振动建议
    vibration_data = dimensions['持续振动']
    if vibration_data['score'] < 70:
        if vibration_data['a_v'] > 0.8:
            dimension_suggestions.append({
                "dimension": "持续振动",
                "issue": f"总加权加速度值过高 ({vibration_data['a_v']:.3f} m/s²)",
                "suggestion": "优化悬挂系统参数，增加减振器阻尼，改善座椅减振性能，考虑采用主动悬架技术。"
            })
        else:
            dimension_suggestions.append({
                "dimension": "持续振动",
                "issue": f"振动控制不佳 (评分: {vibration_data['score']:.1f})",
                "suggestion": "检查并调整轮胎压力，优化悬挂刚度设置，改善车辆动态响应特性。"
            })

    # 瞬时冲击建议
    impact_data = dimensions['瞬时冲击']
    if impact_data['score'] < 70:
        if impact_data['A_peak'] > 1.0:
            dimension_suggestions.append({
                "dimension": "瞬时冲击",
                "issue": f"加速度峰值过高 ({impact_data['A_peak']:.3f} m/s²)",
                "suggestion": "优化动力总成悬置系统，改善传动系统平顺性，调整油门响应特性，减少突然的加减速冲击。"
            })
        elif impact_data['Jerk_rms'] > 0.3:
            dimension_suggestions.append({
                "dimension": "瞬时冲击",
                "issue": f"急动度RMS值过高 ({impact_data['Jerk_rms']:.3f} m/s³)",
                "suggestion": "优化控制算法参数，改善电机扭矩输出平滑性，增加预控制环节以减少突变。"
            })

    # 运动平顺性建议
    smoothness_data = dimensions['运动平顺性']
    if smoothness_data['score'] < 70:
        if smoothness_data['Jerk_peak'] > 6.0:
            dimension_suggestions.append({
                "dimension": "运动平顺性",
                "issue": f"急动度峰值过高 ({smoothness_data['Jerk_peak']:.3f} m/s³)",
                "suggestion": "优化驱动系统控制策略，采用更平滑的扭矩分配算法，增加驾驶模式的平顺性优先级。"
            })
        elif smoothness_data.get('N_jerk_events', 0) > 200:  # 兼容新旧版本
            dimension_suggestions.append({
                "dimension": "运动平顺性",
                "issue": f"急动度超限事件过多 ({smoothness_data.get('N_jerk_events', 0)} 次)",
                "suggestion": "优化路径规划算法，减少急转弯和急加速/减速情况，改善预瞄系统的准确性。"
            })

    # 角运动舒适性建议
    angular_data = dimensions['角运动舒适性']
    if angular_data['score'] < 70:
        if angular_data['omega_rms'] > 0.1:
            dimension_suggestions.append({
                "dimension": "角运动舒适性",
                "issue": f"角速度RMS过高 ({angular_data['omega_rms']:.3f} rad/s)",
                "suggestion": "优化转向系统调校，改善侧倾刚度分布，增加主动稳定杆系统，提高车身姿态控制精度。"
            })
        elif angular_data['alpha_peak'] > 1.0:
            dimension_suggestions.append({
                "dimension": "角运动舒适性",
                "issue": f"角加速度峰值过高 ({angular_data['alpha_peak']:.3f} rad/s²)",
                "suggestion": "优化侧倾控制策略，改善减振器阻尼特性，采用更先进的车身姿态控制系统。"
            })

    # 巡航平稳性建议
    cruise_data = dimensions['巡航平稳性']
    if cruise_data['score'] < 70:
        if cruise_data['sigma_v'] > 0.1:
            dimension_suggestions.append({
                "dimension": "巡航平稳性",
                "issue": f"巡航速度波动过大 ({cruise_data['sigma_v']:.3f} m/s)",
                "suggestion": "优化巡航控制算法，改进动力系统响应线性度，增加自适应巡航控制功能。"
            })
        elif cruise_data['R_c'] < 0.4:
            dimension_suggestions.append({
                "dimension": "巡航平稳性",
                "issue": f"巡航占比过低 ({cruise_data['R_c']:.1%})",
                "suggestion": "改善动力系统稳定性，优化控制参数，减少不必要的状态切换和调整。"
            })

    return suggestions, dimension_suggestions


def generate_markdown_report(motion_data, comfort_results, output_dir, report_title="RFS乘坐舒适度评价报告"):
    """
    生成完整的Markdown报告
    """

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 计算统计信息
    motion_stats = calculate_motion_statistics(motion_data)

    # 生成舒适度提升建议
    general_suggestions, dimension_suggestions = generate_comfort_improvement_suggestions(comfort_results, motion_stats)

    # 获取评价结果
    results = comfort_results['results']
    overall = results['overall']

    # 定义评价标准映射表
    evaluation_standards = {
        'a_v': {
            'name': '总加权加速度值',
            'unit': 'm/s²',
            'standard': 'ISO 2631-1',
            'thresholds': [0.315, 0.63, 1.0, 1.2],
            'description': '持续振动评价',
            'levels': ['优秀 (≤0.315)', '良好 (0.315-0.63)', '一般 (0.63-1.0)', '较差 (1.0-1.2)', '差 (>1.2)']
        },
        'A_peak': {
            'name': '加速度峰值',
            'unit': 'm/s²',
            'standard': 'RFS经验标准',
            'thresholds': [0.3, 0.6, 1.0, 1.5],
            'description': '瞬时冲击强度评价',
            'levels': ['优秀 (≤0.3)', '良好 (0.3-0.6)', '一般 (0.6-1.0)', '较差 (1.0-1.5)', '差 (>1.5)']
        },
        'Jerk_rms': {
            'name': '急动度RMS',
            'unit': 'm/s³',
            'standard': 'RFS经验标准',
            'thresholds': [0.15, 0.3, 0.5, 0.8],
            'description': '运动平顺性评价',
            'levels': ['优秀 (≤0.15)', '良好 (0.15-0.3)', '一般 (0.3-0.5)', '较差 (0.5-0.8)', '差 (>0.8)']
        },
        'Jerk_peak': {
            'name': '急动度峰值',
            'unit': 'm/s³',
            'standard': 'RFS经验标准',
            'thresholds': [2.0, 4.0, 6.0, 8.0],
            'description': '瞬时冲击评价',
            'levels': ['优秀 (≤2.0)', '良好 (2.0-4.0)', '一般 (4.0-6.0)', '较差 (6.0-8.0)', '差 (>8.0)']
        },
        'omega_rms': {
            'name': '角速度RMS',
            'unit': 'rad/s',
            'standard': 'RFS经验标准',
            'thresholds': [0.03, 0.08, 0.12, 0.18],
            'description': '角运动平顺性评价',
            'levels': ['优秀 (≤0.03)', '良好 (0.03-0.08)', '一般 (0.08-0.12)', '较差 (0.12-0.18)', '差 (>0.18)']
        },
        'alpha_peak': {
            'name': '角加速度峰值',
            'unit': 'rad/s²',
            'standard': 'RFS经验标准',
            'thresholds': [0.3, 0.8, 1.5, 2.5],
            'description': '角运动冲击评价',
            'levels': ['优秀 (≤0.3)', '良好 (0.3-0.8)', '一般 (0.8-1.5)', '较差 (1.5-2.5)', '差 (>2.5)']
        },
        'sigma_v': {
            'name': '巡航速度标准差',
            'unit': 'm/s',
            'standard': 'RFS经验标准',
            'thresholds': [0.03, 0.08, 0.12, 0.18],
            'description': '巡航稳定性评价',
            'levels': ['优秀 (≤0.03)', '良好 (0.03-0.08)', '一般 (0.08-0.12)', '较差 (0.12-0.18)', '差 (>0.18)']
        },
        'R_c': {
            'name': '巡航占比',
            'unit': '%',
            'standard': 'RFS经验标准',
            'thresholds': [0.8, 0.6, 0.4, 0.2],
            'description': '巡航平稳性评价',
            'levels': ['优秀 (≥80%)', '良好 (60-80%)', '一般 (40-60%)', '较差 (20-40%)', '差 (<20%)']
        }
    }

    # 首先生成Markdown文本内容（确保文字内容在前面）
    markdown_content = f"""# {report_title}

## 🎯 总体评价结果

### 评价概要

| 评价指标 | 评分结果 |
|---------|---------|
| **RFS总分** | **{overall['rfs_score']:.1f}/100** |
| **舒适度等级** | **{overall['comfort_level']}** |
| **评价时间** | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |

### 评分等级说明

| RFS评分 | 舒适度等级 | 评价说明 |
|---------|------------|----------|
| 90-100 | 卓越 | 极佳的乘坐体验，各项指标均达优秀水平 |
| 80-89 | 优秀 | 良好的乘坐舒适度，个别指标有优化空间 |
| 70-79 | 良好 | 可接受的舒适度水平，存在改进需求 |
| 60-69 | 合格 | 基本的舒适度要求，需要针对性优化 |
| <60 | 有待改进 | 舒适度较差，需要系统性的改进措施 |

---

## 📈 舒适度提升建议

### 总体建议

**当前水平: {general_suggestions[0]['level']}**

{general_suggestions[0]['content']}

### 针对性改进建议

"""

    if dimension_suggestions:
        for suggestion in dimension_suggestions:
            markdown_content += f"""#### {suggestion['dimension']}改进建议

**问题识别:** {suggestion['issue']}

**改进措施:** {suggestion['suggestion']}

"""
    else:
        markdown_content += "✅ 各项指标表现良好，暂无紧急改进需求。\n\n"

    markdown_content += f"""---

## 📋 详细分析报告

### 报告信息

- **评价标准**: {comfort_results.get('evaluation_standard', 'RFS舒适度评价标准')}
- **基于标准**: {comfort_results.get('based_on', 'ISO 2631-1')}
- **评价时长**: {comfort_results.get('evaluation_time', 0):.2f} 秒

---

## 车辆轨迹图

"""

    # 现在生成图表（在文字内容生成之后）
    trajectory_path = os.path.join(output_dir, 'trajectory.png')
    trajectory_info = create_trajectory_plot(motion_data['positions'], motion_data['timestamps'], trajectory_path)

    motion_plots = create_motion_plots(motion_data, output_dir)

    if trajectory_info:
        markdown_content += f"""![Vehicle Trajectory](trajectory.png)

{trajectory_info}

"""
    else:
        markdown_content += "**注意**: 由于缺少matplotlib库，无法生成轨迹图。\n\n"

    # 添加运动参数图表
    if motion_plots:
        markdown_content += "## 运动参数图表\n\n"
        for plot_name, plot_path in motion_plots:
            plot_filename = os.path.basename(plot_path)
            markdown_content += f"### {plot_name}\n\n![{plot_name}]({plot_filename})\n\n"

    # 运动统计信息
    markdown_content += """## 运动参数统计

### 轨迹统计
"""

    if 'trajectory' in motion_stats:
        traj_stats = motion_stats['trajectory']
        markdown_content += f"""- **总里程**: {traj_stats['total_distance']:.2f} m
- **起点坐标**: ({traj_stats['start_pos'][0]:.2f}, {traj_stats['start_pos'][1]:.2f}, {traj_stats['start_pos'][2]:.2f})
- **终点坐标**: ({traj_stats['end_pos'][0]:.2f}, {traj_stats['end_pos'][1]:.2f}, {traj_stats['end_pos'][2]:.2f})
- **行驶时间**: {traj_stats['duration']:.2f} s
- **平均速度**: {traj_stats['total_distance']/traj_stats['duration']:.2f} m/s

"""

    # 速度统计
    vel_stats = motion_stats['velocity']
    markdown_content += """### 速度统计 (m/s)

| 轴向 | 平均值 | 最大值 | 最小值 |
|------|--------|--------|--------|
| X轴(纵向) | {:.3f} | {:.3f} | {:.3f} |
| Y轴(横向) | {:.3f} | {:.3f} | {:.3f} |
| Z轴(垂直) | {:.3f} | {:.3f} | {:.3f} |

""".format(vel_stats['x']['mean'], vel_stats['x']['max'], vel_stats['x']['min'],
           vel_stats['y']['mean'], vel_stats['y']['max'], vel_stats['y']['min'],
           vel_stats['z']['mean'], vel_stats['z']['max'], vel_stats['z']['min'])

    # 加速度统计
    acc_stats = motion_stats['acceleration']
    markdown_content += """### 加速度统计 (m/s²)

| 轴向 | 平均值 | 最大值 | 最小值 | RMS值 |
|------|--------|--------|--------|-------|
| X轴(纵向) | {:.3f} | {:.3f} | {:.3f} | {:.3f} |
| Y轴(横向) | {:.3f} | {:.3f} | {:.3f} | {:.3f} |
| Z轴(垂直) | {:.3f} | {:.3f} | {:.3f} | {:.3f} |

""".format(acc_stats['x']['mean'], acc_stats['x']['max'], acc_stats['x']['min'], acc_stats['x']['rms'],
           acc_stats['y']['mean'], acc_stats['y']['max'], acc_stats['y']['min'], acc_stats['y']['rms'],
           acc_stats['z']['mean'], acc_stats['z']['max'], acc_stats['z']['min'], acc_stats['z']['rms'])

    # 急动度统计
    jerk_stats = motion_stats['jerk']
    markdown_content += """### 急动度统计 (m/s³)

| 轴向 | 平均值 | 最大值 | 最小值 | RMS值 |
|------|--------|--------|--------|-------|
| X轴(纵向) | {:.3f} | {:.3f} | {:.3f} | {:.3f} |
| Y轴(横向) | {:.3f} | {:.3f} | {:.3f} | {:.3f} |
| Z轴(垂直) | {:.3f} | {:.3f} | {:.3f} | {:.3f} |

""".format(jerk_stats['x']['mean'], jerk_stats['x']['max'], jerk_stats['x']['min'], jerk_stats['x']['rms'],
           jerk_stats['y']['mean'], jerk_stats['y']['max'], jerk_stats['y']['min'], jerk_stats['y']['rms'],
           jerk_stats['z']['mean'], jerk_stats['z']['max'], jerk_stats['z']['min'], jerk_stats['z']['rms'])

    # 角速度统计
    omega_stats = motion_stats['angular_velocity']
    markdown_content += """### 角速度统计 (rad/s)

| 轴向 | 平均值 | 最大值 | 最小值 | RMS值 |
|------|--------|--------|--------|-------|
| X轴(横滚) | {:.4f} | {:.4f} | {:.4f} | {:.4f} |
| Y轴(俯仰) | {:.4f} | {:.4f} | {:.4f} | {:.4f} |
| Z轴(偏航) | {:.4f} | {:.4f} | {:.4f} | {:.4f} |

""".format(omega_stats['x']['mean'], omega_stats['x']['max'], omega_stats['x']['min'], omega_stats['x']['rms'],
           omega_stats['y']['mean'], omega_stats['y']['max'], omega_stats['y']['min'], omega_stats['y']['rms'],
           omega_stats['z']['mean'], omega_stats['z']['max'], omega_stats['z']['min'], omega_stats['z']['rms'])

    # 角加速度统计
    alpha_stats = motion_stats['angular_acceleration']
    markdown_content += """### 角加速度统计 (rad/s²)

| 轴向 | 平均值 | 最大值 | 最小值 | RMS值 |
|------|--------|--------|--------|-------|
| X轴(横滚) | {:.4f} | {:.4f} | {:.4f} | {:.4f} |
| Y轴(俯仰) | {:.4f} | {:.4f} | {:.4f} | {:.4f} |
| Z轴(偏航) | {:.4f} | {:.4f} | {:.4f} | {:.4f} |

""".format(alpha_stats['x']['mean'], alpha_stats['x']['max'], alpha_stats['x']['min'], alpha_stats['x']['rms'],
           alpha_stats['y']['mean'], alpha_stats['y']['max'], alpha_stats['y']['min'], alpha_stats['y']['rms'],
           alpha_stats['z']['mean'], alpha_stats['z']['max'], alpha_stats['z']['min'], alpha_stats['z']['rms'])

    # RFS舒适度评价结果
    results = comfort_results['results']
    overall = results['overall']
    dimensions = results['dimensions']

    markdown_content += f"""---

## RFS乘坐舒适度评价结果

### 总体评价

| 指标 | 评分 |
|------|------|
| **RFS总分** | **{overall['rfs_score']:.1f}/100** |
| **舒适度等级** | **{overall['comfort_level']}** |

### 评价权重分配

| 评价维度 | 权重 | 说明 |
|----------|------|------|
"""

    weights = comfort_results.get('weights', {})
    for dim_name, weight in weights.items():
        markdown_content += f"| {dim_name} | {weight*100:.0f}% | 基于工程经验调整 |\n"

    markdown_content += "\n### 详细评价结果\n\n"

    # 各维度详细结果
    for dim_name, dim_data in dimensions.items():
        weight = dim_data['weight'] * 100
        score = dim_data['score']

        markdown_content += f"""#### {dim_name} (权重: {weight:.0f}%)

**得分: {score:.1f}/100**

"""

        if dim_name == '持续振动':
            markdown_content += f"- 总加权加速度值 a_v: {dim_data['a_v']:.4f} m/s²\n"
            markdown_content += f"- 说明: {dim_data['note']}\n"

        elif dim_name == '瞬时冲击':
            markdown_content += f"- 加速度峰值 A_peak: {dim_data['A_peak']:.4f} m/s²\n"
            markdown_content += f"- 急动度RMS Jerk_rms: {dim_data['Jerk_rms']:.4f} m/s³\n"

        elif dim_name == '运动平顺性':
            markdown_content += f"- 急动度峰值 Jerk_peak: {dim_data['Jerk_peak']:.4f} m/s³\n"
            if 'N_jerk_per_hour' in dim_data:
                markdown_content += f"- 急动度超限事件数: {dim_data['N_jerk_per_hour']:.1f} 次/小时\n"
            else:
                markdown_content += f"- 急动度超限事件数: {dim_data.get('N_jerk_events', 0)} 次 ({dim_data.get('Jerk_exceed_ratio', 0):.2%})\n"

        elif dim_name == '角运动舒适性':
            markdown_content += f"- 角速度RMS ω_rms: {dim_data['omega_rms']:.4f} rad/s\n"
            markdown_content += f"- 角加速度峰值 α_peak: {dim_data['alpha_peak']:.4f} rad/s²\n"

        elif dim_name == '巡航平稳性':
            markdown_content += f"- 巡航速度标准差 σ_v: {dim_data['sigma_v']:.4f} m/s\n"
            markdown_content += f"- 巡航占比 R_c: {dim_data['R_c']:.1%}\n"

        markdown_content += "\n"

    # 添加评分标准说明
    markdown_content += """---

## 评分标准说明

### ISO 2631-1频率加权

"""

    freq_weighting = comfort_results.get('frequency_weighting', {})
    markdown_content += f"""- **Z轴(垂直振动)**: {freq_weighting.get('Z_axis', 'Wk filter')}
- **X/Y轴(水平振动)**: {freq_weighting.get('X_Y_axis', 'Wd filter')}
- **实现方式**: {freq_weighting.get('implementation', 'FFT + ISO传递函数')}

### 舒适度等级标准

| RFS评分 | 舒适度等级 | 说明 |
|---------|------------|------|
| 90-100 | 卓越 | 极佳的乘坐体验 |
| 80-89 | 优秀 | 良好的乘坐舒适度 |
| 70-79 | 良好 | 可接受的舒适度水平 |
| 60-69 | 合格 | 基本的舒适度要求 |
| <60 | 有待改进 | 需要优化改进 |

### 各维度评价标准映射表

#### 1. 持续振动评价 (总加权加速度值 a_v)

| 评分等级 | a_v 范围 (m/s²) | 评价标准 | 说明 |
|----------|-----------------|----------|------|
"""

    # 持续振动评价标准
    a_v_standard = evaluation_standards['a_v']
    for i, level in enumerate(a_v_standard['levels']):
        if i == 0:
            range_desc = f"≤ {a_v_standard['thresholds'][0]}"
        elif i == len(a_v_standard['levels']) - 1:
            range_desc = f"> {a_v_standard['thresholds'][-1]}"
        else:
            range_desc = f"{a_v_standard['thresholds'][i-1]} - {a_v_standard['thresholds'][i]}"

        markdown_content += f"| {level.split(' ')[0]} | {range_desc} | {a_v_standard['standard']} | {'几乎无不适感' if i == 0 else '轻微不适感' if i == 1 else '中等不适感' if i == 2 else '明显不适感' if i == 3 else '严重不适感'} |\n"

    markdown_content += """
#### 2. 瞬时冲击评价 (加速度峰值 A_peak + 急动度RMS Jerk_rms)

**加速度峰值 (A_peak):**
| 评分等级 | A_peak 范围 (m/s²) | 说明 |
|----------|-------------------|------|
"""

    # 加速度峰值评价标准
    A_peak_standard = evaluation_standards['A_peak']
    impact_descriptions = ['冲击极小', '轻微冲击', '中等冲击', '明显冲击', '强烈冲击']
    for i, level in enumerate(A_peak_standard['levels']):
        if i == 0:
            range_desc = f"≤ {A_peak_standard['thresholds'][0]}"
        elif i == len(A_peak_standard['levels']) - 1:
            range_desc = f"> {A_peak_standard['thresholds'][-1]}"
        else:
            range_desc = f"{A_peak_standard['thresholds'][i-1]} - {A_peak_standard['thresholds'][i]}"

        markdown_content += f"| {level.split(' ')[0]} | {range_desc} | {impact_descriptions[i]} |\n"

    markdown_content += """
**急动度RMS (Jerk_rms):**
| 评分等级 | Jerk_rms 范围 (m/s³) | 说明 |
|----------|---------------------|------|
"""

    # 急动度RMS评价标准
    Jerk_rms_standard = evaluation_standards['Jerk_rms']
    jerk_descriptions = ['变化极平缓', '变化较平缓', '中等变化率', '变化较剧烈', '变化极剧烈']
    for i, level in enumerate(Jerk_rms_standard['levels']):
        if i == 0:
            range_desc = f"≤ {Jerk_rms_standard['thresholds'][0]}"
        elif i == len(Jerk_rms_standard['levels']) - 1:
            range_desc = f"> {Jerk_rms_standard['thresholds'][-1]}"
        else:
            range_desc = f"{Jerk_rms_standard['thresholds'][i-1]} - {Jerk_rms_standard['thresholds'][i]}"

        markdown_content += f"| {level.split(' ')[0]} | {range_desc} | {jerk_descriptions[i]} |\n"

    markdown_content += """
#### 3. 运动平顺性评价 (急动度峰值 Jerk_peak + 超限事件)

**急动度峰值 (Jerk_peak):**
| 评分等级 | Jerk_peak 范围 (m/s³) | 说明 |
|----------|-----------------------|------|
"""

    # 急动度峰值评价标准
    Jerk_peak_standard = evaluation_standards['Jerk_peak']
    transient_descriptions = ['瞬态变化极小', '轻微瞬态变化', '中等瞬态变化', '明显瞬态变化', '强烈瞬态变化']
    for i, level in enumerate(Jerk_peak_standard['levels']):
        if i == 0:
            range_desc = f"≤ {Jerk_peak_standard['thresholds'][0]}"
        elif i == len(Jerk_peak_standard['levels']) - 1:
            range_desc = f"> {Jerk_peak_standard['thresholds'][-1]}"
        else:
            range_desc = f"{Jerk_peak_standard['thresholds'][i-1]} - {Jerk_peak_standard['thresholds'][i]}"

        markdown_content += f"| {level.split(' ')[0]} | {range_desc} | {transient_descriptions[i]} |\n"

    markdown_content += """
#### 4. 角运动舒适性评价 (角速度RMS ω_rms + 角加速度峰值 α_peak)

**角速度RMS (ω_rms):**
| 评分等级 | ω_rms 范围 (rad/s) | 说明 |
|----------|-------------------|------|
"""

    # 角速度RMS评价标准
    omega_rms_standard = evaluation_standards['omega_rms']
    angular_motion_descriptions = ['角运动极平稳', '角运动较平稳', '中等角运动', '角运动较剧烈', '角运动极剧烈']
    for i, level in enumerate(omega_rms_standard['levels']):
        if i == 0:
            range_desc = f"≤ {omega_rms_standard['thresholds'][0]}"
        elif i == len(omega_rms_standard['levels']) - 1:
            range_desc = f"> {omega_rms_standard['thresholds'][-1]}"
        else:
            range_desc = f"{omega_rms_standard['thresholds'][i-1]} - {omega_rms_standard['thresholds'][i]}"

        markdown_content += f"| {level.split(' ')[0]} | {range_desc} | {angular_motion_descriptions[i]} |\n"

    markdown_content += """
**角加速度峰值 (α_peak):**
| 评分等级 | α_peak 范围 (rad/s²) | 说明 |
|----------|---------------------|------|
"""

    # 角加速度峰值评价标准
    alpha_peak_standard = evaluation_standards['alpha_peak']
    angular_impact_descriptions = ['角冲击极小', '轻微角冲击', '中等角冲击', '明显角冲击', '强烈角冲击']
    for i, level in enumerate(alpha_peak_standard['levels']):
        if i == 0:
            range_desc = f"≤ {alpha_peak_standard['thresholds'][0]}"
        elif i == len(alpha_peak_standard['levels']) - 1:
            range_desc = f"> {alpha_peak_standard['thresholds'][-1]}"
        else:
            range_desc = f"{alpha_peak_standard['thresholds'][i-1]} - {alpha_peak_standard['thresholds'][i]}"

        markdown_content += f"| {level.split(' ')[0]} | {range_desc} | {angular_impact_descriptions[i]} |\n"

    markdown_content += """
#### 5. 巡航平稳性评价 (巡航速度标准差 σ_v + 巡航占比 R_c)

**巡航速度标准差 (σ_v):**
| 评分等级 | σ_v 范围 (m/s) | 说明 |
|----------|----------------|------|
"""

    # 巡航速度标准差评价标准
    sigma_v_standard = evaluation_standards['sigma_v']
    speed_stability_descriptions = ['速度极稳定', '速度较稳定', '中等速度波动', '速度波动较大', '速度极不稳定']
    for i, level in enumerate(sigma_v_standard['levels']):
        if i == 0:
            range_desc = f"≤ {sigma_v_standard['thresholds'][0]}"
        elif i == len(sigma_v_standard['levels']) - 1:
            range_desc = f"> {sigma_v_standard['thresholds'][-1]}"
        else:
            range_desc = f"{sigma_v_standard['thresholds'][i-1]} - {sigma_v_standard['thresholds'][i]}"

        markdown_content += f"| {level.split(' ')[0]} | {range_desc} | {speed_stability_descriptions[i]} |\n"

    markdown_content += """
**巡航占比 (R_c):**
| 评分等级 | R_c 范围 (%) | 说明 |
|----------|-------------|------|
"""

    # 巡航占比评价标准
    R_c_standard = evaluation_standards['R_c']
    cruise_ratio_descriptions = ['稳定巡航为主', '巡航占比较高', '中等巡航比例', '巡航占比较低', '很少稳定巡航']
    for i, level in enumerate(R_c_standard['levels']):
        if i == 0:
            range_desc = f"≥ {R_c_standard['thresholds'][0]*100:.0f}%"
        elif i == len(R_c_standard['levels']) - 1:
            range_desc = f"< {R_c_standard['thresholds'][-1]*100:.0f}%"
        else:
            range_desc = f"{R_c_standard['thresholds'][i]*100:.0f} - {R_c_standard['thresholds'][i-1]*100:.0f}%"

        markdown_content += f"| {level.split(' ')[0]} | {range_desc} | {cruise_ratio_descriptions[i]} |\n"

    markdown_content += """
---

## 技术说明

本报告基于以下技术和标准：

1. **数据采集**: 车载传感器高频采集
2. **坐标系统**: 车辆坐标系（X-纵向，Y-横向，Z-垂直）
3. **评价标准**: ISO 2631-1:1997 + RFS工程经验调整
4. **频率加权**: Wk/Wd滤波器，符合国际标准
5. **评价维度**: 5个维度综合评价，覆盖全面

---

*报告由RFS舒适度评价系统自动生成*
"""

    # 保存Markdown文件
    report_path = os.path.join(output_dir, 'comfort_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    # 调试信息：确认报告结构
    print(f"✅ Markdown报告结构:")
    print(f"   1. 总体评价结果 (置顶)")
    print(f"   2. 舒适度提升建议")
    print(f"   3. 详细分析报告 (包含轨迹图)")
    print(f"   4. 技术说明")
    print(f"   报告已保存至: {report_path}")

    return report_path


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description='RFS舒适度评价Markdown报告生成器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python3 markdown_report_generator.py motion.csv comfort.json
  python3 markdown_report_generator.py motion.csv comfort.json -o output_dir
  python3 markdown_report_generator.py motion.csv comfort.json --title "自定义标题"
        """
    )

    parser.add_argument(
        'motion_csv',
        help='运动参数CSV文件路径'
    )

    parser.add_argument(
        'comfort_json',
        help='舒适度评价JSON文件路径'
    )

    parser.add_argument(
        '-o', '--output',
        default='rfs_report',
        help='输出目录路径 (默认: rfs_report)'
    )

    parser.add_argument(
        '--title',
        default='RFS乘坐舒适度评价报告',
        help='报告标题'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='RFS舒适度评价Markdown报告生成器 v1.0.0'
    )

    return parser.parse_args()


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_arguments()

    print("RFS舒适度评价Markdown报告生成器")
    print("=" * 50)

    # 检查输入文件
    if not os.path.exists(args.motion_csv):
        print(f"错误: 运动参数文件不存在: {args.motion_csv}")
        sys.exit(1)

    if not os.path.exists(args.comfort_json):
        print(f"错误: 舒适度评价文件不存在: {args.comfort_json}")
        sys.exit(1)

    print(f"运动参数文件: {args.motion_csv}")
    print(f"舒适度评价文件: {args.comfort_json}")
    print(f"输出目录: {args.output}")
    print(f"报告标题: {args.title}")
    print()

    # 加载数据
    print("正在加载运动参数数据...")
    motion_data = load_motion_data_from_csv(args.motion_csv)
    if motion_data is None:
        print("错误: 无法加载运动参数数据")
        sys.exit(1)

    print("正在加载舒适度评价结果...")
    comfort_results = load_comfort_results_from_json(args.comfort_json)
    if comfort_results is None:
        print("错误: 无法加载舒适度评价结果")
        sys.exit(1)

    print("正在生成Markdown报告...")
    try:
        report_path = generate_markdown_report(motion_data, comfort_results, args.output, args.title)
        print(f"✅ Markdown报告已生成: {report_path}")

        # 列出生成的文件
        output_files = []
        for root, dirs, files in os.walk(args.output):
            for file in files:
                output_files.append(os.path.join(root, file))

        print("\n📁 生成的文件:")
        for file_path in sorted(output_files):
            print(f"   - {file_path}")

    except Exception as e:
        print(f"❌ 生成报告时出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()