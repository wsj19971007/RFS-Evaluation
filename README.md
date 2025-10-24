# RFS乘坐舒适度评价系统

基于ISO 2631-1标准和工程经验的车辆乘坐舒适度评价系统，支持从原始Odometry数据到完整Markdown报告的全流程自动化处理。

## 功能特性

- **运动参数提取**: 从Odometry JSON数据中提取速度、加速度、急动度、角速度等参数
- **舒适度评价**: 基于ISO 2631-1 Wk/Wd频率加权的5维度综合评价
- **可视化报告**: 自动生成包含车辆轨迹图和运动参数图表的Markdown报告
- **全流程自动化**: 一键运行从原始数据到最终报告的完整流程

## 系统架构

```
RFS舒适度评价系统
├── motion_extraction.py          # 运动参数提取模块
├── rfs_comfort_evaluation.py     # 舒适度评价模块
├── markdown_report_generator.py  # Markdown报告生成器
├── run_rfs_evaluation.py         # 完整流程运行脚本
├── example_usage.sh              # 使用示例脚本
└── README.md                     # 说明文档
```

## 安装依赖

### 必需依赖
- Python 3.6+
- 标准库: json, math, sys, argparse, collections

### 可选依赖（用于图表生成）
```bash
pip install matplotlib
```

**注意**: 图表文字已优化为英文显示，确保在所有系统上都能正常显示，避免中文字体兼容性问题。

## 使用方法

### 1. 完整流程（推荐）

使用一键运行脚本，从原始数据到最终报告：

```bash
# 基本用法
python3 run_rfs_evaluation.py odometry_data.json

# 自定义输出目录和标题
python3 run_rfs_evaluation.py odometry_data.json \
    --output my_report \
    --title "车辆舒适度测试报告"

# 保留中间文件
python3 run_rfs_evaluation.py odometry_data.json --keep-intermediate

# 只执行特定步骤
python3 run_rfs_evaluation.py odometry_data.json --steps extract evaluate
```

### 2. 分步执行

#### 步骤1: 提取运动参数
```bash
python3 motion_extraction.py odometry_data.json -o motion_parameters.csv
```

#### 步骤2: 舒适度评价
```bash
python3 rfs_comfort_evaluation.py motion_parameters.csv -o comfort_evaluation.json
```

#### 步骤3: 生成报告
```bash
python3 markdown_report_generator.py motion_parameters.csv comfort_evaluation.json \
    -o report_dir --title "舒适度评价报告"
```

### 3. 快速示例
```bash
# 运行示例脚本
./example_usage.sh
```

## 输入数据格式

系统需要标准的Odometry JSON格式数据，包含以下字段：

```json
{
  "timestamp": {
    "header": {
      "time_stamp": 1234567890.123
    },
    "pose": {
      "position": {"x": 0.0, "y": 0.0, "z": 0.0},
      "quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
    },
    "twist": {
      "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
      "angular": {"x": 0.0, "y": 0.0, "z": 0.0}
    }
  }
}
```

## 评价体系

### 5个评价维度

1. **持续振动** (权重30%)
   - 总加权加速度值 a_v
   - ISO 2631-1 Wk/Wd频率加权

2. **瞬时冲击** (权重30%)
   - 加速度峰值 A_peak
   - 急动度RMS值

3. **运动平顺性** (权重25%)
   - 急动度峰值
   - 急动度超限事件数

4. **角运动舒适性** (权重10%)
   - 角速度RMS
   - 角加速度峰值

5. **巡航平稳性** (权重5%)
   - 巡航速度标准差
   - 巡航占比

### 舒适度等级

| RFS评分 | 舒适度等级 | 说明 |
|---------|------------|------|
| 90-100 | 卓越 | 极佳的乘坐体验 |
| 80-89 | 优秀 | 良好的乘坐舒适度 |
| 70-79 | 良好 | 可接受的舒适度水平 |
| 60-69 | 合格 | 基本的舒适度要求 |
| <60 | 有待改进 | 需要优化改进 |

## 输出报告

系统生成完整的Markdown报告，按以下结构组织：

### 📊 报告结构

1. **🎯 总体评价结果** (置顶显示)
   - RFS总分和舒适度等级
   - 评分等级说明表格

2. **📈 舒适度提升建议**
   - 总体改进建议
   - 针对性改进措施（按5个评价维度）
   - 问题识别和具体解决方案

3. **📋 详细分析报告**
   - 车辆轨迹图（最前方）
   - 运动参数图表：速度、加速度、急动度、角速度、角加速度等参数的时间序列图
   - 统计信息：各项参数的统计数据
   - 评价结果：详细的RFS舒适度评分和等级
   - **评价标准映射表**：5个维度的详细评价标准（每个指标的评分等级和对应范围）
   - 技术说明：评价标准和技术实现细节

### 🎯 核心特性

- **总体评价置顶**: 首先显示最重要的评价结果
- **智能建议系统**: 根据评分自动生成针对性的改进建议
- **专业图表**: 所有图表都包含完整的标题、图例、坐标轴标签
- **详细统计**: 全面的运动参数统计分析

### 报告文件结构
```
output_directory/
├── comfort_report.md              # 主报告文件
├── trajectory.png                 # 车辆轨迹图
├── vehicle_velocity.png          # 速度时间序列图
├── vehicle_acceleration.png      # 加速度时间序列图
├── vehicle_jerk.png              # 急动度时间序列图
├── angular_velocity.png          # 角速度时间序列图
├── angular_acceleration.png      # 角加速度时间序列图
├── acceleration_magnitude.png    # 合成加速度幅值图
├── jerk_magnitude.png            # 合成急动度幅值图
└── execution_summary.json        # 执行摘要
```

## 命令行参数

### run_rfs_evaluation.py
```bash
python3 run_rfs_evaluation.py <input_json> [options]

选项:
  --output DIR          输出目录 (默认: rfs_evaluation_output)
  --title TITLE         报告标题 (默认: RFS乘坐舒适度评价报告)
  --keep-intermediate   保留中间文件
  --steps STEPS         指定执行步骤 (extract/evaluate/report)
```

### motion_extraction.py
```bash
python3 motion_extraction.py <input_json> [options]

选项:
  -o, --output FILE    输出CSV文件 (默认: motion_parameters_corrected.csv)
  --verbose            显示详细输出
```

### rfs_comfort_evaluation.py
```bash
python3 rfs_comfort_evaluation.py <motion_csv> [options]

选项:
  -o, --output FILE    输出JSON文件
  --verbose            显示详细输出
```

### markdown_report_generator.py
```bash
python3 markdown_report_generator.py <motion_csv> <comfort_json> [options]

选项:
  -o, --output DIR     输出目录
  --title TITLE        报告标题
```

## 技术实现

- **坐标系统**: 车辆坐标系（X-纵向，Y-横向，Z-垂直）
- **频率加权**: ISO 2631-1 Wk/Wd滤波器
- **数据处理**: 低通滤波、中心差分、FFT频域分析
- **评价算法**: 多维度加权评分，结合国际标准与工程经验

## 故障排除

### 常见问题

1. **ImportError: No module named 'matplotlib'**
   ```bash
   pip install matplotlib
   ```

2. **文件不存在错误**
   - 检查输入文件路径是否正确
   - 确保文件权限允许读取

3. **内存不足**
   - 对于大数据文件，考虑分批处理
   - 增加系统虚拟内存

4. **图表生成失败**
   - 检查matplotlib安装
   - 确保输出目录有写入权限

### 调试模式

添加 `--verbose` 参数查看详细执行信息：

```bash
python3 run_rfs_evaluation.py data.json --verbose
```

## 版本信息

- RFS舒适度评价系统 v1.0.0
- 基于ISO 2631-1:1997标准
- 支持工程经验调整

## 许可证

本项目遵循相关开源许可证。

## 贡献

欢迎提交Issue和Pull Request来改进这个系统。