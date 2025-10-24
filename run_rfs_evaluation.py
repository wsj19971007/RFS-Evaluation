#!/usr/bin/env python3
"""
RFS舒适度评价完整流程运行脚本
从原始Odometry JSON数据到最终的Markdown报告
"""

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime

def check_dependencies():
    """
    检查必要的依赖
    """
    required_files = [
        'motion_extraction.py',
        'rfs_comfort_evaluation.py',
        'markdown_report_generator.py'
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print("❌ 缺少必要的文件:")
        for file in missing_files:
            print(f"   - {file}")
        return False

    # 检查Python模块
    try:
        import json
        import math
        import argparse
        from collections import OrderedDict
    except ImportError as e:
        print(f"❌ 缺少必要的Python模块: {e}")
        return False

    # 检查matplotlib（可选）
    try:
        import matplotlib
        print("✅ matplotlib可用，将生成图表")
    except ImportError:
        print("⚠️  matplotlib未安装，将跳过图表生成")
        print("   安装方法: pip install matplotlib")

    return True


def run_motion_extraction(input_json, output_csv):
    """
    运行运动参数提取
    """
    print("🔧 步骤1: 提取运动参数...")

    cmd = ['python3', 'motion_extraction.py', input_json, '-o', output_csv]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ 运动参数提取完成")
        print(f"   输出文件: {output_csv}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 运动参数提取失败:")
        print(f"   错误代码: {e.returncode}")
        print(f"   错误信息: {e.stderr}")
        return False


def run_comfort_evaluation(motion_csv, output_json):
    """
    运行舒适度评价
    """
    print("🔧 步骤2: 进行舒适度评价...")

    cmd = ['python3', 'rfs_comfort_evaluation.py', motion_csv, '-o', output_json]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ 舒适度评价完成")
        print(f"   输出文件: {output_json}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 舒适度评价失败:")
        print(f"   错误代码: {e.returncode}")
        print(f"   错误信息: {e.stderr}")
        return False


def run_markdown_report(motion_csv, comfort_json, output_dir, report_title):
    """
    生成Markdown报告
    """
    print("🔧 步骤3: 生成Markdown报告...")

    cmd = ['python3', 'markdown_report_generator.py', motion_csv, comfort_json,
           '-o', output_dir, '--title', report_title]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Markdown报告生成完成")
        print(f"   输出目录: {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Markdown报告生成失败:")
        print(f"   错误代码: {e.returncode}")
        print(f"   错误信息: {e.stderr}")
        return False


def cleanup_intermediate_files(files_to_remove):
    """
    清理中间文件
    """
    for file_path in files_to_remove:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️  已删除中间文件: {file_path}")
        except Exception as e:
            print(f"⚠️  删除文件失败 {file_path}: {e}")


def generate_summary_report(input_json, output_dir, success, steps_completed):
    """
    生成流程执行摘要
    """
    summary = {
        'execution_time': datetime.now().isoformat(),
        'input_file': input_json,
        'output_directory': output_dir,
        'success': success,
        'steps_completed': steps_completed,
        'total_steps': 3
    }

    summary_path = os.path.join(output_dir, 'execution_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary_path


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description='RFS舒适度评价完整流程运行脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python3 run_rfs_evaluation.py odometry_data.json
  python3 run_rfs_evaluation.py odometry_data.json --output my_report
  python3 run_rfs_evaluation.py odometry_data.json --title "车辆舒适度测试报告"
  python3 run_rfs_evaluation.py odometry_data.json --keep-intermediate
        """
    )

    parser.add_argument(
        'input_json',
        help='输入的Odometry JSON文件路径'
    )

    parser.add_argument(
        '--output',
        default='rfs_evaluation_output',
        help='输出目录名称 (默认: rfs_evaluation_output)'
    )

    parser.add_argument(
        '--title',
        default='RFS乘坐舒适度评价报告',
        help='报告标题'
    )

    parser.add_argument(
        '--keep-intermediate',
        action='store_true',
        help='保留中间文件（运动参数CSV、舒适度评价JSON）'
    )

    parser.add_argument(
        '--steps',
        nargs='+',
        choices=['extract', 'evaluate', 'report'],
        help='指定要执行的步骤 (extract: 提取运动参数, evaluate: 舒适度评价, report: 生成报告)'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='RFS舒适度评价完整流程运行脚本 v1.0.0'
    )

    return parser.parse_args()


def main():
    """
    主函数
    """
    print("RFS舒适度评价完整流程运行脚本")
    print("=" * 60)

    # 解析命令行参数
    args = parse_arguments()

    # 检查输入文件
    if not os.path.exists(args.input_json):
        print(f"❌ 错误: 输入文件不存在: {args.input_json}")
        sys.exit(1)

    print(f"输入文件: {args.input_json}")
    print(f"输出目录: {args.output}")
    print(f"报告标题: {args.title}")
    print()

    # 检查依赖
    if not check_dependencies():
        print("❌ 依赖检查失败，请确保所有必要文件存在")
        sys.exit(1)

    print("✅ 依赖检查通过")
    print()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 确定要执行的步骤
    all_steps = ['extract', 'evaluate', 'report']
    if args.steps:
        steps_to_run = args.steps
    else:
        steps_to_run = all_steps

    print(f"将执行的步骤: {' → '.join(steps_to_run)}")
    print()

    # 定义中间文件路径
    motion_csv = os.path.join(args.output, 'motion_parameters.csv')
    comfort_json = os.path.join(args.output, 'comfort_evaluation.json')

    # 记录执行状态
    steps_completed = []
    overall_success = True

    # 步骤1: 运动参数提取
    if 'extract' in steps_to_run:
        if not run_motion_extraction(args.input_json, motion_csv):
            overall_success = False
        else:
            steps_completed.append('运动参数提取')
        print()
    else:
        print("⏭️  跳过运动参数提取步骤")
        if not os.path.exists(motion_csv):
            print("❌ 错误: 跳过运动参数提取但找不到现有的运动参数文件")
            sys.exit(1)
        print()

    # 步骤2: 舒适度评价
    if 'evaluate' in steps_to_run and overall_success:
        if not run_comfort_evaluation(motion_csv, comfort_json):
            overall_success = False
        else:
            steps_completed.append('舒适度评价')
        print()
    elif 'evaluate' in steps_to_run:
        print("⏭️  跳过舒适度评价步骤（前序步骤失败）")
        print()
    else:
        print("⏭️  跳过舒适度评价步骤")
        if not os.path.exists(comfort_json):
            print("❌ 错误: 跳过舒适度评价但找不到现有的评价结果文件")
            sys.exit(1)
        print()

    # 步骤3: 生成Markdown报告
    if 'report' in steps_to_run and overall_success:
        if not run_markdown_report(motion_csv, comfort_json, args.output, args.title):
            overall_success = False
        else:
            steps_completed.append('Markdown报告生成')
        print()
    elif 'report' in steps_to_run:
        print("⏭️  跳过Markdown报告生成步骤（前序步骤失败）")
        print()
    else:
        print("⏭️  跳过Markdown报告生成步骤")
        print()

    # 生成执行摘要
    summary_path = generate_summary_report(args.input_json, args.output, overall_success, steps_completed)

    # 清理中间文件（如果需要）
    if not args.keep_intermediate:
        intermediate_files = [motion_csv, comfort_json]
        cleanup_intermediate_files(intermediate_files)
    else:
        print("💾 保留中间文件")
        print(f"   - {motion_csv}")
        print(f"   - {comfort_json}")

    # 输出最终结果
    print()
    print("=" * 60)
    if overall_success:
        print("🎉 RFS舒适度评价流程执行成功！")
        print(f"📁 输出目录: {args.output}")
        print(f"📄 执行摘要: {summary_path}")

        # 列出输出目录中的文件
        try:
            output_files = []
            for root, dirs, files in os.walk(args.output):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), args.output)
                    output_files.append(rel_path)

            if output_files:
                print("\n📋 生成的文件:")
                for file_path in sorted(output_files):
                    full_path = os.path.join(args.output, file_path)
                    file_size = os.path.getsize(full_path)
                    print(f"   - {file_path} ({file_size:,} bytes)")
        except Exception as e:
            print(f"⚠️  无法列出输出文件: {e}")

        print(f"\n📖 查看报告: {os.path.join(args.output, 'comfort_report.md')}")
    else:
        print("❌ RFS舒适度评价流程执行失败")
        print(f"📄 执行摘要: {summary_path}")
        print("请检查错误信息并重试")

    print("=" * 60)

    # 返回适当的退出代码
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()