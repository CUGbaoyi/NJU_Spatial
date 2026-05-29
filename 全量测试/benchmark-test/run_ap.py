#!/usr/bin/env python3
"""
AP题目评测脚本

高并发模式 - 快速稳定
"""
import sys
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models import ModelConfig
from src.runner import BenchmarkRunner, BenchmarkConfig
from src.reporter import BenchmarkReporter
from src.visualizer import ReportVisualizer


def load_config():
    """加载配置"""
    config_dir = Path(__file__).parent / "config"

    with open(config_dir / "models.yaml", 'r', encoding='utf-8') as f:
        models_data = yaml.safe_load(f)

    models = [ModelConfig(**m) for m in models_data["models"]]

    with open(config_dir / "judge.yaml", 'r', encoding='utf-8') as f:
        judge_data = yaml.safe_load(f)

    # 只传递ModelConfig需要的字段
    judge_info = judge_data["judge"]
    judge = ModelConfig(
        name=judge_info["name"],
        provider=judge_info["provider"],
        api_base=judge_info["api_base"],
        model_id=judge_info["model_id"]
    )

    return models, judge


def main():
    print("=" * 60)
    print("AP题目评测 - 高并发模式")
    print("=" * 60)

    # 检查API密钥
    if not os.environ.get("NEWAPI_KEY"):
        print("错误: 缺少 NEWAPI_KEY 环境变量")
        return

    # 加载配置
    models, judge = load_config()

    print(f"\n待测模型 ({len(models)}个):")
    for i, m in enumerate(models, 1):
        print(f"  {i}. {m.name}")

    # AP题目高并发配置
    config = BenchmarkConfig(
        models=models,
        judge=judge,
        max_concurrent_per_model=8,   # 高并发
        max_concurrent_models=3,      # 多模型并发
        enable_judge_cache=False,     # AP不需要Judge
        enable_batch_judge=False
    )

    print(f"\n并发配置: {config.max_concurrent_models}模型 × {config.max_concurrent_per_model}并发")
    print(f"数据集: AP Study (1395题选择题)")

    # 创建Runner
    data_dir = Path(__file__).parent / "data"
    runner = BenchmarkRunner(config=config, data_dir=str(data_dir))

    # 运行评测
    print("\n" + "=" * 60)
    summary = runner.run_all(datasets=["apstudy"])

    # 生成报告
    print("\n生成报告...")
    reporter = BenchmarkReporter(
        results=runner.results,
        output_dir=str(Path(__file__).parent / "reports")
    )
    json_path = reporter.generate_json_report()
    md_path = reporter.generate_markdown_report()

    print(f"\n✓ AP题目评测完成!")
    print(f"  JSON报告: {json_path}")
    print(f"  Markdown报告: {md_path}")

    # 打印摘要
    stats = summary.get("statistics", {})
    print(f"\n评测统计:")
    print(f"  完成数: {stats.get('completed', 0)}")
    print(f"  错误数: {stats.get('errors', 0)}")
    print(f"  耗时: {stats.get('elapsed_seconds', 0):.1f}秒")

    # 生成可视化报告
    print(f"\n生成可视化报告...")
    viz_dir = Path(__file__).parent / "reports"
    visualizer = ReportVisualizer(runner.results, str(viz_dir))
    visualizer.generate_all()


if __name__ == "__main__":
    main()
