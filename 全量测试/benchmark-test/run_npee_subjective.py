#!/usr/bin/env python3
"""
NPEE主观题评测 - 填空、名词解释、问答、讨论（串行，使用Judge）
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
    config_dir = Path(__file__).parent / "config"

    with open(config_dir / "models.yaml", 'r', encoding='utf-8') as f:
        models_data = yaml.safe_load(f)
    models = [ModelConfig(
        name=m["name"],
        provider=m["provider"],
        api_base=m["api_base"],
        model_id=m["model_id"]
    ) for m in models_data["models"]]

    with open(config_dir / "judge.yaml", 'r', encoding='utf-8') as f:
        judge_data = yaml.safe_load(f)
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
    print("NPEE主观题评测 - 串行模式")
    print("=" * 60)

    models, judge = load_config()

    print(f"\n待测模型 ({len(models)}个):")
    for i, m in enumerate(models, 1):
        print(f"  {i}. {m.name}")

    # 串行配置 - 单模型内不并发
    config = BenchmarkConfig(
        models=models,
        judge=judge,
        max_concurrent_per_model=1,   # 串行，取消并发
        max_concurrent_models=2,      # 双模型交替（不同API端点）
        enable_judge_cache=True,
        enable_batch_judge=True,
        judge_batch_size=1
    )

    print(f"\n并发配置: {config.max_concurrent_models}模型交替 × {config.max_concurrent_per_model}串行")
    print(f"Judge: {judge.name} (缓存已启用)")

    print(f"\n数据集:")
    print(f"  - 填空题 (150题)")
    print(f"  - 名词解释 (454题) ← 0-6分")
    print(f"  - 问答 (153题) ← 0-10分")
    print(f"  - 讨论 (335题) ← 0-10分")
    print(f"  合计: 1092题")

    data_dir = Path(__file__).parent / "data"
    runner = BenchmarkRunner(config=config, data_dir=str(data_dir))

    print("\n" + "=" * 60)
    datasets = ["npee_completion", "npee_noun", "npee_qa", "npee_discussion"]
    summary = runner.run_all(datasets=datasets)

    reporter = BenchmarkReporter(
        results=runner.results,
        output_dir=str(Path(__file__).parent / "reports")
    )
    json_path = reporter.generate_json_report()

    print(f"\n✓ 主观题评测完成!")
    print(f"  JSON报告: {json_path}")

    # 生成可视化
    print(f"\n生成可视化报告...")
    viz_dir = Path(__file__).parent / "reports"
    visualizer = ReportVisualizer(runner.results, str(viz_dir))
    visualizer.generate_all()


if __name__ == "__main__":
    main()
