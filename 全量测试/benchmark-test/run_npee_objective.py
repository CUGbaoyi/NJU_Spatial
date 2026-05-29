#!/usr/bin/env python3
"""
NPEE客观题评测 - 选择题、判断题（高并发，无需Judge）
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
    print("NPEE客观题评测 - 选择题 + 判断题")
    print("=" * 60)

    models, judge = load_config()

    print(f"\n待测模型 ({len(models)}个):")
    for i, m in enumerate(models, 1):
        print(f"  {i}. {m.name}")

    # 高并发配置
    config = BenchmarkConfig(
        models=models,
        judge=judge,
        max_concurrent_per_model=8,   # 高并发
        max_concurrent_models=3,      # 多模型并发
        enable_judge_cache=False,
        enable_batch_judge=False
    )

    print(f"\n并发配置: {config.max_concurrent_models}模型 × {config.max_concurrent_per_model}并发")
    print(f"数据集: 选择题(182) + 判断题(134) = 316题")

    data_dir = Path(__file__).parent / "data"
    runner = BenchmarkRunner(config=config, data_dir=str(data_dir))

    print("\n" + "=" * 60)
    summary = runner.run_all(datasets=["npee_choice", "npee_tf"])

    reporter = BenchmarkReporter(
        results=runner.results,
        output_dir=str(Path(__file__).parent / "reports")
    )
    json_path = reporter.generate_json_report()

    print(f"\n✓ 客观题评测完成!")
    print(f"  JSON报告: {json_path}")


if __name__ == "__main__":
    main()
