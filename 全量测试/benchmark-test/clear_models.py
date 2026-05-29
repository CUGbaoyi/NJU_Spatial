#!/usr/bin/env python3
"""清除指定模型的数据，准备重新测试"""
import json
from pathlib import Path

checkpoint_path = Path(__file__).parent / "results" / "eval_checkpoint.json"

with open(checkpoint_path, 'r') as f:
    data = json.load(f)

models_to_clear = ['kimi-k2.5', 'deepseek-v3.2']

print("清除前:")
for m in models_to_clear:
    if m in data:
        print(f"  {m}: {len(data[m])} 条")

for m in models_to_clear:
    if m in data:
        del data[m]
        print(f"✓ 已清除 {m}")

with open(checkpoint_path, 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("\n现在可以运行 python3 run_4_models.py 重新测试")
