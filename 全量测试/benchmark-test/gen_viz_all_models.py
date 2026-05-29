#!/usr/bin/env python3
"""包含所有模型的可视化"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import json
from eval_viz import ResultVisualizer

# 加载checkpoint
checkpoint_path = Path(__file__).parent / "results" / "eval_checkpoint.json"
with open(checkpoint_path, 'r') as f:
    data = json.load(f)

# 所有要可视化的模型
models_to_viz = [
    'deepseek-v3.2',
    'kimi-k2.5',
    'claude-opus-4.6',
    'gemini-3-pro-preview',
    'qwen3-32b',
    'qwen3.5-plus',
    'qwen3.6-plus',
    'qwen3.5-397b-a17b',
    'qwen3.6-35b-a3b',
]

results_by_model = {}

for model in models_to_viz:
    if model in data:
        model_results = []
        for qid, state in data[model].items():
            if state.get('status') == 'completed':
                # 从question_id提取task_type
                task_type = 'unknown'
                for tt in ['choice', 'tf', 'completion', 'noun', 'qa', 'discussion']:
                    if tt in qid:
                        task_type = tt
                        break
                model_results.append({
                    'question_id': qid,
                    'task_type': task_type,
                    'score': state.get('score', 0),
                    'judge_reason': state.get('judge_reason', '')
                })
        if model_results:
            results_by_model[model] = model_results
            print(f'{model}: {len(model_results)} 条结果')
    else:
        print(f'警告: {model} 数据不存在')

total_count = sum(len(r) for r in results_by_model.values())
print(f'\n总共: {len(results_by_model)} 个模型，{total_count} 条结果')

# 生成可视化
viz = ResultVisualizer(results_by_model, output_dir='reports')
viz.generate_all()

print('\n完成！图表保存在 reports/ 目录')
