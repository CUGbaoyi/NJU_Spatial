"""
评测可视化模块 - 生成评测报告图表
"""
import os
import re
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter
import numpy as np

import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 中文字体配置 - 按优先级排序（Arial Unicode MS 优先）
CN_FONTS = [
    'Arial Unicode MS',     # 通用，支持所有Unicode字符 ✓
    'Hiragino Sans GB',     # macOS 冬青黑体
    'PingFang HK',          # macOS 苹方港版
    'STHeiti',              # 华文黑体
    'Heiti TC',             # 繁体黑体
    'Microsoft YaHei',      # Windows 微软雅黑
    'SimHei',               # Windows 黑体
]

# 检测可用字体
available_fonts = [f.name for f in fm.fontManager.ttflist]
selected_font = None
for font in CN_FONTS:
    if font in available_fonts:
        selected_font = font
        break

# 设置字体
if selected_font:
    plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
else:
    # 直接指定macOS系统字体路径作为fallback
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

import seaborn as sns
sns.set_style("whitegrid")


class ResultVisualizer:
    """评测结果可视化"""

    # 题型配置
    TASK_CONFIG = {
        "choice": {"name": "选择题", "max_score": 2, "count": 182},
        "tf": {"name": "判断题", "max_score": 2, "count": 134},
        "completion": {"name": "填空题", "max_score": 5, "count": 150},
        "noun": {"name": "名词解释", "max_score": 6, "count": 454},
        "qa": {"name": "简答题", "max_score": 10, "count": 153},
        "discussion": {"name": "论述题", "max_score": 10, "count": 335}
    }

    # 能力维度定义
    DIMENSIONS = {
        "专业术语能力": ["noun", "completion"],
        "基础事实辨析": ["choice", "tf"],
        "过程机理描述": ["qa"],
        "地学综合分析": ["discussion"]
    }

    def __init__(self, results, output_dir: str = "reports"):
        """
        Args:
            results: 可以是单个模型的结果列表，或多模型字典 {model_name: results_list}
        """
        # 强制设置中文字体（在实例化时重新设置，确保生效）
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans GB', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 检测是否为多模型数据
        if isinstance(results, dict):
            self.is_multi_model = True
            self.results_by_model = results
            # 合并所有结果用于旧逻辑兼容
            self.results = []
            for model, model_results in results.items():
                for r in model_results:
                    r_copy = r.copy()
                    r_copy['model'] = model
                    self.results.append(r_copy)
            self.models = list(results.keys())
        else:
            self.is_multi_model = False
            self.results = results
            self.results_by_model = None
            self.models = []

    def _calculate_stats(self) -> Dict[str, Any]:
        """计算统计数据"""
        stats = {
            "by_task": {task: {"scores": [], "total": 0, "max": 0} for task in self.TASK_CONFIG},
            "by_dimension": {dim: {"scores": [], "total": 0, "max": 0} for dim in self.DIMENSIONS},
            "judge_reasons": []
        }

        total_score = 0
        total_max = 0

        for r in self.results:
            task = r.get("task_type", "")
            score = r.get("score", 0)
            reason = r.get("judge_reason", "")

            if task not in self.TASK_CONFIG:
                continue

            cfg = self.TASK_CONFIG[task]
            stats["by_task"][task]["scores"].append(score)
            stats["by_task"][task]["total"] += score
            stats["by_task"][task]["max"] += cfg["max_score"]

            if reason:
                stats["judge_reasons"].append(reason)

            total_score += score
            total_max += cfg["max_score"]

        # 计算维度得分
        for dim_name, tasks in self.DIMENSIONS.items():
            for task in tasks:
                task_stats = stats["by_task"][task]
                stats["by_dimension"][dim_name]["scores"].extend(task_stats["scores"])
                stats["by_dimension"][dim_name]["total"] += task_stats["total"]
                stats["by_dimension"][dim_name]["max"] += task_stats["max"]

        stats["total_score"] = total_score
        stats["total_max"] = total_max

        return stats

    def plot_radar_chart(self, stats: Dict[str, Any]):
        """能力维度雷达图"""
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

        dimensions = list(self.DIMENSIONS.keys())
        N = len(dimensions)

        # 计算每个维度的得分率
        values = []
        for dim in dimensions:
            dim_stats = stats["by_dimension"][dim]
            rate = dim_stats["total"] / dim_stats["max"] if dim_stats["max"] > 0 else 0
            values.append(rate)

        # 闭合雷达图
        values += values[:1]
        angles = [n / N * 2 * np.pi for n in range(N)] + [0]

        ax.plot(angles, values, 'o-', linewidth=2, color='#1f77b4')
        ax.fill(angles, values, alpha=0.25, color='#1f77b4')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title("能力维度雷达图", fontsize=16, pad=20)

        plt.tight_layout()
        plt.savefig(self.output_dir / "01_radar_chart.png", dpi=150, bbox_inches='tight')
        plt.close()

    def plot_task_scores(self, stats: Dict[str, Any]):
        """题型得分率柱状图"""
        fig, ax = plt.subplots(figsize=(12, 6))

        tasks = []
        rates = []
        counts = []

        for task, cfg in self.TASK_CONFIG.items():
            task_stats = stats["by_task"][task]
            if task_stats["max"] > 0:
                rate = task_stats["total"] / task_stats["max"]
                tasks.append(cfg["name"])
                rates.append(rate)
                counts.append(cfg["count"])

        x = np.arange(len(tasks))
        bars = ax.bar(x, rates, color='#2ecc71', alpha=0.8)

        # 添加数值标签
        for i, (bar, rate, count) in enumerate(zip(bars, rates, counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                   f'{rate*100:.1f}%\n({count}题)',
                   ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('题型', fontsize=12)
        ax.set_ylabel('得分率', fontsize=12)
        ax.set_title('各题型得分率对比', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "02_task_scores.png", dpi=150, bbox_inches='tight')
        plt.close()

    def plot_score_distribution(self, stats: Dict[str, Any]):
        """主观题得分分布直方图（10分题）"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # QA题分布
        qa_scores = stats["by_task"]["qa"]["scores"]
        if qa_scores:
            bins = [-0.5, 2, 4, 6, 8, 10.5]
            labels = ['0-2', '3-4', '5-6', '7-8', '9-10']
            axes[0].hist(qa_scores, bins=bins, color='#3498db', alpha=0.7, edgecolor='black')
            axes[0].set_xticks([1, 3, 5, 7, 9])
            axes[0].set_xticklabels(labels)
            axes[0].set_xlabel('分数区间', fontsize=11)
            axes[0].set_ylabel('题数', fontsize=11)
            axes[0].set_title('简答题得分分布 (n=153)', fontsize=13)
            axes[0].grid(alpha=0.3)

        # 论述题分布
        disc_scores = stats["by_task"]["discussion"]["scores"]
        if disc_scores:
            axes[1].hist(disc_scores, bins=bins, color='#e74c3c', alpha=0.7, edgecolor='black')
            axes[1].set_xticks([1, 3, 5, 7, 9])
            axes[1].set_xticklabels(labels)
            axes[1].set_xlabel('分数区间', fontsize=11)
            axes[1].set_ylabel('题数', fontsize=11)
            axes[1].set_title('论述题得分分布 (n=335)', fontsize=13)
            axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "03_score_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()

    def extract_reason_keywords(self, reasons: List[str]) -> List[tuple]:
        """提取扣分理由关键词"""
        if not reasons:
            return []

        # 简单分词和计数
        text = " ".join(reasons)
        # 移除常见停用词
        stop_words = {'的', '了', '是', '在', '有', '和', '与', '等', '或', '但', '而', '对'}
        words = re.findall(r'[一-龥]{2,4}', text)
        words = [w for w in words if w not in stop_words and len(w) >= 2]

        return Counter(words).most_common(5)

    def print_summary(self, stats: Dict[str, Any]):
        """打印控制台汇总看板 (简化格式避免乱码)"""
        print("\n" + "=" * 60)
        print("Evaluation Summary")
        print("=" * 60)

        # 总分
        total = stats["total_score"]
        total_max = stats["total_max"]
        rate = total / total_max * 100 if total_max > 0 else 0
        print(f"\nTotal Score: {total:.0f} / {total_max:.0f} ({rate:.1f}%)")

        # 客观题
        obj_score = stats["by_task"]["choice"]["total"] + stats["by_task"]["tf"]["total"]
        obj_max = stats["by_task"]["choice"]["max"] + stats["by_task"]["tf"]["max"]
        obj_rate = obj_score / obj_max * 100 if obj_max > 0 else 0
        print(f"Objective Accuracy: {obj_rate:.1f}% ({obj_score:.0f}/{obj_max:.0f})")

        # 主观题
        subj_score = (stats["by_task"]["completion"]["total"] +
                     stats["by_task"]["noun"]["total"] +
                     stats["by_task"]["qa"]["total"] +
                     stats["by_task"]["discussion"]["total"])
        subj_max = (stats["by_task"]["completion"]["max"] +
                   stats["by_task"]["noun"]["max"] +
                   stats["by_task"]["qa"]["max"] +
                   stats["by_task"]["discussion"]["max"])
        subj_rate = subj_score / subj_max * 100 if subj_max > 0 else 0
        print(f"Subjective Score Rate: {subj_rate:.1f}% ({subj_score:.0f}/{subj_max:.0f})")

        # 能力维度
        print(f"\nDimension Scores:")
        for dim, tasks in self.DIMENSIONS.items():
            dim_stats = stats["by_dimension"][dim]
            dim_rate = dim_stats["total"] / dim_stats["max"] * 100 if dim_stats["max"] > 0 else 0
            print(f"  - {dim}: {dim_rate:.1f}%")

        # 扣分理由Top 5
        keywords = self.extract_reason_keywords(stats["judge_reasons"])
        if keywords:
            print(f"\nCommon Issues Top 5:")
            for i, (word, count) in enumerate(keywords, 1):
                print(f"  {i}. {word} ({count}x)")

        print("\n" + "=" * 60)

    def plot_model_comparison(self, stats_by_model: Dict[str, Dict]):
        """多模型对比柱状图"""
        if not self.is_multi_model:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        # 任务类型映射
        task_labels = {
            "choice": "选择题",
            "tf": "判断题",
            "completion": "填空题",
            "noun": "名词解释",
            "qa": "简答题",
            "discussion": "论述题"
        }

        tasks = list(self.TASK_CONFIG.keys())

        # 为每个任务类型绘制子图
        for idx, task in enumerate(tasks[:4]):
            ax = axes[idx]
            models = []
            rates = []

            for model in self.models:
                model_stats = stats_by_model[model]["by_task"][task]
                if model_stats["max"] > 0:
                    rate = model_stats["total"] / model_stats["max"]
                    models.append(model)
                    rates.append(rate)

            if models:
                x = np.arange(len(models))
                colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
                bars = ax.bar(x, rates, color=colors)
                ax.set_xticks(x)
                ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
                ax.set_ylabel('得分率', fontsize=10)
                ax.set_title(f'{task_labels.get(task, task)}', fontsize=12)
                ax.set_ylim(0, 1.1)
                ax.grid(axis='y', alpha=0.3)

                # 添加数值标签
                for bar, rate in zip(bars, rates):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                           f'{rate*100:.1f}%', ha='center', va='bottom', fontsize=7)

        plt.suptitle('各模型得分率对比', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "model_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()

    def plot_model_overall(self, stats_by_model: Dict[str, Dict]):
        """多模型总得分对比"""
        if not self.is_multi_model:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        models = []
        total_rates = []
        obj_rates = []
        subj_rates = []

        for model in self.models:
            stats = stats_by_model[model]

            # 总体得分率
            total_rate = stats["total_score"] / stats["total_max"] if stats["total_max"] > 0 else 0

            # 客观题准确率
            obj_score = stats["by_task"]["choice"]["total"] + stats["by_task"]["tf"]["total"]
            obj_max = stats["by_task"]["choice"]["max"] + stats["by_task"]["tf"]["max"]
            obj_rate = obj_score / obj_max if obj_max > 0 else 0

            # 主观题得分率
            subj_score = (stats["by_task"]["completion"]["total"] +
                         stats["by_task"]["noun"]["total"] +
                         stats["by_task"]["qa"]["total"] +
                         stats["by_task"]["discussion"]["total"])
            subj_max = (stats["by_task"]["completion"]["max"] +
                       stats["by_task"]["noun"]["max"] +
                       stats["by_task"]["qa"]["max"] +
                       stats["by_task"]["discussion"]["max"])
            subj_rate = subj_score / subj_max if subj_max > 0 else 0

            models.append(model)
            total_rates.append(total_rate)
            obj_rates.append(obj_rate)
            subj_rates.append(subj_rate)

        x = np.arange(len(models))
        width = 0.25

        ax.bar(x - width, total_rates, width, label='总体', color='#1f77b4')
        ax.bar(x, obj_rates, width, label='客观题', color='#2ecc71')
        ax.bar(x + width, subj_rates, width, label='主观题', color='#e74c3c')

        ax.set_xlabel('模型', fontsize=12)
        ax.set_ylabel('得分率', fontsize=12)
        ax.set_title('各模型得分率总览', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        plt.savefig(self.output_dir / "model_overall.png", dpi=150, bbox_inches='tight')
        plt.close()

    def print_multi_model_summary(self, stats_by_model: Dict[str, Dict]):
        """打印多模型对比汇总 (简化格式避免乱码)"""
        if not self.is_multi_model:
            return

        print("\n" + "=" * 70)
        print("Model Comparison")
        print("=" * 70)

        # 按总体得分率排序
        sorted_models = sorted(
            self.models,
            key=lambda m: stats_by_model[m]["total_score"] / stats_by_model[m]["total_max"] if stats_by_model[m]["total_max"] > 0 else 0,
            reverse=True
        )

        for i, model in enumerate(sorted_models, 1):
            stats = stats_by_model[model]

            total_rate = stats["total_score"] / stats["total_max"] if stats["total_max"] > 0 else 0

            obj_score = stats["by_task"]["choice"]["total"] + stats["by_task"]["tf"]["total"]
            obj_max = stats["by_task"]["choice"]["max"] + stats["by_task"]["tf"]["max"]
            obj_rate = obj_score / obj_max if obj_max > 0 else 0

            subj_score = (stats["by_task"]["completion"]["total"] +
                         stats["by_task"]["noun"]["total"] +
                         stats["by_task"]["qa"]["total"] +
                         stats["by_task"]["discussion"]["total"])
            subj_max = (stats["by_task"]["completion"]["max"] +
                       stats["by_task"]["noun"]["max"] +
                       stats["by_task"]["qa"]["max"] +
                       stats["by_task"]["discussion"]["max"])
            subj_rate = subj_score / subj_max if subj_max > 0 else 0

            print(f"{i}. {model}")
            print(f"   Overall: {total_rate*100:.1f}% | Objective: {obj_rate*100:.1f}% | Subjective: {subj_rate*100:.1f}%")

        print("\n" + "=" * 70)

    def generate_all(self):
        """生成所有图表"""
        stats = self._calculate_stats()

        print("正在生成可视化报告...")

        if self.is_multi_model:
            # 多模式：计算每个模型的统计数据
            stats_by_model = {}
            for model in self.models:
                model_results = [r for r in self.results if r.get('model') == model]
                # 临时创建一个包含model字段的results
                original_results = self.results
                self.results = model_results
                stats_by_model[model] = self._calculate_stats()
                self.results = original_results

            # 生成多模型对比图表
            self.plot_model_overall(stats_by_model)
            print("  ✓ 多模型总览: model_overall.png")

            self.plot_model_comparison(stats_by_model)
            print("  ✓ 模型对比: model_comparison.png")

            # 打印多模型对比汇总
            self.print_multi_model_summary(stats_by_model)
        else:
            # 单模型模式
            self.plot_radar_chart(stats)
            print("  ✓ 雷达图: 01_radar_chart.png")

            self.plot_task_scores(stats)
            print("  ✓ 柱状图: 02_task_scores.png")

            self.plot_score_distribution(stats)
            print("  ✓ 直方图: 03_score_distribution.png")

            self.print_summary(stats)

        print(f"\n报告已保存至: {self.output_dir}/")
