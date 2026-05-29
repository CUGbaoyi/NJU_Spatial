"""
可视化报告生成器 - 生成图表
"""
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import json


class ReportVisualizer:
    """报告可视化生成器"""

    def __init__(self, results: Dict[str, List], output_dir: str):
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 计算统计
        self.stats = self._calculate_stats()

    def _calculate_stats(self) -> Dict:
        """计算统计数据"""
        stats = {
            "models": {},
            "task_types": set()  # 用set来收集题型
        }

        for model_name, results in self.results.items():
            model_stats = {
                "by_task": {},
                "total": len(results),
                "errors": 0
            }

            for result in results:
                task_type = result.task_type
                stats["task_types"].add(task_type)

                if task_type not in model_stats["by_task"]:
                    model_stats["by_task"][task_type] = {
                        "count": 0,
                        "correct": 0,
                        "total_score": 0,
                        "max_score": 1 if task_type in ["choice", "tf", "completion"]
                                   else (6 if task_type == "noun" else 10)
                    }

                model_stats["by_task"][task_type]["count"] += 1
                model_stats["by_task"][task_type]["total_score"] += result.score

                if result.error:
                    model_stats["errors"] += 1
                elif result.is_correct:
                    model_stats["by_task"][task_type]["correct"] += 1

            # 计算准确率/平均分
            for task_type, task_stat in model_stats["by_task"].items():
                count = task_stat["count"]
                task_stat["accuracy"] = task_stat["correct"] / count if count > 0 else 0
                task_stat["avg_score"] = task_stat["total_score"] / count if count > 0 else 0

            stats["models"][model_name] = model_stats

        # 最后转换成list
        stats["task_types"] = sorted(list(stats["task_types"]))

        return stats

    def generate_all(self) -> List[str]:
        """生成所有可视化图表"""
        output_files = []

        # 1. 总体准确率对比
        output_files.append(self._plot_overall_accuracy())

        # 2. 各题型对比（柱状图）
        output_files.append(self._plot_task_type_comparison())

        # 3. 模型雷达图
        output_files.append(self._plot_radar_chart())

        # 4. 错误率对比
        output_files.append(self._plot_error_rate())

        # 5. 主观题得分对比
        output_files.append(self._plot_subjective_scores())

        print(f"\n✓ 已生成 {len(output_files)} 个可视化图表")
        for f in output_files:
            print(f"  {f}")

        return output_files

    def _plot_overall_accuracy(self) -> str:
        """总体准确率对比柱状图"""
        fig, ax = plt.subplots(figsize=(12, 6))

        models = []
        choice_acc = []
        tf_acc = []
        completion_acc = []

        for model_name in sorted(self.stats["models"].keys()):
            by_task = self.stats["models"][model_name]["by_task"]
            models.append(model_name)

            choice_acc.append(by_task.get("choice", {}).get("accuracy", 0) * 100)
            tf_acc.append(by_task.get("tf", {}).get("accuracy", 0) * 100)
            completion_acc.append(by_task.get("completion", {}).get("accuracy", 0) * 100)

        x = np.arange(len(models))
        width = 0.25

        ax.bar(x - width, choice_acc, width, label='选择题', color='#3498db')
        ax.bar(x, tf_acc, width, label='判断题', color='#2ecc71')
        ax.bar(x + width, completion_acc, width, label='填空题', color='#e74c3c')

        ax.set_xlabel('模型', fontsize=12)
        ax.set_ylabel('准确率 (%)', fontsize=12)
        ax.set_title('各模型客观题准确率对比', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 100)

        plt.tight_layout()
        output_file = self.output_dir / "01_overall_accuracy.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        return str(output_file)

    def _plot_task_type_comparison(self) -> str:
        """各题型详细对比（柱状图）"""
        task_types = [t for t in self.stats["task_types"] if t in self.stats["models"].get(list(self.stats["models"].keys())[0], {}).get("by_task", {})]

        task_names = {
            "choice": "选择题",
            "tf": "判断题",
            "completion": "填空题",
            "noun": "名词解释",
            "qa": "问答",
            "discussion": "讨论"
        }

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        colors = plt.cm.Set3(np.linspace(0, 1, len(self.stats["models"])))

        for idx, task_type in enumerate(task_types):
            ax = axes[idx]
            models = []
            values = []

            for model_name in sorted(self.stats["models"].keys()):
                by_task = self.stats["models"][model_name]["by_task"]
                if task_type in by_task:
                    models.append(model_name)
                    if task_type in ["choice", "tf", "completion"]:
                        values.append(by_task[task_type]["accuracy"] * 100)
                    else:
                        max_score = by_task[task_type]["max_score"]
                        values.append(by_task[task_type]["avg_score"] / max_score * 100)

            x = np.arange(len(models))
            bars = ax.bar(x, values, color=colors[:len(models)])
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('分数/准确率 (%)' if idx == 0 else '')
            ax.set_title(task_names.get(task_type, task_type), fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 100)

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=7)

        plt.suptitle('各题型详细对比', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = self.output_dir / "02_task_type_comparison.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        return str(output_file)

    def _plot_radar_chart(self) -> str:
        """模型能力雷达图"""
        # 只测试有数据的题型
        task_types = ["choice", "tf", "completion", "noun", "qa", "discussion"]
        task_labels = ["选择题", "判断题", "填空题", "名词解释", "问答", "讨论"]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(task_types), endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        # 为每个模型绘制雷达图
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.stats["models"])))

        for idx, (model_name, color) in enumerate(zip(sorted(self.stats["models"].keys()), colors)):
            by_task = self.stats["models"][model_name]["by_task"]
            values = []

            for task_type in task_types:
                if task_type in by_task:
                    if task_type in ["choice", "tf", "completion"]:
                        values.append(by_task[task_type]["accuracy"] * 100)
                    else:
                        max_score = by_task[task_type]["max_score"]
                        values.append(by_task[task_type]["avg_score"] / max_score * 100)
                else:
                    values.append(0)

            values += values[:1]  # 闭合

            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(task_labels)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.set_title('模型能力雷达图', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        output_file = self.output_dir / "03_radar_chart.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        return str(output_file)

    def _plot_error_rate(self) -> str:
        """错误率对比"""
        fig, ax = plt.subplots(figsize=(10, 6))

        models = []
        error_rates = []
        error_counts = []

        for model_name in sorted(self.stats["models"].keys()):
            model_stats = self.stats["models"][model_name]
            total = model_stats["total"]
            errors = model_stats["errors"]

            models.append(model_name)
            error_rates.append(errors / total * 100 if total > 0 else 0)
            error_counts.append(errors)

        x = np.arange(len(models))
        colors = ['#e74c3c' if rate > 10 else '#f39c12' if rate > 5 else '#27ae60' for rate in error_rates]

        bars = ax.bar(x, error_rates, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('错误率 (%)', fontsize=12)
        ax.set_title('各模型错误率对比', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for bar, count in zip(bars, error_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%\n({count}个)', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        output_file = self.output_dir / "04_error_rate.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        return str(output_file)

    def _plot_subjective_scores(self) -> str:
        """主观题得分对比"""
        subjective_tasks = ["noun", "qa", "discussion"]
        task_labels = ["名词解释\n(0-6分)", "问答\n(0-10分)", "讨论\n(0-10分)"]
        task_max = [6, 10, 10]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, (task_type, label, max_score) in enumerate(zip(subjective_tasks, task_labels, task_max)):
            ax = axes[idx]

            models = []
            scores = []

            for model_name in sorted(self.stats["models"].keys()):
                by_task = self.stats["models"][model_name]["by_task"]
                if task_type in by_task:
                    models.append(model_name)
                    scores.append(by_task[task_type]["avg_score"])

            if not models:
                ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes)
                continue

            x = np.arange(len(models))
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))

            bars = ax.bar(x, scores, color=colors)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('平均分', fontsize=10)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.set_ylim(0, max_score)
            ax.grid(axis='y', alpha=0.3)
            ax.axhline(y=max_score * 0.6, color='r', linestyle='--', alpha=0.5, label='及格线')

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)

            if idx == 2:
                ax.legend(fontsize=8)

        plt.suptitle('主观题得分对比', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = self.output_dir / "05_subjective_scores.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        return str(output_file)


def generate_visual_report(results: Dict[str, List], output_dir: str) -> List[str]:
    """生成可视化报告的便捷函数"""
    visualizer = ReportVisualizer(results, output_dir)
    return visualizer.generate_all()
