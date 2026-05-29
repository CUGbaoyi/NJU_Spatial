"""
报告生成器 - 生成评测报告
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


class BenchmarkReporter:
    """Benchmark报告生成器"""

    def __init__(self, results: Dict[str, List], output_dir: str | None = None):
        """
        Args:
            results: 评测结果 {model_name: [EvaluationResult, ...]}
            output_dir: 输出目录
        """
        self.results = results
        self.output_dir = Path(output_dir) if output_dir else Path("reports")
        self.output_dir.mkdir(exist_ok=True)

    def generate_json_report(self) -> str:
        """生成JSON格式报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": self._calculate_summary(),
            "detailed_results": self._format_detailed_results()
        }

        output_file = self.output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        return str(output_file)

    def generate_markdown_report(self) -> str:
        """生成Markdown格式报告"""
        summary = self._calculate_summary()

        lines = [
            "# GeoBenchmark 评测报告",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**模型数量**: {len(self.results)}",
            "",
            "## 1. 总体概览",
            "",
        ]

        # 生成总体对比表
        lines.extend(self._generate_summary_table(summary))

        lines.extend([
            "",
            "## 2. 各题型详细结果",
            "",
        ])

        # 各题型详细结果
        lines.extend(self._generate_task_type_details(summary))

        lines.extend([
            "",
            "## 3. 模型排名",
            "",
        ])

        # 模型排名
        lines.extend(self._generate_rankings(summary))

        output_file = self.output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        return str(output_file)

    def _calculate_summary(self) -> Dict[str, Any]:
        """计算汇总统计"""
        summary = {
            "models": {},
            "task_types": set(),
            "overall_stats": {}
        }

        for model_name, results in self.results.items():
            model_stats = {
                "total": len(results),
                "errors": 0,
                "by_task": {},
                "total_score": 0,
                "weighted_accuracy": 0
            }

            task_stats = {}

            for result in results:
                task_type = result.task_type
                summary["task_types"].add(task_type)

                if task_type not in task_stats:
                    task_stats[task_type] = {
                        "count": 0,
                        "correct": 0,
                        "total_score": 0,
                        "max_score_per_question": 1 if task_type in ["choice", "tf", "completion"] else (6 if task_type == "noun" else 10)
                    }

                task_stats[task_type]["count"] += 1

                if result.error:
                    model_stats["errors"] += 1
                else:
                    task_stats[task_type]["total_score"] += result.score
                    if result.is_correct is not None and result.is_correct:
                        task_stats[task_type]["correct"] += 1

            # 计算各题型准确率和平均分
            for task_type, stats in task_stats.items():
                count = stats["count"]
                accuracy = stats["correct"] / count if count > 0 else 0
                avg_score = stats["total_score"] / count if count > 0 else 0

                model_stats["by_task"][task_type] = {
                    "count": count,
                    "accuracy": accuracy,
                    "avg_score": avg_score,
                    "max_score": stats["max_score_per_question"]
                }

            summary["models"][model_name] = model_stats

        summary["task_types"] = sorted(list(summary["task_types"]))

        return summary

    def _generate_summary_table(self, summary: Dict) -> List[str]:
        """生成汇总表格"""
        lines = [
            "| 模型 | 总题数 | 错误数 | 选择题准确率 | 填空题准确率 | 判断题准确率 | 名词解释均分 | 问答均分 | 讨论题均分 |",
            "|------|--------|--------|--------------|--------------|--------------|--------------|----------|------------|"
        ]

        for model_name, stats in sorted(summary["models"].items()):
            by_task = stats["by_task"]

            choice_acc = by_task.get("choice", {}).get("accuracy", 0)
            tf_acc = by_task.get("tf", {}).get("accuracy", 0)
            completion_acc = by_task.get("completion", {}).get("accuracy", 0)
            noun_score = by_task.get("noun", {}).get("avg_score", 0)
            qa_score = by_task.get("qa", {}).get("avg_score", 0)
            discussion_score = by_task.get("discussion", {}).get("avg_score", 0)

            lines.append(
                f"| {model_name} | {stats['total']} | {stats['errors']} | "
                f"{choice_acc:.2%} | {completion_acc:.2%} | {tf_acc:.2%} | "
                f"{noun_score:.2f}/6 | {qa_score:.2f}/10 | {discussion_score:.2f}/10 |"
            )

        return lines

    def _generate_task_type_details(self, summary: Dict) -> List[str]:
        """生成各题型详细结果"""
        lines = []

        task_type_names = {
            "choice": "选择题",
            "completion": "填空题",
            "tf": "判断题",
            "noun": "名词解释",
            "qa": "问答",
            "discussion": "讨论题"
        }

        for task_type in summary["task_types"]:
            lines.append(f"### {task_type_names.get(task_type, task_type)}")
            lines.append("")
            lines.append("| 模型 | 题数 | 准确率/均分 |")
            lines.append("|------|------|-------------|")

            for model_name, stats in sorted(summary["models"].items()):
                task_stats = stats["by_task"].get(task_type, {})
                if not task_stats:
                    continue

                count = task_stats["count"]

                if task_type in ["choice", "tf", "completion"]:
                    metric = f"{task_stats['accuracy']:.2%}"
                else:
                    max_score = task_stats["max_score"]
                    metric = f"{task_stats['avg_score']:.2f}/{max_score}"

                lines.append(f"| {model_name} | {count} | {metric} |")

            lines.append("")

        return lines

    def _generate_rankings(self, summary: Dict) -> List[str]:
        """生成模型排名"""
        lines = []

        # 选择题排名
        choice_ranking = []
        for model_name, stats in summary["models"].items():
            acc = stats["by_task"].get("choice", {}).get("accuracy", 0)
            count = stats["by_task"].get("choice", {}).get("count", 0)
            if count > 0:
                choice_ranking.append((model_name, acc))

        choice_ranking.sort(key=lambda x: x[1], reverse=True)

        lines.append("### 选择题排名")
        lines.append("")
        for i, (model, acc) in enumerate(choice_ranking, 1):
            lines.append(f"{i}. **{model}**: {acc:.2%}")
        lines.append("")

        # 名词解释排名
        noun_ranking = []
        for model_name, stats in summary["models"].items():
            score = stats["by_task"].get("noun", {}).get("avg_score", 0)
            count = stats["by_task"].get("noun", {}).get("count", 0)
            if count > 0:
                noun_ranking.append((model_name, score))

        noun_ranking.sort(key=lambda x: x[1], reverse=True)

        lines.append("### 名词解释排名")
        lines.append("")
        for i, (model, score) in enumerate(noun_ranking, 1):
            lines.append(f"{i}. **{model}**: {score:.2f}/6")
        lines.append("")

        # 问答排名
        qa_ranking = []
        for model_name, stats in summary["models"].items():
            score = stats["by_task"].get("qa", {}).get("avg_score", 0)
            count = stats["by_task"].get("qa", {}).get("count", 0)
            if count > 0:
                qa_ranking.append((model_name, score))

        qa_ranking.sort(key=lambda x: x[1], reverse=True)

        lines.append("### 问答题排名")
        lines.append("")
        for i, (model, score) in enumerate(qa_ranking, 1):
            lines.append(f"{i}. **{model}**: {score:.2f}/10")
        lines.append("")

        return lines

    def _format_detailed_results(self) -> Dict[str, Any]:
        """格式化详细结果"""
        detailed = {}

        for model_name, results in self.results.items():
            model_results = []

            for result in results:
                model_results.append({
                    "question_id": result.question_id,
                    "task_type": result.task_type,
                    "model_answer": result.model_answer[:500],  # 限制长度
                    "reference_answer": result.reference_answer[:500],
                    "score": result.score,
                    "is_correct": result.is_correct,
                    "details": result.details,
                    "error": result.error
                })

            detailed[model_name] = model_results

        return detailed
