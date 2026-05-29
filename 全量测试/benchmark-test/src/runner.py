"""
测试控制器 - 协调模型调用和评测

优化特性:
1. 模型级并发 - 多个模型同时测试
2. 持久化Judge缓存 - 跨模型、跨会话复用评判结果
3. 实时进度跟踪
4. 检查点保存
"""
import yaml
import asyncio
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from .data_loader import GeoBenchmarkLoader, Question
from .models import create_client, ModelConfig
from .evaluators import ExactMatchEvaluator, FuzzyMatchEvaluator, LLMJudgeEvaluator, EvaluationResult
from .cache import JudgeCache


@dataclass
class BenchmarkConfig:
    """Benchmark配置"""
    models: List[ModelConfig]
    judge: ModelConfig
    max_concurrent_per_model: int = 3
    max_concurrent_judge: int = 5
    # 同时测试的模型数量
    max_concurrent_models: int = 3
    # Judge缓存配置
    enable_judge_cache: bool = True
    judge_cache_dir: str | None = None
    # 批量评判配置
    enable_batch_judge: bool = True
    judge_batch_size: int = 5


class BenchmarkRunner:
    """Benchmark测试控制器（优化版）"""

    def __init__(self, config: BenchmarkConfig, data_dir: str):
        self.config = config
        self.data_dir = data_dir
        self.loader = GeoBenchmarkLoader(data_dir)

        # 加载所有提示词模板（包括task和judge）
        self._all_prompts = self._load_prompts()
        self.task_prompts = self._all_prompts.get("task_prompts", {})
        self.judge_prompts = self._all_prompts.get("judge_prompts", {})

        # 初始化评测器
        self.exact_evaluator = ExactMatchEvaluator()
        self.fuzzy_evaluator = FuzzyMatchEvaluator()

        # Judge客户端和评测器（延迟初始化）
        self._judge_client = None
        self._llm_evaluator = None
        self._evaluator_lock = threading.Lock()

        # 持久化Judge缓存（跨模型共享）
        self._judge_cache: JudgeCache | None = None
        if config.enable_judge_cache:
            cache_dir = config.judge_cache_dir or str(Path(__file__).parent.parent / "results" / "cache")
            self._judge_cache = JudgeCache(cache_dir=cache_dir)
            print(f"✓ Judge缓存已启用: {self._judge_cache.db_path}")

        # 结果存储
        self.results: Dict[str, List[EvaluationResult]] = {}

        # 进度跟踪
        self.progress = {
            "total": 0,
            "completed": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None
        }

    def _load_prompts(self) -> Dict[str, str]:
        """加载提示词模板"""
        prompt_path = Path(__file__).parent.parent / "config" / "prompts.yaml"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data

    def _get_judge_client(self):
        """获取Judge客户端（延迟初始化，线程安全）"""
        if self._judge_client is None:
            with self._evaluator_lock:
                if self._judge_client is None:
                    self._judge_client = create_client(self.config.judge)
        return self._judge_client

    def _get_llm_evaluator(self):
        """获取LLM评测器（延迟初始化，线程安全，共享缓存）"""
        if self._llm_evaluator is None:
            with self._evaluator_lock:
                if self._llm_evaluator is None:
                    self._llm_evaluator = LLMJudgeEvaluator(
                        judge_client=self._get_judge_client(),
                        prompt_templates=self.judge_prompts,
                        cache=self._judge_cache,  # 共享持久化缓存
                        enable_batch=self.config.enable_batch_judge,
                        batch_size=self.config.judge_batch_size
                    )
        return self._llm_evaluator

    def _get_evaluator(self, question: Question):
        """根据题型获取评测器（复用评测器实例）"""
        task_type = question.task_type

        # 选择题和判断题用精确匹配
        if task_type in ["choice", "tf"]:
            return self.exact_evaluator

        # 填空题用模糊匹配
        elif task_type == "completion":
            return self.fuzzy_evaluator

        # 开放式问题用LLM Judge（复用实例，共享缓存）
        elif task_type in ["noun", "qa", "discussion"]:
            return self._get_llm_evaluator()

        else:
            raise ValueError(f"未知题型: {task_type}")

    def _format_prompt(self, question: Question) -> str:
        """格式化问题提示词（使用缓存的模板）"""
        task_type = question.task_type
        template = self.task_prompts.get(task_type, "{question}")

        # 处理选择题
        if task_type == "choice":
            if question.choices:
                # AP Study格式：有单独的choices字段
                choices_text = "\n".join([
                    f"{c['label']}. {c['text']}"
                    for c in question.choices
                ])
                return template.format(
                    question=question.question,
                    choices=choices_text
                )
            else:
                # NPEE格式：选项已在问题文本中
                # 使用安全替换，避免KeyError
                import re
                # 检查模板是否有{choices}占位符
                if '{choices}' in template or '{ choices }' in template:
                    # 有{choices}但choices为None，替换为空字符串
                    prompt = template.replace('{choices}', '').replace('{ choices }', '')
                    return prompt.format(question=question.question)
                return template.format(question=question.question)

        return template.format(question=question.question)

    def _evaluate_single_question(
        self,
        model_client,
        question: Question,
        model_name: str,
        max_retries: int = 3
    ) -> EvaluationResult:
        """评测单个问题（带重试机制）"""
        import time

        # 格式化提示词
        prompt = self._format_prompt(question)

        # 重试逻辑
        for attempt in range(max_retries):
            try:
                # 调用模型
                model_answer = model_client.generate(
                    prompt=prompt,
                    system_prompt="你是地理学专家。请根据题目要求准确回答问题。",
                    max_tokens=4000,
                    temperature=0.7
                )

                # 获取评测器
                evaluator = self._get_evaluator(question)

                # 评测
                result = evaluator.evaluate(
                    question=question,
                    model_answer=model_answer,
                    reference_answer=question.reference_answer
                )

                # 填充模型名称
                result.model_name = model_name

                return result

            except Exception as e:
                # 最后一次尝试失败
                if attempt == max_retries - 1:
                    error_msg = str(e)
                    # 记录重试次数
                    if attempt > 0:
                        error_msg = f"重试{attempt}次后失败: {error_msg}"
                    return EvaluationResult(
                        question_id=question.id,
                        model_name=model_name,
                        task_type=question.task_type,
                        model_answer="",
                        reference_answer=question.reference_answer,
                        score=0,
                        is_correct=False,
                        details={},
                        error=error_msg
                    )

                # 指数退避等待后重试
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                print(f"  题目 {question.id} 第{attempt + 1}次尝试失败: {str(e)[:50]}... {wait_time}秒后重试")
                time.sleep(wait_time)

    def run_model(self, model_config: ModelConfig, questions: List[Question]) -> List[EvaluationResult]:
        """运行单个模型的评测（支持并发）"""
        print(f"\n开始测试模型: {model_config.name}")

        # 创建客户端
        client = create_client(model_config)

        results = []
        total = len(questions)

        # 创建jsonl文件用于实时写入
        results_dir = Path(__file__).parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        jsonl_path = results_dir / f"{model_config.name}.jsonl"

        # 使用并发加速
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        import json
        from datetime import datetime

        # 线程锁（用于进度更新和文件写入）
        lock = threading.Lock()

        def write_result_to_jsonl(result):
            """将结果实时写入jsonl文件"""
            try:
                with open(jsonl_path, 'a', encoding='utf-8') as f:
                    json_line = {
                        "timestamp": datetime.now().isoformat(),
                        "question_id": result.question_id,
                        "task_type": result.task_type,
                        "model_answer": result.model_answer,
                        "reference_answer": result.reference_answer,
                        "score": result.score,
                        "is_correct": result.is_correct,
                        "details": result.details,
                        "error": result.error,
                        "cached": getattr(result, 'cached', False)  # 标记是否来自缓存
                    }
                    f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
                    f.flush()
            except Exception as e:
                print(f"  警告: 写入jsonl失败 - {e}")

        def evaluate_with_progress(question):
            result = self._evaluate_single_question(client, question, model_config.name)

            # 实时写入jsonl
            with lock:
                write_result_to_jsonl(result)
                self.progress["completed"] += 1
                if result.error:
                    self.progress["errors"] += 1
            return result

        # 加载已完成的题目ID（断点续跑）
        completed_ids = set()
        if jsonl_path.exists():
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        # 只记录成功的，失败的重跑
                        if not data.get('error'):
                            completed_ids.add(data['question_id'])
                    except:
                        pass
            if completed_ids:
                print(f"  断点续跑: 跳过已完成的 {len(completed_ids)} 题")

        # 过滤未完成的题目
        remaining_questions = [q for q in questions if q.id not in completed_ids]
        failed_count = len([q for q in questions if q.id in completed_ids])  # 这里逻辑有问题，修复
        failed_count = sum(1 for q in questions if q.id not in completed_ids)

        print(f"  待测试: {len(remaining_questions)} 题 (包括重跑失败的)")
        if not remaining_questions:
            print(f"  所有题目已完成!")
            # 返回已有结果
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        results.append(EvaluationResult(
                            question_id=data['question_id'],
                            model_name=model_config.name,
                            task_type=data['task_type'],
                            model_answer=data.get('model_answer', ''),
                            reference_answer=data.get('reference_answer', ''),
                            score=data.get('score', 0),
                            is_correct=data.get('is_correct'),
                            details=data.get('details', {}),
                            error=data.get('error')
                        ))
                    except:
                        pass
            return results

        # 注意：不清空旧文件，追加新结果
        # 最后会通过去重处理

        # 并发执行
        max_workers = self.config.max_concurrent_per_model
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_question = {
                executor.submit(evaluate_with_progress, q): q
                for q in questions
            }

            # 收集结果
            for i, future in enumerate(as_completed(future_to_question)):
                question = future_to_question[future]
                try:
                    result = future.result()
                    results.append(result)

                    # 每10题打印一次进度
                    if len(results) % 10 == 0:
                        elapsed = (datetime.now() - self.progress["start_time"]).total_seconds()
                        rate = self.progress["completed"] / elapsed if elapsed > 0 else 0
                        print(f"  进度: {len(results)}/{total} ({len(results)/total*100:.1f}%) - 速度: {rate:.2f}题/秒")
                except Exception as e:
                    print(f"  题目 {question.id} 执行失败: {e}")
                    error_result = EvaluationResult(
                        question_id=question.id,
                        model_name=model_config.name,
                        task_type=question.task_type,
                        model_answer="",
                        reference_answer=question.reference_answer,
                        score=0,
                        is_correct=False,
                        details={},
                        error=str(e)
                    )
                    # 也写入错误结果
                    with lock:
                        write_result_to_jsonl(error_result)
                    results.append(error_result)

        print(f"  完成: {model_config.name} ({len(results)}/{total})")
        print(f"  结果已保存到: {jsonl_path}")
        return results

    def run_all(self, datasets: List[str] | None = None) -> Dict[str, Any]:
        """运行所有模型的评测（支持模型级并发 + Judge缓存）"""
        # 加载数据
        all_data = self.loader.load_all()

        # 选择要测试的数据集
        if datasets is None:
            datasets = list(all_data.keys())

        # 合并所有问题
        all_questions = []
        for dataset_name in datasets:
            all_questions.extend(all_data[dataset_name])

        self.progress["total"] = len(all_questions) * len(self.config.models)
        self.progress["start_time"] = datetime.now()

        # 打印初始缓存状态
        if self._judge_cache:
            cache_stats = self._judge_cache.get_stats()
            print(f"\nJudge缓存状态:")
            print(f"  现有缓存条目: {cache_stats['total_entries']}")
            print(f"  缓存大小: {cache_stats['cache_size_mb']:.2f} MB")

        print(f"\n开始评测:")
        print(f"  模型数: {len(self.config.models)}")
        print(f"  问题数: {len(all_questions)}")
        print(f"  总评测数: {self.progress['total']}")
        print(f"  模型级并发数: {self.config.max_concurrent_models}")

        # 使用模型级并发
        all_results = {}
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        # 用于线程安全的锁
        results_lock = threading.Lock()

        def test_single_model(model_config):
            """测试单个模型"""
            results = self.run_model(model_config, all_questions)
            with results_lock:
                all_results[model_config.name] = results
                self._save_checkpoint(model_config.name, results)
            return model_config.name, len(results)

        # 并发执行多个模型
        max_model_workers = min(self.config.max_concurrent_models, len(self.config.models))
        print(f"\n同时测试 {max_model_workers} 个模型...\n")

        with ThreadPoolExecutor(max_workers=max_model_workers) as executor:
            # 提交所有模型测试任务
            future_to_model = {
                executor.submit(test_single_model, m): m
                for m in self.config.models
            }

            # 收集结果
            for future in as_completed(future_to_model):
                model_config = future_to_model[future]
                try:
                    model_name, count = future.result()
                    print(f"\n✓ {model_name} 完成 ({count}题)")
                except Exception as e:
                    print(f"\n✗ {model_config.name} 失败: {e}")

        self.progress["end_time"] = datetime.now()
        self.results = all_results

        # 打印Judge缓存统计
        if self._llm_evaluator:
            self._llm_evaluator.print_stats()

        return self._generate_summary()

    def _save_checkpoint(self, model_name: str, results: List[EvaluationResult]):
        """保存检查点"""
        checkpoint_dir = Path(__file__).parent.parent / "results"
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_file = checkpoint_dir / f"{model_name}_checkpoint.json"

        data = [
            {
                "question_id": r.question_id,
                "task_type": r.task_type,
                "model_answer": r.model_answer,
                "reference_answer": r.reference_answer,
                "score": r.score,
                "is_correct": r.is_correct,
                "details": r.details,
                "error": r.error,
                "cached": getattr(r, 'cached', False)
            }
            for r in results
        ]

        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _generate_summary(self) -> Dict[str, Any]:
        """生成评测摘要"""
        elapsed = (
            self.progress["end_time"] - self.progress["start_time"]
        ).total_seconds() if self.progress["end_time"] else 0

        summary = {
            "timestamp": datetime.now().isoformat(),
            "models": list(self.results.keys()),
            "model_summary": {},
            "statistics": {
                "total_questions": self.progress["total"] // len(self.config.models) if self.config.models else 0,
                "total_evaluations": self.progress["total"],
                "completed": self.progress["completed"],
                "errors": self.progress["errors"],
                "elapsed_seconds": elapsed,
                "evaluations_per_second": self.progress["completed"] / elapsed if elapsed > 0 else 0
            }
        }

        # Judge缓存统计
        if self._judge_cache:
            cache_stats = self._judge_cache.get_stats()
            summary["statistics"]["judge_cache"] = {
                "total_entries": cache_stats["total_entries"],
                "total_accesses": cache_stats["total_accesses"],
                "cache_size_mb": cache_stats["cache_size_mb"]
            }

        for model_name, results in self.results.items():
            model_summary = {
                "total": len(results),
                "errors": sum(1 for r in results if r.error),
                "by_task_type": {},
                "overall_accuracy": 0
            }

            # 按题型统计
            task_counts = {}
            task_correct = {}
            task_scores = {}  # 用于开放式题型

            for result in results:
                task_type = result.task_type

                if task_type not in task_counts:
                    task_counts[task_type] = 0
                    task_correct[task_type] = 0
                    task_scores[task_type] = []

                task_counts[task_type] += 1

                if result.is_correct is not None:
                    if result.is_correct:
                        task_correct[task_type] += 1

                if result.score is not None:
                    task_scores[task_type].append(result.score)

            # 计算准确率和平均分
            for task_type in task_counts:
                total = task_counts[task_type]
                correct = task_correct[task_type]
                accuracy = correct / total if total > 0 else 0
                avg_score = sum(task_scores[task_type]) / len(task_scores[task_type]) if task_scores[task_type] else 0

                model_summary["by_task_type"][task_type] = {
                    "total": total,
                    "correct": correct,
                    "accuracy": accuracy,
                    "avg_score": avg_score
                }

            summary["model_summary"][model_name] = model_summary

        return summary

    def get_judge_cache(self) -> JudgeCache | None:
        """获取Judge缓存实例"""
        return self._judge_cache

    def export_judge_cache(self, output_path: str | None = None) -> str | None:
        """导出Judge缓存"""
        if self._judge_cache:
            return self._judge_cache.export_cache(output_path)
        return None

    def clear_judge_cache(self, task_type: str | None = None, older_than_days: int | None = None):
        """清理Judge缓存"""
        if self._judge_cache:
            deleted = self._judge_cache.clear(task_type, older_than_days)
            print(f"已清理 {deleted} 条Judge缓存")
