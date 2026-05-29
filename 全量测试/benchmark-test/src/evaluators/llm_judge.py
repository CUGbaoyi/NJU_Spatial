"""
LLM Judge评测器 - 使用GLM-5.1评判开放式问题

优化特性:
1. 持久化缓存 (SQLite)
2. 批量评判支持
3. 线程安全
4. 缓存统计
"""
import json
import re
import threading
from typing import Any, Optional, List, Dict, Tuple
from .base import BaseEvaluator, EvaluationResult
from ..cache import JudgeCache


class LLMJudgeEvaluator(BaseEvaluator):
    """LLM辅助评测器（持久化缓存 + 批量处理）"""

    def __init__(
        self,
        judge_client,
        prompt_templates: dict,
        cache: Optional[JudgeCache] = None,
        max_retries: int = 3,
        enable_batch: bool = True,
        batch_size: int = 5
    ):
        """
        Args:
            judge_client: Judge模型客户端
            prompt_templates: 各题型的评判提示词模板
            cache: 持久化缓存实例 (None时自动创建)
            max_retries: 最大重试次数
            enable_batch: 是否启用批量评判
            batch_size: 批量评判大小
        """
        self.judge_client = judge_client
        self.prompt_templates = prompt_templates
        self.max_retries = max_retries
        self.enable_batch = enable_batch
        self.batch_size = batch_size

        # 持久化缓存
        if cache is None:
            self.cache = JudgeCache()
        else:
            self.cache = cache

        # 统计信息
        self.stats = {
            "total_evaluations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0
        }
        self._stats_lock = threading.Lock()

    def _parse_json_response(self, response_text: str) -> dict[str, Any]:
        """解析JSON响应"""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # 尝试提取JSON块
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            # 返回默认评分
            return {"total": 0, "reasoning": "JSON解析失败"}

    def _get_prompt(self, task_type: str, question: str, reference: str, answer: str) -> str:
        """获取评判提示词（安全处理缺失的占位符）"""
        template = self.prompt_templates.get(task_type)
        if not template:
            raise ValueError(f"未找到题型 {task_type} 的评判模板")

        # 构建可用参数字典
        format_kwargs = {
            "question": question,
            "reference": reference,
            "answer": answer
        }

        # 使用 safe_substitute 避免 KeyError：缺失的占位符会被保留或移除
        from string import Template
        # 先将 {var} 转换为 $var 格式用于 Template
        import re
        # 检查模板中是否有 format_kwargs 之外的占位符
        placeholder_pattern = r'\{([^{}]+)\}'
        all_placeholders = set(re.findall(placeholder_pattern, template))
        missing_placeholders = all_placeholders - set(format_kwargs.keys())

        if missing_placeholders:
            # 移除缺失的占位符（替换为空字符串）
            for ph in missing_placeholders:
                template = template.replace('{' + ph + '}', '')

        return template.format(**format_kwargs)

    def _update_stats(self, **kwargs):
        """更新统计信息（线程安全）"""
        with self._stats_lock:
            for key, value in kwargs.items():
                if key in self.stats:
                    self.stats[key] += value

    def evaluate(self, question, model_answer: str, reference_answer: str) -> EvaluationResult:
        """
        使用LLM评测答案（持久化缓存）

        Args:
            question: 问题对象
            model_answer: 模型答案
            reference_answer: 参考答案

        Returns:
            评测结果
        """
        task_type = question.task_type
        self._update_stats(total_evaluations=1)

        # 检查缓存
        cached_result = self.cache.get(
            task_type=task_type,
            question=question.question,
            reference=reference_answer,
            answer=model_answer
        )

        if cached_result is not None:
            self._update_stats(cache_hits=1)
            return EvaluationResult(
                question_id=question.id,
                model_name="",
                task_type=task_type,
                model_answer=model_answer,
                reference_answer=reference_answer,
                score=cached_result.get("total", 0),
                is_correct=None,
                details=cached_result,
                error=None,
                cached=True  # 标记来自缓存
            )

        self._update_stats(cache_misses=1)

        # 获取评判提示词
        judge_prompt = self._get_prompt(
            task_type,
            question.question,
            reference_answer,
            model_answer
        )

        # 调用Judge（重试机制）
        judge_result = None
        for attempt in range(self.max_retries):
            try:
                response = self.judge_client.generate_json(
                    prompt=judge_prompt,
                    system_prompt="你是地理学评测专家。请严格按照要求以JSON格式输出评分结果。",
                    max_tokens=2000,
                    temperature=0.1
                )
                judge_result = response
                break
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self._update_stats(errors=1)
                    return EvaluationResult(
                        question_id=question.id,
                        model_name="",
                        task_type=task_type,
                        model_answer=model_answer,
                        reference_answer=reference_answer,
                        score=0,
                        is_correct=None,
                        details={},
                        error=f"Judge调用失败: {str(e)}"
                    )

        # 解析评分结果
        if not judge_result:
            self._update_stats(errors=1)
            return EvaluationResult(
                question_id=question.id,
                model_name="",
                task_type=task_type,
                model_answer=model_answer,
                reference_answer=reference_answer,
                score=0,
                is_correct=None,
                details={},
                error="Judge返回空结果"
            )

        total_score = judge_result.get("total", 0)

        # 存入缓存
        self.cache.set(
            task_type=task_type,
            question=question.question,
            reference=reference_answer,
            answer=model_answer,
            result=judge_result
        )

        return EvaluationResult(
            question_id=question.id,
            model_name="",
            task_type=task_type,
            model_answer=model_answer,
            reference_answer=reference_answer,
            score=total_score,
            is_correct=None,  # 开放式问题没有二元正确性
            details=judge_result,
            cached=False
        )

    def evaluate_batch(
        self,
        questions: List,
        model_answers: List[str],
        reference_answers: List[str]
    ) -> List[EvaluationResult]:
        """
        批量评测多个答案

        Args:
            questions: 问题对象列表
            model_answers: 模型答案列表
            reference_answers: 参考答案列表

        Returns:
            评测结果列表
        """
        if len(questions) != len(model_answers) or len(questions) != len(reference_answers):
            raise ValueError("questions, model_answers, reference_answers长度必须一致")

        results = []

        # 分批处理
        for i in range(0, len(questions), self.batch_size):
            batch_questions = questions[i:i + self.batch_size]
            batch_answers = model_answers[i:i + self.batch_size]
            batch_references = reference_answers[i:i + self.batch_size]

            # 先检查缓存
            uncached_indices = []
            for j, (q, a, r) in enumerate(zip(batch_questions, batch_answers, batch_references)):
                cached = self.cache.get(q.task_type, q.question, r, a)
                if cached:
                    results.append(EvaluationResult(
                        question_id=q.id,
                        model_name="",
                        task_type=q.task_type,
                        model_answer=a,
                        reference_answer=r,
                        score=cached.get("total", 0),
                        is_correct=None,
                        details=cached,
                        cached=True
                    ))
                    self._update_stats(total_evaluations=1, cache_hits=1)
                else:
                    uncached_indices.append((j, q, a, r))
                    self._update_stats(total_evaluations=1, cache_misses=1)

            # 批量处理未命中的
            if uncached_indices:
                batch_results = self._evaluate_batch_internal(
                    [item[1] for item in uncached_indices],
                    [item[2] for item in uncached_indices],
                    [item[3] for item in uncached_indices]
                )

                # 将结果插入正确位置
                for (idx, q, a, r), result in zip(uncached_indices, batch_results):
                    results.insert(idx, result)

        return results

    def _evaluate_batch_internal(
        self,
        questions: List,
        model_answers: List[str],
        reference_answers: List[str]
    ) -> List[EvaluationResult]:
        """内部批量评测实现"""
        results = []

        for q, a, r in zip(questions, model_answers, reference_answers):
            result = self.evaluate(q, a, r)
            results.append(result)

        return results

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        cache_stats = self.cache.get_stats()
        return {
            **self.stats,
            "cache_hit_rate": (
                self.stats["cache_hits"] / self.stats["total_evaluations"]
                if self.stats["total_evaluations"] > 0 else 0
            ),
            "cache_entries": cache_stats["total_entries"],
            "cache_size_mb": cache_stats["cache_size_mb"]
        }

    def print_stats(self):
        """打印统计信息"""
        stats = self.get_cache_stats()
        print("\n=== Judge评测统计 ===")
        print(f"  总评测数: {stats['total_evaluations']}")
        print(f"  缓存命中: {stats['cache_hits']}")
        print(f"  缓存未命中: {stats['cache_misses']}")
        print(f"  缓存命中率: {stats['cache_hit_rate']:.1%}")
        print(f"  错误数: {stats['errors']}")
        print(f"  缓存条目数: {stats['cache_entries']}")
        print(f"  缓存大小: {stats['cache_size_mb']:.2f} MB")
        print("=====================\n")

    def clear_cache(self, task_type: str | None = None, older_than_days: int | None = None):
        """清理缓存"""
        deleted = self.cache.clear(task_type, older_than_days)
        print(f"已清理 {deleted} 条缓存")

    @property
    def name(self) -> str:
        return "llm_judge"
