"""
评测器基类
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class EvaluationResult:
    """评测结果"""
    question_id: str
    model_name: str
    task_type: str
    model_answer: str
    reference_answer: str
    score: float | int  # 分数或布尔值
    is_correct: bool | None  # 是否正确（选择题/判断题）
    details: dict[str, Any]  # 详细评分信息
    error: str | None = None
    cached: bool = False  # 是否来自缓存（Judge评测）


class BaseEvaluator(ABC):
    """评测器基类"""

    @abstractmethod
    def evaluate(self, question, model_answer: str, reference_answer: str) -> EvaluationResult:
        """评测单个答案"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """评测器名称"""
        pass
