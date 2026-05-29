"""
精确匹配评测器 - 用于选择题和判断题
"""
import re
from .base import BaseEvaluator, EvaluationResult


class ExactMatchEvaluator(BaseEvaluator):
    """精确匹配评测器"""

    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize: 是否标准化答案（去除多余空格、统一大小写）
        """
        self.normalize = normalize

    def _normalize(self, answer: str | None) -> str:
        """标准化答案"""
        if not answer:
            return ""

        if not self.normalize:
            return answer.strip()

        # 去除多余空格和换行
        answer = re.sub(r'\s+', ' ', answer.strip())

        # 统一大小写（对于T/F）
        if answer.upper() in ['TRUE', 'FALSE', 'T', 'F']:
            return answer.upper()

        # 对于选择题，保持大写字母
        if len(answer) == 1 and answer.isalpha():
            return answer.upper()

        return answer

    def evaluate(self, question, model_answer: str, reference_answer: str) -> EvaluationResult:
        """评测答案"""
        normalized_model = self._normalize(model_answer)
        normalized_reference = self._normalize(reference_answer)

        is_correct = normalized_model == normalized_reference

        return EvaluationResult(
            question_id=question.id,
            model_name="",  # 由runner填充
            task_type=question.task_type,
            model_answer=model_answer,
            reference_answer=reference_answer,
            score=1.0 if is_correct else 0.0,
            is_correct=is_correct,
            details={
                "normalized_model": normalized_model,
                "normalized_reference": normalized_reference,
                "method": "exact_match"
            }
        )

    @property
    def name(self) -> str:
        return "exact_match"
