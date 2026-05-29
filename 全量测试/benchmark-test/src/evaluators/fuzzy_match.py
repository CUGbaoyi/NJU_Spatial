"""
模糊匹配评测器 - 用于填空题
"""
import re
import warnings
from .base import BaseEvaluator, EvaluationResult

# 禁用正则表达式警告
warnings.filterwarnings("ignore", category=SyntaxWarning)


class FuzzyMatchEvaluator(BaseEvaluator):
    """模糊匹配评测器（基于关键词）"""

    def __init__(self):
        pass

    def _extract_keywords(self, answer: str | None) -> list[str]:
        """提取关键词（中文和英文单词）"""
        if not answer:
            return []
        # 移除标点符号（使用原始字符串避免转义警告）
        answer = re.sub(r'[，。、；：""''（）\[\]{}(),.;:]', ' ', answer)
        # 分割成词
        words = [w.strip() for w in answer.split() if w.strip()]
        return words

    def _contains_all_keywords(self, model_answer: str | None, reference_keywords: list[str]) -> tuple[bool, list[str]]:
        """检查模型答案是否包含所有关键词"""
        if not model_answer:
            return False, reference_keywords.copy()
        model_lower = model_answer.lower()
        missing = []
        for keyword in reference_keywords:
            if keyword.lower() not in model_lower:
                missing.append(keyword)
        return len(missing) == 0, missing

    def evaluate(self, question, model_answer: str, reference_answer: str) -> EvaluationResult:
        """评测答案"""
        reference_keywords = self._extract_keywords(reference_answer)
        has_all, missing = self._contains_all_keywords(model_answer, reference_keywords)

        # 计算覆盖率
        coverage = len(reference_keywords) - len(missing)
        coverage_ratio = coverage / len(reference_keywords) if reference_keywords else 0

        # 覆盖率>=80%算正确
        is_correct = coverage_ratio >= 0.8

        return EvaluationResult(
            question_id=question.id,
            model_name="",  # 由runner填充
            task_type=question.task_type,
            model_answer=model_answer,
            reference_answer=reference_answer,
            score=coverage_ratio,
            is_correct=is_correct,
            details={
                "reference_keywords": reference_keywords,
                "missing_keywords": missing,
                "coverage": coverage,
                "coverage_ratio": coverage_ratio,
                "method": "keyword_fuzzy_match"
            }
        )

    @property
    def name(self) -> str:
        return "fuzzy_match"
