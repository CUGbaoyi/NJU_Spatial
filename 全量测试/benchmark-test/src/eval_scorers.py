"""
评分器 - 客观题机改 + 主观题GLM裁判
"""
import re
import json
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """评测结果"""
    score: float
    reason: Optional[str] = None
    is_correct: Optional[bool] = None
    raw_answer: Optional[str] = None


class ObjectiveScorer:
    """客观题评分器（机改）"""

    @staticmethod
    def normalize_answer(answer: str) -> str:
        """标准化答案"""
        answer = answer.strip().upper()
        # 移除标点和空格
        answer = re.sub(r'[^\w]', '', answer)
        return answer

    @staticmethod
    def extract_choice(text: str) -> Optional[str]:
        """提取选择题答案（首字母）"""
        text = text.strip()
        # 匹配开头的字母 A-Z
        match = re.match(r'^\s*([A-Z])', text.upper())
        if match:
            return match.group(1)
        # 尝试匹配 "答案是A" 格式
        match = re.search(r'[答对]?案?[是为][:：]?\s*([A-Z])', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return None

    @staticmethod
    def extract_tf(text: str) -> Optional[str]:
        """提取判断题答案"""
        text = text.strip().upper()
        # 直接匹配 T/F
        if text.startswith('T') or 'TRUE' in text or '正确' in text or '对' in text:
            return 'T'
        if text.startswith('F') or 'FALSE' in text or '错误' in text or '错' in text:
            return 'F'
        return None

    @classmethod
    def score_choice(cls, model_answer: str, reference_answer: str) -> EvalResult:
        """评分选择题"""
        extracted = cls.extract_choice(model_answer)
        if not extracted:
            return EvalResult(score=0, reason="无法提取选项", raw_answer=model_answer[:50])
        reference = cls.normalize_answer(reference_answer)
        is_correct = extracted == reference
        return EvalResult(
            score=2 if is_correct else 0,
            reason=f"模型回答: {extracted}, 参考答案: {reference}",
            is_correct=is_correct,
            raw_answer=model_answer[:50]
        )

    @classmethod
    def score_tf(cls, model_answer: str, reference_answer: str) -> EvalResult:
        """评分判断题"""
        extracted = cls.extract_tf(model_answer)
        if not extracted:
            return EvalResult(score=0, reason="无法提取T/F", raw_answer=model_answer[:50])
        reference = 'T' if cls.normalize_answer(reference_answer) in ['T', 'TRUE', '正确'] else 'F'
        is_correct = extracted == reference
        return EvalResult(
            score=2 if is_correct else 0,
            reason=f"模型回答: {extracted}, 参考答案: {reference}",
            is_correct=is_correct,
            raw_answer=model_answer[:50]
        )


class SubjectiveJudge:
    """主观题裁判（GLM）"""

    # 裁判系统提示词
    SYSTEM_PROMPT = """你是一位严谨的地球科学教授。请按照以下标准评判学生答案：

1. 科学性：原理是否有明显错误
2. 术语精度：专业名词使用是否准确（填空题允许规范同义词）
3. 完整性：是否涵盖核心要点

请给出客观评分，不要因为答案简短而扣分，只要核心内容正确即给满分。

重要：只返回JSON格式，不要返回任何其他解释或推理过程。"""

    TASK_PROMPTS = {
        "completion": """这是一道填空题。请评判以下答案：

【题目】
{question}

【标准答案】
{reference}

【学生答案】
{model_answer}

填空题允许使用规范的专业同义词。如果答案核心概念正确但表述不同，应给满分。

请以JSON格式返回评分：
{{
  "score": 0-5的整数,
  "reason": "简要评语（指出优缺点或扣分原因）"
}}""",

        "noun": """这是一道名词解释题。请评判以下答案：

【术语】
{question}

【标准答案】
{reference}

【学生答案】
{model_answer}

评分标准（0-6分）：
- 5-6分：定义准确，核心要素完整
- 3-4分：定义基本正确，但有遗漏
- 1-2分：部分正确
- 0分：错误或答非所问

请以JSON格式返回评分：
{{
  "score": 0-6的整数,
  "reason": "简要评语"
}}""",

        "qa": """这是一道简答题。请评判以下答案：

【题目】
{question}

【标准答案】
{reference}

【学生答案】
{model_answer}

评分标准（0-10分）：
- 9-10分：准确、完整、有条理
- 7-8分：基本正确，略有遗漏
- 5-6分：部分正确
- 0-4分：错误或严重不完整

请以JSON格式返回评分：
{{
  "score": 0-10的整数,
  "reason": "简要评语"
}}""",

        "discussion": """这是一道综合论述题。请评判以下答案：

【主题】
{question}

【标准答案】
{reference}

【学生答案】
{model_answer}

评分标准（0-10分）：
- 9-10分：深入、全面、逻辑清晰
- 7-8分：内容基本正确，但深度或完整性有欠缺
- 5-6分：部分正确，论述不够充分
- 0-4分：错误或严重偏离主题

请以JSON格式返回评分：
{{
  "score": 0-10的整数,
  "reason": "简要评语"
}}"""
    }

    SCORE_RANGES = {
        "completion": (0, 5),
        "noun": (0, 6),
        "qa": (0, 10),
        "discussion": (0, 10)
    }

    @classmethod
    def build_prompt(cls, task_type: str, question: str, reference: str, model_answer: str) -> str:
        """构建裁判提示词"""
        template = cls.TASK_PROMPTS.get(task_type)
        if not template:
            template = """请评判以下答案：

【题目】{question}
【标准答案】{reference}
【学生答案】{model_answer}

请以JSON格式返回评分：{{"score": 整数, "reason": "评语"}}"""
        return template.format(
            question=question,
            reference=reference,
            model_answer=model_answer
        )

    @classmethod
    def parse_response(cls, response: str, task_type: str) -> EvalResult:
        """解析GLM响应（支持推理模型输出）"""
        # 处理空响应
        if not response or not response.strip():
            logger.warning("GLM返回空响应")
            return EvalResult(score=0, reason="GLM返回空")

        response = response.strip()

        # 优先尝试提取最后的JSON块（推理模型的最终结论通常在最后）
        # 支持中文key和多种格式
        json_patterns = [
            r'\{[\s\S]*?"score"[\s\S]*?\}',           # 标准 "score"
            r'\{[\s\S]*?"分数"[\s\S]*?\}',            # 中文 "分数"
            r'\{[\s\S]*?score[\s\S]*?\}',            # 无引号score
            r'["\']?\{["\']?score["\']?\s*:\s*\d+',  # 最简单的{"score":数字}
            r'["\']?\{["\']?分数["\']?\s*[:：]\s*\d+', # 中文分数
        ]

        for pattern in json_patterns:
            matches = list(re.finditer(pattern, response))
            if matches:
                # 取最后一个匹配（通常是最终结论）
                last_match = matches[-1]
                try:
                    data = json.loads(last_match.group())
                    score = int(data.get("score", 0))
                    reason = data.get("reason", "")
                    # 验证分数范围
                    min_score, max_score = cls.SCORE_RANGES.get(task_type, (0, 10))
                    score = max(min_score, min(max_score, score))
                    return EvalResult(score=score, reason=reason)
                except json.JSONDecodeError:
                    continue

        # 如果没有找到JSON，尝试正则提取分数
        score_patterns = [
            r'(?:确定分数|最终评分|分数|给.*?分)[:：]\s*(\d+)分',
            r'["\']?score["\']?\s*[:：]\s*(\d+)',
            r'给\s*(\d+)\s*分',
        ]

        for pattern in score_patterns:
            score_match = re.search(pattern, response, re.IGNORECASE)
            if score_match:
                score = int(score_match.group(1))
                # 提取附近的原因文本
                reason_match = re.search(r'(?:原因|理由|评语|评述)[:：](.{10,200})', response)
                reason = reason_match.group(1).strip() if reason_match else response[:100]
                # 验证分数范围
                min_score, max_score = cls.SCORE_RANGES.get(task_type, (0, 10))
                score = max(min_score, min(max_score, score))
                return EvalResult(score=score, reason=reason)

        # 完全无法解析，输出部分响应用于调试
        logger.warning(f"无法解析裁判响应，响应长度: {len(response)}")
        # 保存原始响应用于调试
        debug_file = Path("results") / "failed_response_debug.txt"
        debug_file.parent.mkdir(exist_ok=True)
        with open(debug_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\nTask: {task_type}\n{response}\n{'='*60}\n")
        logger.info(f"原始响应已保存到: {debug_file}")

        # 尝试从响应中提取任何数字作为分数
        all_numbers = re.findall(r'\d+', response)
        if all_numbers:
            # 取最后一个较大的数字（可能是分数）
            for num in reversed(all_numbers):
                score = int(num)
                min_score, max_score = cls.SCORE_RANGES.get(task_type, (0, 10))
                if min_score <= score <= max_score:
                    return EvalResult(score=score, reason="从推理中提取")
        return EvalResult(score=0, reason="无法解析")

        return EvalResult(score=score, reason=reason)

    @classmethod
    async def judge(
        cls,
        client,
        task_type: str,
        question: str,
        reference: str,
        model_answer: str
    ) -> EvalResult:
        """调用裁判模型"""
        prompt = cls.build_prompt(task_type, question, reference, model_answer)
        try:
            response = await client.generate(
                prompt=prompt,
                system_prompt=cls.SYSTEM_PROMPT,
                max_tokens=500,
                temperature=0.3
            )
            return cls.parse_response(response, task_type)
        except Exception as e:
            logger.error(f"裁判失败: {e}")
            return EvalResult(score=0, reason=f"裁判失败: {str(e)}")


async def score_question(
    task_type: str,
    question: str,
    reference: str,
    model_answer: str,
    judge_client=None
) -> EvalResult:
    """统一评分入口"""
    if task_type == "choice":
        return ObjectiveScorer.score_choice(model_answer, reference)
    elif task_type == "tf":
        return ObjectiveScorer.score_tf(model_answer, reference)
    elif task_type in ["completion", "noun", "qa", "discussion"]:
        if judge_client is None:
            return EvalResult(score=0, reason="缺少裁判客户端")
        return await SubjectiveJudge.judge(judge_client, task_type, question, reference, model_answer)
    else:
        return EvalResult(score=0, reason=f"未知题型: {task_type}")
