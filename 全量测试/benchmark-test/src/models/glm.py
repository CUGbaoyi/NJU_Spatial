"""
智谱GLM API客户端 (用于LLM Judge)
"""
import json
import re
from .base import BaseModelClient, ModelConfig


class GLMClient(BaseModelClient):
    """智谱GLM客户端 - 使用官方SDK"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            from zhipuai import ZhipuAI
            self._client = ZhipuAI(api_key=config.api_key)
        except ImportError:
            raise ImportError("请安装 zhipuai: pip install zhipuai")

    def generate(self, prompt: str, system_prompt: str | None = None,
                 max_tokens: int = 4000, temperature: float = 0.7) -> str:
        """生成回答"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.config.model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        # 处理响应
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message'):
                message = choice.message
                # 优先使用 content，如果为空则使用 reasoning_content（思维模型）
                content = getattr(message, 'content', '') or ''
                reasoning = getattr(message, 'reasoning_content', '') or ''

                # 如果 content 为空，返回 reasoning 的最后部分（通常是答案）
                if not content and reasoning:
                    # 从 reasoning 中提取最后的答案
                    lines = reasoning.strip().split('\n')
                    # 取最后一行非空内容
                    for line in reversed(lines):
                        line = line.strip()
                        if line:
                            return line
                    return reasoning

                return content

        return ""

    def generate_json(self, prompt: str, system_prompt: str | None = None,
                      max_tokens: int = 2000, temperature: float = 0.1) -> dict:
        """生成JSON格式回答"""
        if system_prompt is None:
            system_prompt = "请以JSON格式输出回答，不要包含任何其他内容。"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        response = self._client.chat.completions.create(
            model=self.config.model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        # 处理响应
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message'):
                message = choice.message
                content = getattr(message, 'content', '') or ''
                reasoning = getattr(message, 'reasoning_content', '') or ''

                # 使用 content 或 reasoning
                text = content or reasoning

                # 解析JSON
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    json_match = re.search(r'\{.*\}', text, re.DOTALL)
                    if json_match:
                        try:
                            return json.loads(json_match.group())
                        except:
                            pass

        return {}
