"""
Anthropic Claude API客户端
"""
import json
from typing import Dict, Any
from .base import BaseModelClient, ModelConfig


class AnthropicClient(BaseModelClient):
    """Anthropic Claude客户端"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=config.api_key)
        except ImportError:
            raise ImportError("请安装 anthropic: pip install anthropic")

    def generate(self, prompt: str, system_prompt: str | None = None,
                 max_tokens: int = 4000, temperature: float = 0.7) -> str:
        """生成回答"""
        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": self.config.model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = self._client.messages.create(**kwargs)
        return response.content[0].text

    def generate_json(self, prompt: str, system_prompt: str | None = None,
                      max_tokens: int = 2000, temperature: float = 0.1) -> Dict[str, Any]:
        """生成JSON格式回答"""
        if system_prompt is None:
            system_prompt = "请以JSON格式输出回答，不要包含任何其他内容。"

        response_text = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

        # 尝试解析JSON
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # 尝试提取JSON块
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"无法解析JSON响应: {response_text}")
