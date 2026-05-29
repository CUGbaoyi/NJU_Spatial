"""
OpenAI兼容API客户端（用于Qwen、Moonshot、DeepSeek等）
"""
import json
import httpx
from typing import Dict, Any
from .base import BaseModelClient, ModelConfig


class OpenAICompatClient(BaseModelClient):
    """OpenAI兼容API客户端"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            from openai import OpenAI
            import httpx
            # 增加超时时间
            self._client = OpenAI(
                api_key=config.api_key,
                base_url=config.api_base,
                timeout=600.0,  # 10分钟超时
                http_client=httpx.Client(
                    proxy=None,
                    timeout=httpx.Timeout(600.0, connect=60.0)
                )
            )
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

    def generate(self, prompt: str, system_prompt: str | None = None,
                 max_tokens: int = 4000, temperature: float = 0.7) -> str:
        """生成回答"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self.config.model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )

            # 打印响应类型用于调试
            if not hasattr(response, 'choices'):
                raise ValueError(f"API返回响应类型错误: {type(response)}, 无choices属性")

            # 检查响应是否有效
            if not response.choices or len(response.choices) == 0:
                raise ValueError("API返回空响应 (no choices)")

            choice = response.choices[0]
            if not hasattr(choice, 'message'):
                raise ValueError(f"choice无message属性, choice类型: {type(choice)}")

            if not choice.message:
                raise ValueError("API返回空响应 (no message)")

            if not hasattr(choice.message, 'content'):
                raise ValueError(f"message无content属性, message类型: {type(choice.message)}")

            if choice.message.content is None:
                raise ValueError("API返回空内容 (no content)")

            return choice.message.content

        except (ValueError, KeyError, IndexError, AttributeError) as e:
            raise ValueError(f"API响应格式错误: {str(e)}")
        except Exception as e:
            raise type(e)(f"API调用失败: {str(e)}")

    def generate_json(self, prompt: str, system_prompt: str | None = None,
                      max_tokens: int = 2000, temperature: float = 0.1) -> Dict[str, Any]:
        """生成JSON格式回答"""
        if system_prompt is None:
            system_prompt = "请以JSON格式输出回答，不要包含任何其他内容。"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self._client.chat.completions.create(
                model=self.config.model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )

            # 检查响应是否有效
            if not response.choices:
                raise ValueError("API返回空响应 (no choices)")

            choice = response.choices[0]
            if not hasattr(choice, 'message') or not choice.message:
                raise ValueError("API返回空响应 (no message)")

            if not hasattr(choice.message, 'content') or choice.message.content is None:
                raise ValueError("API返回空内容 (no content)")

            content = choice.message.content
            return json.loads(content)

        except (ValueError, KeyError, IndexError, AttributeError) as e:
            raise ValueError(f"API响应格式错误: {str(e)}")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON解析失败: {str(e)}")
        except Exception as e:
            raise
