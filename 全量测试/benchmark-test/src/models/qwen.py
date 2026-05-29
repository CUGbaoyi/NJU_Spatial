"""
Qwen API客户端 (使用OpenAI兼容接口)
"""
from .openai_compat import OpenAICompatClient
from .base import ModelConfig


class QwenClient(OpenAICompatClient):
    """Qwen客户端"""
    pass
