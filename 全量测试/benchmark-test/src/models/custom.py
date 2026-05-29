"""
自定义API网关客户端 (用于newapi.geos3ai.com等OpenAI兼容网关)
"""
from .openai_compat import OpenAICompatClient
from .base import ModelConfig


class CustomClient(OpenAICompatClient):
    """自定义API网关客户端"""
    pass
