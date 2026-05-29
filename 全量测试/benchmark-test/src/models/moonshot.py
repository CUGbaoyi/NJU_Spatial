"""
Moonshot API客户端
"""
from .openai_compat import OpenAICompatClient
from .base import ModelConfig


class MoonshotClient(OpenAICompatClient):
    """Moonshot客户端"""
    pass
