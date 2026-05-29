"""
模型API基类和工厂
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from dataclasses import dataclass
import os


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    provider: str
    api_base: str
    model_id: str
    api_key: Optional[str] = None

    def __post_init__(self):
        if self.api_key is None:
            # 从环境变量获取API key
            env_key_map = {
                "anthropic": "ANTHROPIC_API_KEY",
                "google": "GOOGLE_API_KEY",
                "qwen": "DASHSCOPE_API_KEY",
                "moonshot": "MOONSHOT_API_KEY",
                "deepseek": "DEEPSEEK_API_KEY",
                "zhipu": "ZHIPUAI_API_KEY",
                "custom": "NEWAPI_KEY",  # newapi.geos3ai.com网关
            }
            env_key = env_key_map.get(self.provider)
            if env_key:
                self.api_key = os.environ.get(env_key)
            else:
                # 尝试用provider大写作为key
                self.api_key = os.environ.get(f"{self.provider.upper()}_API_KEY")


class BaseModelClient(ABC):
    """模型客户端基类"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._client = None

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None,
                 max_tokens: int = 4000, temperature: float = 0.7) -> str:
        """生成回答"""
        pass

    @abstractmethod
    def generate_json(self, prompt: str, system_prompt: str | None = None,
                      max_tokens: int = 2000, temperature: float = 0.1) -> Dict[str, Any]:
        """生成JSON格式回答"""
        pass

    @property
    def name(self) -> str:
        return self.config.name


def create_client(config: ModelConfig) -> BaseModelClient:
    """工厂方法：根据provider创建客户端"""
    provider_map = {
        "anthropic": ("anthropic", "AnthropicClient"),
        "google": ("google", "GoogleClient"),
        "qwen": ("qwen", "QwenClient"),
        "moonshot": ("moonshot", "MoonshotClient"),
        "deepseek": ("deepseek", "DeepSeekClient"),
        "zhipu": ("glm", "GLMClient"),
        "custom": ("custom", "CustomClient"),
    }

    module_info = provider_map.get(config.provider)
    if not module_info:
        raise ValueError(f"Unknown provider: {config.provider}")

    module_name, class_name = module_info
    # 动态导入
    import importlib
    module = importlib.import_module(f"src.models.{module_name}")
    client_class = getattr(module, class_name)
    return client_class(config)
