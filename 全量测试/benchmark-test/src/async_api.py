"""
异步API客户端 - 支持OpenAI兼容格式和智谱GLM
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIError(Exception):
    """API调用错误"""
    pass


class RateLimitError(APIError):
    """API限流错误"""
    pass


class AsyncAPIClient:
    """异步API客户端基类"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 120.0,
        max_retries: int = 3
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((RateLimitError, httpx.TimeoutException)),
        reraise=True
    )
    async def _request(
        self,
        method: str,
        endpoint: str,
        headers: Dict[str, str],
        json_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """发送HTTP请求（带重试）"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            response = await self._client.request(
                method=method,
                url=url,
                headers=headers,
                json=json_data
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning(f"Rate limit hit: {url}")
                raise RateLimitError(f"Rate limit: {e}")
            raise APIError(f"HTTP {e.response.status_code}: {e.response.text}")
        except httpx.TimeoutException:
            logger.warning(f"Request timeout: {url}")
            raise
        except Exception as e:
            raise APIError(f"Request failed: {e}")

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> str:
        """生成文本 - 子类实现"""
        raise NotImplementedError


class OpenAICompatClient(AsyncAPIClient):
    """OpenAI兼容API客户端（用于Qwen、DeepSeek、Claude等）"""

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self._request(
                method="POST",
                endpoint="chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json_data={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            )
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise APIError(f"Invalid response format: {e}")


class ZhipuClient(AsyncAPIClient):
    """智谱GLM客户端（用于裁判）"""

    # 推理模型列表
    REASONING_MODELS = {"glm-5.1", "glm-5.1-preview"}

    def __init__(self, api_key: str, model: str = "glm-4-plus", **kwargs):
        super().__init__(api_key, base_url="https://open.bigmodel.cn/api/paas/v4", model=model, **kwargs)
        self.is_reasoning = model in self.REASONING_MODELS

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.3,  # 裁判使用较低温度
        json_mode: bool = False
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = await self._request(
                method="POST",
                endpoint="chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json_data=payload
            )
            message = response.get("choices", [{}])[0].get("message", {})
            # 推理模型：内容在 reasoning_content；非推理模型：内容在 content
            if self.is_reasoning:
                content = message.get("reasoning_content") or message.get("content", "")
            else:
                content = message.get("content", "")
            if not content:
                logger.warning(f"GLM返回空内容: {response}")
                return ""
            return content
        except (KeyError, IndexError) as e:
            raise APIError(f"Invalid response format: {e}")


async def generate_batch(
    client: AsyncAPIClient,
    prompts: List[str],
    system_prompt: Optional[str] = None,
    max_tokens: int = 2000,
    temperature: float = 0.7,
    semaphore: Optional[asyncio.Semaphore] = None
) -> List[str]:
    """批量生成（并发控制）"""

    async def generate_with_semaphore(prompt: str, idx: int) -> tuple[int, str]:
        if semaphore:
            async with semaphore:
                result = await client.generate(prompt, system_prompt, max_tokens, temperature)
                return idx, result
        else:
            result = await client.generate(prompt, system_prompt, max_tokens, temperature)
            return idx, result

    tasks = [generate_with_semaphore(p, i) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    output = [None] * len(prompts)
    for r in results:
        if isinstance(r, Exception):
            logger.error(f"Generate error: {r}")
            continue
        idx, text = r
        output[idx] = text
    return output
