#!/usr/bin/env python3
"""
测试指定模型: kimi-k2.5, deepseek-v3.2, qwen3.5-plus, qwen3.6-plus, qwen3.5-27b
"""
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 添加路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import GeoBenchmarkLoader, Question
from async_api import OpenAICompatClient
from checkpoint import CheckpointManager
from eval_scorers import score_question
from tqdm import tqdm
from typing import Dict, List, Any

# 加载环境变量
from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

# 只测试这5个模型
SELECTED_MODELS = [
    {"name": "kimi-k2.5", "api_key": os.getenv("NEWAPI_KEY", ""),
     "base_url": "http://newapi.geos3ai.com/v1", "model_id": "moonshotai/kimi-k2.5"},
    {"name": "deepseek-v3.2", "api_key": os.getenv("NEWAPI_KEY", ""),
     "base_url": "http://newapi.geos3ai.com/v1", "model_id": "deepseek/deepseek-v3.2"},
    {"name": "qwen3.5-plus", "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
     "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model_id": "qwen-plus"},
    {"name": "qwen3.6-plus", "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
     "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model_id": "qwen-plus"},
    {"name": "qwen3.5-27b", "api_key": os.getenv("NEWAPI_KEY", ""),
     "base_url": "http://newapi.geos3ai.com/v1", "model_id": "qwen/qwen3.5-27b"},
]

JUDGE_CONFIG = {
    "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
    "model": "qwen-max"
}

SYSTEM_PROMPT = "你是一个地理学专家。请根据题目要求准确回答问题。"

TASK_PROMPTS = {
    "choice": "请回答以下地理选择题，只输出选项字母（如A、B、C、D），不要输出任何其他内容。\n\n题目：{question}\n\n你的答案是：",
    "tf": "请判断以下地理陈述的对错，只输出'True'或'False'，不要输出任何其他内容。\n\n题目：{question}\n\n你的答案是：",
    "completion": "请回答以下地理填空题，简洁准确地给出答案。\n\n题目：{question}\n\n答案：",
    "noun": "请解释以下地理学术语，要求准确、简洁。\n\n术语：{question}\n\n解释：",
    "qa": "请回答以下地理问题，要求准确、有条理。\n\n题目：{question}\n\n答案：",
    "discussion": "请深入讨论以下地理话题，要求内容准确、逻辑清晰。\n\n主题：{question}\n\n答案：",
}


class EvalRunner:
    def __init__(self):
        results_dir = Path(__file__).parent / "results"
        self.checkpoint = CheckpointManager(str(results_dir / "eval_checkpoint.json"))
        data_dir = Path(__file__).parent / "data"
        self.loader = GeoBenchmarkLoader(str(data_dir))
        self.judge_client = None

    def _format_prompt(self, question: Question) -> str:
        template = TASK_PROMPTS.get(question.task_type, "{question}")
        return template.format(question=question.question)

    async def _evaluate_single(self, client, question: Question, model_name: str, semaphore, pbar) -> Dict[str, Any]:
        async with semaphore:
            if self.checkpoint.is_completed(model_name, question.id):
                state = self.checkpoint.get_state(model_name, question.id)
                pbar.update(1)
                return {"question_id": question.id, "task_type": question.task_type,
                        "question": question.question, "reference_answer": question.reference_answer,
                        "model_answer": state.model_answer, "score": state.score,
                        "judge_reason": state.judge_reason, "cached": True}

            try:
                prompt = self._format_prompt(question)
                model_answer = await client.generate(prompt, SYSTEM_PROMPT, max_tokens=2000, temperature=0.7)
                result = await score_question(question.task_type, question.question, question.reference_answer, model_answer, self.judge_client)

                self.checkpoint.mark_completed(question.id, model_name, result.score, model_answer, result.reason)
                pbar.update(1)
                return {"question_id": question.id, "task_type": question.task_type,
                        "question": question.question, "reference_answer": question.reference_answer,
                        "model_answer": model_answer, "score": result.score, "judge_reason": result.reason, "cached": False}
            except Exception as e:
                self.checkpoint.mark_failed(question.id, model_name, str(e))
                pbar.update(1)
                return {"question_id": question.id, "task_type": question.task_type, "error": str(e)}

    async def _evaluate_model(self, model_config: Dict, questions: List[Question], semaphore):
        model_name = model_config["name"]
        completed = self.checkpoint.get_completed(model_name)
        pending = [q for q in questions if q.id not in completed]

        if not pending:
            print(f"  {model_name}: 已完成，跳过")
            return []

        print(f"  {model_name}: 测试 {len(pending)} 题 (已完成 {len(completed)})")

        async with OpenAICompatClient(api_key=model_config["api_key"], base_url=model_config["base_url"], model=model_config["model_id"]) as client:
            pbar = tqdm(total=len(pending), desc=f"  {model_name}", leave=False)
            tasks = [self._evaluate_single(client, q, model_name, semaphore, pbar) for q in pending]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            pbar.close()
        return results

    async def run(self):
        print("=" * 60)
        print("测试指定模型: kimi-k2.5, deepseek-v3.2, qwen3.5-plus, qwen3.6-plus, qwen3.5-27b")
        print("=" * 60)

        print("\n正在加载数据...")
        all_data = self.loader.load_npee()
        questions = []
        for task_type, qs in all_data.items():
            questions.extend(qs)
        print(f"  NPEE 数据集: {len(questions)} 题")

        valid_models = [m for m in SELECTED_MODELS if m["api_key"]]
        print(f"\n待测模型: {[m['name'] for m in valid_models]}")

        if not valid_models:
            print("错误: API Key 未配置")
            return

        if JUDGE_CONFIG["api_key"]:
            self.judge_client = OpenAICompatClient(JUDGE_CONFIG["api_key"], "https://dashscope.aliyuncs.com/compatible-mode/v1", JUDGE_CONFIG["model"])
            await self.judge_client.__aenter__()
        else:
            print("警告: 裁判 API Key 未配置")

        semaphore = asyncio.Semaphore(5)

        for model_config in valid_models:
            await self._evaluate_model(model_config, questions, semaphore)

        if self.judge_client:
            await self.judge_client.__aexit__(None, None, None)

        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)


async def main():
    runner = EvalRunner()
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
