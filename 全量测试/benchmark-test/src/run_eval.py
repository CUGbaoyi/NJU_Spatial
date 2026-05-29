"""
GeoEval-Benchmark 主入口
异步评测框架 - 支持断点续传和高并发
"""
import asyncio
import os
import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from typing import Dict, List, Any

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import GeoBenchmarkLoader, Question
from async_api import OpenAICompatClient, ZhipuClient
from checkpoint import CheckpointManager
from eval_scorers import score_question
from eval_viz import ResultVisualizer

# 加载环境变量
load_dotenv()


class EvalConfig:
    """评测配置"""

    # 获取脚本所在目录
    SCRIPT_DIR = Path(__file__).parent.parent

    # 被测模型配置
    MODELS = [
        {"name": "qwen3.5-27b", "api_key": os.getenv("QWEN_API_KEY", ""),
         "base_url": "http://newapi.geos3ai.com/v1", "model_id": "qwen/qwen3.5-27b"},
        {"name": "qwen3.6-35b-a3b", "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
         "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model_id": "qwen-plus-latest"},
        {"name": "kimi-k2.5", "api_key": os.getenv("NEWAPI_KEY", ""),
         "base_url": "http://newapi.geos3ai.com/v1", "model_id": "moonshotai/kimi-k2.5"},
        {"name": "qwen3.5-397b-a17b", "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
         "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model_id": "qwen-plus-latest"},
        {"name": "deepseek-v3.2", "api_key": os.getenv("NEWAPI_KEY", ""),
         "base_url": "http://newapi.geos3ai.com/v1", "model_id": "deepseek/deepseek-v3.2"},
        {"name": "qwen3-32b", "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
         "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model_id": "qwen-plus"},
        {"name": "claude-opus-4.6", "api_key": os.getenv("NEWAPI_KEY", ""),
         "base_url": "http://newapi.geos3ai.com/v1", "model_id": "anthropic/claude-opus-4.6"},
        {"name": "qwen3.6-plus", "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
         "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model_id": "qwen-plus"},
        {"name": "gemini-3-pro-preview", "api_key": os.getenv("NEWAPI_KEY", ""),
         "base_url": "http://newapi.geos3ai.com/v1", "model_id": "google/gemini-3.1-flash-lite-preview"},
        {"name": "qwen3.5-plus", "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
         "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model_id": "qwen-plus"},
    ]

    # 裁判配置 (qwen-max - 通义千问旗舰，参数最大)
    JUDGE_CONFIG = {
        "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
        "model": "qwen-max"
    }

    # 并发配置
    MAX_CONCURRENT_PER_MODEL = 5  # 每个模型的最大并发数
    MAX_CONCURRENT_MODELS = 3      # 同时测试的模型数

    # 提示词配置
    SYSTEM_PROMPT = "你是一个地理学专家。请根据题目要求准确回答问题。"

    # 题型提示词模板
    TASK_PROMPTS = {
        "choice": "请回答以下地理选择题，只输出选项字母（如A、B、C、D），不要输出任何其他内容。\n\n题目：{question}\n\n你的答案是：",
        "tf": "请判断以下地理陈述的对错，只输出'True'或'False'，不要输出任何其他内容。\n\n题目：{question}\n\n你的答案是：",
        "completion": "请回答以下地理填空题，简洁准确地给出答案。\n\n题目：{question}\n\n答案：",
        "noun": "请解释以下地理学术语，要求准确、简洁。\n\n术语：{question}\n\n解释：",
        "qa": "请回答以下地理问题，要求准确、有条理。\n\n题目：{question}\n\n答案：",
        "discussion": "请深入讨论以下地理话题，要求内容准确、逻辑清晰。\n\n主题：{question}\n\n答案：",
    }


class EvalRunner:
    """评测运行器"""

    def __init__(self, config: EvalConfig):
        self.config = config
        # 使用绝对路径
        results_dir = Path(__file__).parent.parent / "results"
        self.checkpoint = CheckpointManager(str(results_dir / "eval_checkpoint.json"))
        # 获取脚本所在目录的data目录
        script_dir = Path(__file__).parent.parent
        data_dir = script_dir / "data"
        self.loader = GeoBenchmarkLoader(str(data_dir))
        self.judge_client: OpenAICompatClient = None

    def _format_prompt(self, question: Question) -> str:
        """格式化提示词"""
        template = self.config.TASK_PROMPTS.get(question.task_type, "{question}")
        return template.format(question=question.question)

    async def _evaluate_single(
        self,
        client: OpenAICompatClient,
        question: Question,
        model_name: str,
        semaphore: asyncio.Semaphore,
        pbar: tqdm
    ) -> Dict[str, Any]:
        """评测单个问题"""
        async with semaphore:
            # 检查是否已完成
            if self.checkpoint.is_completed(model_name, question.id):
                state = self.checkpoint.get_state(model_name, question.id)
                pbar.update(1)
                return {
                    "question_id": question.id,
                    "task_type": question.task_type,
                    "question": question.question,
                    "reference_answer": question.reference_answer,
                    "model_answer": state.model_answer,
                    "score": state.score,
                    "judge_reason": state.judge_reason,
                    "cached": True
                }

            # 调用模型生成答案
            try:
                prompt = self._format_prompt(question)
                model_answer = await client.generate(
                    prompt=prompt,
                    system_prompt=self.config.SYSTEM_PROMPT,
                    max_tokens=2000,
                    temperature=0.7
                )

                # 评分
                result = await score_question(
                    task_type=question.task_type,
                    question=question.question,
                    reference=question.reference_answer,
                    model_answer=model_answer,
                    judge_client=self.judge_client
                )

                # 保存到断点
                self.checkpoint.mark_completed(
                    question_id=question.id,
                    model_name=model_name,
                    score=result.score,
                    model_answer=model_answer,
                    judge_reason=result.reason
                )

                pbar.update(1)
                return {
                    "question_id": question.id,
                    "task_type": question.task_type,
                    "question": question.question,
                    "reference_answer": question.reference_answer,
                    "model_answer": model_answer,
                    "score": result.score,
                    "judge_reason": result.reason,
                    "cached": False
                }

            except Exception as e:
                self.checkpoint.mark_failed(question.id, model_name, str(e))
                pbar.update(1)
                return {
                    "question_id": question.id,
                    "task_type": question.task_type,
                    "question": question.question,
                    "reference_answer": question.reference_answer,
                    "model_answer": None,
                    "score": 0,
                    "judge_reason": None,
                    "error": str(e)
                }

    async def _evaluate_model(
        self,
        model_config: Dict[str, str],
        questions: List[Question],
        semaphore: asyncio.Semaphore
    ) -> List[Dict[str, Any]]:
        """评测单个模型"""
        model_name = model_config["name"]

        # 过滤已完成的问题
        completed = self.checkpoint.get_completed(model_name)
        pending = [q for q in questions if q.id not in completed]

        if not pending:
            print(f"  {model_name}: 已全部完成，跳过")
            return self.checkpoint.get_all_results(model_name)

        print(f"  {model_name}: 测试 {len(pending)} 题（已完成 {len(completed)} 题）")

        async with OpenAICompatClient(
            api_key=model_config["api_key"],
            base_url=model_config["base_url"],
            model=model_config["model_id"]
        ) as client:
            # 创建进度条
            pbar = tqdm(total=len(pending), desc=f"  {model_name}", leave=False)

            # 创建任务
            tasks = [
                self._evaluate_single(client, q, model_name, semaphore, pbar)
                for q in pending
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)
            pbar.close()

        # 合并已完成的结果
        all_results = []
        for q in questions:
            state = self.checkpoint.get_state(model_name, q.id)
            if state and state.status == "completed":
                all_results.append({
                    "question_id": q.id,
                    "task_type": q.task_type,
                    "question": q.question,
                    "reference_answer": q.reference_answer,
                    "model_answer": state.model_answer,
                    "score": state.score,
                    "judge_reason": state.judge_reason
                })
        return all_results

    async def run(self):
        """运行评测"""
        print("=" * 60)
        print("GeoEval-Benchmark 评测框架")
        print("=" * 60)

        # 加载数据
        print("\n正在加载数据...")
        all_data = self.loader.load_npee()
        questions = []
        for task_type, qs in all_data.items():
            questions.extend(qs)
        print(f"  NPEE 数据集: {len(questions)} 题")

        # 过滤有效模型
        valid_models = [m for m in self.config.MODELS if m["api_key"]]
        print(f"\n待测模型: {len(valid_models)} 个")
        if not valid_models:
            print("错误: 没有配置有效的 API Key")
            return

        # 初始化裁判
        if not self.config.JUDGE_CONFIG["api_key"]:
            print("警告: 未配置裁判 API Key，主观题将无法评分")
        else:
            # qwen-max 使用 OpenAI 兼容接口
            self.judge_client = OpenAICompatClient(
                api_key=self.config.JUDGE_CONFIG["api_key"],
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                model=self.config.JUDGE_CONFIG["model"]
            )
            await self.judge_client.__aenter__()

        # 信号量控制并发
        semaphore = asyncio.Semaphore(self.config.MAX_CONCURRENT_PER_MODEL)

        # 评测所有模型
        all_results = {}
        for i, model_config in enumerate(valid_models):
            if i > 0 and i % self.config.MAX_CONCURRENT_MODELS == 0:
                await asyncio.sleep(2)  # 短暂延迟

            results = await self._evaluate_model(model_config, questions, semaphore)
            all_results[model_config["name"]] = results

        # 关闭裁判
        if self.judge_client:
            await self.judge_client.__aexit__(None, None, None)

        # 生成报告
        print("\n" + "=" * 60)
        print("生成报告...")

        # 汇总所有模型的结果
        for model_name, results in all_results.items():
            print(f"\n【{model_name}】")
            visualizer = ResultVisualizer(results, output_dir=f"reports/{model_name}")
            visualizer.generate_all()

            # 导出 CSV
            self.checkpoint.export_to_csv(
                model_name,
                {q.id: {"task_type": q.task_type, "question": q.question,
                       "reference_answer": q.reference_answer} for q in questions},
                f"results/{model_name}_report.csv"
            )

        print("\n" + "=" * 60)
        print("评测完成！")
        print("=" * 60)


async def main():
    """主函数"""
    config = EvalConfig()
    runner = EvalRunner(config)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
