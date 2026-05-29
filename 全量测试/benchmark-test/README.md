# GeoBenchmark 测试框架

地理知识评测框架，支持多模型API调用和LLM辅助评判。

## 项目结构

```
benchmark-test/
├── config/
│   ├── models.yaml          # 待测模型配置
│   ├── judge.yaml           # Judge模型配置
│   └── prompts.yaml         # 提示词模板
├── src/
│   ├── models/              # 模型API客户端
│   ├── evaluators/          # 评测器
│   ├── data_loader.py       # 数据加载
│   ├── runner.py            # 测试控制器
│   └── reporter.py          # 报告生成
├── data/                    # 数据集(软链接)
├── results/                 # 测试结果
├── reports/                 # 评测报告
├── run.py                   # 主入口
└── requirements.txt         # 依赖
```

## 安装

```bash
cd benchmark-test
pip install -r requirements.txt
```

## 配置

1. 复制环境变量模板:
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，填入各API密钥

## 运行评测

```bash
python run.py
```

## 待测模型列表

### 开源系列
- qwen3.5-27b
- kimi-k2.5
- deepseek-v3.2
- qwen3.6-35b-a3b
- qwen3.5-122b-a10b
- qwen3.5-397b-a17b
- qwen3-235b-a22b
- qwen3-32b

### 不开源系列
- claude-opus-4.6
- gemini-3-pro-preview
- qwen3-max
- qwen-plus
- qwen3.6-plus
- qwen3.5-plus
- qwen3.6-flash
- qwen3.5-flash

## 评测说明

| 题型 | 数据集 | 题量 | 评测方式 |
|------|--------|------|----------|
| 选择题 | AP Study | 1,395 | 精确匹配 |
| 选择题 | NPEE | 182 | 精确匹配 |
| 判断题 | NPEE | 134 | 精确匹配 |
| 填空题 | NPEE | 150 | 关键词模糊匹配 |
| 名词解释 | NPEE | 454 | GLM-5.1评分 (0-6分) |
| 问答 | NPEE | 153 | GLM-5.1评分 (0-10分) |
| 讨论 | NPEE | 335 | GLM-5.1评分 (0-10分) |

## 输出

评测完成后会在 `reports/` 目录生成:
- `benchmark_report_*.json` - 详细JSON结果
- `benchmark_report_*.md` - Markdown格式报告
