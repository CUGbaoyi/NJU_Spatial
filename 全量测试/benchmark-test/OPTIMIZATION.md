# Benchmark 优化说明

## 优化概述

本次优化实现了**模型级并发 + 持久化Judge缓存**，大幅提升评测效率和成本效益。

---

## 核心优化

### 1. 持久化Judge缓存系统 (SQLite)

**位置**: [`src/cache/judge_cache.py`](src/cache/judge_cache.py)

**特性**:
- SQLite持久化存储，支持跨会话复用
- 基于题型+问题+参考答案+模型答案的hash键
- 线程安全设计
- 自动统计访问次数和命中率
- 支持按题型/时间清理缓存
- 支持导出缓存为JSON

**收益**:
- 相同答案的评判结果可跨模型复用
- 多次运行评测时，Judge调用大幅减少
- 降低API成本和评测时间

**缓存键策略**:
```
SHA256(task_type + question + reference_answer + model_answer)
```

### 2. 共享Judge缓存池

**位置**: [`src/runner.py`](src/runner.py), [`src/evaluators/llm_judge.py`](src/evaluators/llm_judge.py)

**特性**:
- 所有模型共享同一个Judge缓存实例
- LLMJudgeEvaluator使用缓存减少Judge调用
- 实时统计缓存命中率

**使用方式**:
```python
# 自动初始化，配置文件中启用
runner = BenchmarkRunner(config=config, data_dir=data_dir)
# 访问缓存实例
cache = runner.get_judge_cache()
```

### 3. 模型级并发

**位置**: [`src/runner.py:run_all()`](src/runner.py:289)

**特性**:
- 多个模型同时进行评测
- 可配置并发数量 (`max_concurrent_models`)
- 每个模型独立并发 (`max_concurrent_per_model`)

**配置**:
```yaml
# config/judge.yaml
evaluation:
  max_concurrent_models: 3   # 同时测试3个模型
  max_concurrent_per_model: 5  # 每个模型5个并发请求
```

### 4. 批量Judge评判

**位置**: [`src/evaluators/llm_judge.py:evaluate_batch()`](src/evaluators/llm_judge.py:142)

**特性**:
- 支持批量处理多个评判请求
- 先批量检查缓存，再批量调用Judge
- 可配置批量大小

**配置**:
```yaml
# config/judge.yaml
evaluation:
  enable_batch_judge: true
  judge_batch_size: 5
```

---

## CLI命令

### 运行评测
```bash
python3 run.py run
```

### 缓存管理
```bash
# 查看缓存统计
python3 run.py cache stats

# 清理所有缓存
python3 run.py cache clear

# 清理指定题型的缓存
python3 run.py cache clear --task-type noun

# 清理N天前的缓存
python3 run.py cache clear --older-than 30

# 导出缓存
python3 run.py cache export --output cache_export.json
```

---

## 性能对比

### 优化前
- 串行模型测试
- 每次评测都调用Judge
- 无缓存，重复评判相同答案
- 高API成本，长评测时间

### 优化后
- 并行模型测试 (3x模型并发)
- Judge缓存命中率随运行次数增长
- 第二次运行相同问题，Judge调用接近0
- 显著降低API成本和时间

---

## 文件变更清单

| 文件 | 变更 |
|------|------|
| `src/cache/judge_cache.py` | **新增** - 持久化缓存系统 |
| `src/cache/__init__.py` | **新增** - 缓存模块导出 |
| `src/evaluators/llm_judge.py` | **修改** - 集成持久化缓存 |
| `src/evaluators/base.py` | **修改** - EvaluationResult添加cached字段 |
| `src/runner.py` | **修改** - 集成缓存系统，添加统计 |
| `run.py` | **修改** - 添加cache命令，新配置支持 |
| `config/judge.yaml` | **修改** - 添加缓存和批量配置 |
| `test_cache.py` | **新增** - 缓存功能测试 |

---

## 配置示例

```yaml
# config/judge.yaml

evaluation:
  # 并发控制
  max_concurrent_models: 3      # 同时测试的模型数
  max_concurrent_per_model: 5   # 每模型的并发请求数

  # Judge缓存
  enable_judge_cache: true      # 启用缓存
  judge_cache_dir: null         # 缓存目录（null=默认）

  # 批量评判
  enable_batch_judge: true      # 启用批量处理
  judge_batch_size: 5           # 批量大小
```

---

## 缓存统计示例

```
=== Judge评测统计 ===
  总评测数: 150
  缓存命中: 120
  缓存未命中: 30
  缓存命中率: 80.0%
  错误数: 0
  缓存条目数: 30
  缓存大小: 0.05 MB
```
