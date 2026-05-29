# 加速测试方案

## 方案对比

| 方案 | 加速比 | 风险 | 实现难度 |
|------|--------|------|----------|
| **1. 模型级并发** | 5-10x | API限流 | 中 |
| **2. 自适应并发** | 1.5-2x | 低 | 低 |
| **3. Judge结果缓存** | 2-3x | 低 | 中 |
| **4. 分题层优先** | - | 无 | 低 |

---

## 方案1: 模型级并发 (推荐)

**核心思想**: 多个模型同时测试，而不是串行

```python
# 当前: 串行
for model in models:
    test(model, questions)  # 10个模型串行

# 优化: 并发
with ThreadPoolExecutor(max_workers=3) as executor:
    executor.map(test_model, models)  # 3个模型同时测
```

**加速比**: 3-5倍 (取决于同时测几个模型)

**风险控制**:
- 每个API的并发限制
- 本地资源限制
- 建议: 同时测2-3个模型

---

## 方案2: Judge结果缓存

**核心思想**: 相同答案复用Judge结果

```python
# 用答案内容的hash作为key
answer_hash = md5(model_answer)
if answer_hash in cache:
    return cache[answer_hash]
else:
    result = judge(model_answer)
    cache[answer_hash] = result
    return result
```

**加速比**: 2-3倍 (很多模型答案相似)

---

## 方案3: 分题型优先级

**核心思想**: 先跑完快速题型(选择题)，再跑慢速题型

```python
# 第一轮: 所有模型的选择题 (快速)
# 第二轮: 所有模型的填空题
# 第三轮: 所有模型的开放式题
```

**好处**: 可以更快看到部分结果

---

## 推荐组合方案

```
1. 模型级并发 (3个模型同时)
2. Judge结果缓存
3. 分题型优先级
```

**预期加速**: 5-8倍
**原时间**: 4-5小时 → **优化后**: 30-60分钟
