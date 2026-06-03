# MGWR 多尺度地理加权回归 — 美国县级人均收入

## 目标

使用 MGWR（Multiscale Geographically Weighted Regression）捕捉美国县级人均收入影响因素的空间异质性，比较不同变量的空间尺度差异。

## MGWR vs GWR

- **GWR**：所有变量共用一个带宽
- **MGWR**：每个变量有独立的最优带宽，能捕捉多尺度空间异质性

## 分析流程

1. **数据加载与预处理**：加载 SHP + CSV，合并县级数据
2. **OLS 全局模型**：作为基准对比
3. **MGWR 模型**：
   - 使用 `mgwr` 包的 `MGWR` 类
   - 每个变量独立带宽选择（`Sel_BW` with `multi=True`）
   - 基于 AICc 准则
4. **结果可视化**：
   - 各变量最优带宽对比图
   - 局部 R² 空间分布图
   - 自变量局部系数空间分布图
   - MGWR vs OLS 系数对比图
   - R² 改善空间分布图
5. **分析与讨论**

## 数据

- 县边界：`us_counties.shp`
- 属性数据：`us_counties_mgwr.csv`
- 因变量：`income_per_capita`（人均收入）
- 自变量：`pop_density`、`manufacturing_share`、`finance_share`、`services_share`

## 运行

```bash
cd Desktop/us_counties
pip install mgwr  # 如果未安装
python mgwr_analysis.py
```

或打开 `mgwr_analysis.ipynb` 进行交互式演示。

## 输出

- `output_01_ols_diagnostics.png` — OLS 残差诊断
- `output_02_mgwr_bandwidths.png` — MGWR 各变量带宽对比
- `output_03_local_r2.png` — 局部 R² 空间分布
- `output_04_local_coefficients.png` — 局部系数空间分布
- `output_05_mgwr_vs_ols.png` — MGWR vs OLS 系数对比
- `output_06_r2_improvement.png` — R² 改善空间分布
- `output_07_coef_distributions.png` — 系数分布直方图
