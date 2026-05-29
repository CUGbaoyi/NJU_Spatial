#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算全国对外开放水平的Moran's I值（含完整可视化）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# 空间分析核心库
import geopandas as gpd
from libpysal.weights import Queen, KNN
from esda.moran import Moran, Moran_Local
from esda.getisord import G_Local
from scipy import stats

# 中文显示设置
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("全国对外开放水平的Moran's I分析")
print("=" * 60)

# 数据路径
data_dir = 'data/china'

# 加载城市边界 shapefile
shp_path = os.path.join(data_dir, 'city.shp')
gdf = gpd.read_file(shp_path, encoding='utf-8')
print(f"\n[1] Shapefile 加载完成: {len(gdf)} 个城市")

# 加载属性数据
csv_path = os.path.join(data_dir, '地级市数据.csv')
df_attr = pd.read_csv(csv_path, encoding='utf-8')
print(f"[2] CSV 加载完成: {df_attr.shape}")
print(f"    年份范围: {df_attr['year'].min()} ~ {df_attr['year'].max()}")

# 选择最新年份（2021）
ANALYSIS_YEAR = 2021
df_year = df_attr[df_attr['year'] == ANALYSIS_YEAR].copy()
print(f"\n筛选 {ANALYSIS_YEAR} 年数据: {len(df_year)} 个城市")

# 行政区划代码匹配
df_year['adcode'] = df_year['行政区划代码'].astype(str).str.zfill(6)
MUNICIPALITY_MAP = {'110000': '110100', '120000': '120100', '310000': '310100', '500000': '500100'}
df_year['adcode_mapped'] = df_year['adcode'].map(lambda c: MUNICIPALITY_MAP.get(c, c))

# 合并数据
gdf_merged = gdf.merge(
    df_year,
    left_on='ct_adcode',
    right_on='adcode_mapped',
    how='left',
    suffixes=('', '_csv')
)

print(f"[3] 数据合并完成: {len(gdf_merged)} 个空间单元")
print(f"    成功匹配属性: {gdf_merged['year'].notna().sum()} 个")

# 选择对外开放水平变量
VAR_NAME = '对外开放水平'
VAR_LABEL = '对外开放水平'

# 处理缺失值
gdf_valid = gdf_merged.dropna(subset=[VAR_NAME]).copy()
y_values = gdf_valid[VAR_NAME].values.astype(float)

print(f"\n研究变量: {VAR_LABEL}")
print(f"有效数据量: {len(y_values)} 个城市")

# 基本统计
print(f"\n【基本统计量】")
print(f"  样本量:   {len(y_values)}")
print(f"  均值:     {np.mean(y_values):.4f}")
print(f"  中位数:   {np.median(y_values):.4f}")
print(f"  标准差:   {np.std(y_values):.4f}")
print(f"  最小值:   {np.min(y_values):.4f}")
print(f"  最大值:   {np.max(y_values):.4f}")
print(f"  偏度:     {stats.skew(y_values):.4f}")
print(f"  峰度:     {stats.kurtosis(y_values):.4f}")

# %% ============================================================
# 图1: EDA可视化（直方图 + 箱线图 + QQ图）
# ============================================================
print("\n生成图1: EDA分布图...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) 直方图 + KDE
axes[0].hist(y_values, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='white')
kde_x = np.linspace(y_values.min(), y_values.max(), 200)
kde = stats.gaussian_kde(y_values)
axes[0].plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE')
axes[0].axvline(np.mean(y_values), color='orange', linestyle='--', linewidth=2,
                label=f'均值={np.mean(y_values):.2f}')
axes[0].axvline(np.median(y_values), color='green', linestyle='-.', linewidth=2,
                label=f'中位数={np.median(y_values):.2f}')
axes[0].set_xlabel(VAR_LABEL, fontsize=12)
axes[0].set_ylabel('密度', fontsize=12)
axes[0].set_title(f'{VAR_NAME} 分布直方图', fontsize=14)
axes[0].legend(fontsize=10)

# (b) 箱线图
axes[1].boxplot(y_values, vert=True, patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2),
                flierprops=dict(marker='o', markerfacecolor='red', markersize=5))
axes[1].set_ylabel(VAR_LABEL, fontsize=12)
axes[1].set_title(f'{VAR_NAME} 箱线图', fontsize=14)
axes[1].set_xticklabels([VAR_NAME])

# (c) QQ图（检验正态性）
stats.probplot(y_values, plot=axes[2])
axes[2].set_title('Q-Q 图（正态性检验）', fontsize=14)
axes[2].get_lines()[0].set_markerfacecolor('steelblue')
axes[2].get_lines()[0].set_markersize(3)

plt.tight_layout()
plt.savefig('openness_01_eda_distribution.png', dpi=150, bbox_inches='tight')
print("  保存: openness_01_eda_distribution.png")
plt.close()

# %% ============================================================
# 图2: 空间分布图
# ============================================================
print("\n生成图2: 空间分布图...")
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# 底图：所有城市边界
gdf.plot(ax=ax, facecolor='none', edgecolor='#cccccc', linewidths=0.3)
# 叠加分析图层
gdf_valid.plot(column=VAR_NAME, cmap='RdYlGn_r', legend=True,
               legend_kwds={'label': VAR_LABEL, 'shrink': 0.8},
               edgecolors='gray', linewidths=0.2, ax=ax)

ax.set_title(f'{ANALYSIS_YEAR}年 中国地级市{VAR_LABEL}空间分布',
             fontsize=14)
ax.set_xlabel('经度')
ax.set_ylabel('纬度')

plt.tight_layout()
plt.savefig('openness_02_spatial_distribution.png', dpi=150, bbox_inches='tight')
print("  保存: openness_02_spatial_distribution.png")
plt.close()

# %% ============================================================
# 构建权重矩阵
# ============================================================
print("\n" + "=" * 60)
print("构建空间权重矩阵")
print("=" * 60)

# Queen邻接矩阵
w_queen = Queen.from_dataframe(gdf_valid)
w_queen.transform = 'r'
avg_neighbors = np.mean([len(v) for v in w_queen.neighbors.values()])
print(f"Queen 邻接矩阵: 平均邻居数 = {avg_neighbors:.1f}")

# KNN(4)矩阵
w_knn4 = KNN.from_dataframe(gdf_valid, k=4)
w_knn4.transform = 'r'
print(f"KNN(4) 矩阵: 每个单元 4 个邻居")

weight_schemes = {'Queen': w_queen, 'KNN(4)': w_knn4}

# %% ============================================================
# 计算Moran's I
# ============================================================
print("\n" + "=" * 60)
print("Global Moran's I 计算结果")
print("=" * 60)

moran_results = {}

for name, w in weight_schemes.items():
    moran = Moran(y_values, w, permutations=9999)
    moran_results[name] = moran

    print(f"\n{'─' * 50}")
    print(f"权重方案: {name}")
    print(f"  Moran's I      = {moran.I:.4f}")
    print(f"  期望值 E[I]    = {moran.EI:.6f}")
    print(f"  Z 值           = {moran.z_norm:.4f}")

    # 显著性判断
    if moran.p_sim < 0.001:
        sig = "p < 0.001 ***"
    elif moran.p_sim < 0.01:
        sig = "p < 0.01 **"
    elif moran.p_sim < 0.05:
        sig = "p < 0.05 *"
    else:
        sig = "不显著"

    print(f"  p 值（排列检验）= {moran.p_sim:.6f}")
    print(f"  显著性: {sig}")

    # 结果解读
    if moran.I > 0:
        direction = "正空间自相关（聚集）"
        meaning = "高值城市倾向与高值城市为邻，低值城市倾向与低值城市为邻"
    else:
        direction = "负空间自相关（离散）"
        meaning = "高值与低值交替分布"

    print(f"  → 方向: {direction}")
    print(f"  → 含义: {meaning}")

# %% ============================================================
# 图3: Moran散点图
# ============================================================
print("\n生成图3: Moran散点图...")
fig, axes = plt.subplots(1, len(moran_results), figsize=(6 * len(moran_results), 5))
if len(moran_results) == 1:
    axes = [axes]

for idx, (name, moran) in enumerate(moran_results.items()):
    ax = axes[idx]
    w = weight_schemes[name]

    # 标准化变量
    y_std = (y_values - y_values.mean()) / y_values.std()
    # 计算空间滞后
    lag = w.sparse.dot(y_std)

    ax.scatter(y_std, lag, s=15, alpha=0.5, c='steelblue',
               edgecolors='white', linewidths=0.3)

    # 趋势线（斜率 ≈ Moran's I）
    fit = np.polyfit(y_std, lag, 1)
    x_line = np.linspace(y_std.min(), y_std.max(), 100)
    ax.plot(x_line, np.polyval(fit, x_line), 'r-', linewidth=2,
            label=f"拟合斜率 = {moran.I:.4f}")

    # 象限分割线
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)

    # 标注四个象限
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(xlim[1]*0.5, ylim[1]*0.7, 'HH', fontsize=14, color='red',
            fontweight='bold', ha='center')
    ax.text(xlim[0]*0.5, ylim[1]*0.7, 'LH', fontsize=14, color='blue',
            fontweight='bold', ha='center')
    ax.text(xlim[0]*0.5, ylim[0]*0.7, 'LL', fontsize=14, color='darkblue',
            fontweight='bold', ha='center')
    ax.text(xlim[1]*0.5, ylim[0]*0.7, 'HL', fontsize=14, color='orange',
            fontweight='bold', ha='center')

    ax.set_xlabel(f'{VAR_LABEL}（标准化）', fontsize=12)
    ax.set_ylabel('空间滞后值（邻居均值）', fontsize=12)
    ax.set_title(f"Moran 散点图（{name}）\nI = {moran.I:.4f}, p = {moran.p_sim:.4f}",
                 fontsize=13)
    ax.legend(fontsize=10, loc='upper left')

plt.tight_layout()
plt.savefig('openness_03_moran_scatterplot.png', dpi=150, bbox_inches='tight')
print("  保存: openness_03_moran_scatterplot.png")
plt.close()

# 统计各象限的点数
print("\n【Moran 散点图各象限统计】")
for name, moran in moran_results.items():
    w = weight_schemes[name]
    y_std = (y_values - y_values.mean()) / y_values.std()
    lag = w.sparse.dot(y_std)

    hh = np.sum((y_std > 0) & (lag > 0))
    lh = np.sum((y_std < 0) & (lag > 0))
    ll = np.sum((y_std < 0) & (lag < 0))
    hl = np.sum((y_std > 0) & (lag < 0))

    print(f"\n  {name}:")
    print(f"    HH（高-高）: {hh} ({hh/len(y_std)*100:.1f}%)")
    print(f"    LL（低-低）: {ll} ({ll/len(y_std)*100:.1f}%)")
    print(f"    HL（高-低）: {hl} ({hl/len(y_std)*100:.1f}%)")
    print(f"    LH（低-高）: {lh} ({lh/len(y_std)*100:.1f}%)")

# %% ============================================================
# 图4: LISA聚类图
# ============================================================
print("\n生成图4: LISA聚类图...")

# 使用Queen权重作为主方案
primary_w_name = 'Queen'
primary_w = weight_schemes[primary_w_name]

lisa = Moran_Local(y_values, primary_w, permutations=999)

# 分类：基于象限 + 显著性（p < 0.05）
labels = lisa.q  # 象限 (1=HH, 2=LH, 3=LL, 4=HL)
significant = lisa.p_sim <= 0.05

lisa_class = np.zeros(len(y_values), dtype=int)
lisa_class[significant & (labels == 1)] = 1  # HH
lisa_class[significant & (labels == 2)] = 2  # LH
lisa_class[significant & (labels == 3)] = 3  # LL
lisa_class[significant & (labels == 4)] = 4  # HL

class_names = {
    0: '不显著', 1: 'HH（高-高）', 2: 'LH（低-高）',
    3: 'LL（低-低）', 4: 'HL（高-低）'
}

print(f"\nLISA 聚类结果（p < 0.05 显著性水平）:")
for c in [1, 3, 2, 4, 0]:
    count = np.sum(lisa_class == c)
    pct = count / len(y_values) * 100
    print(f"  {class_names[c]}: {count} 个城市 ({pct:.1f}%)")

# 列出 HH 和 LL 的具体城市
if np.sum(lisa_class == 1) > 0:
    hh_cities = gdf_valid.loc[gdf_valid.index[lisa_class == 1], 'ct_name'].tolist()
    print(f"\n  HH 热点城市（示例）: {', '.join(hh_cities[:15])}")
if np.sum(lisa_class == 3) > 0:
    ll_cities = gdf_valid.loc[gdf_valid.index[lisa_class == 3], 'ct_name'].tolist()
    print(f"  LL 冷点城市（示例）: {', '.join(ll_cities[:15])}")

# 绘制LISA聚类图
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# 底图：所有城市边界
gdf.plot(ax=ax, facecolor='none', edgecolor='#cccccc', linewidths=0.3)

colors = {
    0: '#d9d9d9',   # 不显著 - 浅灰
    1: '#d7191c',   # HH - 红色
    2: '#abd9e9',   # LH - 浅蓝
    3: '#2c7bb6',   # LL - 深蓝
    4: '#fdae61',   # HL - 橙色
}

gdf_valid['lisa_class'] = lisa_class
for cls, color in colors.items():
    mask = gdf_valid['lisa_class'] == cls
    if mask.any():
        gdf_valid[mask].plot(ax=ax, facecolor=color,
                              edgecolors='gray', linewidths=0.3,
                              label=class_names[cls], alpha=0.85)

legend_elements = [Patch(facecolor=colors[c], edgecolor='gray', label=class_names[c])
                   for c in [1, 3, 2, 4, 0] if np.sum(lisa_class == c) > 0]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11,
          title='LISA 聚类（p < 0.05）', title_fontsize=12)

ax.set_title(f'LISA 聚类图（{primary_w_name}）\n'
             f'{ANALYSIS_YEAR}年 中国地级市{VAR_LABEL}——局部空间自相关',
             fontsize=14)
ax.set_xlabel('经度')
ax.set_ylabel('纬度')

plt.tight_layout()
plt.savefig('openness_04_lisa_cluster.png', dpi=150, bbox_inches='tight')
print("  保存: openness_04_lisa_cluster.png")
plt.close()

# %% ============================================================
# 图5: Gi*热点图
# ============================================================
print("\n生成图5: Gi*热点图...")

gi_star = G_Local(y_values, primary_w, star=True, permutations=999)
gi_z = gi_star.Zs
gi_p = gi_star.p_sim

print(f"Gi* Z值范围: [{np.nanmin(gi_z):.4f}, {np.nanmax(gi_z):.4f}]")

print(f"\nGi* 热点/冷点统计:")
hot_99 = np.sum((gi_z > 0) & (gi_p < 0.01))
hot_95 = np.sum((gi_z > 0) & (gi_p >= 0.01) & (gi_p < 0.05))
hot_90 = np.sum((gi_z > 0) & (gi_p >= 0.05) & (gi_p < 0.10))
cold_99 = np.sum((gi_z < 0) & (gi_p < 0.01))
cold_95 = np.sum((gi_z < 0) & (gi_p >= 0.01) & (gi_p < 0.05))
cold_90 = np.sum((gi_z < 0) & (gi_p >= 0.05) & (gi_p < 0.10))
not_sig = np.sum(gi_p >= 0.10)

print(f"  热点（99%置信）: {hot_99} 个城市")
print(f"  热点（95%置信）: {hot_95} 个城市")
print(f"  热点（90%置信）: {hot_90} 个城市")
print(f"  冷点（99%置信）: {cold_99} 个城市")
print(f"  冷点（95%置信）: {cold_95} 个城市")
print(f"  冷点（90%置信）: {cold_90} 个城市")
print(f"  不显著:         {not_sig} 个城市")

# 绘制Gi*热点图
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# 底图：所有城市边界
gdf.plot(ax=ax, facecolor='none', edgecolor='#cccccc', linewidths=0.3)

gi_class = np.zeros(len(y_values), dtype=int)
gi_class[(gi_z > 0) & (gi_p < 0.01)] = 3
gi_class[(gi_z > 0) & (gi_p >= 0.01) & (gi_p < 0.05)] = 2
gi_class[(gi_z > 0) & (gi_p >= 0.05) & (gi_p < 0.10)] = 1
gi_class[(gi_z < 0) & (gi_p < 0.01)] = -3
gi_class[(gi_z < 0) & (gi_p >= 0.01) & (gi_p < 0.05)] = -2
gi_class[(gi_z < 0) & (gi_p >= 0.05) & (gi_p < 0.10)] = -1

gi_colors = {
    -3: '#2166ac', -2: '#67a9cf', -1: '#d1e5f0',
    0:  '#f7f7f7',
    1:  '#fddbc7', 2:  '#ef8a62', 3:  '#b2182b',
}
gi_labels = {
    -3: '冷点（99%置信）', -2: '冷点（95%置信）', -1: '冷点（90%置信）',
    0:  '不显著',
    1:  '热点（90%置信）', 2:  '热点（95%置信）', 3:  '热点（99%置信）',
}

gdf_valid['gi_class'] = gi_class
for cls in [3, 2, 1, -1, -2, -3, 0]:
    mask = gdf_valid['gi_class'] == cls
    if mask.any():
        gdf_valid[mask].plot(ax=ax, facecolor=gi_colors[cls],
                              edgecolors='gray', linewidths=0.3,
                              label=gi_labels[cls], alpha=0.85)

legend_elements = [Patch(facecolor=gi_colors[c], edgecolor='gray', label=gi_labels[c])
                   for c in [3, 2, 1, -1, -2, -3, 0] if np.sum(gi_class == c) > 0]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
          title='Gi* 热点分析', title_fontsize=11)

ax.set_title(f'Getis-Ord Gi* 热点图（{primary_w_name}）\n'
             f'{ANALYSIS_YEAR}年 中国地级市{VAR_LABEL}',
             fontsize=14)
ax.set_xlabel('经度')
ax.set_ylabel('纬度')

plt.tight_layout()
plt.savefig('openness_05_gistar_hotspot.png', dpi=150, bbox_inches='tight')
print("  保存: openness_05_gistar_hotspot.png")
plt.close()

# %% ============================================================
# 图6: 稳健性检验对比图
# ============================================================
print("\n生成图6: 稳健性检验对比图...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

names = list(moran_results.keys())
i_values = [moran_results[n].I for n in names]
p_values = [moran_results[n].p_sim for n in names]

# (a) Moran's I 对比柱状图
colors_bar = ['#d7191c' if v > 0 else '#2c7bb6' for v in i_values]
bars = axes[0].bar(names, i_values, color=colors_bar, edgecolor='white', alpha=0.8)
axes[0].axhline(y=0, color='black', linewidth=0.8)
for bar, val in zip(bars, i_values):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[0].set_ylabel("Moran's I", fontsize=12)
axes[0].set_title('不同权重方案下的 Moran\'s I 对比', fontsize=13)

# (b) p 值对比柱状图
axes[1].bar(names, p_values, color='steelblue', edgecolor='white', alpha=0.8)
axes[1].axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
axes[1].axhline(y=0.01, color='orange', linestyle='--', linewidth=1.5, label='α = 0.01')
axes[1].axhline(y=0.001, color='green', linestyle='--', linewidth=1, label='α = 0.001')
for i_p, (n, p) in enumerate(zip(names, p_values)):
    axes[1].text(i_p, max(p, 0.0001) + 0.0005, f'{p:.4f}', ha='center', fontsize=10)
axes[1].set_ylabel('p 值（排列检验）', fontsize=12)
axes[1].set_title('不同权重方案下的 p 值对比', fontsize=13)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('openness_06_robustness.png', dpi=150, bbox_inches='tight')
print("  保存: openness_06_robustness.png")
plt.close()

# %% ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("分析完成！共生成 6 张图：")
print("  1. openness_01_eda_distribution.png    - EDA分布图")
print("  2. openness_02_spatial_distribution.png - 空间分布图")
print("  3. openness_03_moran_scatterplot.png   - Moran散点图")
print("  4. openness_04_lisa_cluster.png        - LISA聚类图")
print("  5. openness_05_gistar_hotspot.png      - Gi*热点图")
print("  6. openness_06_robustness.png          - 稳健性检验图")
print("=" * 60)

# 主要发现总结
primary_moran = moran_results[primary_w_name]
print(f"\n【主要发现】")
print(f"全国地级市{VAR_LABEL}存在", end=" ")
if primary_moran.I > 0 and primary_moran.p_sim < 0.05:
    print(f"显著的正空间自相关")
    print(f"  Moran's I = {primary_moran.I:.4f}, p = {primary_moran.p_sim:.6f}")
    print(f"  含义：对外开放水平相似的城市在空间上聚集分布")
elif primary_moran.I < 0 and primary_moran.p_sim < 0.05:
    print(f"显著的负空间自相关")
    print(f"  Moran's I = {primary_moran.I:.4f}, p = {primary_moran.p_sim:.6f}")
    print(f"  含义：对外开放水平高低交替分布，存在空间竞争")
else:
    print(f"不显著的空间自相关")
    print(f"  Moran's I = {primary_moran.I:.4f}, p = {primary_moran.p_sim:.6f}")
