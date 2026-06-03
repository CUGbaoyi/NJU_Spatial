# %% [markdown]
# # 多尺度地理加权回归 MGWR — 美国县级人均收入影响因素的空间异质性
#
# 本代码完整实现以下分析流程：
# 1. 数据加载与预处理
# 2. OLS 全局回归模型（作为基准对比）
# 3. MGWR 模型：每个变量独立带宽选择 + 模型拟合
# 4. 结果可视化：
#    - 各变量最优带宽对比图
#    - 局部 R² 空间分布图
#    - 自变量局部系数空间分布图
#    - MGWR 局部系数 vs OLS 全局系数对比图
# 5. 空间异质性分析与讨论
#
# ---
# **核心研究问题：**
# 1. 影响美国县级人均收入的因素是否存在空间异质性？
# 2. 不同因素的空间尺度是否不同（带宽差异）？
# 3. MGWR 相比 OLS 全局模型，拟合优度提升了多少？
#
# **MGWR vs GWR 的关键区别：**
# - GWR：所有变量共用一个带宽
# - MGWR：每个变量有独立的最优带宽，能捕捉多尺度空间异质性

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
import warnings
import math
# 仅抑制 matplotlib 中文字体缺失的警告，保留其他警告
warnings.filterwarnings('ignore', message='.*Glyph.*missing from current font.*')
warnings.filterwarnings('ignore', message='.*findfont: Font family.*not found.*')

# 空间分析核心库
import geopandas as gpd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 显示中文（Mac系统）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("多尺度地理加权回归 MGWR")
print("案例：美国县级人均收入影响因素的空间异质性")
print("=" * 60)


def subplot_grid(n_items, n_cols=2):
    n_rows = math.ceil(n_items / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5.5 * n_rows))
    axes = np.array(axes).reshape(-1)
    return fig, axes


def diverging_limits(values):
    vmax = max(abs(np.nanmin(values)), abs(np.nanmax(values)))
    return (-vmax, vmax) if vmax > 0 else (-1, 1)

# %% ---------------------------------------------------------------
# 第一步：数据加载与合并
# ---------------------------------------------------------------
import os

data_dir = os.path.dirname(os.path.abspath(__file__))

# --- 加载属性数据（CSV）---
csv_path = os.path.join(data_dir, 'us_counties_mgwr.csv')
df = pd.read_csv(csv_path, dtype={'FIPS': str})
print(f"\n[1] CSV 数据加载完成")
print(f"    县数: {len(df)}")
print(f"    列: {list(df.columns)}")

# --- 加载县边界 shapefile ---
shp_path = os.path.join(data_dir, 'us_counties.shp')
gdf = gpd.read_file(shp_path)
print(f"\n[2] Shapefile 加载完成")
print(f"    县数: {len(gdf)}")

# 处理 FIPS
gdf['FIPS'] = gdf['FIPS'].astype(str).str.replace('US', '', regex=False).str.zfill(5)

# 合并
gdf_merged = gdf.merge(df, on='FIPS', how='inner')
print(f"\n[3] 数据合并完成: {len(gdf_merged)} 个县")

# 健壮性检查：文件和字段
required_cols = ['FIPS', 'lon', 'lat', 'income_per_capita']
missing_cols = [c for c in required_cols if c not in gdf_merged.columns]
if missing_cols:
    raise ValueError(f"缺少必要字段: {missing_cols}")

# %% ---------------------------------------------------------------
# 第二步：变量选择与数据准备
# ---------------------------------------------------------------
VAR_Y = 'income_per_capita'
EXCLUDE_COLS = {'FIPS', 'lon', 'lat', VAR_Y}
PREFERRED_X_ORDER = [
    'pop_density',
    'agriculture_share',
    'mining_share',
    'construction_share',
    'manufacturing_share',
    'finance_share',
    'services_share',
    'government_share',
]
numeric_x_cols = [
    col for col in df.columns
    if col not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(gdf_merged[col])
]
VAR_X = [col for col in PREFERRED_X_ORDER if col in numeric_x_cols]
VAR_X += [col for col in numeric_x_cols if col not in VAR_X]

if not VAR_X:
    raise ValueError("未找到可用于 MGWR 的数值型自变量")

print(f"\n因变量: {VAR_Y}")
print(f"自变量（全部数值型解释变量，共 {len(VAR_X)} 个）: {', '.join(VAR_X)}")

all_vars = [VAR_Y] + VAR_X
gdf_valid = gdf_merged.dropna(subset=all_vars).copy()

# 健壮性检查：样本量
n_vars = len(VAR_X) + 1  # 含截距
if len(gdf_valid) < n_vars + 1:
    raise ValueError(f"有效样本 ({len(gdf_valid)}) 不足以拟合 {n_vars} 个变量的模型")
print(f"有效数据量: {len(gdf_valid)} 个县")

y = gdf_valid[VAR_Y].values.astype(float).reshape(-1, 1)
X = gdf_valid[VAR_X].values.astype(float)
coords = np.array(list(zip(gdf_valid['lon'].values, gdf_valid['lat'].values)))
print(f"坐标范围: 经度 [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]")
print(f"           纬度 [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")

# z-score 标准化（MGWR 和 OLS 均使用标准化后的数据）
y_mean, y_std_val = y.mean(), y.std()
y_std = (y - y_mean) / y_std_val
X_mean, X_std_val = X.mean(axis=0), X.std(axis=0)
if y_std_val == 0:
    raise ValueError(f"因变量 {VAR_Y} 标准差为 0，无法标准化")
zero_std_vars = [var for var, std in zip(VAR_X, X_std_val) if std == 0]
if zero_std_vars:
    raise ValueError(f"以下自变量标准差为 0，无法标准化: {zero_std_vars}")
X_std = (X - X_mean) / X_std_val
print(f"\n已完成 z-score 标准化（y 和 X）")

# %% ---------------------------------------------------------------
# 第三步：OLS 全局回归模型（基准对比）
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("步骤一：OLS 全局回归模型（基准对比）")
print("=" * 60)

X_ols = sm.add_constant(X_std)
ols_model = sm.OLS(y_std, X_ols).fit()

print(f"\nOLS 回归结果:")
print(f"  R² = {ols_model.rsquared:.4f}")
print(f"  调整 R² = {ols_model.rsquared_adj:.4f}")
print(f"  AIC = {ols_model.aic:.2f}")
print(f"  F 统计量 = {ols_model.fvalue:.4f}, p = {ols_model.f_pvalue:.6f}")
print(f"\n回归系数:")
ols_params = {}
for i, var in enumerate(['常数项'] + VAR_X):
    coef = ols_model.params[i]
    pval = ols_model.pvalues[i]
    sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else ''))
    print(f"  {var:25s}: {coef:>12.4f}  (p = {pval:.4f} {sig})")
    ols_params[var] = coef

# VIF
print(f"\nVIF 检验:")
for i, var in enumerate(VAR_X):
    vif = variance_inflation_factor(X_ols, i + 1)
    print(f"  {var}: VIF = {vif:.4f} {'(⚠️ 多重共线性)' if vif > 10 else ''}")

# OLS 残差诊断图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(ols_model.fittedvalues, ols_model.resid, s=10, alpha=0.4, c='steelblue')
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[0].set_xlabel('Fitted Values', fontsize=12)
axes[0].set_ylabel('Residuals', fontsize=12)
axes[0].set_title('OLS Residuals vs Fitted', fontsize=13)

from scipy import stats
stats.probplot(ols_model.resid, plot=axes[1])
axes[1].set_title('OLS Residuals Q-Q Plot', fontsize=13)
axes[1].get_lines()[0].set_markerfacecolor('steelblue')
axes[1].get_lines()[0].set_markersize(2)

plt.tight_layout()
plt.savefig('output_01_ols_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# %% ---------------------------------------------------------------
# 第四步：MGWR 模型 — 多尺度带宽选择
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("步骤二：MGWR 模型 — 多尺度带宽选择")
print("=" * 60)

from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW

# MGWR 内部自动处理截距项，不需要手动添加常数列
# 使用标准化后的数据
X_mgwr = X_std

print("\n正在搜索各变量的最优带宽（MGWR 每个变量独立带宽）...")
print("  这可能需要较长时间，请耐心等待...\n")

# MGWR 带宽选择：multi=True 表示多尺度，spherical=True 处理经纬度距离
min_bw = max(40, len(VAR_X) * 5)
max_bw = min(len(y) - 1, 1500)
selector = Sel_BW(coords, y_std, X_mgwr, kernel='gaussian', multi=True, spherical=True)
bw = selector.search(
    criterion='AICc',
    tol=1e-5,
    max_iter=80,
    tol_multi=1e-4,
    max_iter_multi=30,
    bws_same_times=3,
    multi_bw_min=[min_bw] * (len(VAR_X) + 1),
    multi_bw_max=[max_bw] * (len(VAR_X) + 1),
    verbose=True,
)

print(f"\nMGWR 多尺度带宽选择结果:")
print(f"  {'变量':25s} {'最优带宽':>10s} {'占样本比':>10s} {'空间尺度':>10s}")
print(f"  {'─' * 60}")

var_names = ['常数项'] + VAR_X
bw_dict = {}
for i, var in enumerate(var_names):
    b = bw[i]
    pct = b / len(y) * 100
    if pct > 60:
        scale = '全局/宏观'
    elif pct > 30:
        scale = '区域/中观'
    else:
        scale = '局部/微观'
    print(f"  {var:25s} {b:>10.0f} {pct:>9.1f}% {scale:>10s}")
    bw_dict[var] = b

# 带宽对比可视化
fig, ax = plt.subplots(figsize=(10, 5))
colors = plt.cm.tab10(np.linspace(0, 1, len(var_names)))
bars = ax.bar(var_names, [bw_dict[v] for v in var_names], color=colors, edgecolor='white')
ax.axhline(y=len(y), color='gray', linestyle='--', linewidth=1, label=f'Total Samples ({len(y)})')
for bar, var in zip(bars, var_names):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
            f'{bw_dict[var]:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('Bandwidth (Number of Neighbors)', fontsize=12)
ax.set_title('MGWR Optimal Bandwidth per Variable', fontsize=14)
ax.tick_params(axis='x', rotation=35)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('output_02_mgwr_bandwidths.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# %% ---------------------------------------------------------------
# 第五步：MGWR 模型拟合
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("步骤三：MGWR 模型拟合")
print("=" * 60)

mgwr_model = MGWR(coords, y_std, X_mgwr, selector, kernel='gaussian', spherical=True).fit()

print(f"\nMGWR 模型拟合完成:")
print(f"  全局 R² = {mgwr_model.R2:.4f}")
print(f"  调整 R² = {mgwr_model.adj_R2:.4f}")
print(f"  AICc = {mgwr_model.aicc:.2f}")
print(f"\nMGWR 局部系数统计摘要:")
print(f"  {'变量':25s} {'全局均值':>12s} {'最小值':>12s} {'最大值':>12s} {'标准差':>12s}")
print(f"  {'─' * 75}")

mgwr_coef_stats = {}
for i, var in enumerate(var_names):
    coefs = mgwr_model.params[:, i]
    mgwr_coef_stats[var] = {
        'mean': np.mean(coefs),
        'std': np.std(coefs),
        'min': np.min(coefs),
        'max': np.max(coefs),
        'values': coefs
    }
    print(f"  {var:25s} {np.mean(coefs):>12.4f} {np.min(coefs):>12.4f} "
          f"{np.max(coefs):>12.4f} {np.std(coefs):>12.4f}")

# 局部 R²（MGWR 不支持 localR2，手动用 kNN 加权回归计算）
from scipy.spatial.distance import cdist

def compute_local_r2(y, X, coords, k=50):
    """用 kNN 空间加权回归计算每个观测的局部 R²"""
    n = len(y)
    dists = cdist(coords, coords)
    local_r2 = np.zeros(n)
    y_flat = y.flatten()
    for i in range(n):
        idx = np.argsort(dists[i])[:k]
        w = np.exp(-0.5 * (dists[i, idx] / dists[i, idx].mean()) ** 2)
        W = np.diag(w)
        X_local = np.column_stack([np.ones(k), X[idx]])
        try:
            beta = np.linalg.solve(X_local.T @ W @ X_local, X_local.T @ W @ y_flat[idx])
            y_hat_local = X_local @ beta
            ss_res = np.sum(w * (y_flat[idx] - y_hat_local) ** 2)
            ss_tot = np.sum(w * (y_flat[idx] - np.average(y_flat[idx], weights=w)) ** 2)
            local_r2[i] = max(0, 1 - ss_res / ss_tot) if ss_tot > 0 else 0
        except np.linalg.LinAlgError:
            local_r2[i] = np.nan
    return local_r2

local_r2 = compute_local_r2(y_std, X_std, coords, k=50)
print(f"\n局部 R² 统计:")
print(f"  均值: {np.mean(local_r2):.4f}")
print(f"  最小值: {np.min(local_r2):.4f}")
print(f"  最大值: {np.max(local_r2):.4f}")
print(f"  中位数: {np.median(local_r2):.4f}")

r2_improvement = mgwr_model.R2 - ols_model.rsquared
print(f"\nMGWR vs OLS R² 改善: {r2_improvement:.4f} "
      f"({'+' if r2_improvement > 0 else ''}{r2_improvement/ols_model.rsquared*100:.1f}%)")

# 写入结果
gdf_valid['local_r2'] = local_r2
for i, var in enumerate(VAR_X):
    gdf_valid[f'coef_{var}'] = mgwr_model.params[:, i + 1]

# %% ---------------------------------------------------------------
# 第六步：结果可视化
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("步骤四：结果可视化")
print("=" * 60)

# --- 图1：局部 R² 空间分布图 ---
fig, ax = plt.subplots(figsize=(14, 10))
gdf.plot(ax=ax, facecolor='none', edgecolor='#cccccc', linewidths=0.1)
gdf_valid.plot(column='local_r2', cmap='RdYlGn', legend=True,
               legend_kwds={'label': 'Local R²', 'shrink': 0.8, 'pad': 0.02},
               edgecolors='none', linewidths=0, ax=ax, vmin=0, vmax=1)

worst_r2_idx = gdf_valid['local_r2'].nsmallest(5).index
for idx in worst_r2_idx:
    row = gdf_valid.loc[idx]
    cx, cy = row['lon'], row['lat']
    name = row.get('NAME', row.get('FIPS', ''))
    ax.annotate(f"{name}\nR²={row['local_r2']:.2f}",
                xy=(cx, cy), fontsize=6, ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))

ax.set_title(f'MGWR Local R² Spatial Distribution\n'
             f'(Mean={np.mean(local_r2):.3f}, Range=[{np.min(local_r2):.3f}, {np.max(local_r2):.3f}])',
             fontsize=14)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_xlim([-130, -65])
ax.set_ylim([24, 50])
plt.tight_layout()
plt.savefig('output_03_local_r2.png', dpi=150, bbox_inches='tight')
plt.close(fig)

print("\n[图3] 局部 R² 空间分布 → output_03_local_r2.png")

# --- 图2：各自变量局部系数空间分布图 ---
fig, axes = subplot_grid(len(VAR_X), n_cols=2)
for idx, var in enumerate(VAR_X):
    ax = axes[idx]
    col_name = f'coef_{var}'
    gdf.plot(ax=ax, facecolor='none', edgecolor='#cccccc', linewidths=0.1)
    vmin, vmax = diverging_limits(gdf_valid[col_name].values)
    gdf_valid.plot(column=col_name, cmap='RdBu_r', legend=True,
                   legend_kwds={'shrink': 0.7, 'pad': 0.02},
                   edgecolors='none', linewidths=0, ax=ax, vmin=vmin, vmax=vmax)
    ols_val = ols_params[var]
    ax.set_title(f'{var}\n'
                 f'OLS Coef = {ols_val:.4f} | MGWR Range = [{mgwr_coef_stats[var]["min"]:.4f}, {mgwr_coef_stats[var]["max"]:.4f}]\n'
                 f'Bandwidth = {bw_dict[var]:.0f}',
                 fontsize=11)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xlim([-130, -65])
    ax.set_ylim([24, 50])

for ax in axes[len(VAR_X):]:
    ax.axis('off')

fig.suptitle('MGWR Local Coefficient Spatial Distribution', fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig('output_04_local_coefficients.png', dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"\n[图4] 局部系数空间分布 → output_04_local_coefficients.png")

# --- 图3：MGWR 局部系数 vs OLS 全局系数对比 ---
fig, axes = subplot_grid(len(VAR_X), n_cols=2)
for idx, var in enumerate(VAR_X):
    ax = axes[idx]
    col_name = f'coef_{var}'
    local_vals = gdf_valid[col_name].values
    ols_val = ols_params[var]

    bp = ax.boxplot(local_vals, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=3))
    ax.axhline(y=ols_val, color='green', linestyle='--', linewidth=2,
               label=f'OLS Global = {ols_val:.4f}')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

    n_positive = np.sum(local_vals > 0)
    n_negative = np.sum(local_vals < 0)
    ax.text(0.02, 0.98,
            f'Positive: {n_positive} ({n_positive/len(local_vals)*100:.1f}%)\n'
            f'Negative: {n_negative} ({n_negative/len(local_vals)*100:.1f}%)',
            transform=ax.transAxes, ha='left', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_ylabel(f'{var} Coefficient', fontsize=11)
    ax.set_title(f'{var} (BW={bw_dict[var]:.0f})', fontsize=11)
    ax.set_xticklabels(['MGWR Local Coefficients'])
    ax.legend(fontsize=9, loc='lower right')

for ax in axes[len(VAR_X):]:
    ax.axis('off')

fig.suptitle('MGWR Local Coefficients vs OLS Global (Green Dashed)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('output_05_mgwr_vs_ols.png', dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"\n[图5] MGWR vs OLS 对比 → output_05_mgwr_vs_ols.png")

# --- 图4：R² 改善空间分布图 ---
fig, ax = plt.subplots(figsize=(14, 10))
gdf_valid['r2_improvement'] = gdf_valid['local_r2'] - ols_model.rsquared
gdf.plot(ax=ax, facecolor='none', edgecolor='#cccccc', linewidths=0.1)
vmax_imp = max(abs(gdf_valid['r2_improvement'].min()), abs(gdf_valid['r2_improvement'].max()))
gdf_valid.plot(column='r2_improvement', cmap='RdYlGn', legend=True,
               legend_kwds={'label': 'R² Improvement (MGWR - OLS)', 'shrink': 0.8, 'pad': 0.02},
               edgecolors='none', linewidths=0, ax=ax, vmin=-vmax_imp, vmax=vmax_imp)
ax.set_title(f'MGWR vs OLS: Local R² Improvement\n(OLS R² = {ols_model.rsquared:.4f})',
             fontsize=14)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_xlim([-130, -65])
ax.set_ylim([24, 50])
plt.tight_layout()
plt.savefig('output_06_r2_improvement.png', dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"\n[图6] R² 改善空间分布 → output_06_r2_improvement.png")

# --- 图5：系数分布直方图 ---
fig, axes = subplot_grid(len(VAR_X), n_cols=2)
from scipy.stats import gaussian_kde
for idx, var in enumerate(VAR_X):
    ax = axes[idx]
    col_name = f'coef_{var}'
    local_vals = gdf_valid[col_name].values
    ols_val = ols_params[var]

    ax.hist(local_vals, bins=30, color='steelblue', edgecolor='white', alpha=0.8, density=True)
    if np.nanstd(local_vals) > 0:
        kde_x = np.linspace(local_vals.min(), local_vals.max(), 200)
        kde = gaussian_kde(local_vals)
        ax.plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE')
    ax.axvline(x=ols_val, color='green', linestyle='--', linewidth=2, label=f'OLS = {ols_val:.4f}')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel(f'{var} Coefficient', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'{var} (BW={bw_dict[var]:.0f})\n'
                 f'Range=[{mgwr_coef_stats[var]["min"]:.3f}, {mgwr_coef_stats[var]["max"]:.3f}]',
                 fontsize=11)
    ax.legend(fontsize=9)

for ax in axes[len(VAR_X):]:
    ax.axis('off')

fig.suptitle('MGWR Local Coefficient Distributions (Green = OLS Global)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('output_07_coef_distributions.png', dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"\n[图7] 系数分布直方图 → output_07_coef_distributions.png")

# %% ---------------------------------------------------------------
# 第七步：空间异质性分析与讨论
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("步骤五：空间异质性分析与讨论")
print("=" * 60)

print("\n【分析一】局部 R² 较低的区域")
print("─" * 50)
low_r2_threshold = np.percentile(local_r2, 10)
low_r2_mask = local_r2 < low_r2_threshold
low_r2_names = gdf_valid.loc[low_r2_mask, 'NAME'].tolist() if 'NAME' in gdf_valid.columns else []
print(f"  局部 R² 最低 10% 阈值: {low_r2_threshold:.4f}")
print(f"  低 R² 县数: {np.sum(low_r2_mask)} 个")
if low_r2_names:
    print(f"  低 R² 县: {low_r2_names[:10]}...")

print(f"\n【分析二】自变量系数的空间异质性（符号变化）")
print("─" * 50)
for var in VAR_X:
    col_name = f'coef_{var}'
    vals = gdf_valid[col_name].values
    ols_val = ols_params[var]
    n_pos = np.sum(vals > 0)
    n_neg = np.sum(vals < 0)
    pct_neg = n_neg / len(vals) * 100
    cv = np.std(vals) / abs(np.mean(vals)) * 100 if np.mean(vals) != 0 else np.inf

    print(f"\n  {var} (BW={bw_dict[var]:.0f}):")
    print(f"    OLS Global: {ols_val:.4f}")
    print(f"    MGWR Range: [{np.min(vals):.4f}, {np.max(vals):.4f}]")
    print(f"    CV: {cv:.1f}%")
    print(f"    Positive: {n_pos} ({n_pos/len(vals)*100:.1f}%)")
    print(f"    Negative: {n_neg} ({pct_neg:.1f}%)")
    if pct_neg > 5:
        print(f"    ⚠️ Significant sign reversal!")

print(f"\n【分析三】MGWR vs OLS 模型比较")
print("─" * 50)
print(f"\n  模型比较:")
print(f"  {'指标':20s} {'OLS':>12s} {'MGWR':>12s}")
print(f"  {'─' * 47}")
print(f"  {'R²':20s} {ols_model.rsquared:>12.4f} {mgwr_model.R2:>12.4f}")
print(f"  {'调整 R²':20s} {ols_model.rsquared_adj:>12.4f} {mgwr_model.adj_R2:>12.4f}")
print(f"  {'AICc':20s} {'—':>12s} {mgwr_model.aicc:>12.2f}")

print(f"\n  MGWR 的优势:")
print(f"    1. 每个变量独立带宽，捕捉多尺度空间异质性")
print(f"    2. R² 从 {ols_model.rsquared:.4f} 提升至 {mgwr_model.R2:.4f}")
print(f"    3. 不同变量的空间尺度差异揭示了不同过程的作用范围")

print(f"\n  带宽解读:")
for var in VAR_X:
    b = bw_dict[var]
    pct = b / len(y) * 100
    if pct > 60:
        print(f"    {var}: 带宽大 ({b:.0f}, {pct:.1f}%) → 全局性影响，空间变异小")
    elif pct > 30:
        print(f"    {var}: 带宽中 ({b:.0f}, {pct:.1f}%) → 区域性影响")
    else:
        print(f"    {var}: 带宽小 ({b:.0f}, {pct:.1f}%) → 局部性影响，空间异质性强")

# %% ---------------------------------------------------------------
# 最终总结
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("分析总结")
print("=" * 60)

print(f"1. MGWR R² ({mgwr_model.R2:.4f}) > OLS R² ({ols_model.rsquared:.4f})，说明考虑空间异质性后模型拟合改善。")
print(f"2. 本次使用全部 {len(VAR_X)} 个数值型自变量: {', '.join(VAR_X)}。")
print("3. 各变量带宽差异如下，带宽越小表示作用越局部、空间异质性越强:")
for var in VAR_X:
    print(f"   · {var}: BW={bw_dict[var]:.0f} ({bw_dict[var] / len(y) * 100:.1f}%)")
print("4. MGWR 能展示每个变量在不同县域的局部系数差异，但仍不能替代空间误差/空间滞后模型对空间自相关的检验。")

print("=" * 60)
print(f"分析完成！共生成 7 张结果图（output_01 ~ output_07.png）")
print("=" * 60)
