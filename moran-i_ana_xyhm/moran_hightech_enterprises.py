from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import KNN, Queen


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


DEFAULT_YEAR = 2022
MUNICIPALITY_MAP = {
    "110000": "110100",
    "120000": "120100",
    "310000": "310100",
    "500000": "500100",
}


def first_matching_column(columns: Iterable[str], keywords: list[str]) -> str:
    column_list = [str(col) for col in columns]
    lowered = {col: col.lower() for col in column_list}

    for keyword in keywords:
        for column in column_list:
            if keyword.lower() == lowered[column]:
                return column

    for keyword in keywords:
        for column in column_list:
            if keyword.lower() in lowered[column]:
                return column

    raise KeyError(f"未找到匹配列，候选关键词: {keywords}；现有列: {column_list}")


def choose_year(df: pd.DataFrame, year_col: str, year: int | None) -> int:
    numeric_year = pd.to_numeric(df[year_col], errors="coerce")
    available_years = sorted(int(item) for item in numeric_year.dropna().unique())
    if not available_years:
        raise ValueError("数据中没有可用年份。")
    return year if year is not None else available_years[-1]


def build_weights(gdf: gpd.GeoDataFrame, method: str) -> object:
    if method == "queen":
        weights = Queen.from_dataframe(gdf)
        if any(cardinality == 0 for cardinality in weights.cardinalities.values()):
            weights = KNN.from_dataframe(gdf, k=4)
    elif method == "knn":
        weights = KNN.from_dataframe(gdf, k=4)
    else:
        raise ValueError(f"不支持的权重方法: {method}")

    weights.transform = "r"
    return weights


def is_standardized_hightech_csv(df: pd.DataFrame) -> bool:
    cols = {str(col) for col in df.columns}
    required = {"city", "行政区划代码", "year", "省份", "高新技术企业数"}
    return required.issubset(cols)


def standardize_hightech_csv(
    raw_csv_path: Path,
    reference_panel_path: Path,
    output_csv_path: Path,
    year: int = DEFAULT_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing = pd.read_csv(raw_csv_path, encoding="utf-8-sig")
    if is_standardized_hightech_csv(existing):
        panel_df = existing.copy()
    else:
        raw_df = pd.read_csv(raw_csv_path, header=None, encoding="utf-8-sig")
        reference_df = pd.read_csv(reference_panel_path, encoding="utf-8")

        ref_city_col = first_matching_column(reference_df.columns, ["city"])
        ref_code_col = first_matching_column(reference_df.columns, ["行政区划代码"])
        ref_province_col = first_matching_column(reference_df.columns, ["省份"])

        city_map = (
            reference_df[[ref_city_col, ref_code_col, ref_province_col]]
            .dropna(subset=[ref_city_col, ref_code_col])
            .drop_duplicates()
            .copy()
        )
        province_names = set(city_map[ref_province_col].astype(str))

        rows = raw_df.iloc[4:].copy()
        rows.columns = [
            "raw_city",
            "city_en",
            "高新技术企业数",
            "高新技术企业数_市辖区",
            "技术合同成交额_万元_全市",
            "技术合同成交额_万元_市辖区",
        ]
        rows["raw_city"] = rows["raw_city"].astype(str).str.strip()
        rows = rows[
            rows["raw_city"].notna()
            & rows["raw_city"].ne("nan")
            & ~rows["raw_city"].str.contains("续表", na=False)
            & rows["raw_city"].ne("城市")
        ].copy()

        rows["is_province"] = rows["raw_city"].isin(province_names)
        rows["省份"] = rows["raw_city"].where(rows["is_province"]).ffill()
        city_rows = rows.loc[~rows["is_province"]].copy()

        merged = city_rows.merge(
            city_map,
            left_on="raw_city",
            right_on=ref_city_col,
            how="left",
            suffixes=("_raw", ""),
            indicator=True,
        )
        unmatched_df = merged.loc[
            merged["_merge"] != "both",
            ["raw_city", "city_en", "省份_raw"],
        ].copy()
        unmatched_df = unmatched_df.rename(columns={"省份_raw": "省份"})
        matched_df = merged.loc[merged["_merge"] == "both"].copy()

        numeric_cols = [
            "高新技术企业数",
            "高新技术企业数_市辖区",
            "技术合同成交额_万元_全市",
            "技术合同成交额_万元_市辖区",
        ]
        for column in numeric_cols:
            matched_df[column] = pd.to_numeric(matched_df[column], errors="coerce")

        panel_df = pd.DataFrame(
            {
                "city": matched_df[ref_city_col].astype(str),
                "行政区划代码": matched_df[ref_code_col].astype(int).astype(str).str.zfill(6),
                "year": year,
                "省份": matched_df[ref_province_col].astype(str),
                "高新技术企业数": matched_df["高新技术企业数"],
                "高新技术企业数_市辖区": matched_df["高新技术企业数_市辖区"],
                "技术合同成交额_万元_全市": matched_df["技术合同成交额_万元_全市"],
                "技术合同成交额_万元_市辖区": matched_df["技术合同成交额_万元_市辖区"],
                "city_en": matched_df["city_en"].astype(str),
                "数据来源": "2-18 高新技术企业数和技术合同成交额",
            }
        ).sort_values(["省份", "行政区划代码"], ignore_index=True)

        unmatched_path = output_csv_path.with_name(f"{output_csv_path.stem}_未匹配城市.csv")
        unmatched_df.to_csv(unmatched_path, index=False, encoding="utf-8-sig")

    panel_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    return panel_df, existing


def prepare_dataset(
    shp_path: Path,
    csv_path: Path,
    analysis_year: int | None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, str, str, str, int]:
    gdf = gpd.read_file(shp_path, encoding="utf-8")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    city_col = first_matching_column(df.columns, ["city", "城市", "地级市"])
    code_col = first_matching_column(df.columns, ["行政区划代码", "adcode", "地区代码"])
    year_col = first_matching_column(df.columns, ["year", "年份"])
    hightech_col = first_matching_column(
        df.columns,
        ["高新技术企业数", "高新技术企业数量", "高新技术企业", "高企数量", "高企数"],
    )

    target_year = choose_year(df, year_col, analysis_year)
    df = df.copy()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df = df[df[year_col] == target_year].copy()

    code_numeric = pd.to_numeric(df[code_col], errors="coerce")
    df["adcode"] = pd.Series(pd.NA, index=df.index, dtype="object")
    valid_code_mask = code_numeric.notna()
    df.loc[valid_code_mask, "adcode"] = (
        code_numeric.loc[valid_code_mask].astype(int).astype(str).str.zfill(6)
    )
    df = df.loc[df["adcode"].notna()].copy()
    df["adcode_mapped"] = df["adcode"].map(lambda code: MUNICIPALITY_MAP.get(code, code))
    df["hightech_value"] = pd.to_numeric(df[hightech_col], errors="coerce")

    gdf_merged = gdf.merge(
        df[[city_col, "adcode_mapped", "hightech_value"]],
        left_on="ct_adcode",
        right_on="adcode_mapped",
        how="left",
    )
    gdf_valid = gdf_merged.dropna(subset=["hightech_value"]).copy()

    if gdf_valid.empty:
        raise ValueError("合并后没有可用于 Moran's I 计算的数据，请检查代码列和企业数列。")

    return gdf_merged, gdf_valid, city_col, code_col, hightech_col, target_year


def plot_choropleth(
    gdf_all: gpd.GeoDataFrame,
    gdf_valid: gpd.GeoDataFrame,
    out_path: Path,
    year: int,
    value_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 8.5))

    # 底图：中国地级市整体范围，先画浅灰底面形成完整中国轮廓。
    gdf_all.plot(
        ax=ax,
        facecolor="#f3f4f6",
        edgecolor="#c7cbd1",
        linewidth=0.35,
    )

    # 叠加全国外边界，让中国轮廓更明确。
    national_outline = gdf_all.dissolve()
    national_outline.boundary.plot(
        ax=ax,
        color="#4b5563",
        linewidth=0.9,
    )

    # 最上层叠加有数据的城市专题图。
    gdf_valid.plot(
        column="hightech_value",
        cmap="YlOrRd",
        linewidth=0.28,
        edgecolor="#5b616b",
        legend=True,
        legend_kwds={"label": value_label, "shrink": 0.72},
        ax=ax,
    )

    ax.set_title(f"{year}年中国地级市高新技术企业数空间分布", fontsize=15)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_moran_simulation(moran: Moran, out_path: Path, year: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.hist(moran.sim, bins=30, color="#93c5fd", edgecolor="white")
    ax.axvline(moran.I, color="#dc2626", linewidth=2, label=f"Observed I = {moran.I:.4f}")
    ax.set_title(f"{year}年高新技术企业数 Moran's I 置换检验", fontsize=14)
    ax.set_xlabel("模拟 Moran's I")
    ax.set_ylabel("频数")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_moran_scatter(gdf: gpd.GeoDataFrame, weights: object, out_path: Path, year: int) -> None:
    y = gdf["hightech_value"].to_numpy(dtype=float)
    z = (y - y.mean()) / y.std(ddof=0)
    lag_z = np.asarray(weights.sparse @ z).ravel()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(z, lag_z, s=28, alpha=0.8, color="#2563eb", edgecolor="white", linewidth=0.4)
    slope, intercept = np.polyfit(z, lag_z, 1)
    x_limit = np.max(np.abs(z)) * 1.08
    y_limit = np.max(np.abs(lag_z)) * 1.08
    xs = np.linspace(-x_limit, x_limit, 200)
    ax.plot(xs, slope * xs + intercept, color="#dc2626", linewidth=2)
    ax.axhline(0, color="#666666", linewidth=1, linestyle="--")
    ax.axvline(0, color="#666666", linewidth=1, linestyle="--")
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{year}年高新技术企业数 Moran 散点图", fontsize=14)
    ax.set_xlabel("标准化高新技术企业数")
    ax.set_ylabel("空间滞后值")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="整理高新技术企业 CSV，并计算 Moran's I。")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/china/高新技术企业数_标准化.csv"),
        help="高新技术企业 CSV 文件路径。",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("data/china/地级市数据.csv"),
        help="用于补充行政区划代码和省份的参考 CSV。",
    )
    parser.add_argument(
        "--shp",
        type=Path,
        default=Path("data/china/city.shp"),
        help="地级市矢量边界文件路径。",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help="分析年份；若原始高新技术企业表没有年份，将写入该年份。",
    )
    parser.add_argument(
        "--weights",
        choices=["queen", "knn"],
        default="queen",
        help="空间权重构建方式，默认 queen。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hightech_moran"),
        help="图表和结果文件输出目录。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="置换检验随机种子，默认 42。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    standardized_df, original_df = standardize_hightech_csv(
        raw_csv_path=args.csv,
        reference_panel_path=args.reference_csv,
        output_csv_path=args.csv,
        year=args.year,
    )

    gdf_all, gdf, city_col, code_col, hightech_col, target_year = prepare_dataset(
        args.shp,
        args.csv,
        args.year,
    )

    weights = build_weights(gdf, args.weights)
    values = gdf["hightech_value"].to_numpy(dtype=float)
    moran = Moran(values, weights, permutations=999)

    summary = pd.DataFrame(
        [
            {
                "year": target_year,
                "n_cities": len(gdf),
                "weights_method": args.weights,
                "moran_i": moran.I,
                "expected_i": moran.EI,
                "z_score": moran.z_sim,
                "p_value": moran.p_sim,
                "seed": args.seed,
                "hightech_column": hightech_col,
                "city_column": city_col,
                "code_column": code_col,
                "standardized_rows": len(standardized_df),
                "raw_rows": len(original_df),
            }
        ]
    )
    summary_path = args.output_dir / "moran_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    merged_path = args.output_dir / "hightech_moran_merged.geojson"
    gdf.to_file(merged_path, driver="GeoJSON", encoding="utf-8")

    plot_choropleth(
        gdf_all,
        gdf,
        args.output_dir / "hightech_spatial_distribution.png",
        target_year,
        hightech_col,
    )
    plot_moran_simulation(
        moran,
        args.output_dir / "moran_simulation_histogram.png",
        target_year,
    )
    plot_moran_scatter(
        gdf,
        weights,
        args.output_dir / "moran_scatter.png",
        target_year,
    )

    print("=" * 60)
    print("高新技术企业数空间自相关分析完成")
    print("=" * 60)
    print(f"分析年份: {target_year}")
    print(f"标准化后城市数: {len(standardized_df)}")
    print(f"参与 Moran's I 的城市数: {len(gdf)}")
    print(f"企业数字段: {hightech_col}")
    print(f"权重方法: {args.weights}")
    print(f"Moran's I: {moran.I:.6f}")
    print(f"Expected I: {moran.EI:.6f}")
    print(f"Z-score: {moran.z_sim:.6f}")
    print(f"P-value: {moran.p_sim:.6f}")
    print(f"标准化 CSV: {args.csv}")
    print(f"结果汇总: {summary_path}")
    print(f"空间分布图: {args.output_dir / 'hightech_spatial_distribution.png'}")
    print(f"置换检验图: {args.output_dir / 'moran_simulation_histogram.png'}")
    print(f"Moran 散点图: {args.output_dir / 'moran_scatter.png'}")
    print(f"合并后空间数据: {merged_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
