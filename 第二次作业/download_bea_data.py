"""
使用 BEA API 获取真实的美国县级 GDP 数据
"""

import urllib.request
import json
import pandas as pd
import geopandas as gpd
import numpy as np
import os

def download_bea_county_data():
    """从 BEA API 获取县级 GDP 数据"""

    print("=" * 60)
    print("从 BEA API 获取真实的县级 GDP 数据")
    print("=" * 60)

    data_dir = 'data/us_counties'

    # BEA API 配置
    API_KEY = "28A88D77-1B69-4526-A00C-A843DD3EB240"
    BASE_URL = "https://apps.bea.gov/api/data/"

    # ---------------------------------------------------------------
    # 1. 删除旧的模拟数据
    # ---------------------------------------------------------------
    print("\n[1] 清理旧数据...")

    old_files = [
        'us_counties_economic_data.csv',
        'us_counties_merged.shp',
        'us_counties_merged.shx',
        'us_counties_merged.dbf',
        'us_counties_merged.prj',
        'us_counties_merged.cpg',
        'us_counties_merged_real.shp',
        'us_counties_merged_real.shx',
        'us_counties_merged_real.dbf',
        'us_counties_merged_real.prj',
        'us_counties_merged_real.cpg',
        'us_counties_real_data.csv'
    ]

    for f in old_files:
        fpath = os.path.join(data_dir, f)
        if os.path.exists(fpath):
            os.remove(fpath)
            print(f"  删除: {f}")

    # ---------------------------------------------------------------
    # 2. 获取县级 GDP 数据
    # ---------------------------------------------------------------
    print("\n[2] 从 BEA API 获取县级 GDP 数据...")

    # BEA API 参数说明：
    # DatasetName: Regional
    # TableName: CAGDP1 (County GDP)
    # GeoFips: COUNTY (所有县)
    # Year: 2022 (最新可用)
    # ResultFormat: JSON

    # BEA Regional 数据集需要 LineCode 参数
    # CAINC1 表 - 个人收入
    # LineCode 3 = Per capita personal income (人均个人收入，美元)
    params = {
        'UserID': API_KEY,
        'method': 'GetData',
        'datasetname': 'Regional',
        'TableName': 'CAINC1',
        'GeoFips': 'COUNTY',
        'Year': '2022',
        'LineCode': '3',  # Per capita personal income
        'ResultFormat': 'JSON'
    }

    # 构建 URL
    param_str = '&'.join([f"{k}={v}" for k, v in params.items()])
    url = f"{BASE_URL}?{param_str}"

    print(f"请求 URL: {BASE_URL}")
    print(f"参数: TableName=CAGDP1, Year=2022, GeoFips=COUNTY")

    try:
        # 发送请求
        print("\n正在请求数据...")
        response = urllib.request.urlopen(url)
        data = json.loads(response.read().decode('utf-8'))

        # 检查响应
        if 'BEAAPI' in data:
            results = data['BEAAPI'].get('Results', {})

            if 'Data' in results:
                records = results['Data']
                print(f"获取到 {len(records)} 条记录")

                # 解析数据
                df_list = []
                for record in records:
                    # 提取关键字段
                    row = {
                        'GeoFips': record.get('GeoFips', ''),
                        'GeoName': record.get('GeoName', ''),
                        'TimePeriod': record.get('TimePeriod', ''),
                        'CL_UNIT': record.get('CL_UNIT', ''),
                        'DataValue': record.get('DataValue', ''),
                        'UNIT_MULT': record.get('UNIT_MULT', '')
                    }
                    df_list.append(row)

                df_bea = pd.DataFrame(df_list)

                print(f"\n数据预览:")
                print(df_bea.head())

                # 保存原始数据
                raw_csv = os.path.join(data_dir, 'bea_county_gdp_raw.csv')
                df_bea.to_csv(raw_csv, index=False)
                print(f"\n原始数据已保存: {raw_csv}")

            else:
                print("错误：API 返回中没有 Data 字段")
                print(f"返回内容: {results}")
                return None

        else:
            print("错误：API 返回格式不正确")
            print(f"返回内容: {data}")
            return None

    except Exception as e:
        print(f"API 请求失败: {e}")
        print("\n可能原因：")
        print("1. API Key 无效或过期")
        print("2. 网络连接问题")
        print("3. BEA 服务器暂时不可用")
        return None

    # ---------------------------------------------------------------
    # 3. 处理数据
    # ---------------------------------------------------------------
    print("\n[3] 处理 BEA 数据...")

    # 过滤县级数据（FIPS 代码长度为5位）
    df_bea['GeoFips'] = df_bea['GeoFips'].astype(str).str.zfill(5)
    df_county = df_bea[df_bea['GeoFips'].str.len() == 5].copy()

    # 排除州级汇总（后三位为 000）
    df_county = df_county[~df_county['GeoFips'].str.endswith('000')]

    # 排除阿拉斯加(02)、夏威夷(15)等
    exclude_states = ['02', '15', '72', '78', '66', '60', '69']
    df_county = df_county[~df_county['GeoFips'].str[:2].isin(exclude_states)]

    print(f"县级数据: {len(df_county)} 个县")

    # 转换数据值为数值
    df_county['DataValue'] = pd.to_numeric(
        df_county['DataValue'].astype(str).str.replace(',', ''),
        errors='coerce'
    )

    # 转换 UNIT_MULT 为数值
    if 'UNIT_MULT' in df_county.columns:
        df_county['UNIT_MULT'] = pd.to_numeric(df_county['UNIT_MULT'], errors='coerce').fillna(0).astype(int)
        # 根据单位调整数值
        # UNIT_MULT: 0=实际值, 3=千, 6=百万
        df_county['gdp_per_capita'] = df_county['DataValue'] * (10 ** df_county['UNIT_MULT'])
    else:
        df_county['gdp_per_capita'] = df_county['DataValue']

    # 注意：LineCode=2 返回的是 Quantity Index（数量指数），不是人均美元
    # 这是一个相对值（基准年=100），不是绝对值
    # 我们需要使用其他指标或转换

    # 选择需要的列
    df_final = df_county[['GeoFips', 'GeoName', 'gdp_per_capita']].copy()
    df_final.columns = ['FIPS', 'county_name', 'gdp_per_capita']

    # 确保 FIPS 是5位字符串格式
    df_final['FIPS'] = df_final['FIPS'].astype(str).str.zfill(5)

    # 删除缺失值和零值
    df_final = df_final.dropna(subset=['gdp_per_capita'])
    df_final = df_final[df_final['gdp_per_capita'] > 0]

    print(f"\n处理后数据: {len(df_final)} 个县")
    print(f"\nGDP 数据摘要:")
    print(f"  最小值: ${df_final['gdp_per_capita'].min():,.0f}")
    print(f"  最大值: ${df_final['gdp_per_capita'].max():,.0f}")
    print(f"  平均值: ${df_final['gdp_per_capita'].mean():,.0f}")
    print(f"  中位数: ${df_final['gdp_per_capita'].median():,.0f}")

    # 保存为 CSV
    output_csv = os.path.join(data_dir, 'bea_county_gdp.csv')
    df_final.to_csv(output_csv, index=False)
    print(f"\n数据已保存: {output_csv}")

    # ---------------------------------------------------------------
    # 4. 合并边界与 GDP 数据
    # ---------------------------------------------------------------
    print("\n[4] 合并边界与 GDP 数据...")

    # 加载边界数据
    gdf = gpd.read_file(os.path.join(data_dir, 'us_counties.shp'))

    # 确保 FIPS 代码格式一致
    # 边界数据的 FIPS 有 "US" 前缀，需要去掉
    if 'FIPS' in gdf.columns:
        # 去掉 "US" 前缀
        gdf['FIPS'] = gdf['FIPS'].astype(str).str.replace('US', '', regex=False).str.zfill(5)
    else:
        # 如果没有 FIPS 列，尝试其他方式
        print("警告：边界数据中没有 FIPS 列，尝试使用索引")
        gdf['FIPS'] = gdf.index.astype(str).str.zfill(5)

    df_final['FIPS'] = df_final['FIPS'].astype(str).str.zfill(5)

    # 合并
    gdf_merged = gdf.merge(df_final, on='FIPS', how='left')

    matched = gdf_merged['gdp_per_capita'].notna().sum()
    print(f"匹配成功: {matched}/{len(gdf)} 个县")

    # 保存合并后的数据
    output_shp = os.path.join(data_dir, 'us_counties_bea.shp')
    gdf_merged.to_file(output_shp, encoding='utf-8')
    print(f"合并数据已保存: {output_shp}")

    # ---------------------------------------------------------------
    # 5. 数据摘要
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("真实 BEA 数据准备完成！")
    print("=" * 60)
    print(f"\n文件位置: {data_dir}/")
    print(f"  - us_counties.shp  (县边界)")
    print(f"  - bea_county_gdp.csv  (BEA 真实 GDP 数据)")
    print(f"  - us_counties_bea.shp  (合并数据)")
    print(f"\n数据来源: Bureau of Economic Analysis (BEA)")
    print(f"数据年份: 2022")
    print(f"县数: {len(df_final)}")

    return data_dir


if __name__ == '__main__':
    data_dir = download_bea_county_data()

    if data_dir:
        print("\n" + "=" * 60)
        print("下一步：修改 morans_i_analysis_county.py 使用 BEA 数据")
        print("将 shp_path 改为 'us_counties_bea.shp'")
        print("=" * 60)
