# -*- coding: utf-8 -*-
"""
Excel 处理服务
支持智能列识别和批量坐标转换
"""
import os
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd

# 延迟导入智谱 AI
try:
    from zhipuai import ZhipuAI
    ZHIPU_AVAILABLE = True
except ImportError:
    ZHIPU_AVAILABLE = False

from backend.services.convert_service import CoordTransformService
from backend.models.schemas import CoordSystem, TaskStatus


class ExcelProcessingService:
    """Excel 处理服务类"""

    def __init__(self):
        # 加载环境变量
        env_path = Path(__file__).parent.parent.parent / ".env"
        load_dotenv(env_path)

        self.api_key = os.getenv("ZHIPU_API_KEY")
        if self.api_key and ZHIPU_AVAILABLE:
            try:
                self.ai_client = ZhipuAI(api_key=self.api_key)
            except Exception:
                self.ai_client = None
        else:
            self.ai_client = None

        # 存储任务状态的字典（生产环境应使用 Redis）
        self.tasks = {}

        # 文件路径配置
        self.upload_dir = Path(__file__).parent.parent.parent / "uploads"
        self.export_dir = Path(__file__).parent.parent.parent / "exports"

        # 确保目录存在
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    async def analyze_excel_columns(self, file_path: str) -> dict:
        """
        使用 AI 分析 Excel 文件，识别经纬度列

        Args:
            file_path: Excel 文件路径

        Returns:
            包含识别结果的字典
        """
        try:
            # 读取前 3 行数据用于分析
            df = pd.read_excel(file_path, nrows=3)

            # 获取列名
            columns = df.columns.tolist()

            # 构建提示词
            sample_data = df.head(3).to_dict()

            prompt = f"""请分析以下 Excel 数据，识别哪一列是经度 (Longitude)，哪一列是纬度 (Latitude)。

列名: {columns}

样本数据:
{json_format(sample_data)}

请返回 JSON 格式结果:
{{
    "longitude_column": "经度列名",
    "latitude_column": "纬度列名",
    "confidence": "识别置信度（high/medium/low）",
    "reason": "识别理由"
}}

注意：
- 经度通常在 -180 到 180 之间
- 纬度通常在 -90 到 90 之间
- 常见的列名如：经度/longitude/lon/lng/_lng/经度，纬度/latitude/lat/lat/_lat/纬度
- 如果列名是中文，请返回中文列名"""

            if self.ai_client:
                response = self.ai_client.chat.completions.create(
                    model="glm-4",
                    messages=[
                        {"role": "system", "content": "你是一个专业的数据分析助手，擅长识别表格中的地理坐标数据。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )

                result = response.choices[0].message.content

                # 尝试解析 JSON
                import json
                try:
                    # 提取 JSON 部分（处理可能的前后缀）
                    if "```json" in result:
                        result = result.split("```json")[1].split("```")[0].strip()
                    elif "```" in result:
                        result = result.split("```")[1].split("```")[0].strip()

                    analysis = json.loads(result)

                    # 验证列名是否存在
                    if analysis.get("longitude_column") not in columns:
                        analysis["longitude_column"] = None
                    if analysis.get("latitude_column") not in columns:
                        analysis["latitude_column"] = None

                    return {
                        "success": True,
                        "columns": columns,
                        "sample_data": sample_data,
                        "analysis": analysis
                    }

                except json.JSONDecodeError:
                    return {
                        "success": False,
                        "error": "AI 返回结果解析失败",
                        "raw_response": result
                    }
            else:
                # 没有 AI 时的回退方案：简单规则匹配
                lng_col = self._find_column_by_name(columns, ["经度", "longitude", "lon", "lng", "_lng"])
                lat_col = self._find_column_by_name(columns, ["纬度", "latitude", "lat", "_lat"])

                return {
                    "success": True,
                    "columns": columns,
                    "sample_data": sample_data,
                    "analysis": {
                        "longitude_column": lng_col,
                        "latitude_column": lat_col,
                        "confidence": "low",
                        "reason": "基于规则匹配（无 AI 服务）"
                    }
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _find_column_by_name(self, columns: list, keywords: list) -> Optional[str]:
        """根据关键词查找列名"""
        for col in columns:
            col_lower = str(col).lower()
            for keyword in keywords:
                if keyword.lower() in col_lower:
                    return col
        return None

    def create_batch_task(
        self,
        file_path: str,
        longitude_column: str,
        latitude_column: str,
        from_coord: CoordSystem,
        to_coord: CoordSystem
    ) -> str:
        """
        创建批量转换任务

        Returns:
            任务 ID
        """
        task_id = str(uuid.uuid4())

        self.tasks[task_id] = {
            "task_id": task_id,
            "status": TaskStatus.PENDING,
            "total_rows": 0,
            "processed_rows": 0,
            "file_path": file_path,
            "longitude_column": longitude_column,
            "latitude_column": latitude_column,
            "from_coord": from_coord,
            "to_coord": to_coord,
            "download_urls": None,
            "error": None,
            "created_at": datetime.now()
        }

        return task_id

    async def process_batch_task(self, task_id: str, progress_callback=None):
        """
        处理批量转换任务

        Args:
            task_id: 任务 ID
            progress_callback: 进度回调函数
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task not found: {task_id}")

        task = self.tasks[task_id]
        task["status"] = TaskStatus.PROCESSING

        try:
            # 读取 Excel 文件
            df = pd.read_excel(task["file_path"])
            task["total_rows"] = len(df)

            # 创建结果列
            result_lng_col = f"{task['to_coord'].value}_lng"
            result_lat_col = f"{task['to_coord'].value}_lat"

            df[result_lng_col] = None
            df[result_lat_col] = None

            # 逐行转换
            for idx, row in df.iterrows():
                try:
                    lng = float(row[task["longitude_column"]])
                    lat = float(row[task["latitude_column"]])

                    converted = CoordTransformService.convert(
                        lng, lat,
                        task["from_coord"],
                        task["to_coord"]
                    )

                    df.at[idx, result_lng_col] = converted[0]
                    df.at[idx, result_lat_col] = converted[1]

                except (ValueError, TypeError) as e:
                    # 标记转换失败的行
                    df.at[idx, result_lng_col] = None
                    df.at[idx, result_lat_col] = None

                task["processed_rows"] = idx + 1

                # 触发进度回调
                if progress_callback:
                    await progress_callback(task_id, idx + 1, task["total_rows"])

                # 每 100 行让出控制权
                if (idx + 1) % 100 == 0:
                    await asyncio.sleep(0)

            # 生成导出文件
            base_name = Path(task["file_path"]).stem
            export_files = {}

            # 1. Excel 格式
            excel_path = self.export_dir / f"{base_name}_converted_{task_id[:8]}.xlsx"
            df.to_excel(excel_path, index=False)
            export_files["excel"] = str(excel_path)

            # 2. GeoJSON 格式
            geojson_path = self.export_dir / f"{base_name}_converted_{task_id[:8]}.geojson"
            self._export_geojson(df, result_lng_col, result_lat_col, geojson_path)
            export_files["geojson"] = str(geojson_path)

            # 3. KML 格式
            kml_path = self.export_dir / f"{base_name}_converted_{task_id[:8]}.kml"
            self._export_kml(df, result_lng_col, result_lat_col, kml_path)
            export_files["kml"] = str(kml_path)

            task["download_urls"] = export_files
            task["status"] = TaskStatus.COMPLETED

            # 设置 24 小时后过期
            task["expires_at"] = datetime.now() + timedelta(hours=24)

        except Exception as e:
            task["status"] = TaskStatus.FAILED
            task["error"] = str(e)

    def _export_geojson(self, df: pd.DataFrame, lng_col: str, lat_col: str, output_path: Path):
        """导出为 GeoJSON 格式"""
        features = []

        for idx, row in df.iterrows():
            lng = row.get(lng_col)
            lat = row.get(lat_col)

            if pd.notna(lng) and pd.notna(lat):
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(lng), float(lat)]
                    },
                    "properties": {k: v for k, v in row.items()
                                 if k not in [lng_col, lat_col] and pd.notna(v)}
                }
                features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)

    def _export_kml(self, df: pd.DataFrame, lng_col: str, lat_col: str, output_path: Path):
        """导出为 KML 格式"""
        kml_template = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Converted Coordinates</name>
    <description>Coordinates exported from Geo-AI Expert System</description>
{placemarks}
  </Document>
</kml>'''

        placemarks = []
        for idx, row in df.iterrows():
            lng = row.get(lng_col)
            lat = row.get(lat_col)

            if pd.notna(lng) and pd.notna(lat):
                props = "\n".join([f"      {k}: {v}" for k, v in row.items()
                                  if k not in [lng_col, lat_col] and pd.notna(v)])

                placemark = f'''    <Placemark>
      <name>Point {idx + 1}</name>
      <description>
{props}
      </description>
      <Point>
        <coordinates>{lng},{lat},0</coordinates>
      </Point>
    </Placemark>'''
                placemarks.append(placemark)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(kml_template.format(placemarks="\n".join(placemarks)))

    def _generate_geo_report(self, df: pd.DataFrame, lng_col: str, lat_col: str) -> str:
        """生成地理分析报告"""
        from backend.services.coordinate_validator import CoordinateValidator

        lines = []
        lines.append("=" * 50)
        lines.append("地理坐标分析报告")
        lines.append("=" * 50)
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 基本信息
        total_rows = len(df)
        lines.append("📊 数据统计")
        lines.append(f"  总行数: {total_rows}")
        lines.append(f"  经度列: {lng_col}")
        lines.append(f"  纬度列: {lat_col}")
        lines.append("")

        # 坐标验证
        lines.append("🔍 坐标验证")
        valid_coords = []
        for idx, row in df.iterrows():
            try:
                lng = float(row[lng_col])
                lat = float(row[lat_col])
                if not math.isnan(lng) and not math.isnan(lat):
                    valid_coords.append({'lng': lng, 'lat': lat})
            except (ValueError, TypeError):
                pass

        validation = CoordinateValidator.validate_batch(valid_coords)
        lines.append(f"  有效坐标: {validation['valid']}")
        lines.append(f"  异常坐标: {validation['invalid']}")
        lines.append("")

        # 统计信息
        if validation.get('statistics'):
            stats = validation['statistics']
            lines.append("📍 分布统计")
            lines.append(f"  经度范围: {stats['lng_range']['min']:.4f} ~ {stats['lng_range']['max']:.4f}")
            lines.append(f"  纬度范围: {stats['lat_range']['min']:.4f} ~ {stats['lat_range']['max']:.4f}")
            lines.append(f"  中心点: ({stats['lng_center']:.4f}, {stats['lat_center']:.4f})")
            lines.append("")

        # 地理区域推断
        lines.append("🌍 区域分析")
        center_lat = validation['statistics']['lat_center']
        center_lng = validation['statistics']['lng_center']

        # 粗略判断所在城市
        city_hints = []
        if 30.5 < center_lat < 32.5 and 119 < center_lng < 121:
            city_hints.append("可能在苏州市附近")
        if 39.5 < center_lat < 41.5 and 115.5 < center_lng < 117.5:
            city_hints.append("可能在北京附近")
        if 30.5 < center_lat < 31.5 and 120 < center_lng < 121.5:
            city_hints.append("可能在杭州附近")

        if city_hints:
            for hint in city_hints:
                lines.append(f"  - {hint}")

        lines.append("")
        lines.append("💡 建议")
        if validation['invalid'] > 0:
            invalid_rate = validation['invalid'] / total_rows * 100
            if invalid_rate > 10:
                lines.append(f"  ⚠️  有 {validation['invalid']} 个异常坐标（{invalid_rate:.1f}%），请检查数据")
                lines.append(f"  建议：核对原始坐标格式，确保经度在前，纬度在后")
            else:
                lines.append(f"  ✓ 数据质量良好，异常坐标占比 {invalid_rate:.1f}%")
        else:
            lines.append("  ✓ 所有坐标格式正确")

        lines.append("")
        lines.append("=" * 50)

        return "\n".join(lines)

    def get_task_status(self, task_id: str) -> Optional[dict]:
        """获取任务状态"""
        if task_id not in self.tasks:
            return None

        task = self.tasks[task_id].copy()

        # 清理过期任务
        if "expires_at" in task and task["expires_at"] < datetime.now():
            self._cleanup_task(task_id)
            return None

        return task

    def _cleanup_task(self, task_id: str):
        """清理任务和相关文件"""
        if task_id in self.tasks:
            task = self.tasks[task_id]

            # 删除导出文件
            if task.get("download_urls"):
                for file_path in task["download_urls"].values():
                    try:
                        Path(file_path).unlink(missing_ok=True)
                    except Exception:
                        pass

            # 删除任务记录
            del self.tasks[task_id]

    def cleanup_expired_tasks(self):
        """清理所有过期任务"""
        now = datetime.now()
        expired_ids = [
            task_id for task_id, task in self.tasks.items()
            if task.get("expires_at") and task["expires_at"] < now
        ]

        for task_id in expired_ids:
            self._cleanup_task(task_id)


def json_format(data: dict) -> str:
    """格式化字典数据为可读字符串"""
    import json
    return json.dumps(data, ensure_ascii=False, indent=2)


# 导出便捷函数
def get_excel_service() -> ExcelProcessingService:
    """获取 Excel 处理服务实例"""
    return ExcelProcessingService()
