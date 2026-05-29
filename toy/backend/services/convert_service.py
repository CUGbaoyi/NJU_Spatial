# -*- coding: utf-8 -*-
"""
坐标转换服务
基于 coordtransform_utils.py 实现核心转换逻辑
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from coordtransform_utils import (
    wgs_to_gcj, gcj_to_wgs,
    gcj_to_bd, bd_to_gcj,
    wgs_to_bd, bd_to_wgs,
    Transform,
    out_of_china
)
from backend.models.schemas import CoordSystem


class CoordTransformService:
    """坐标转换服务类"""

    # 转换函数映射表
    TRANSFORM_MAP = {
        (CoordSystem.WGS84, CoordSystem.GCJ02): wgs_to_gcj,
        (CoordSystem.GCJ02, CoordSystem.WGS84): gcj_to_wgs,
        (CoordSystem.GCJ02, CoordSystem.BD09): gcj_to_bd,
        (CoordSystem.BD09, CoordSystem.GCJ02): bd_to_gcj,
        (CoordSystem.WGS84, CoordSystem.BD09): wgs_to_bd,
        (CoordSystem.BD09, CoordSystem.WGS84): bd_to_wgs,
    }

    @classmethod
    def convert(
        cls,
        longitude: float,
        latitude: float,
        from_coord: CoordSystem,
        to_coord: CoordSystem
    ) -> tuple[float, float]:
        """
        执行坐标转换

        Args:
            longitude: 经度
            latitude: 纬度
            from_coord: 源坐标系
            to_coord: 目标坐标系

        Returns:
            转换后的坐标 (经度, 纬度)

        Raises:
            ValueError: 不支持的坐标系转换
        """
        # 相同坐标系直接返回
        if from_coord == to_coord:
            return longitude, latitude

        # 获取转换函数
        transform_key = (from_coord, to_coord)
        transform_func = cls.TRANSFORM_MAP.get(transform_key)

        if transform_func is None:
            # 尝试通过 GCJ02 中转
            if from_coord != CoordSystem.GCJ02 and to_coord != CoordSystem.GCJ02:
                # 例如: BD09 -> WGS84 通过 GCJ02 中转
                mid_lon, mid_lat = cls.TRANSFORM_MAP[(from_coord, CoordSystem.GCJ02)](longitude, latitude)
                return cls.TRANSFORM_MAP[(CoordSystem.GCJ02, to_coord)](mid_lon, mid_lat)
            raise ValueError(f"不支持的坐标系转换: {from_coord} -> {to_coord}")

        return transform_func(longitude, latitude)

    @classmethod
    def batch_convert(
        cls,
        coordinates: list[tuple[float, float]],
        from_coord: CoordSystem,
        to_coord: CoordSystem
    ) -> list[tuple[float, float]]:
        """
        批量坐标转换

        Args:
            coordinates: 坐标列表 [(lng1, lat1), (lng2, lat2), ...]
            from_coord: 源坐标系
            to_coord: 目标坐标系

        Returns:
            转换后的坐标列表
        """
        return [
            cls.convert(lng, lat, from_coord, to_coord)
            for lng, lat in coordinates
        ]

    @classmethod
    def is_out_of_china(cls, longitude: float, latitude: float) -> bool:
        """检查坐标是否在中国境外"""
        return out_of_china(longitude, latitude)

    @classmethod
    def get_transform_class(cls) -> Transform:
        """获取 Transform 类实例（面向对象接口）"""
        return Transform()


# 导出便捷函数
def convert_coord(
    longitude: float,
    latitude: float,
    from_coord: str,
    to_coord: str
) -> tuple[float, float]:
    """
    便捷的坐标转换函数

    Args:
        longitude: 经度
        latitude: 纬度
        from_coord: 源坐标系 ('wgs84', 'gcj02', 'bd09')
        to_coord: 目标坐标系 ('wgs84', 'gcj02', 'bd09')

    Returns:
        转换后的坐标 (经度, 纬度)
    """
    from_sys = CoordSystem(from_coord.lower())
    to_sys = CoordSystem(to_coord.lower())
    return CoordTransformService.convert(longitude, latitude, from_sys, to_sys)
