# -*- coding: utf-8 -*-
"""
坐标验证服务 - 纠错专家
"""
import math
from typing import Dict, List, Any, Optional


class CoordinateValidator:
    """坐标验证器 - 纠错专家"""

    # 中国大陆边界范围（粗略）
    CHINA_BOUNDS = {
        'lng_min': 72.004,
        'lng_max': 137.8347,
        'lat_min': 0.8293,
        'lat_max': 55.8271
    }

    # 中国主要城市的大致坐标范围，用于验证
    CITY_REGIONS = {
        '北京': {'lng': (115.7, 117.4), 'lat': (39.4, 41.05)},
        '上海': {'lng': (120.8, 122.2), 'lat': (30.7, 31.8)},
        '广州': {'lng': (113.0, 114.5), 'lat': (22.5, 24.0)},
        '深圳': {'lng': (113.7, 114.7), 'lat': (22.4, 22.9)},
        '苏州': {'lng': (119.5, 121.5), 'lat': (30.5, 32.0)},
        '南京': {'lng': (118.0, 119.5), 'lat': (31.0, 32.5)},
        '杭州': {'lng': (119.5, 120.9), 'lat': (29.8, 30.6)},
        '成都': {'lng': (103.5, 104.9), 'lat': (30.0, 30.8)},
        '武汉': {'lng': (113.6, 115.1), 'lat': (29.8, 31.0)},
        '西安': {'lng': (107.5, 109.5), 'lat': (33.5, 34.5)},
        '天津': {'lng': (116.8, 118.2), 'lat': (38.5, 40.5)},
    }

    @classmethod
    def validate_coordinate(cls, lng: float, lat: float) -> Dict[str, Any]:
        """
        验证单个坐标，返回验证结果

        Returns:
            {
                'valid': bool,
                'issues': list[str],
                'warnings': list[str],
                'suggestions': list[str]
            }
        """
        issues = []
        warnings = []
        suggestions = []

        # 1. 检查基本范围
        if lng < -180 or lng > 180:
            issues.append(f"经度超出有效范围 (-180 到 180)，当前值: {lng}")
            suggestions.append("经度应在 -180° 到 180° 之间")

        if lat < -90 or lat > 90:
            issues.append(f"纬度超出有效范围 (-90 到 90)，当前值: {lat}")
            suggestions.append("纬度应在 -90° 到 90° 之间")

        # 2. 检查是否在境外
        if (cls.CHINA_BOUNDS['lng_min'] <= lng <= cls.CHINA_BOUNDS['lng_max'] and
            cls.CHINA_BOUNDS['lat_min'] <= lat <= cls.CHINA_BOUNDS['lat_max']):
            # 在中国境内，检查是否在合理区域内
            if lat < 18:
                warnings.append("坐标位于南海海域附近")
            # 检查是否在海里（粗略判断，远离大陆）
            if cls._is_in_ocean(lng, lat):
                issues.append("坐标位于海域，可能不在中国陆地上")
                suggestions.append("请确认坐标是否正确，经纬度可能写反了")
        else:
            issues.append("坐标位于中国境外")
            suggestions.append("境外坐标各坐标系一致，不需要转换")

        # 3. 检查常见错误模式
        cls._check_common_errors(lng, lat, issues, suggestions)

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'suggestions': suggestions,
            'location_type': cls._get_location_type(lng, lat)
        }

    @classmethod
    def validate_batch(cls, coordinates: List[Dict]) -> Dict[str, Any]:
        """
        批量验证坐标，并生成分析报告

        Args:
            coordinates: 坐标列表 [{'lng': x, 'lat': y}, ...]

        Returns:
            {
                'total': int,
                'valid': int,
                'invalid': int,
                'issues': list,
                'summary': str,
                'statistics': dict
            }
        """
        total = len(coordinates)
        valid_count = 0
        invalid_list = []
        warning_list = []

        all_lats = []
        all_lngs = []

        for idx, coord in enumerate(coordinates):
            lng = coord.get('lng')
            lat = coord.get('lat')

            if lng is None or lat is None:
                invalid_list.append({'index': idx, 'error': '坐标缺失'})
                continue

            try:
                lng = float(lng)
                lat = float(lat)
                all_lngs.append(lng)
                all_lats.append(lat)

                result = cls.validate_coordinate(lng, lat)

                if result['valid']:
                    valid_count += 1
                    if result['warnings']:
                        warning_list.append({
                            'index': idx,
                            'lng': lng,
                            'lat': lat,
                            'warnings': result['warnings']
                        })
                else:
                    invalid_list.append({
                        'index': idx,
                        'lng': lng,
                        'lat': lat,
                        'issues': result['issues']
                    })
            except (ValueError, TypeError):
                invalid_list.append({'index': idx, 'error': '坐标格式错误'})

        # 生成统计报告
        statistics = cls._generate_statistics(all_lngs, all_lats, valid_count, total)

        # 生成总结
        summary = cls._generate_summary(total, valid_count, invalid_list, warning_list, statistics)

        return {
            'total': total,
            'valid': valid_count,
            'invalid': len(invalid_list),
            'invalid_list': invalid_list,
            'warning_list': warning_list,
            'issues': invalid_list,
            'summary': summary,
            'statistics': statistics
        }

    @classmethod
    def _is_in_ocean(cls, lng: float, lat: float) -> bool:
        """粗略判断坐标是否在海域"""
        # 远离大陆的区域
        china_center_lng = 104.0
        china_center_lat = 35.0

        distance = math.sqrt((lng - china_center_lng)**2 + (lat - china_center_lat)**2)

        # 距离中国中心超过 15 度的可能是海域
        return distance > 15

    @classmethod
    def _check_common_errors(cls, lng: float, lat: float, issues: List, suggestions: List):
        """检查常见的坐标错误"""
        # 检查是否可能是经纬度写反
        if 70 <= lat <= 90 and 0 <= lng <= 180:
            issues.append("纬度异常高，经纬度可能写反了")
            suggestions.append(f"当前坐标 ({lng}, {lat})，请检查是否应该为 ({lat}, {lng})")

        # 检查是否在小数点位置错误
        if abs(lng) > 1000 or abs(lat) > 1000:
            issues.append("坐标值异常大，可能单位错误")
            suggestions.append("请确认使用十进制度数，如 116.404, 39.915")

    @classmethod
    def _get_location_type(cls, lng: float, lat: float) -> str:
        """判断坐标位置类型"""
        if cls.CHINA_BOUNDS['lng_min'] <= lng <= cls.CHINA_BOUNDS['lng_max']:
            if cls.CHINA_BOUNDS['lat_min'] <= lat <= cls.CHINA_BOUNDS['lat_max']:
                return "中国境内"
        return "境外"

    @classmethod
    def _generate_statistics(cls, lngs: List, lats: List, valid_count: int, total: int) -> Dict:
        """生成统计信息"""
        if not lngs:
            return {}

        lngs_sorted = sorted(lngs)
        lats_sorted = sorted(lats)

        return {
            'lng_range': {'min': lngs_sorted[0], 'max': lngs_sorted[-1]},
            'lat_range': {'min': lats_sorted[0], 'max': lats_sorted[-1]},
            'lng_center': (lngs_sorted[0] + lngs_sorted[-1]) / 2,
            'lat_center': (lats_sorted[0] + lats_sorted[-1]) / 2,
            'valid_rate': f"{valid_count / total * 100:.1f}%" if total > 0 else "0%"
        }

    @classmethod
    def _generate_summary(cls, total: int, valid: int, invalid: List,
                         warnings: List, stats: Dict) -> str:
        """生成可读的总结报告"""
        lines = ["📊 坐标验证报告", ""]

        # 总体情况
        lines.append(f"总计: {total} 个坐标点")
        lines.append(f"✅ 有效: {valid} 个")
        lines.append(f"❌ 异常: {len(invalid)} 个")

        if warnings:
            lines.append(f"⚠️  警告: {len(warnings)} 个")

        lines.append("")

        # 异常详情
        if invalid:
            lines.append("🔍 异常坐标:")
            for item in invalid[:5]:  # 只显示前5个
                if 'error' in item:
                    lines.append(f"  - 第{item['index']+1}个: {item['error']}")
                else:
                    lines.append(f"  - 第{item['index']+1}个: {item['lng']}, {item['lat']}")
                    if item.get('issues'):
                        lines.append(f"    问题: {item['issues'][0]}")
            if len(invalid) > 5:
                lines.append(f"  ... 还有 {len(invalid)-5} 个异常坐标")
            lines.append("")

        # 统计信息
        if stats.get('lng_range'):
            lines.append("📍 坐标分布:")
            lines.append(f"  经度范围: {stats['lng_range']['min']:.2f} ~ {stats['lng_range']['max']:.2f}")
            lines.append(f"  纬度范围: {stats['lat_range']['min']:.2f} ~ {stats['lat_range']['max']:.2f}")
            lines.append(f"  中心点: ({stats['lng_center']:.2f}, {stats['lat_center']:.2f})")
            lines.append(f"  有效率: {stats['valid_rate']}")
            lines.append("")

        # 智能建议
        lines.append("💡 智能建议:")
        if invalid:
            swap_suggestions = []
            for item in invalid:
                if '写反' in str(item.get('issues', [])):
                    swap_suggestions.append(item)
            if swap_suggestions:
                lines.append("  - 部分坐标经纬度可能写反，AI 可以帮你自动调换")
            lines.append("  - 建议仔细核对原始坐标数据")

        return "\n".join(lines)


def get_validator() -> CoordinateValidator:
    """获取验证器实例"""
    return CoordinateValidator()
