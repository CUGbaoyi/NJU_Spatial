# -*- coding: utf-8 -*-
"""
智谱 AI HTTP API 服务
增强版：支持地址查询、坐标转换、距离计算、状态管理、模糊意图解析
"""
import os
import httpx
import re
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import deque
from dotenv import load_dotenv

# 加载环境变量
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

from backend.services.convert_service import CoordTransformService
from backend.models.schemas import CoordSystem
from backend.services.coordinate_validator import CoordinateValidator


class SessionManager:
    """会话状态管理器 - 记录用户操作历史"""

    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.sessions: Dict[str, Dict] = {}

    def get_session(self, session_id: str) -> Dict:
        """获取或创建会话"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'markers': deque(maxlen=self.max_history),  # 存储标记历史
                'last_action': None,
                'created_at': datetime.now()
            }
        return self.sessions[session_id]

    def add_marker(self, session_id: str, lng: float, lat: float, label: str = ""):
        """添加标记到历史"""
        session = self.get_session(session_id)
        session['markers'].append({
            'lng': lng,
            'lat': lat,
            'label': label,
            'time': datetime.now()
        })

    def get_recent_markers(self, session_id: str, count: int = 10) -> List[Dict]:
        """获取最近的标记"""
        session = self.get_session(session_id)
        markers = list(session['markers'])
        return markers[-count:] if count else markers

    def clear_session(self, session_id: str):
        """清除会话历史"""
        if session_id in self.sessions:
            del self.sessions[session_id]


# 全局会话管理器
session_manager = SessionManager()


class ZhipuAIHTTPService:
    """智谱 AI HTTP API 服务类（增强版）"""

    def __init__(self):
        self.api_key = os.getenv("ZHIPU_API_KEY")
        self.amap_key = os.getenv("AMAP_API_KEY")
        self.base_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self.available = bool(self.api_key)
        self.model = os.getenv("ZHIPU_MODEL", "glm-4-plus")
        self.amap_geocode_url = "https://restapi.amap.com/v3/geocode/geo"
        self.amap_distance_url = "https://restapi.amap.com/v3/distance"

    # ==================== 工具函数 ====================

    def _calculate_distance(self, lng1: float, lat1: float, lng2: float, lat2: float) -> dict:
        """计算两点之间的距离（米）"""
        # Haversine 公式
        R = 6371000  # 地球半径（米）
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lng2 - lng1)

        a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c

        # 计算方位角
        y = math.sin(delta_lambda) * math.cos(phi2)
        x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
        bearing = math.atan2(y, x)
        bearing = math.degrees(bearing)
        if bearing < 0:
            bearing += 360

        return {
            "distance_m": round(distance, 2),
            "distance_km": round(distance / 1000, 2),
            "bearing": round(bearing, 2)
        }

    def _format_distance(self, distance_m: float) -> str:
        """格式化距离显示"""
        if distance_m < 1000:
            return f"{round(distance_m)} 米"
        elif distance_m < 100000:
            return f"{round(distance_m / 1000, 2)} 公里"
        else:
            return f"{round(distance_m / 1000)} 公里"

    async def call_amap_geocode(self, address: str, city: str = None) -> dict:
        """调用高德地理编码 API"""
        params = {"key": self.amap_key, "address": address}
        if city:
            params["city"] = city

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(self.amap_geocode_url, params=params)
                data = response.json()

                if data.get("status") == "1" and data.get("geocodes"):
                    geocode = data["geocodes"][0]
                    location = geocode["location"].split(",")
                    return {
                        "success": True,
                        "address": geocode.get("formatted_address", address),
                        "lng": float(location[0]),
                        "lat": float(location[1]),
                        "level": geocode.get("level"),
                        "adcode": geocode.get("adcode"),
                        "coord_system": "GCJ02"
                    }
                else:
                    return {"success": False, "error": data.get("info", "地址未找到")}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def call_amap_distance(self, origins: str, destination: str, type: str = "1") -> dict:
        """调用高德距离计算 API"""
        params = {
            "key": self.amap_key,
            "origins": origins,
            "destination": destination,
            "type": type  # 0:直线距离, 1:驾车距离, 3:步行距离
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(self.amap_distance_url, params=params)
                data = response.json()

                if data.get("status") == "1" and data.get("results"):
                    result = data["results"][0]
                    return {
                        "success": True,
                        "distance": int(result.get("distance", 0)),
                        "duration": int(result.get("duration", 0))
                    }
                else:
                    return {"success": False, "error": data.get("info", "距离计算失败")}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==================== AI 对话主函数 ====================

    async def chat(self, message: str, conversation_history: list = None) -> dict:
        """与 AI 对话"""
        if not self.available:
            return self._fallback_chat(message)

        messages = []

        # 系统提示词
        system_prompt = """你是一个专业的地理信息助手，精通中国地理、坐标系统和地图服务。

## 你可以处理的问题类型：

### 1. 📍 地址与坐标
- 查询城市/地点的经纬度
- 坐标转地址（逆地理编码）
- 判断坐标所属的坐标系

### 2. 🔄 坐标转换
- WGS84(GPS) ↔ GCJ02(高德/腾讯)
- WGS84/GCJ02 ↔ BD09(百度)
- 自动识别坐标系类型

### 3. 📏 距离计算
- 两点之间的直线距离
- 城市之间的距离
- 驾车/步行路线距离

### 4. 🧠 智能操作（高级功能）
- 模糊意图解析：理解"刚才那些点往北移50米"这类指令
- 坐标偏移计算：根据方向和距离计算新坐标
- 坐标纠错：检测异常坐标并主动提示

### 5. 📊 数据分析
- 点位分布分析
- 统计报告生成

## 坐标系知识：
- WGS84: 国际GPS标准，未加密
- GCJ02: 中国强制加密标准（高德/腾讯使用）
- BD09: 百度专用坐标系（在GCJ02基础上再次加密）

## 坐标纠错规则：
- 纬度 > 50° 且在中国范围内：可能经纬度写反
- 坐标在海域：主动询问是否需要调换
- 经纬度异常大（>1000）：提示单位错误

## 方向偏移计算（1度约111km）：
- 向北移1km：纬度 + 0.009
- 向南移1km：纬度 - 0.009
- 向东移1km：经度 + 0.01（近似，与纬度有关）
- 向西移1km：经度 - 0.01

请用友好、专业的方式回答，适当使用emoji让回复更生动。当检测到可能的坐标错误时，主动提醒用户。"""

        messages.append({"role": "system", "content": system_prompt})

        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": message})

        # 定义工具
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_address_coord",
                    "description": "查询地址的经纬度坐标。支持城市、街道、POI等地址查询。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "address": {"type": "string", "description": "地址名称，如'苏州市'、'故宫'"},
                            "city": {"type": "string", "description": "城市名（可选），如'苏州'"}
                        },
                        "required": ["address"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "convert_coords",
                    "description": "转换坐标系。支持WGS84(GPS)、GCJ02(高德)、BD09(百度)互转。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "lng": {"type": "number", "description": "经度"},
                            "lat": {"type": "number", "description": "纬度"},
                            "from": {"type": "string", "enum": ["wgs84", "gcj02", "bd09"], "description": "源坐标系"},
                            "to": {"type": "string", "enum": ["wgs84", "gcj02", "bd09"], "description": "目标坐标系"}
                        },
                        "required": ["lng", "lat", "from", "to"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_distance",
                    "description": "计算两点之间的直线距离（单位：米/公里）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "lng1": {"type": "number", "description": "第一点经度"},
                            "lat1": {"type": "number", "description": "第一点纬度"},
                            "lng2": {"type": "number", "description": "第二点经度"},
                            "lat2": {"type": "number", "description": "第二点纬度"}
                        },
                        "required": ["lng1", "lat1", "lng2", "lat2"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_city_distance",
                    "description": "计算两个城市之间的距离。会先查询城市坐标，再计算距离。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city1": {"type": "string", "description": "第一个城市名"},
                            "city2": {"type": "string", "description": "第二个城市名"}
                        },
                        "required": ["city1", "city2"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_recent_markers",
                    "description": "获取用户最近添加的标记点（用于模糊意图，如'刚才那些点'）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "count": {"type": "number", "description": "获取最近的标记数量", "default": 10}
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "shift_coordinates",
                    "description": "将坐标向指定方向偏移一定距离。用于处理'往北移50米'这类指令。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "lng": {"type": "number", "description": "原经度"},
                            "lat": {"type": "number", "description": "原纬度"},
                            "direction": {"type": "string", "enum": ["北", "南", "东", "西", "northeast", "northwest", "southeast", "southwest"], "description": "方向"},
                            "distance_meters": {"type": "number", "description": "偏移距离（米）"}
                        },
                        "required": ["lng", "lat", "direction", "distance_meters"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "validate_coordinate",
                    "description": "验证坐标是否正确，检测常见错误（如经纬度写反、坐标在海里等）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "lng": {"type": "number", "description": "经度"},
                            "lat": {"type": "number", "description": "纬度"}
                        },
                        "required": ["lng", "lat"]
                    }
                }
            }
        ]

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "tools": tools,
                        "tool_choice": "auto",
                        "temperature": 0.7,
                        "max_tokens": 2048
                    }
                )

                if response.status_code != 200:
                    print(f"智谱 AI API 错误: {response.status_code}")
                    return self._fallback_chat(message)

                result = response.json()

                if "choices" not in result or len(result["choices"]) == 0:
                    return self._fallback_chat(message)

                assistant_message = result["choices"][0]["message"]
                ai_response = assistant_message.get("content", "")
                tool_calls = assistant_message.get("tool_calls")

                # 处理工具调用
                if tool_calls:
                    tool_results = []
                    tool_outputs = []

                    for tool_call in tool_calls:
                        function_name = tool_call["function"]["name"]
                        function_args = json.loads(tool_call["function"]["arguments"])

                        # 执行函数
                        if function_name == "get_address_coord":
                            func_result = await self.call_amap_geocode(
                                function_args.get("address"),
                                function_args.get("city")
                            )
                        elif function_name == "convert_coords":
                            func_result = self._convert_coords(
                                function_args.get("lng"),
                                function_args.get("lat"),
                                function_args.get("from"),
                                function_args.get("to")
                            )
                        elif function_name == "calculate_distance":
                            func_result = self._calculate_distance(
                                function_args.get("lng1"),
                                function_args.get("lat1"),
                                function_args.get("lng2"),
                                function_args.get("lat2")
                            )
                        elif function_name == "get_city_distance":
                            func_result = await self._get_city_distance(
                                function_args.get("city1"),
                                function_args.get("city2")
                            )
                        elif function_name == "get_recent_markers":
                            func_result = self._get_recent_markers(
                                function_args.get("count", 10)
                            )
                        elif function_name == "shift_coordinates":
                            func_result = self._shift_coordinates(
                                function_args.get("lng"),
                                function_args.get("lat"),
                                function_args.get("direction"),
                                function_args.get("distance_meters")
                            )
                        elif function_name == "validate_coordinate":
                            func_result = self._validate_coordinate(
                                function_args.get("lng"),
                                function_args.get("lat")
                            )
                        else:
                            func_result = {"error": f"Unknown function: {function_name}"}

                        tool_results.append(func_result)
                        tool_outputs.append({
                            "tool_call_id": tool_call["id"],
                            "role": "tool",
                            "name": function_name,
                            "content": json.dumps(func_result, ensure_ascii=False)
                        })

                    # 调用 AI 获取最终回复
                    messages.append(assistant_message)
                    messages.extend(tool_outputs)

                    final_response = await client.post(
                        self.base_url,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": self.model,
                            "messages": messages,
                            "temperature": 0.7,
                            "max_tokens": 2048
                        }
                    )

                    if final_response.status_code == 200:
                        final_result = final_response.json()
                        if "choices" in final_result and len(final_result["choices"]) > 0:
                            ai_response = final_result["choices"][0]["message"].get("content", ai_response)

                    return {
                        "response": ai_response,
                        "tool_calls": [{"name": tc["name"], "result": tr} for tc, tr in zip(tool_calls, tool_results)]
                    }

                return {
                    "response": ai_response,
                    "tool_calls": None
                }

        except Exception as e:
            print(f"智谱 AI 调用失败: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_chat(message)

    async def _get_city_distance(self, city1: str, city2: str) -> dict:
        """获取两个城市之间的距离"""
        # 查询两个城市的坐标
        coord1 = await self.call_amap_geocode(city1)
        coord2 = await self.call_amap_geocode(city2)

        if not coord1.get("success") or not coord2.get("success"):
            return {
                "success": False,
                "error": f"城市查询失败: {city1} 或 {city2}"
            }

        # 计算距离
        distance_info = self._calculate_distance(
            coord1["lng"], coord1["lat"],
            coord2["lng"], coord2["lat"]
        )

        return {
            "success": True,
            "city1": {"name": city1, "lng": coord1["lng"], "lat": coord1["lat"]},
            "city2": {"name": city2, "lng": coord2["lng"], "lat": coord2["lat"]},
            "distance_km": distance_info["distance_km"],
            "distance_m": distance_info["distance_m"],
            "bearing": distance_info["bearing"]
        }

    def _convert_coords(self, lng: float, lat: float, from_coord: str, to_coord: str) -> dict:
        """执行坐标转换"""
        try:
            result = CoordTransformService.convert(
                lng, lat,
                CoordSystem(from_coord),
                CoordSystem(to_coord)
            )
            return {
                "success": True,
                "original": {"lng": lng, "lat": lat, "coord": from_coord},
                "converted": {"lng": result[0], "lat": result[1], "coord": to_coord}
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_recent_markers(self, count: int = 10) -> dict:
        """获取最近的标记点"""
        # 从会话中获取历史标记
        # 这里简化实现，返回空列表，实际应该从前端传递会话ID
        return {
            "success": True,
            "markers": [],
            "note": "历史标记功能需要前端传递会话ID"
        }

    def _shift_coordinates(self, lng: float, lat: float, direction: str, distance_meters: float) -> dict:
        """计算坐标偏移"""
        # 1度约等于 111km (纬度)
        # 经度随纬度变化: 1度经度 = 111km * cos(纬度)

        lat_rad = math.radians(lat)
        km_per_degree_lat = 111.0
        km_per_degree_lng = 111.0 * math.cos(lat_rad)

        # 计算偏移量
        d_lat = 0
        d_lng = 0

        distance_km = distance_meters / 1000

        direction = direction.lower()
        if 'north' in direction:
            d_lat = distance_km / km_per_degree_lat
        elif 'south' in direction:
            d_lat = -distance_km / km_per_degree_lat

        if 'east' in direction:
            d_lng = distance_km / km_per_degree_lng
        elif 'west' in direction:
            d_lng = -distance_km / km_per_degree_lng

        new_lng = lng + d_lng
        new_lat = lat + d_lat

        # 检查新坐标是否有效
        validation = CoordinateValidator.validate_coordinate(new_lng, new_lat)

        return {
            "success": True,
            "original": {"lng": lng, "lat": lat},
            "new": {"lng": round(new_lng, 6), "lat": round(new_lat, 6)},
            "distance_m": distance_meters,
            "direction": direction,
            "validation": validation
        }

    def _validate_coordinate(self, lng: float, lat: float) -> dict:
        """验证坐标"""
        result = CoordinateValidator.validate_coordinate(lng, lat)

        # 如果有严重问题，建议调换
        if not result['valid'] and '写反' in str(result.get('issues', [])):
            result['suggestion'] = f"坐标可能在海域或境外，是否需要调换为 ({lat}, {lng})？"

        return result

    def _fallback_chat(self, message: str) -> dict:
        """AI 不可用时的增强规则匹配"""
        message_lower = message.lower()

        # 城市距离计算
        if '距离' in message or '多远' in message:
            # 提取城市名
            cities = []
            common_cities = ['苏州', '北京', '上海', '杭州', '南京', '武汉', '成都', '西安', '广州', '深圳', '天津', '重庆', '青岛', '大连', '厦门', '长沙', '郑州', '沈阳', '哈尔滨', '济南', '昆明', '贵阳', '南宁', '海口', '兰州', '银川', '西宁', '拉萨', '乌鲁木齐', '呼和浩特', '石家庄', '太原', '合肥', '南昌', '福州', '台北', '香港', '澳门']

            for city in common_cities:
                if city in message:
                    cities.append(city)

            if len(cities) >= 2:
                # 使用预存的城市中心坐标
                city_coords = {
                    '苏州': (120.585, 31.298), '北京': (116.407, 39.904),
                    '上海': (121.473, 31.230), '杭州': (120.153, 30.287),
                    '南京': (118.767, 32.041), '武汉': (114.305, 30.593),
                    '成都': (104.066, 30.572), '西安': (108.940, 34.341),
                    '广州': (113.264, 23.129), '深圳': (114.085, 22.547),
                    '天津': (117.200, 39.084), '重庆': (106.551, 29.563),
                    '青岛': (120.382, 36.067), '大连': (121.618, 38.914),
                    '厦门': (118.089, 24.479), '长沙': (112.982, 28.194),
                    '郑州': (113.625, 34.746), '沈阳': (123.458, 41.617),
                    '哈尔滨': (126.534, 45.803), '济南': (117.120, 36.650),
                    '昆明': (102.832, 24.880), '贵阳': (106.713, 26.578),
                    '南宁': (108.366, 22.817), '海口': (110.331, 20.031),
                    '兰州': (103.823, 36.058), '银川': (106.230, 38.487),
                    '西宁': (101.778, 36.617), '拉萨': (91.132, 29.660),
                    '乌鲁木齐': (87.616, 43.825), '呼和浩特': (111.749, 40.842),
                    '石家庄': (114.514, 38.042), '太原': (112.548, 37.857),
                    '合肥': (117.227, 31.820), '南昌': (115.857, 28.682),
                    '福州': (119.296, 26.074), '香港': (114.173, 22.320),
                    '澳门': (113.549, 22.198)
                }

                c1, c2 = cities[0], cities[1]
                if c1 in city_coords and c2 in city_coords:
                    lng1, lat1 = city_coords[c1]
                    lng2, lat2 = city_coords[c2]
                    dist = self._calculate_distance(lng1, lat1, lng2, lat2)

                    return {
                        "response": f"📍 {c1} 与 {c2} 的距离（直线距离）：\n\n{self._format_distance(dist['distance_m'])}\n\n方位角: {dist['bearing']}°\n\n注意：这是两个城市中心点之间的直线距离，实际交通距离会有所不同。",
                        "tool_calls": None
                    }

        # 城市坐标查询
        if any(kw in message for kw in ['经纬度', '坐标', '在哪里', '位置']):
            city_coords = {
                '苏州': (120.585, 31.298), '北京': (116.407, 39.904),
                '上海': (121.473, 31.230), '杭州': (120.153, 30.287),
                '南京': (118.767, 32.041), '武汉': (114.305, 30.593),
                '成都': (104.066, 30.572), '西安': (108.940, 34.341),
                '广州': (113.264, 23.129), '深圳': (114.085, 22.547),
                '天津': (117.200, 39.084), '重庆': (106.551, 29.563)
            }

            for city, (lng, lat) in city_coords.items():
                if city in message:
                    return {
                        "response": f"📍 {city}的坐标（GCJ02/高德坐标系）：\n\n经度: {lng}\n纬度: {lat}\n\n这是{city}市中心的大致坐标。",
                        "tool_calls": None
                    }

        # 默认回复
        return {
            "response": """我可以帮你：🌍

📍 **地址查询** - "北京的经纬度是多少？"
🔄 **坐标转换** - "把116.404,39.915转成GPS坐标"
📏 **距离计算** - "北京到上海有多远？"
📚 **地理知识** - "什么是GCJ02坐标系？"

请告诉我你需要什么帮助！""",
            "tool_calls": None
        }


# 同步包装器
class ZhipuAIService:
    """同步版本的 AI 服务"""
    def __init__(self):
        self.http_service = ZhipuAIHTTPService()
        self.available = self.http_service.available

    def chat(self, message: str, conversation_history: list = None):
        """同步版本的 chat 方法"""
        import asyncio
        import threading

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                result = [None]
                exception = [None]

                def run_in_new_loop():
                    try:
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        result[0] = new_loop.run_until_complete(
                            self.http_service.chat(message, conversation_history)
                        )
                        new_loop.close()
                    except Exception as e:
                        exception[0] = e

                thread = threading.Thread(target=run_in_new_loop)
                thread.start()
                thread.join(timeout=20)

                if exception[0]:
                    raise exception[0]
                return result[0]
            else:
                return loop.run_until_complete(
                    self.http_service.chat(message, conversation_history)
                )
        except RuntimeError:
            return asyncio.run(
                self.http_service.chat(message, conversation_history)
            )


def get_ai_service():
    """获取 AI 服务实例"""
    return ZhipuAIService()
