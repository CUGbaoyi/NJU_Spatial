# -*- coding: utf-8 -*-
"""
智谱 AI 服务
实现 Function Calling 机制
"""
import os
import json
from typing import Optional, Any
from pathlib import Path

# 加载 .env 文件
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

# 延迟导入，如果不可用则禁用 AI 功能
try:
    from zhipuai import ZhipuAI
    ZHIPU_AVAILABLE = True
except ImportError:
    ZHIPU_AVAILABLE = False

from backend.services.convert_service import CoordTransformService
from backend.models.schemas import CoordSystem


class ZhipuAIService:
    """智谱 AI 服务类"""

    def __init__(self):
        if not ZHIPU_AVAILABLE:
            self.client = None
            self.available = False
            return

        self.api_key = os.getenv("ZHIPU_API_KEY")
        if not self.api_key:
            self.client = None
            self.available = False
            return

        try:
            self.client = ZhipuAI(api_key=self.api_key)
            self.available = True
        except Exception:
            self.client = None
            self.available = False

    # Function Calling 定义的工具
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "search_location",
                "description": "查询地点的准确坐标和地址信息。使用高德地图地理编码 API 获取真实数据，避免 AI 幻觉。当用户询问某个地点、建筑、学校、公司等的位置时使用此工具。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "要查询的地点名称或地址，如'南京大学苏州校区'、'北京天安门'等"
                        },
                        "city": {
                            "type": "string",
                            "description": "指定查询的城市，可以提供更准确的结果，如'苏州'、'北京'等，可选"
                        }
                    },
                    "required": ["keyword"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "single_convert",
                "description": "转换单个坐标点的坐标系。支持 WGS84、GCJ02（火星坐标/高德）、BD09（百度）之间的互相转换。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "longitude": {
                            "type": "number",
                            "description": "经度，范围 -180 到 180"
                        },
                        "latitude": {
                            "type": "number",
                            "description": "纬度，范围 -90 到 90"
                        },
                        "from_coord": {
                            "type": "string",
                            "enum": ["wgs84", "gcj02", "bd09"],
                            "description": "源坐标系：wgs84(GPS/地球坐标), gcj02(高德/腾讯火星坐标), bd09(百度坐标)"
                        },
                        "to_coord": {
                            "type": "string",
                            "enum": ["wgs84", "gcj02", "bd09"],
                            "description": "目标坐标系：wgs84(GPS/地球坐标), gcj02(高德/腾讯火星坐标), bd09(百度坐标)"
                        }
                    },
                    "required": ["longitude", "latitude", "from_coord", "to_coord"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "explain_coord_system",
                "description": "解释各个坐标系的含义、用途以及它们之间的区别。帮助用户理解应该使用哪种坐标系。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "system_name": {
                            "type": "string",
                            "description": "要查询的坐标系名称，如 wgs84、gcj02、bd09，或者留空查询所有",
                            "enum": ["wgs84", "gcj02", "bd09", "all"]
                        }
                    },
                    "required": []
                }
            }
        }
    ]

    def call_single_convert(self, longitude: float, latitude: float,
                           from_coord: str, to_coord: str) -> dict:
        """执行单次坐标转换"""
        try:
            result = CoordTransformService.convert(
                longitude=longitude,
                latitude=latitude,
                from_coord=CoordSystem(from_coord),
                to_coord=CoordSystem(to_coord)
            )
            return {
                "success": True,
                "original": {"lng": longitude, "lat": latitude, "coord": from_coord},
                "converted": {"lng": result[0], "lat": result[1], "coord": to_coord}
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def call_explain_coord_system(self, system_name: str = "all") -> dict:
        """解释坐标系"""
        explanations = {
            "wgs84": {
                "name": "WGS84",
                "description": "World Geodetic System 1984，世界大地坐标系",
                "usage": "GPS 设备使用的原始坐标系，国际标准",
                "features": "地球上统一的坐标系统，没有经过加密偏移",
                "typical_use": "iPhone/Android 原始 GPS、Google Earth（中国境外）"
            },
            "gcj02": {
                "name": "GCJ02",
                "description": "国测局坐标系，俗称火星坐标系",
                "usage": "中国法律规定的必须使用的加密坐标系",
                "features": "在 WGS84 基础上进行加密偏移，由国测局制定",
                "typical_use": "高德地图、腾讯地图、阿里云地图"
            },
            "bd09": {
                "name": "BD09",
                "description": "百度坐标系",
                "usage": "百度地图专用的坐标系",
                "features": "在 GCJ02 基础上再次进行偏移",
                "typical_use": "百度地图、百度定位服务"
            }
        }

        if system_name == "all":
            return {
                "systems": explanations,
                "conversion_tips": {
                    "GPS转高德": "WGS84 -> GCJ02",
                    "GPS转百度": "WGS84 -> BD09",
                    "高德转百度": "GCJ02 -> BD09",
                    "百度转高德": "BD09 -> GCJ02"
                }
            }
        return explanations.get(system_name, {})

    def call_search_location(self, keyword: str, city: str = None) -> dict:
        """使用高德地理编码 API 查询地点的准确坐标和地址信息"""
        import requests

        try:
            api_key = os.getenv("AMAP_API_KEY")
            if not api_key:
                return {"success": False, "error": "高德 API Key 未配置"}

            # 构建请求参数
            params = {
                "key": api_key,
                "keywords": keyword,
                "output": "json"
            }
            if city:
                params["city"] = city

            # 调用高德地理编码 API
            url = "https://restapi.amap.com/v3/place/text"
            response = requests.get(url, params=params, timeout=5)
            data = response.json()

            if data.get("status") == "1" and data.get("pois"):
                # 取第一个最相关的结果
                poi = data["pois"][0]
                location = poi.get("location", "").split(",")
                if len(location) == 2:
                    lng, lat = float(location[0]), float(location[1])

                    # 获取行政区划逆地理编码信息
                    address_response = requests.get(
                        "https://restapi.amap.com/v3/geocode/regeo",
                        params={"key": api_key, "location": poi["location"], "output": "json"},
                        timeout=5
                    )
                    address_data = address_response.json()

                    formatted_address = ""
                    if address_data.get("status") == "1":
                        formatted_address = address_data.get("regeocode", {}).get("formatted_address", "")

                    return {
                        "success": True,
                        "name": poi.get("name", keyword),
                        "address": poi.get("address", ""),
                        "full_address": formatted_address or poi.get("pname", "") + poi.get("cityname", "") + poi.get("adname", ""),
                        "lng": lng,
                        "lat": lat,
                        "coord_system": "GCJ02（高德地图坐标系）",
                        "province": poi.get("pname", ""),
                        "city": poi.get("cityname", ""),
                        "district": poi.get("adname", "")
                    }

            return {
                "success": False,
                "error": f"未找到地点 '{keyword}' 的相关信息"
            }

        except requests.RequestException as e:
            return {
                "success": False,
                "error": f"网络请求失败: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"查询失败: {str(e)}"
            }

    def chat(self, message: str, conversation_history: Optional[list] = None) -> dict:
        """
        与 AI 对话，支持 Function Calling

        Args:
            message: 用户消息
            conversation_history: 对话历史

        Returns:
            包含回复内容和可能的函数调用结果
        """
        # 检查 AI 服务是否可用
        if not self.available:
            # 简单的规则匹配回退
            return self._fallback_chat(message)

        # 构建消息列表
        messages = []

        # 系统提示词
        system_prompt = """你是一个专业的地理坐标转换助手。你可以帮助用户：
1. 在不同坐标系之间转换坐标（WGS84、GCJ02、BD09）
2. 解释各种坐标系的区别和使用场景
3. 判断坐标应该属于哪个坐标系
4. 查询地点的准确位置信息

**重要规则（必须遵守）：**

1. **语言要求：** 你必须始终使用中文回答用户的问题，除非用户明确要求使用其他语言。

2. **地点查询规则：** 当用户询问某个地点、建筑、学校、公司等的位置、地址、坐标时，**必须使用 search_location 函数**查询真实数据，**绝不要**凭记忆或训练数据回答！AI 的记忆可能是错误的或过时的。

3. **工具使用：**
   - 用户询问地点位置/地址 → 使用 search_location 函数
   - 用户需要转换坐标 → 使用 single_convert 函数
   - 用户询问坐标系知识 → 使用 explain_coord_system 函数

坐标系识别技巧：
- 如果用户说"GPS坐标"、"原始坐标"、"Google Earth坐标"，通常是 WGS84
- 如果用户说"高德坐标"、"腾讯坐标"、"火星坐标"，通常是 GCJ02
- 如果用户说"百度坐标"，通常是 BD09"""
        messages.append({"role": "system", "content": system_prompt})

        # 添加历史对话
        if conversation_history:
            messages.extend(conversation_history)

        # 添加当前消息
        messages.append({"role": "user", "content": message})

        try:
            response = self.client.chat.completions.create(
                model="glm-4",
                messages=messages,
                tools=self.TOOLS,
                tool_choice="auto"
            )

            assistant_message = response.choices[0].message

            # 检查是否有工具调用
            if assistant_message.tool_calls:
                tool_calls = assistant_message.tool_calls
                tool_results = []

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    # 执行对应的函数
                    if function_name == "search_location":
                        result = self.call_search_location(**function_args)
                    elif function_name == "single_convert":
                        result = self.call_single_convert(**function_args)
                    elif function_name == "explain_coord_system":
                        result = self.call_explain_coord_system(**function_args)
                    else:
                        result = {"error": f"Unknown function: {function_name}"}

                    tool_results.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(result, ensure_ascii=False)
                    })

                # 将函数结果返回给模型获取最终回复
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in tool_calls
                    ]
                })
                messages.extend(tool_results)

                # 添加中文回复要求
                messages.append({
                    "role": "user",
                    "content": "请用中文友好地回答，将转换结果以清晰的格式展示给用户。"
                })

                final_response = self.client.chat.completions.create(
                    model="glm-4",
                    messages=messages
                )

                return {
                    "response": final_response.choices[0].message.content,
                    "tool_calls": [
                        {
                            "name": tc["name"],
                            "result": json.loads(tc["content"])
                        }
                        for tc in tool_results
                    ]
                }

            return {
                "response": assistant_message.content or "",
                "tool_calls": None
            }

        except Exception as e:
            return {
                "response": f"AI 服务出错: {str(e)}",
                "tool_calls": None,
                "error": True
            }

    def _fallback_chat(self, message: str) -> dict:
        """AI 不可用时的简单规则匹配"""
        import re

        message_lower = message.lower()

        # 尝试匹配坐标转换请求
        # 匹配: "116.404, 39.915" 或 "经度116.404 纬度39.915" 等
        coord_pattern = r'(\d+\.?\d*)[°\s,，]*?(\d+\.?\d*)'
        coords = re.findall(coord_pattern, message)

        if coords and len(coords[0]) == 2:
            lng, lat = float(coords[0][0]), float(coords[0][1])

            # 判断是否是有效经纬度
            if -180 <= lng <= 180 and -90 <= lat <= 90:
                # 尝试判断源坐标系
                from_coord = "gcj02"  # 默认
                to_coord = "wgs84"

                if "wgs" in message_lower or "gps" in message_lower:
                    from_coord = "wgs84"
                elif "百度" in message or "bd" in message_lower:
                    from_coord = "bd09"
                elif "高德" in message or "火星" in message or "gcj" in message_lower:
                    from_coord = "gcj02"

                # 判断目标坐标系
                if "转百度" in message or "->bd" in message_lower:
                    to_coord = "bd09"
                elif "转高德" in message or "->gcj" in message_lower:
                    to_coord = "gcj02"
                elif "转wgs" in message or "转gps" in message_lower:
                    to_coord = "wgs84"

                result = self.call_single_convert(lng, lat, from_coord, to_coord)

                if result.get("success"):
                    return {
                        "response": f"坐标转换完成：\n原始坐标 ({from_coord.upper()})：{lng}, {lat}\n转换结果 ({to_coord.upper()})：{result['converted']['lng']}, {result['converted']['lat']}",
                        "tool_calls": [{"name": "single_convert", "result": result}]
                    }

        # 默认回复
        return {
            "response": "AI 服务当前不可用（智谱 AI SDK 与 Python 3.14 不兼容）。\n\n你可以使用以下功能：\n• 左侧快速转换工具进行坐标转换\n• 上传 Excel 文件进行批量转换\n• 直接调用 API 接口",
            "tool_calls": None,
            "error": False
        }


# 导出便捷函数
def get_ai_service() -> ZhipuAIService:
    """获取 AI 服务实例"""
    return ZhipuAIService()
