# -*- coding: utf-8 -*-
"""
坐标转换 API 路由
"""
import os
import json
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.models.schemas import (
    CoordSystem,
    SingleConvertRequest,
    SingleConvertResponse,
    BatchConvertRequest,
    ChatRequest,
    ChatResponse,
    TaskStatus,
    ColumnMapping,
)
from backend.services.convert_service import CoordTransformService
from backend.services.ai_service import get_ai_service
from backend.services.excel_service import get_excel_service


router = APIRouter(prefix="/api", tags=["坐标转换"])

# 全局服务实例
ai_service = get_ai_service()
excel_service = get_excel_service()


# ==================== 坐标系信息 ====================

@router.get("/coord-systems")
async def get_coord_systems():
    """获取支持的坐标系列表及说明"""
    return {
        "coord_systems": [
            {
                "code": "wgs84",
                "name": "WGS84",
                "description": "世界大地坐标系 1984",
                "usage": "GPS 原始坐标，国际标准",
                "typical_for": ["iPhone/Android GPS", "Google Earth（境外）"]
            },
            {
                "code": "gcj02",
                "name": "GCJ02",
                "description": "国测局坐标系（火星坐标）",
                "usage": "中国法定加密坐标系",
                "typical_for": ["高德地图", "腾讯地图", "阿里云地图"]
            },
            {
                "code": "bd09",
                "name": "BD09",
                "description": "百度坐标系",
                "usage": "百度地图专用坐标系",
                "typical_for": ["百度地图", "百度定位"]
            }
        ],
        "conversion_matrix": {
            "wgs84": ["gcj02", "bd09"],
            "gcj02": ["wgs84", "bd09"],
            "bd09": ["wgs84", "gcj02"]
        }
    }


# ==================== 单次转换 ====================

@router.post("/convert/single", response_model=SingleConvertResponse)
async def convert_single(request: SingleConvertRequest):
    """
    单个坐标点转换

    - **longitude**: 经度 (-180 到 180)
    - **latitude**: 纬度 (-90 到 90)
    - **from_coord**: 源坐标系 (wgs84/gcj02/bd09)
    - **to_coord**: 目标坐标系 (wgs84/gcj02/bd09)
    """
    try:
        converted = CoordTransformService.convert(
            request.longitude,
            request.latitude,
            request.from_coord,
            request.to_coord
        )

        return SingleConvertResponse(
            success=True,
            from_coord=request.from_coord,
            to_coord=request.to_coord,
            original=(request.longitude, request.latitude),
            converted=converted
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/convert/single")
async def convert_single_get(
    lng: float,
    lat: float,
    from_coord: str,
    to_coord: str
):
    """GET 方式的单次转换（便于调试）

    注意：坐标转换只在中国境内有效，境外坐标各坐标系是一致的。
    """
    try:
        # 检查是否在中国境外
        is_outside = CoordTransformService.is_out_of_china(lng, lat)

        result = CoordTransformService.convert(
            lng,
            lat,
            CoordSystem(from_coord.lower()),
            CoordSystem(to_coord.lower())
        )

        response = {
            "success": True,
            "original": {"lng": lng, "lat": lat},
            "converted": {"lng": result[0], "lat": result[1]},
            "from_coord": from_coord,
            "to_coord": to_coord
        }

        # 添加中国境内提示
        if is_outside:
            response["note"] = "该坐标位于中国境外，境外各坐标系（WGS84/GCJ02/BD09）是一致的，不需要转换。"

        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== AI 聊天 ====================

class ChatMessageRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    conversation_history: Optional[list] = None


@router.post("/chat")
async def chat(request: ChatMessageRequest):
    """
    AI 智能对话接口

    支持的功能：
    - 自然语言进行坐标转换
    - 坐标系知识问答
    - 自动识别用户意图
    - 上下文记忆（通过 conversation_history 参数）
    """
    # 使用前端传入的对话历史
    conversation_history = request.conversation_history or []

    result = ai_service.chat(request.message, conversation_history)

    return {
        "response": result["response"],
        "conversation_id": request.conversation_id or "default",
        "tool_calls": result.get("tool_calls"),
        "error": result.get("error", False)
    }


# ==================== Excel 批量处理 ====================

@router.post("/excel/analyze")
async def analyze_excel(file: UploadFile = File(...)):
    """
    上传 Excel 文件，AI 自动识别经纬度列

    返回：列名分析和样本数据
    """
    # 保存上传的文件
    file_path = f"/Users/chenao/Desktop/toy/uploads/{file.filename}"

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # AI 分析
    analysis = await excel_service.analyze_excel_columns(file_path)

    return analysis


@router.post("/excel/convert")
async def convert_excel(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    longitude_column: str = None,
    latitude_column: str = None,
    from_coord: str = "wgs84",
    to_coord: str = "gcj02"
):
    """
    批量转换 Excel 文件中的坐标

    - **file**: Excel 文件
    - **longitude_column**: 经度列名（如未提供，会先进行 AI 识别）
    - **latitude_column**: 纬度列名
    - **from_coord**: 源坐标系
    - **to_coord**: 目标坐标系

    返回：任务 ID，可用于查询进度和下载结果
    """
    # 保存文件
    file_path = f"/Users/chenao/Desktop/toy/uploads/{file.filename}"

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # 如果没有指定列名，先进行 AI 识别
    if not longitude_column or not latitude_column:
        analysis = await excel_service.analyze_excel_columns(file_path)
        if analysis.get("success") and analysis.get("analysis"):
            longitude_column = analysis["analysis"].get("longitude_column")
            latitude_column = analysis["analysis"].get("latitude_column")

        if not longitude_column or not latitude_column:
            raise HTTPException(
                status_code=400,
                detail="无法自动识别经纬度列，请手动指定列名"
            )

    # 创建转换任务
    task_id = excel_service.create_batch_task(
        file_path=file_path,
        longitude_column=longitude_column,
        latitude_column=latitude_column,
        from_coord=CoordSystem(from_coord),
        to_coord=CoordSystem(to_coord)
    )

    # 后台处理
    background_tasks.add_task(excel_service.process_batch_task, task_id)

    return {
        "task_id": task_id,
        "status": "pending",
        "message": "任务已创建，正在后台处理中"
    }


@router.get("/excel/task/{task_id}")
async def get_task_status(task_id: str):
    """查询批量转换任务状态"""
    task = excel_service.get_task_status(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")

    return {
        "task_id": task["task_id"],
        "status": task["status"].value,
        "total_rows": task["total_rows"],
        "processed_rows": task["processed_rows"],
        "progress": round(task["processed_rows"] / task["total_rows"] * 100, 2) if task["total_rows"] > 0 else 0,
        "download_urls": task.get("download_urls"),
        "error": task.get("error")
    }


@router.get("/excel/download/{task_id}/{file_type}")
async def download_result(task_id: str, file_type: str):
    """
    下载转换结果文件

    - **task_id**: 任务 ID
    - **file_type**: 文件类型 (excel/geojson/kml)
    """
    task = excel_service.get_task_status(task_id)

    if not task or task["status"] != TaskStatus.COMPLETED:
        raise HTTPException(status_code=404, detail="任务未完成或不存在")

    download_urls = task.get("download_urls", {})
    file_path = download_urls.get(file_type)

    if not file_path:
        raise HTTPException(status_code=404, detail="文件不存在")

    filename = os.path.basename(file_path)
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )


# ==================== 高德地图集成 ====================

@router.get("/amap/geocode")
async def amap_geocode(address: str, city: Optional[str] = None):
    """
    高德地图地理编码：地址转坐标

    需要在 .env 中配置 AMAP_API_KEY
    """
    api_key = os.getenv("AMAP_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="高德 API Key 未配置")

    import httpx

    url = "https://restapi.amap.com/v3/geocode/geo"
    params = {
        "key": api_key,
        "address": address
    }
    if city:
        params["city"] = city

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        return response.json()


@router.get("/amap/regeocode")
async def amap_regeocode(lng: float, lat: float):
    """
    高德地图逆地理编码：坐标转地址

    可用于验证坐标系：如果用 GCJ02 调用能得到正确地址，说明原坐标是 GCJ02
    """
    api_key = os.getenv("AMAP_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="高德 API Key 未配置")

    import httpx

    url = "https://restapi.amap.com/v3/geocode/regeo"
    params = {
        "key": api_key,
        "location": f"{lng},{lat}",
        "extensions": "all"
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        return response.json()


# ==================== 纠错专家 API ====================

class CoordinateValidateRequest(BaseModel):
    """坐标验证请求"""
    lng: float
    lat: float


class BatchValidateRequest(BaseModel):
    """批量验证请求"""
    coordinates: list[dict]  # [{"lng": x, "lat": y}, ...]


@router.post("/validate/single")
async def validate_single(request: CoordinateValidateRequest):
    """
    纠错专家：验证单个坐标

    - **lng**: 经度
    - **lat**: 纬度

    返回验证结果，包括问题诊断和智能建议
    """
    from backend.services.coordinate_validator import CoordinateValidator

    result = CoordinateValidator.validate_coordinate(request.lng, request.lat)

    return {
        "lng": request.lng,
        "lat": request.lat,
        **result
    }


@router.post("/validate/batch")
async def validate_batch(request: BatchValidateRequest):
    """
    纠错专家：批量验证坐标并生成分析报告

    - **coordinates**: 坐标列表 [{"lng": x, "lat": y}, ...]

    返回验证报告，包括统计信息、异常坐标列表和智能建议
    """
    from backend.services.coordinate_validator import CoordinateValidator

    result = CoordinateValidator.validate_batch(request.coordinates)

    return result


@router.post("/ai/enhanced")
async def ai_enhanced(message: str):
    """
    AI 增强对话 - 支持状态管理和模糊意图

    - **message**: 用户消息
    """
    result = ai_service.chat(message)
    return result
