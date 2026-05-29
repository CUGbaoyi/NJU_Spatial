# -*- coding: utf-8 -*-
from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum


class CoordSystem(str, Enum):
    """支持的坐标系类型"""
    WGS84 = "wgs84"
    GCJ02 = "gcj02"
    BD09 = "bd09"


class SingleConvertRequest(BaseModel):
    """单次坐标转换请求"""
    longitude: float = Field(..., description="经度", ge=-180, le=180)
    latitude: float = Field(..., description="纬度", ge=-90, le=90)
    from_coord: CoordSystem = Field(..., description="源坐标系")
    to_coord: CoordSystem = Field(..., description="目标坐标系")


class SingleConvertResponse(BaseModel):
    """单次坐标转换响应"""
    success: bool
    from_coord: CoordSystem
    to_coord: CoordSystem
    original: tuple[float, float]
    converted: tuple[float, float]
    message: Optional[str] = None


class BatchConvertRequest(BaseModel):
    """批量转换请求"""
    from_coord: CoordSystem = Field(..., description="源坐标系")
    to_coord: CoordSystem = Field(..., description="目标坐标系")


class ColumnMapping(BaseModel):
    """Excel 列映射"""
    longitude_column: str = Field(..., description="经度列名")
    latitude_column: str = Field(..., description="纬度列名")


class ChatMessage(BaseModel):
    """聊天消息"""
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    """聊天请求"""
    message: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    """聊天响应"""
    response: str
    conversation_id: str
    function_call: Optional[dict] = None


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class BatchTaskInfo(BaseModel):
    """批量任务信息"""
    task_id: str
    status: TaskStatus
    total_rows: int
    processed_rows: int
    download_urls: Optional[dict] = None
    error: Optional[str] = None
