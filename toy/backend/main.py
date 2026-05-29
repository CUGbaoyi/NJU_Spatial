# -*- coding: utf-8 -*-
"""
Geo-AI Expert System - 主应用
智能地理坐标专家系统
"""
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from backend.api import api_router
from backend.models.schemas import CoordSystem

# 创建 FastAPI 应用
app = FastAPI(
    title="Geo-AI Expert System",
    description="智能地理坐标专家系统 - 支持 WGS84/GCJ02/BD09 坐标转换与 AI 智能对话",
    version="1.0.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录（前端）
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    # 根路径返回首页
    from fastapi.responses import FileResponse
    index_path = frontend_dir / "index.html"

    @app.get("/")
    async def read_root():
        return FileResponse(str(index_path))

# 注册 API 路由
app.include_router(api_router)


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "service": "Geo-AI Expert System",
        "version": "1.0.0",
        "coord_systems_supported": ["wgs84", "gcj02", "bd09"]
    }


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
