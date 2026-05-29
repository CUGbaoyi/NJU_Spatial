# -*- coding: utf-8 -*-
from fastapi import APIRouter
from .convert import router as convert_router

api_router = APIRouter()

# 包含所有子路由
api_router.include_router(convert_router)
