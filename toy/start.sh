#!/bin/bash

# Geo-AI Expert System 启动脚本

echo "======================================"
echo "  Geo-AI Expert System 启动中..."
echo "======================================"

# 检查 Python 版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python 版本: $python_version"

# 安装依赖
echo ""
echo "正在安装依赖包..."
pip3 install -r requirements.txt

# 启动服务器
echo ""
echo "======================================"
echo "  服务器启动成功！"
echo "======================================"
echo ""
echo "访问地址: http://localhost:8000"
echo "API 文档: http://localhost:8000/docs"
echo ""
echo "按 Ctrl+C 停止服务器"
echo ""

python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
