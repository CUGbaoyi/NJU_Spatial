# Geo-AI Expert System - 智能地理坐标专家系统

## 项目结构

```
toy/
├── backend/                    # 后端代码 (FastAPI)
│   ├── main.py                # 主应用入口
│   ├── api/                   # API 路由
│   │   ├── __init__.py
│   │   └── convert.py         # 坐标转换 API
│   ├── models/                # 数据模型
│   │   ├── __init__.py
│   │   └── schemas.py         # Pydantic 模型定义
│   └── services/              # 业务服务
│       ├── __init__.py
│       ├── convert_service.py # 坐标转换服务（调用 coordtransform_utils.py）
│       ├── ai_service.py      # 智谱 AI 服务（Function Calling）
│       └── excel_service.py   # Excel 批量处理服务
├── frontend/                   # 前端代码
│   ├── index.html             # 主页面
│   ├── css/
│   │   └── style.css          # 样式文件
│   └── js/
│       └── app.js             # 前端逻辑 + 高德地图集成
├── uploads/                    # 上传文件目录
├── exports/                    # 导出文件目录（24小时后自动清理）
├── coordtransform_utils.py    # 核心坐标转换工具
├── .env                        # 环境变量配置
├── requirements.txt            # Python 依赖
├── start.sh                    # 启动脚本
└── venv/                       # 虚拟环境
```

## 功能特性

### 1. 坐标系转换
- WGS84 (GPS/地球坐标)
- GCJ02 (高德/腾讯火星坐标)
- BD09 (百度坐标)
- 支持三种坐标系之间的任意双向转换

### 2. AI 智能对话
- 基于 Function Calling 的坐标转换
- 坐标系知识问答
- 自然语言输入解析

### 3. Excel 批量处理
- AI 自动识别经纬度列
- 批量坐标转换
- 导出格式：Excel, GeoJSON, KML
- 实时进度反馈

### 4. 高德地图集成
- 卫星图/标准图切换
- 批量打点（MassMarks）
- 轨迹连线
- 转换前后对比显示

## 启动方式

### 方式一：使用启动脚本
```bash
./start.sh
```

### 方式二：手动启动
```bash
# 激活虚拟环境
source venv/bin/activate

# 安装依赖（首次运行）
pip install -r requirements.txt

# 启动服务器
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

## 访问地址

- 主页面: http://localhost:8000
- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health

## API 端点

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/api/coord-systems` | 获取支持的坐标系 |
| GET/POST | `/api/convert/single` | 单次坐标转换 |
| POST | `/api/chat` | AI 对话 |
| POST | `/api/excel/analyze` | 分析 Excel 列 |
| POST | `/api/excel/convert` | Excel 批量转换 |
| GET | `/api/excel/task/{task_id}` | 查询任务状态 |
| GET | `/api/excel/download/{task_id}/{type}` | 下载结果 |
| GET | `/api/amap/geocode` | 高德地理编码 |
| GET | `/api/amap/regeocode` | 高德逆地理编码 |

## 环境变量

```env
# 智谱 AI
ZHIPU_API_KEY=your_key_here

# 高德地图
AMAP_API_KEY=your_key_here
```

## 注意事项

1. **Python 版本兼容性**: 智谱 AI SDK 当前与 Python 3.14 存在兼容性问题，AI 功能会自动降级为规则匹配模式
2. **文件保留**: 导出的文件会在服务器保留 24 小时后自动删除
3. **坐标系识别**: AI 列识别基于智谱 GLM-4 模型，确保 API Key 配置正确

## 下一步开发建议

1. [ ] 添加用户认证系统
2. [ ] 实现对话历史存储（Redis/数据库）
3. [ ] 添加坐标轨迹回放功能
4. [ ] 支持更多导出格式（Shapefile 等）
5. [ ] 添加坐标系自动检测功能
