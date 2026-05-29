/**
 * Geo-AI Expert System - Frontend Application
 */

console.log('=== 开始加载 app.js ===');

const CONFIG = {
    API_BASE: '/api',
    AMAP_KEY: 'b3849009b91c04ef3541c555b4146376'
};

const state = {
    map: null,
    satelliteLayer: null,
    roadNetLayer: null,
    markers: [],
    lastConvert: null,
    useSatellite: false,
    tempMarker: null,  // 临时点击标记
    infoWindow: null,  // 信息窗口
    conversationHistory: [],  // AI 对话历史
    conversationId: 'conv-' + Date.now()  // 会话ID
};

// ==================== 检查坐标是否在中国境内 ====================
function isInsideChina(lng, lat) {
    return lng >= 72.004 && lng <= 137.8347 && lat >= 0.8293 && lat <= 55.8271;
}

// ==================== API ====================
const API = {
    async convertSingle(lng, lat, fromCoord, toCoord) {
        const params = new URLSearchParams({
            lng: String(lng),
            lat: String(lat),
            from_coord: fromCoord,
            to_coord: toCoord
        });
        const response = await fetch(`${CONFIG.API_BASE}/convert/single?${params}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    },

    async chat(message, conversationHistory = []) {
        const response = await fetch(`${CONFIG.API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                conversation_history: conversationHistory,
                conversation_id: state.conversationId
            })
        });
        return await response.json();
    },

    async analyzeExcel(file) {
        const formData = new FormData();
        formData.append('file', file);
        const response = await fetch(`${CONFIG.API_BASE}/excel/analyze`, {
            method: 'POST',
            body: formData
        });
        return await response.json();
    },

    async convertExcel(file, options) {
        const formData = new FormData();
        formData.append('file', file);
        if (options.longitude_column) formData.append('longitude_column', options.longitude_column);
        if (options.latitude_column) formData.append('latitude_column', options.latitude_column);
        formData.append('from_coord', options.from_coord || 'wgs84');
        formData.append('to_coord', options.to_coord || 'gcj02');

        const response = await fetch(`${CONFIG.API_BASE}/excel/convert`, {
            method: 'POST',
            body: formData
        });
        return await response.json();
    },

    async getTaskStatus(taskId) {
        const response = await fetch(`${CONFIG.API_BASE}/excel/task/${taskId}`);
        return await response.json();
    },

    downloadFile(taskId, fileType) {
        window.location.href = `${CONFIG.API_BASE}/excel/download/${taskId}/${fileType}`;
    }
};

// ==================== 地图 ====================
function initMap() {
    console.log('开始初始化地图...');

    const mapEl = document.getElementById('map');
    if (!mapEl) {
        console.error('找不到地图容器元素 #map');
        return false;
    }

    if (typeof AMap === 'undefined') {
        console.error('AMap 未定义');
        mapEl.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#ff4d4f;">' +
            '<div style="text-align:center;"><div style="font-size:48px;">⚠️</div><div>地图加载失败</div></div></div>';
        return false;
    }

    try {
        // 创建地图 - 简化配置
        state.map = new AMap.Map(mapEl, {
            zoom: 12,
            center: [116.397428, 39.90923],
            viewMode: '2D',
            dragEnable: true,
            keyboardEnable: true,
            doubleClickZoom: true,
            scrollWheel: true,
            showLabel: true,
            zooms: [3, 18]
        });

        // 创建卫星图层
        state.satelliteLayer = new AMap.TileLayer.Satellite({
            zIndex: 10
        });

        // 创建路网图层
        state.roadNetLayer = new AMap.TileLayer.RoadNet({
            zIndex: 11
        });

        // 添加控件
        AMap.plugin(['AMap.ToolBar', 'AMap.Scale'], function() {
            state.map.addControl(new AMap.ToolBar({
                position: { top: '110px', right: '40px' }
            }));
            state.map.addControl(new AMap.Scale());
        });

        // 添加地图点击事件 - 获取经纬度
        state.map.on('click', function(e) {
            const lng = e.lnglat.getLng();
            const lat = e.lnglat.getLat();

            console.log('点击位置:', lng, lat);

            // 移除之前的临时标记
            if (state.tempMarker) {
                state.map.remove(state.tempMarker);
            }

            // 创建临时标记 - 调整锚点位置
            state.tempMarker = new AMap.Marker({
                position: [lng, lat],
                icon: new AMap.Icon({
                    size: new AMap.Size(24, 32),
                    image: `data:image/svg+xml;base64,${btoa(`
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="32" viewBox="0 0 24 32">
                            <path d="M12 0C5.373 0 0 5.373 0 12c0 9 12 20 12 20s12-11 12-20c0-6.627-5.373-12-12-12z" fill="#ff6b6b"/>
                            <circle cx="12" cy="12" r="4" fill="white"/>
                        </svg>
                    `)}`,
                    imageSize: new AMap.Size(24, 32)
                }),
                // 关键：设置锚点在图标的底部中心
                anchor: 'bottom-center'
            });
            state.map.add(state.tempMarker);

            // 创建信息窗口 - 调整偏移量
            if (state.infoWindow) {
                state.infoWindow.close();
            }

            const content = `
                <div style="padding:10px;">
                    <div style="font-weight:bold;margin-bottom:8px;">📍 点击位置信息</div>
                    <div style="font-size:12px;line-height:1.8;">
                        <div>经度: <span style="color:#1890ff;font-family:monospace;">${lng.toFixed(6)}</span></div>
                        <div>纬度: <span style="color:#1890ff;font-family:monospace;">${lat.toFixed(6)}</span></div>
                        <div style="margin-top:8px;">
                            <button onclick="copyCoordinate(${lng}, ${lat})" style="padding:4px 12px;background:#1890ff;color:white;border:none;border-radius:4px;cursor:pointer;font-size:12px;">
                                📋 复制坐标
                            </button>
                        </div>
                    </div>
                </div>
            `;

            state.infoWindow = new AMap.InfoWindow({
                content: content,
                // 调整偏移，让窗口显示在标记上方
                offset: new AMap.Pixel(0, -40),
                closeWhenClickMap: true
            });

            state.infoWindow.open(state.map, [lng, lat]);

            // 更新左侧输入框
            document.getElementById('lngInput').value = lng.toFixed(6);
            document.getElementById('latInput').value = lat.toFixed(6);
        });

        console.log('地图初始化成功');
        return true;
    } catch (e) {
        console.error('地图初始化异常:', e);
        mapEl.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#ff4d4f;padding:20px;text-align:center;">' +
            `地图初始化失败: ${e.message}</div>`;
        return false;
    }
}

// 切换卫星图
window.toggleSatellite = function(useSatellite) {
    if (!state.map) return;
    state.useSatellite = useSatellite;

    if (useSatellite) {
        state.map.add(state.satelliteLayer);
        const showRoadNet = document.getElementById('showRoadNet')?.checked;
        if (showRoadNet) {
            state.map.add(state.roadNetLayer);
        }
    } else {
        state.map.remove(state.satelliteLayer);
        state.map.remove(state.roadNetLayer);
    }
};

window.toggleRoadNet = function(show) {
    if (!state.map || !state.useSatellite) return;
    if (show) {
        state.map.add(state.roadNetLayer);
    } else {
        state.map.remove(state.roadNetLayer);
    }
};

window.toggleLabels = function(show) {
    if (!state.map) return;
    state.map.setShowLabel(show);
};

window.clearMarkers = function() {
    if (!state.map) return;
    state.markers.forEach(m => state.map.remove(m));
    state.markers = [];
};

function addMarker(lng, lat, label) {
    if (!state.map) return;
    const marker = new AMap.Marker({
        position: [lng, lat],
        title: label,
        // 设置锚点在底部中心，确保标记位置准确
        anchor: 'bottom-center'
    });
    state.map.add(marker);
    state.markers.push(marker);
    state.map.setCenter([lng, lat]);
}

// ==================== 聊天 ====================
async function sendMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    if (!message) return;

    addChatMessage('user', message);
    input.value = '';

    // 将用户消息添加到历史
    state.conversationHistory.push({
        role: 'user',
        content: message
    });

    const loadingId = addChatMessage('assistant', '正在思考...');

    try {
        const response = await API.chat(message, state.conversationHistory);
        removeMessage(loadingId);
        addChatMessage('assistant', response.response);

        // 将AI回复添加到历史
        state.conversationHistory.push({
            role: 'assistant',
            content: response.response
        });

        // 如果有纠错提示，显示警告
        if (response.warnings && response.warnings.length > 0) {
            setTimeout(() => {
                addChatMessage('system', `⚠️ 检测到可能的问题：\n${response.warnings.join('\n')}`);
            }, 500);
        }
    } catch (error) {
        removeMessage(loadingId);
        addChatMessage('assistant', `出错了：${error.message}`);
    }
}

function addChatMessage(role, content) {
    const container = document.getElementById('chatMessages');
    const id = 'msg-' + Date.now();
    const div = document.createElement('div');
    div.id = id;
    div.className = `message ${role}`;
    div.innerHTML = `<div class="message-content">${content}</div>`;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return id;
}

function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// 清空对话历史
window.clearChatHistory = function() {
    state.conversationHistory = [];
    state.conversationId = 'conv-' + Date.now();
    const container = document.getElementById('chatMessages');
    container.innerHTML = '';
    addChatMessage('system', '对话历史已清空');
};

// ==================== 快速转换 ====================
async function quickConvert() {
    const lng = parseFloat(document.getElementById('lngInput').value);
    const lat = parseFloat(document.getElementById('latInput').value);
    const fromCoord = document.getElementById('fromCoord').value;
    const toCoord = document.getElementById('toCoord').value;

    if (isNaN(lng) || isNaN(lat)) {
        alert('请输入有效的经纬度');
        return;
    }

    if (!isInsideChina(lng, lat)) {
        document.getElementById('convertNote').classList.remove('hidden');
        document.getElementById('convertNote').innerHTML = '<small style="color:var(--warning-color)">⚠️ 该坐标位于中国境外</small>';
    } else {
        document.getElementById('convertNote').classList.add('hidden');
    }

    const btn = document.getElementById('convertBtn');
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<span>⏳ 转换中...</span>';

    try {
        const result = await API.convertSingle(lng, lat, fromCoord, toCoord);
        if (result.success) {
            document.getElementById('originalCoord').textContent = `${result.original.lng.toFixed(6)}, ${result.original.lat.toFixed(6)}`;
            document.getElementById('convertedCoord').textContent = `${result.converted.lng.toFixed(6)}, ${result.converted.lat.toFixed(6)}`;
            document.getElementById('convertResult').classList.remove('hidden');
            state.lastConvert = { original: result.original, converted: result.converted };
        }
    } catch (error) {
        alert(`转换失败：${error.message}`);
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

function addConvertMarker() {
    if (!state.lastConvert) {
        alert('请先进行坐标转换');
        return;
    }
    clearMarkers();
    addMarker(state.lastConvert.converted.lng, state.lastConvert.converted.lat, '转换结果');
}

// ==================== Excel ====================
let currentUploadTaskId = null;

async function handleFileUpload(file) {
    const progressDiv = document.getElementById('uploadProgress');
    const resultDiv = document.getElementById('uploadResult');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');

    progressDiv.classList.remove('hidden');
    resultDiv.classList.add('hidden');
    progressText.textContent = '正在分析...';

    try {
        const analysis = await API.analyzeExcel(file);
        if (!analysis.success) throw new Error(analysis.error);

        const { longitude_column, latitude_column } = analysis.analysis;
        progressText.textContent = `已识别: 经度=${longitude_column}, 纬度=${latitude_column}`;

        const result = await API.convertExcel(file, {
            longitude_column,
            latitude_column,
            from_coord: 'wgs84',
            to_coord: 'gcj02'
        });

        currentUploadTaskId = result.task_id;
        pollTaskStatus(result.task_id);
    } catch (error) {
        progressDiv.classList.add('hidden');
        alert(`上传失败：${error.message}`);
    }
}

function pollTaskStatus(taskId) {
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const resultDiv = document.getElementById('uploadResult');
    const progressDivEl = document.getElementById('uploadProgress');

    const interval = setInterval(async () => {
        try {
            const status = await API.getTaskStatus(taskId);
            progressFill.style.width = `${status.progress}%`;
            progressText.textContent = `转换中 ${status.processed_rows}/${status.total_rows} (${status.progress}%)`;

            if (status.status === 'completed') {
                clearInterval(interval);
                progressDivEl.classList.add('hidden');
                resultDiv.classList.remove('hidden');
                document.getElementById('uploadSummary').textContent = `完成！${status.total_rows} 行`;
            } else if (status.status === 'failed') {
                clearInterval(interval);
                progressDivEl.classList.add('hidden');
                alert(`失败：${status.error}`);
            }
        } catch (error) {
            clearInterval(interval);
            progressDivEl.classList.add('hidden');
        }
    }, 1000);
}

// ==================== 事件绑定 ====================
function bindEvents() {
    // 转换
    const convertBtn = document.getElementById('convertBtn');
    if (convertBtn) {
        const newBtn = convertBtn.cloneNode(true);
        convertBtn.parentNode.replaceChild(newBtn, convertBtn);
        newBtn.addEventListener('click', quickConvert);
    }

    const addMarkerBtn = document.getElementById('addMarkerBtn');
    if (addMarkerBtn) {
        addMarkerBtn.addEventListener('click', addConvertMarker);
    }

    // 聊天
    const sendBtn = document.getElementById('sendBtn');
    if (sendBtn) {
        sendBtn.addEventListener('click', sendMessage);
    }

    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }

    // 地图控制
    document.getElementById('useSatellite').addEventListener('change', (e) => {
        toggleSatellite(e.target.checked);
    });

    document.getElementById('showRoadNet').addEventListener('change', (e) => {
        toggleRoadNet(e.target.checked);
    });

    document.getElementById('showLabels').addEventListener('change', (e) => {
        toggleLabels(e.target.checked);
    });

    // Excel
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('excelFile');

    if (uploadArea && fileInput) {
        uploadArea.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => {
            if (e.target.files[0]) handleFileUpload(e.target.files[0]);
        });

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files[0]) handleFileUpload(e.dataTransfer.files[0]);
        });
    }

    document.querySelectorAll('.btn-download').forEach(btn => {
        btn.addEventListener('click', () => {
            if (currentUploadTaskId) {
                API.downloadFile(currentUploadTaskId, btn.dataset.format);
            }
        });
    });
}

// 复制坐标到剪贴板
window.copyCoordinate = function(lng, lat) {
    const text = `${lng.toFixed(6)}, ${lat.toFixed(6)}`;
    navigator.clipboard.writeText(text).then(() => {
        alert(`坐标已复制: ${text}`);
    }).catch(() => {
        // 降级方案
        const textarea = document.createElement('textarea');
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        alert(`坐标已复制: ${text}`);
    });

    // 关闭信息窗口
    if (state.infoWindow) {
        state.infoWindow.close();
    }
};

// ==================== 初始化 ====================
function initApp() {
    bindEvents();

    if (typeof AMap === 'undefined') {
        let attempts = 0;
        const checkAMap = setInterval(() => {
            attempts++;
            if (typeof AMap !== 'undefined') {
                clearInterval(checkAMap);
                initMap();
                console.log('=== 应用初始化完成 ===');
            } else if (attempts > 50) {
                clearInterval(checkAMap);
                initMap();
            }
        }, 100);
    } else {
        initMap();
        console.log('=== 应用初始化完成 ===');
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}
