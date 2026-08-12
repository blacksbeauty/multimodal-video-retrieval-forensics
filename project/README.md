# 全视智眼 · 自然语言智能视频检索与取证系统

> 用一句话说"黑色汽车 2026-05-07 14:23 闯红灯返回视频"，系统就能找到对应片段。

---

## 1. 项目背景与动机

传统视频检索依赖文件名、时间范围或人工标签筛选。想从一段 10 小时的路口监控里找到"凌晨三点那辆逆行的白色轿车"，要么逐帧回放，要么靠预标注的事件时间点跳转——两种方式都把检索成本压在人力上。

更深层的问题是，关键词匹配只能命中"字面相同"的内容。用户输入"车辆逆行"，系统如果只做字符串比对，就找不到轨迹方向与道路标注方向相反的片段，因为"逆行"这个动作并没有出现在任何文本字段里——它是时空行为，不是标签。

本项目把**事件结构**和**用户意图**作为检索的核心抽象。系统先用 YOLO + ByteTrack 把视频里的运动目标检测出来、跟踪出轨迹，再用可插拔的事件插件判断"逆行""闯红灯""压线"等行为是否发生，最终把这些结构化事件与 CLIP 视觉语义、OCR 文本、检测标签、轨迹方向一起纳入多路径混合检索。用户说"逆行"，系统不是去匹配"逆行"两个字，而是去找轨迹方向与道路方向夹角大于阈值的片段。

一句话概括检索流程：

```
用户中文自然语言
      ↓
意图解析（是找事件？找目标？还是找复合条件？）
      ↓
多路径动态路由（CLIP / Detection / Trajectory / Event / OCR）
      ↓
段级融合与排序
      ↓
返回视频片段
```

---

## 2. 核心功能

### 事件检测与结构化抽取

系统通过可插拔的事件插件引擎检测交通违章行为。每个插件继承 `EventPluginBase`，接收检测元数据和轨迹数据，输出结构化事件记录（事件类型、起止时间、置信度、关联轨迹 ID）。

当前已实现三种事件插件：

- **逆行检测**（`wrong_way_driving`）— 轨迹方向与道路标注方向的点积低于阈值
- **闯红灯**（`red_light_violation`）— 车辆在红灯期间越过停止线
- **压线行驶**（`vehicle_crosses_line`）— 车辆轨迹与车道线段相交

每个插件通过 JSON 配置文件调整阈值，支持 ROI 多边形过滤，无需改代码。

### 查询意图分类

`QueryRewriteService` 解析中文查询，识别实体（汽车、行人、红绿灯）、事件类型（逆行、闯红灯）、属性（颜色、灯态）和空间关系，输出 `QueryIntent` 结构体。系统根据意图类型选择不同的检索路径组合和权重分配。

意图分类器当前支持六种类型：

| 意图类型 | 触发条件 | 典型查询 |
| --- | --- | --- |
| `event` | 命中事件别名 | "车辆逆行" |
| `object` | 仅含实体 | "白色汽车" |
| `motion` | 含方向/动作 | "向右转的车辆" |
| `composite` | 实体 + 属性/方向 | "白色汽车向右转" |
| `relational` | 实体 + 关系 | "汽车在红绿灯附近" |
| `attribute` | 仅含属性 | "红色的" |

### 多路径动态检索路由

`HybridSearchService` 根据查询意图动态分配检索路径和权重。`event` 类型查询直接走事件检索，跳过 CLIP 和检测通道；`composite` 类型查询则五路并行检索后融合。

当前权重配置示例（event 意图）：

| 通道 | 权重 |
| --- | --- |
| CLIP 语义 | 0.05 |
| Detection | 0.10 |
| Trajectory | 0.15 |
| Event | 0.65 |
| OCR | 0.05 |

### 结果融合与排序

`ResultAggregationService` 把帧级检索结果按时间窗口聚合成段级结果，每段保留最佳匹配帧、综合得分和命中的模态列表，避免同一事件被多次返回。

---

## 3. 系统架构

```
视频输入 (RTSP / 文件)
      ↓
Traffic-Aware Preprocessing Layer        ← 交通感知预处理（基于 YOLO 抽样过滤空道路帧）
      ↓
关键帧提取 (FrameExtractor)
      ↓
┌──────────────┬──────────────┬──────────────┐
│              │              │              │
CLIP 编码      YOLOv8 检测    PaddleOCR      │
(embedding)    (detections)  (ocr text)     │
│              │              │              │
│         ByteTrack 跟踪        │              │
│              │              │              │
│      Event Plugin Engine     │              │
│      (逆行/闯红灯/压线)       │              │
│              │              │              │
└──────────────┴──────────────┴──────────────┘
      ↓
HybridSearchService (多路径动态路由 + 段级融合)
      ↓
中文自然语言查询结果
```

所有元数据以 JSON 文件存储，不依赖数据库。向量索引使用 FAISS 或 numpy 矩阵，按视频隔离管理。

---

## 4. 快速开始

### 环境要求

- Python >= 3.10
- 操作系统：Windows / Linux / macOS
- 依赖包见 `requirements.txt`

### 安装

```bash
git clone <your-repo-url>
cd <project-dir>

# 创建虚拟环境（推荐）
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 最小化运行

```bash
cd project

# 启动 API 服务
python main.py
```

服务默认监听 `http://localhost:8000`，API 文档访问 `/docs`。

摄入视频并建立索引：

```bash
# 摄入单个视频
curl -X POST "http://localhost:8000/api/videos/ingest?video_path=MVI_40851.avi&frame_interval=30"

# 中文检索
curl "http://localhost:8000/api/search/hybrid?query=白色汽车逆行&top_k=10"
```

Python SDK 调用示例：

```python
import requests

# 摄入视频
resp = requests.post(
    "http://localhost:8000/api/videos/ingest",
    params={"video_path": "MVI_40851.avi", "frame_interval": 30},
)
print(resp.json())

# 中文自然语言检索
resp = requests.get(
    "http://localhost:8000/api/search/hybrid",
    params={"query": "白色汽车逆行", "top_k": 10},
)
for result in resp.json()["results"]:
    print(f"{result['video_name']} @ {result['start_ts']}s score={result['best_score']:.3f}")
```

---

## 5. 使用说明

### API 端点概览

| 端点 | 方法 | 说明 |
| --- | --- | --- |
| `/api/videos/ingest` | POST | 摄入单个视频，自动提取帧、编码、建索引 |
| `/api/videos/ingest-directory` | POST | 批量摄入目录下所有视频 |
| `/api/search/hybrid` | GET | 多模态混合检索（主入口） |
| `/api/search/clip` | GET | 纯 CLIP 语义检索 |
| `/api/detection/search` | GET | 检测标签检索 |
| `/api/trajectory/search` | GET | 轨迹方向检索 |
| `/api/event/search` | GET | 事件类型检索 |
| `/api/ocr/search` | GET | OCR 文本检索 |
| `/api/detection/ingest-directory` | POST | 批量生成检测元数据 |
| `/api/tracking/generate` | POST | 从检测元数据生成轨迹 |
| `/api/event/generate` | POST | 从检测+轨迹生成事件元数据 |

### 启用交通感知过滤

在 `config.py` 中设置：

```python
enable_traffic_filter: bool = True       # 启用过滤
traffic_sample_interval: int = 30         # 每 30 个采样帧做一次 YOLO 抽样
traffic_retain_window: int = 60           # 检测到目标后保留 60 个采样帧
```

### 新增事件插件

1. 在 `services/event_plugins/` 下新建插件文件，继承 `EventPluginBase`
2. 在 `__init__.py` 中导入并注册
3. 在 `config.py` 的 `event_plugin_names` 列表中添加插件名
4. 在 `configs/events/` 下创建 JSON 配置文件

---

## 6. 技术栈

| 模块 | 技术 / 框架 |
| --- | --- |
| 视觉语义编码 | CN-CLIP ViT-B-16 / OpenCLIP ViT-B-32 |
| 目标检测 | YOLOv8 (ultralytics) |
| 目标跟踪 | ByteTrack |
| OCR 文本提取 | PaddleOCR |
| 向量索引 | FAISS / numpy |
| 事件检测引擎 | 自研可插拔插件架构 |
| 查询意图解析 | 确定性规则引擎（交通 Ontology） |
| Web 框架 | FastAPI + Uvicorn |
| 配置管理 | Pydantic Settings |
| 元数据存储 | 本地 JSON 文件（无数据库） |

---

## 7. 项目优势与适用场景

### 相比传统检索的优势

传统视频检索依赖人工标签或文件名匹配，无法理解"逆行""闯红灯"这类时空行为语义。本项目把检测、跟踪、事件检测和视觉语义编码统一到一个多路径检索框架里，用户用自然语言描述事件，系统自动路由到最相关的检索路径并融合结果。

### 相比 RAG 的区别

RAG 从文档中检索文本片段再喂给 LLM 生成回答。本系统检索的是视频片段，融合的是视觉语义、目标检测结果、轨迹行为和事件规则——四种模态的信号互补，不依赖 LLM 在线推理，适合边缘部署。

### 适用场景

- 交通违法取证（逆行、闯红灯、压线）
- 路口监控视频事件检索
- 多摄像头时空关联查询
- 边缘计算盒子上的轻量化视频分析

---

## 8. 贡献指南与许可证

### 贡献指南

欢迎通过 Issue 和 Pull Request 贡献代码。提交前请：

- 运行 `python -m pytest tests/` 确保测试通过
- 新增事件插件时，在 `tests/` 下添加对应测试
- 遵循现有的 Git 提交格式：`type: description`（如 `feat: add illegal parking event plugin`）
- 不引入微服务、Docker 或数据库依赖

### 许可证

MIT License
