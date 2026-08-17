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

`HybridSearchService` 根据查询意图动态分配检索路径和权重（`generate_dynamic_weights`）。所有通道（CLIP / Detection / Trajectory / Event / OCR）总是并行运行，再按意图权重融合：`event` 意图下 Event 通道权重最高（0.6~0.65），`object`/`attribute` 意图下 CLIP 与 Detection 主导。

当前权重配置示例（高置信 event 意图，无视觉属性）：

| 通道 | 权重 |
| --- | --- |
| CLIP 语义 | 0.05 |
| Detection | 0.10 |
| Trajectory | 0.15 |
| Event | 0.65 |
| OCR | 0.05 |

权重随意图动态变化：带颜色/车型属性的 event 查询会提升 CLIP 权重（0.35），普通目标查询则以 CLIP（0.50）与 Detection（0.25）为主。

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

### CN-CLIP 依赖安装（Windows 必读）

系统默认语义编码后端为 **CN-CLIP**（`config.py` 中 `clip_backend: str = "cnclip"`），中文查询（如"白色汽车""闯红灯"）依赖它。PyPI 上的 `cn-clip` 包在 Windows 上因 `lmdb==1.3.0` 源码构建失败（需要 `patch-ng` 且无预编译 wheel）**无法通过 `pip install` 安装**，需手动部署官方源码：

```bash
# 1. 下载官方源码（master 分支）
curl -L -o Chinese-CLIP.zip "https://codeload.github.com/OFA-Sys/Chinese-CLIP/zip/refs/heads/master"
unzip Chinese-CLIP.zip

# 2. 将纯 Python 的 cn_clip 包放入 site-packages
#    （Windows 示例，路径以你的 Python 环境为准）
cp -r Chinese-CLIP-master/cn_clip D:/software/python311/Lib/site-packages/cn_clip

# 3. 验证
python -c "import cn_clip.clip; from cn_clip.clip import load_from_name; print('cn_clip OK')"
```

> 提示：cn_clip 运行时实际不依赖 lmdb（lmdb 仅用于离线特征缓存），手动部署纯 Python 包即可正常工作。模型权重 `clip_cn_vit-b-16.pt` 放在 `ckpts/` 目录（`config.py` 中 `cnclip_download_root`），加载时自动命中本地文件，无需联网下载。

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

全部业务端点挂在 `/api` 前缀下，可用浏览器访问 `/docs`（Swagger UI）在线调试。

#### 检索 / 资源类

| 方法 | 端点 | 功能 | 主要参数 | 返回 |
| --- | --- | --- | --- | --- |
| GET | `/api/health` | 服务健康检查 + 通道可用性 | 无 | JSON |
| GET | `/api/index/stats` | FAISS 索引统计（帧/视频数） | 无 | JSON |
| POST | `/api/search` | 纯 CLIP 语义检索 | `query`, `top_k` | JSON |
| POST | `/api/search/hybrid` | 多模态混合检索（主入口，5 通道融合） | `query`, `top_k` | JSON（含 `clip_url`） |
| GET | `/api/search/download_clip` | 视频片段下载（FFmpeg 无损剪辑，并发受限 429） | `video_path`, `start_ts`, `end_ts`, `output_name` | MP4 文件流 |
| POST | `/api/detection/search` | 检测标签检索 | `query`, `top_k` | JSON |
| POST | `/api/trajectory/search` | 轨迹检索 | `query`, `direction`, `min_duration_sec`, `top_k` | JSON |
| POST | `/api/event/search` | 事件类型检索（逆行/闯红灯/压线） | `query`, `top_k` | JSON |
| POST | `/api/ocr/search` | OCR 文本检索 | `query`, `top_k` | JSON |

#### 摄入 / 处理类

| 方法 | 端点 | 功能 | 主要参数 | 返回 |
| --- | --- | --- | --- | --- |
| POST | `/api/videos/ingest` | 摄入单个视频（抽帧→编码→建索引） | `video_path`, `frame_interval` | JSON |
| POST | `/api/videos/ingest-directory` | 批量摄入视频目录 | `directory`, `frame_interval` | JSON |
| POST | `/api/datasets/streetscene/ingest` | StreetScene 数据集导入 | `dataset_root`, `split`, `max_sequences`, `frame_step` | JSON |
| POST | `/api/datasets/accident/ingest` | CARLA 事故数据集导入 | `dataset_root`, `scenario_names`, `max_scenarios`, `frame_step` | JSON |
| POST | `/api/detection/ingest-directory` | 批量生成 YOLO 检测元数据 | `frames_dir` | JSON |
| POST | `/api/tracking/ingest-directory` | 从检测元数据生成 ByteTrack 轨迹 | `metadata_dir` | JSON |
| POST | `/api/event/ingest-directory` | 从检测+轨迹生成事件元数据 | `detection_metadata_dir`, `trajectory_metadata_dir`, `plugin_names` | JSON |
| POST | `/api/ocr/ingest-directory` | 批量生成 PaddleOCR 元数据 | `frames_dir` | JSON |

#### 网页路由（非 API）

| 方法 | 路径 | 功能 |
| --- | --- | --- |
| GET | `/` | 检索控制台页面（HTML） |
| POST | `/search` | 页面搜索（兼容入口） |

> 摄入流水线：`videos/ingest → detection/ingest-directory → tracking/ingest-directory → event/ingest-directory`；
> 检索主入口：`POST /api/search/hybrid`（返回段级结果，每项含 `clip_url` 可直接作为 `<video>` 的 src 播放）。

### 视频片段下载 `GET /api/search/download_clip`

对检索结果中的视频片段做无损剪辑（FFmpeg `-c copy` 流拷贝，不重编码），直接返回 MP4 文件流：

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `video_path` | str | 源视频路径（必须存在） |
| `start_ts` | float | 片段开始时间（秒） |
| `end_ts` | float | 片段结束时间（秒），**必须大于 `start_ts`** |
| `output_name` | str? | 下载文件名的 stem（可选，默认取源视频名） |

限制与错误码（阈值见 `config.py` 剪辑配置）：

| 条件 | 响应 |
| --- | --- |
| `end_ts <= start_ts` | `400 Invalid clip range` |
| 片段时长 > `clip_max_duration_sec`（默认 60s） | `400` |
| 源视频不存在 | `404` |
| ffmpeg 超过 `clip_ffmpeg_timeout_sec`（默认 10s）未完成 | `504` |
| 同时运行的剪辑数已达 `clip_max_concurrent`（默认 2） | `429 Too many concurrent clip requests` |
| 其他 ffmpeg 失败 | `500` |

> 端点以同步方式运行于 FastAPI 线程池，ffmpeg 子进程不会阻塞事件循环；并发剪辑由信号量限制，防止恶意请求同时拉起大量 ffmpeg 打满 CPU。临时片段存放于 `/dev/shm`（或系统临时目录），响应发送后异步清理。

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
