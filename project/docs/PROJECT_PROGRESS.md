# 交通事件自然语言检索系统 — 项目进展现状与实现原理

> 更新日期：2026-08-17 · 分支：main · 仓库：`multimodal-video-retrieval-forensics`
> 定位：面向交通违法取证的**多模态视频语义检索系统**——用自然语言检索监控视频中的事件（闯红灯 / 压线 / 逆行），并输出符合取证规范的**三帧关键帧快照**。

---

## 1. 系统架构总览

```
┌────────────────────────────── Web 界面层 ──────────────────────────────┐
│  search.html（检索主页）        key_snapshots_demo.html（三帧取证界面）   │
│  页面路由: /  /search  /key-snapshots  /video-probe                     │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │ fetch POST /api/search/hybrid
┌────────────────────────────── API 层（FastAPI, api/routes.py）─────────┐
│ 检索: /search/hybrid(唯一入口)  摄取: /videos/ingest  /detection/…      │
│       /tracking/…  /event/…  /ocr/…  /ingest/segments/{id}             │
│ 片段: /search/download_clip(ffmpeg 剪辑)   状态: /health  /index/stats  │
└──────────────────────────────────┬──────────────────────────────────────┘
┌────────────────────────────── 服务层（services/, 20+ 服务）────────────┐
│ 检索链路: HybridSearchService → 5 通道(Clip/Detection/Trajectory/      │
│           Event/OCR) → ResultAggregationService → 融合打分             │
│ 摄取链路: VideoService → EmbeddingService(CN-CLIP) → IndexService      │
│           → DetectionService(YOLOv8) → TrackingService(ByteTrack)      │
│           → EventService(3 插件) → SegmentBuildService                 │
│ 基础设施: FaissIndexService(IndexIDMap 增量)  VideoClipService(ffmpeg) │
└──────────────────────────────────┬──────────────────────────────────────┘
┌────────────────────────────── 存储层 ───────────────────────────────────┐
│ videos/  frames/(2171帧)  embeddings/(CN-CLIP 512维向量)                │
│ index/(faiss 二进制 + segment_meta.json + video_mapping.json)           │
│ metadata/{detections,trajectories,events,segments,ocr,datasets}.json    │
└──────────────────────────────────────────────────────────────────────────┘
```

**技术栈**：Python 3.11 · FastAPI · OpenCV · Ultralytics YOLOv8 · ByteTrack · CN-CLIP(ViT-B-16) · FAISS(IndexFlatIP+IndexIDMap) · PaddleOCR · FFmpeg

---

## 2. 数据主链路（一次完整入库 → 检索）

```
视频文件 → 抽帧 → CLIP编码 → FAISS帧级索引
        → YOLO检测 → ByteTrack轨迹 → 事件插件(红绿灯/压线/逆行)
        → 三帧快照 → 段级文本索引 → hybrid检索
```

### 2.1 视频摄取（`services/video_service.py`）
- **抽帧**：`FrameExtractor` 按 `frame_interval`（短视频推荐 12≈2fps，长视频默认 30）从视频抽帧，帧名规范 `{视频名}_{时间戳:.2f}.jpg`（中文名保留，经 safe-name 净化）。
- **CLIP 编码**：`EmbeddingService` 用 CN-CLIP（中文预训练）对帧图编码为 512 维向量（L2 归一化）。
- **增量索引**：`IndexService.upsert_video_records` → FAISS `IndexIDMap.add_with_ids` 只追加新视频向量，**不重建全库**（O(新增) 而非 O(全库)）；批量导入（`ingest_directory` / 数据集）循环内不落盘、结束统一 `save_index()` 一次。
- **持久化**：embeddings 每视频一份 `.npy`+`.json` bundle；索引三文件（.index / segment_meta.json / video_mapping.json）**原子写入**（tmp + `os.replace`）。

### 2.2 FAISS 向量索引（`services/faiss_index_service.py`）
- **结构**：`IndexFlatIP`（内积=余弦相似度，向量已归一化）外包 `IndexIDMap`，外部 id 全局递增（`_next_id`），支持按 video_id 覆盖/删除。
- **append 语义**（关键修复）：帧级与段级共用同一 video_id，段级写入必须 `append=True`（否则覆盖语义会删光同视频帧级向量——历史事故，已修复并单测覆盖）。
- **检索**：`query → normalize → index.search → 关联 metadata`，返回 `{rank, score, index, **frame_metadata}`。
- **rebuild**：启动检测到 metadata 陈旧时，从 embeddings bundle 全量重建（跳过 `_seg_` 段级文件与缺 frame_id 的无效记录）。

### 2.3 目标检测（`services/detection_service.py`）
- **模型**：Ultralytics YOLOv8n（CPU 推理），类别：car / truck / bus / motorcycle / person / traffic light / bicycle / van。
- **`detect_frame(frame_path)`**：读图 → predict → 结构化输出（label / bbox / confidence / class_id）。
- **增量策略**：摄取脚本只对**新视频的帧**调用 detect_frame（按帧名 glob），避免全量重跑旧帧；结果按 video_id 聚合保存 `metadata/detections/{video_id}.json`。
- **中文路径兼容**（修复）：Windows 下 `cv2.imread` 无法打开中文路径，统一改用 `utils/frame_utils.read_image_cv`（`np.fromfile + cv2.imdecode`）。

### 2.4 轨迹构建（`services/tracking_service.py`）
- **算法**：ByteTrack 多目标跟踪，把检测框按帧关联成轨迹；每点含 timestamp / frame_path / bbox / 中心点 / confidence。
- **输出**：`TrajectoryTrackMetadata`（track_id、start/end_ts、duration、avg_confidence、direction、representative_frame、points）。
- **方向判定**：`trajectory_main_direction` 取轨迹中段位移归一化向量（抗首尾抖动），方向枚举：left_to_right / right_to_left / top_to_bottom / bottom_to_top / stationary / unknown。

### 2.5 事件插件（`services/event_plugins/`）
三个规则插件继承 `EventPluginBase`（注册制，`registry.py`），配置在 `configs/events/{plugin}.json`：

| 插件 | 事件 | 判定核心 |
|---|---|---|
| `red_light_violation` | 闯红灯 | 红灯状态 + 车辆越过停止线 + 越线后继续移动 + 持续红灯校验 |
| `vehicle_crosses_line` | 压线 | 车辆 bbox 与虚拟停止线相交（`geometry.find_line_contact`） |
| `wrong_way_driving` | 逆行 | 轨迹主方向与允许方向点积 < 阈值（反向） |

**红绿灯判色（`_classify_traffic_light_state`，重点）**：
- 对 bbox 裁剪区统计**亮灯像素**（HSV V>180，过滤灯壳/环境）的红/黄/绿比例；
- 用颜色像素**垂直重心**约束位置（红灯偏上 / 绿灯偏下 / 黄灯居中），`得分 = 比例 × 位置权重`；
- 遍历该帧**所有**信号灯候选检测、取"判色成功且置信度最高者"（解决远距小灯被大框淹没的问题）；
- confidence 非线性校准 `sqrt(score)×1.4`；`min_red_duration_sec` 校验红灯持续时长（默认 1.5s）。

**代表帧与三帧快照（取证规范）**：
- `representative_frame`：从事件**证据帧列表**取首帧（修复：曾用轨迹级代表帧导致选到事件窗外帧）。
- `key_snapshots`（三帧快照 `extract_three_keyframes`）：Frame_A 越线前（线上方+ts≤锚点、取离停止线最近点帧）/ Frame_B 越线中（点到线段距离≤5px）/ Frame_C 通过后（线下方+ts≥锚点）；逐帧兜底（证据帧首/中/尾）+ 去重补足；无停止线场景按时间均匀取首中尾。上下方以"点到线段投影点 y"为界（兼容竖直/倾斜线）。

### 2.6 段级索引（`services/segment_build_service.py`）
- 事件生成后，按事件时间窗把视频切成**语义段**（segment），用事件中文 description 做段文本；
- 段文本经 CLIP 编码进段级 FAISS（`append=True` 与帧级共存）；段元数据存 `metadata/segments/{video_id}.json` + `embeddings/{segment_id}.npy`。
- 目的：让"闯红灯"等事件文本可直接语义命中，不依赖帧画面相似度。

### 2.7 混合检索（`services/hybrid_search_service.py`）— 系统唯一检索入口
```
POST /api/search/hybrid {query, top_k}
  → QueryRewriteService（意图解析/近义改写/事件类型识别）
  → 五通道并行检索:
      clip        : 帧级语义（CN-CLIP 文本-图像）
      detection   : 检测框属性（类别/置信度）
      trajectory  : 轨迹方向/位移
      event       : 事件元数据（含 key_snapshots 透传）
      ocr         : 文本（当前 enable_ocr=False 关闭）
  → _fuse_results: 按 video_id+5s 时间桶合并多通道命中（段级按 segment_id）
  → ResultAggregationService: 意图加权融合打分、阈值过滤、top_k 截断
  → 附 clip_url（ffmpeg 片段下载）
```
- **通道降级**：某通道异常（模型缺失/数据为空）自动跳过，不影响其他通道。
- **key_snapshots 透传**：事件结果的三帧快照经 event 通道 → 融合 → 聚合 → `HybridSegmentResult` 完整透传到前端（曾因融合时 clip 先建条目导致快照丢失，已修复）。
- **单一入口约定**：5 个旧端点（/search、/detection/search、/trajectory/search、/event/search、/ocr/search）标记 deprecated 保留兼容，新检索一律走 hybrid。

### 2.8 OCR（`services/ocr_service.py`，当前关闭）
- PaddleOCR 中文识别，逐帧提取文本区域（文本/置信度/位置），存 `metadata/ocr/`；检索时 `matched_text` 参与融合。
- 状态：`enable_ocr=False`（元数据未灌入），灌入后改回 True 即可启用。

---

## 3. 视频片段剪辑（`services/clip_service.py`）

- **原理**：FFmpeg `-ss 快进 + -c copy（或 H.264 重编码）` 无损/兼容剪辑，`-movflags +faststart` 支持浏览器流式播放。
- **安全**：源视频路径**白名单**（仅受管 videos/ 目录）；`output_name` 经 `_safe_stem` 净化（防路径穿越）；时长上限 60s；并发信号量（默认 2）超限返回 429；子进程超时 10s；stderr 脱敏（替换源/输出路径）。
- **生命周期**：临时文件在响应发送后由 `BackgroundTask.cleanup` 删除。

---

## 4. Web 界面

| 页面 | 路由 | 说明 |
|---|---|---|
| 三帧取证界面（新主页） | `/` | 左 280px 检索区（自然语言搜索框/事件类型/Top-K）+ 结果列表；中央视频画布 + 时间轴（三帧竖线标记、可拖动）；右 320px 详情（三帧真实缩略图 + 时间戳角标 + 上/下一帧/播放）。深色安防风（#0F141C / 主色 #1677FF / 告警 #D93025），真实接入 `/api/search/hybrid` + `/frames` 静态帧图 |
| 检索主页（兼容） | `/search` | 原 search.html |
| 视频播放诊断 | `/video-probe` | 验证 FFmpeg 片段浏览器播放 |

---

## 5. API 端点清单（前缀 `/api`，页面路由除外）

| 方法 | 路径 | 功能 | 状态 |
|---|---|---|---|
| POST | /search/hybrid | **统一混合检索** | ✅ 推荐 |
| POST | /videos/ingest | 单视频摄取（抽帧+CLIP+索引） | ✅ |
| POST | /videos/ingest-directory | 目录批量摄取 | ✅ |
| POST | /detection/ingest-directory | 检测（全量重跑帧） | ✅ |
| POST | /tracking/ingest-directory | 轨迹 | ✅ |
| POST | /event/ingest-directory | 事件 | ✅ |
| POST | /ingest/segments/{video_id} | 段级索引 | ✅ |
| POST | /ocr/ingest-directory | OCR（关闭） | ⏸ |
| POST | /datasets/streetscene/ingest | StreetScene 数据集导入 | ✅ |
| POST | /datasets/accident/ingest | Accident 数据集导入 | ✅ |
| GET | /search/download_clip | ffmpeg 片段下载 | ✅ |
| GET | /health · /index/stats | 健康/索引统计 | ✅ |
| POST | /search · /detection/search · /trajectory/search · /event/search · /ocr/search | 旧检索端点 | ⚠️ deprecated |

---

## 6. 数据与文件布局

```
project/
├─ videos/          源视频（5 个闯红灯测试 + 14 个 AVI 数据集）
├─ frames/          2171 张抽取帧（中文/英文名）
├─ embeddings/      55 个视频的 CN-CLIP 向量 bundle（.npy + .json）+ 段向量
├─ index/           faiss 二进制 + segment_meta.json + video_mapping.json
├─ metadata/
│  ├─ detections/   18 个视频的 YOLO 检测
│  ├─ trajectories/ 18 个视频的 ByteTrack 轨迹
│  ├─ events/       18 个视频的事件（含 key_snapshots 三帧快照）
│  ├─ segments/     段级元数据
│  ├─ ocr/          OCR（空）
│  └─ datasets/     数据集 source map
├─ configs/events/  三个事件插件配置（含 line 停止线）
├─ web/             HTML 界面
├─ scripts/         ingest_single_video.py 单视频一键入库
└─ tests/           23 个测试文件
```

---

## 7. 关键配置（`config.py`，环境变量可覆盖）

| 分组 | 关键项 |
|---|---|
| 网络 | host/port、cors_allow_origins（白名单）、api_prefix |
| 检索 | max_search_results、clip/ocr/hybrid 分数阈值、segment_window_seconds、candidate_multiplier |
| 模型 | clip_model（CN-CLIP）、detection_model（YOLOv8n）、device |
| 路径 | 上述全部目录 |
| 事件 | tracking_frame_rate、事件配置在 configs/events/ |

---

## 8. 测试覆盖（86 项全部通过）

- 事件插件（判色鲁棒性合成图、越线判定、逆行方向、三帧快照 5 例）
- FAISS 增量（append/覆盖、原子持久化、rebuild）
- 混合检索 / 事件检索 / 轨迹检索 / 统一入口
- 数据集导入（StreetScene/Accident mock）
- 片段剪辑（白名单、范围校验、并发 429）
- OCR / 中文路径读图

---

## 9. 近期关键演进（2026-08-16 ~ 08-17）

1. **索引事故修复**：段级 upsert 覆盖语义曾删光同视频帧级向量 → `add_video(append=True)` + 全库数据恢复（rebuild + 段级重编码）
2. **代表帧修复**：从轨迹级改为事件证据帧首帧（避免选到事件窗外帧）
3. **中文路径读图**：`read_image_cv` 替换 3 处 `cv2.imread`（Windows 中文帧路径无法解码）
4. **三帧取证快照**：`extract_three_keyframes` + `key_snapshots` 字段 + hybrid 透传 + 前端展示
5. **红灯识别增强**：整图亮灯像素比例 + 垂直重心判色、多候选择优、H 区间拓宽——AI 生成测试视频召回从 1/4 提升到 2/4（视频2/3）
6. **工程加固**（第 2 轮 Code Review）：批量落盘、save_index 原子化、图像缓存去无界化、confidence 非线性校准、配置默认值安全化

---

## 10. 已知问题 / 待办

- 视频1（无停止线路口）、视频4（红灯与压线时间错开）不触发闯红灯事件——**内容/场景适配**，需逐视频 line 配置或插件升级（`traffic_synonym_dict._build_attribute_index` 优先级比较对象错误为遗留 P3，影响被消费端排序掩盖）
- OCR 通道关闭（元数据未灌入）
- 全站无鉴权、`/frames` 全量暴露——部署公网前需加访问控制
- FAISS 服务无锁（当前串行无并发，未来并发化需加锁）
- `ingest_segment_pipeline` 重跑会 append 重复段向量（建议按 video_id 先清段再构建）
- 大量未提交改动（20+ 文件）待 commit + push
