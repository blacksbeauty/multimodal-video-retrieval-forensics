from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseModel):
    """系统全局配置（pydantic 模型，get_settings() 单例缓存）。

    目录约定：所有数据目录默认位于项目根（BASE_DIR）下；
    通道开关（enable_*）：关闭某通道后对应检索模块被跳过，系统自动降级
    到剩余可用通道，而不是整体报错（P5 降级体系）。
    """

    project_name: str = "CLIP + FAISS Video Retrieval"
    api_prefix: str = "/api"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    # Code Review Nice to Have：CORS 允许来源白名单。禁止 "allow_origins=*" 与
    # allow_credentials=True 组合（浏览器规范拒绝且任意站点可调用 API）。
    # 本地开发默认放开本机；部署时改为实际前端域名。
    cors_allow_origins: list[str] = Field(
        default_factory=lambda: ["http://127.0.0.1:8000", "http://localhost:8000"]
    )

    videos_dir: Path = Field(default=BASE_DIR / "videos")
    frames_dir: Path = Field(default=BASE_DIR / "frames")
    embeddings_dir: Path = Field(default=BASE_DIR / "embeddings")
    index_dir: Path = Field(default=BASE_DIR / "index")
    metadata_dir: Path = Field(default=BASE_DIR / "metadata")
    dataset_metadata_dir: Path = Field(default=BASE_DIR / "metadata" / "datasets")
    ocr_metadata_dir: Path = Field(default=BASE_DIR / "metadata" / "ocr")
    detection_metadata_dir: Path = Field(default=BASE_DIR / "metadata" / "detections")
    trajectory_metadata_dir: Path = Field(default=BASE_DIR / "metadata" / "trajectories")
    event_metadata_dir: Path = Field(default=BASE_DIR / "metadata" / "events")
    # S4: 段级元数据目录（含段向量 meta 聚合，数据规范 v1.1 §3.8/§3.9）
    segment_metadata_dir: Path = Field(default=BASE_DIR / "metadata" / "segments")
    event_config_dir: Path = Field(default=BASE_DIR / "configs" / "events")
    web_dir: Path = Field(default=BASE_DIR / "web")

    faiss_index_path: Path = Field(default=BASE_DIR / "index" / "video_frames.index")
    # S6: 段级索引元数据文件名（原 frame_metadata.json → segment_meta.json，数据规范 v1.1 §3.9）
    metadata_path: Path = Field(default=BASE_DIR / "index" / "segment_meta.json")

    # S2: 中文编码优先 CN-CLIP（依据：CN-CLIP中文语义泛化评测报告.md /
    # 基于中文CLIP的多模态交通视频语义检索适配方案.md）。
    # 约束：更换编码模型必须全量重建 embeddings/ 与 index/（数据规范 v1.1 §3.9 / §6-6）。
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    clip_backend: str = "cnclip"  # 中文场景默认 cnclip；openclip 仅作图像基线
    cnclip_model_name: str = "ViT-B-16"
    cnclip_download_root: Path = Field(default=BASE_DIR / "ckpts")
    cnclip_use_modelscope: bool = True
    device: str = "auto"

    default_frame_interval: int = 30
    # S3: 跨层时间戳对齐容差（秒）。各层 timestamp 统一 round(秒, 2) 存储；
    # 对齐优先用 frame_index 精确匹配，仅时间戳比较时用 abs(diff) < TIMESTAMP_EPSILON
    # （依据：数据规范 v1.1 §0 / §6-3）。
    timestamp_epsilon: float = 1e-6
    max_search_results: int = 10
    clip_score_threshold: float = 0.2
    ocr_score_threshold: float = 0.6
    hybrid_score_threshold: float = 0.2
    segment_window_seconds: float = 5.0
    hybrid_candidate_multiplier: int = 3
    detection_model_name: str = "yolov8n.pt"
    detection_score_threshold: float = 0.25
    tracking_frame_rate: int = 30
    tracking_high_thresh: float = 0.25
    tracking_low_thresh: float = 0.1
    tracking_new_track_thresh: float = 0.25
    tracking_track_buffer: int = 30
    tracking_match_thresh: float = 0.8
    tracking_fuse_score: bool = True
    trajectory_direction_min_displacement: float = 20.0
    event_plugin_names: list[str] = Field(
        default_factory=lambda: ["vehicle_crosses_line", "wrong_way_driving", "red_light_violation"]
    )
    streetscene_dataset_dir: Path = Field(
        default=BASE_DIR / "datasets" / "streetscene" / "StreetScene" / "StreetScene"
    )
    streetscene_frame_rate: float = 15.0
    streetscene_default_frame_step: int = 1

    accident_dataset_dir: Path = Field(
        default=BASE_DIR.parent / "type1_subtype1_accident"
    )
    accident_frame_rate: float = 20.0
    accident_default_frame_step: int = 1

    # 交通感知视频预处理层配置
    enable_traffic_filter: bool = True
    traffic_sample_interval: int = 30
    traffic_retain_window: int = 60

    # 检索通道开关（P5 降级体系）：关闭某通道后，对应检索模块被跳过，
    # 系统自动降级到剩余可用通道，而不是整体报错。
    enable_clip: bool = True
    enable_faiss: bool = True
    # [收敛至 /api/search/hybrid] OCR 通道临时关闭：metadata/ocr 目前 0 数据，
    # 避免空通道在每次 hybrid 检索时白跑。待灌入 OCR 元数据后改回 True。
    enable_ocr: bool = False
    enable_event: bool = True
    enable_detection: bool = True
    enable_trajectory: bool = True

    # S7: 段级索引开关。True: 检索包含段级记录；False: 一键回滚到帧级
    # （段级数据保留，随时可重新开启；数据规范 v1.1 §3.8 / 段级FAISS索引适配方案 §5）。
    use_segment_index: bool = True

    # 视频剪辑（/api/search/download_clip 端点）
    clip_max_duration_sec: float = 60.0
    clip_ffmpeg_timeout_sec: float = 10.0
    # 同时运行的 ffmpeg 剪辑进程上限；满载时端点返回 429
    clip_max_concurrent: int = 2
    # True: 重编码为 H.264（浏览器 <video> 必播；XVID/mp4v 等编码浏览器不支持）。
    # False: -c copy 无损流拷贝（仅当源编码浏览器可播时使用）。
    clip_reencode: bool = True
    # 剪辑源文件白名单的额外根目录（默认空）。
    # 默认仅允许 videos_dir + 两个数据集目录；第三方/测试视频目录通过此项挂载
    # （安全约束，Code Review Must Fix #8：防 download_clip 读取任意文件）。
    clip_allowed_roots: list[Path] = Field(default_factory=list)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
