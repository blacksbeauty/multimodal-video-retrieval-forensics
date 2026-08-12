from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseModel):
    project_name: str = "CLIP + FAISS Video Retrieval"
    api_prefix: str = "/api"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

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
    event_config_dir: Path = Field(default=BASE_DIR / "configs" / "events")
    web_dir: Path = Field(default=BASE_DIR / "web")

    faiss_index_path: Path = Field(default=BASE_DIR / "index" / "video_frames.index")
    metadata_path: Path = Field(default=BASE_DIR / "index" / "frame_metadata.json")

    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    clip_backend: str = "cnclip"
    cnclip_model_name: str = "ViT-B-16"
    cnclip_download_root: Path = Field(default=BASE_DIR / "ckpts")
    cnclip_use_modelscope: bool = True
    device: str = "auto"

    default_frame_interval: int = 30
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
    enable_traffic_filter: bool = False
    traffic_sample_interval: int = 30
    traffic_retain_window: int = 60


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
