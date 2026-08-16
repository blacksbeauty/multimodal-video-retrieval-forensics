from __future__ import annotations

from dataclasses import dataclass

from config import Settings
from services.accident_dataset_import_service import AccidentDatasetImportService
from services.clip_service import VideoClipService
from services.detection_search_service import DetectionSearchService
from services.detection_service import DetectionService
from services.embedding_service import EmbeddingService
from services.event_search_service import EventSearchService
from services.event_service import EventService
from services.hybrid_search_service import HybridSearchService
from services.index_service import IndexService
from services.ocr_search_service import OCRSearchService
from services.ocr_service import OCRService
from services.query_rewrite_service import QueryRewriteService
from services.result_aggregation_service import ResultAggregationService
from services.search_service import SearchService
from services.segment_build_service import SegmentBuildService
from services.segment_service import SegmentService
from services.streetscene_import_service import StreetSceneImportService
from services.tracking_service import TrackingService
from services.trajectory_search_service import TrajectorySearchService
from services.video_service import VideoService


@dataclass(slots=True)
class AppState:
    """应用级服务容器（FastAPI 全生命周期单例）。

    作用：
      - 在应用启动时一次性构建全部服务并共享给所有请求（依赖注入 get_services）；
      - OCR 模型等重资源以单例持有，避免每个请求重复创建（PaddleOCR 只初始化一次）；
      - 提供通道开关对应的降级路径：某个服务不可用时检索自动降级到其他通道。
    """

    clip_service: EmbeddingService
    video_clip_service: VideoClipService
    video_service: VideoService
    index_service: IndexService
    search_service: SearchService
    streetscene_import_service: StreetSceneImportService
    accident_import_service: AccidentDatasetImportService
    ocr_service: OCRService
    ocr_search_service: OCRSearchService
    detection_service: DetectionService
    detection_search_service: DetectionSearchService
    event_service: EventService
    event_search_service: EventSearchService
    tracking_service: TrackingService
    trajectory_search_service: TrajectorySearchService
    query_rewrite_service: QueryRewriteService
    result_aggregation_service: ResultAggregationService
    hybrid_search_service: HybridSearchService
    segment_service: SegmentService
    segment_build_service: SegmentBuildService

    @classmethod
    def build(cls, settings: Settings) -> "AppState":
        """一次性构建全部服务（FastAPI lifespan 中调用）。

        注意：HybridSearchService 在此注入全部检索通道（clip / detection /
        trajectory / event / ocr），这就是 /api/search/hybrid 单入口能融合
        五通道的装配点。
        """
        clip_service = EmbeddingService(settings)
        video_clip_service = VideoClipService(settings)
        index_service = IndexService(settings)
        video_service = VideoService(settings)
        search_service = SearchService(settings, clip_service, index_service)
        streetscene_import_service = StreetSceneImportService(settings)
        accident_import_service = AccidentDatasetImportService(settings)
        ocr_service = OCRService(settings)
        ocr_search_service = OCRSearchService(settings)
        detection_service = DetectionService(settings)
        query_rewrite_service = QueryRewriteService(
            use_chinese_clip=settings.clip_backend.lower() == "cnclip"
        )
        detection_search_service = DetectionSearchService(settings, query_rewrite_service)
        event_service = EventService(settings)
        event_search_service = EventSearchService(settings, query_rewrite_service)
        tracking_service = TrackingService(settings)
        trajectory_search_service = TrajectorySearchService(settings, query_rewrite_service)
        result_aggregation_service = ResultAggregationService(settings)
        segment_service = SegmentService(settings)
        segment_build_service = SegmentBuildService(settings, segment_service)
        hybrid_search_service = HybridSearchService(
            settings=settings,
            query_rewrite_service=query_rewrite_service,
            clip_search_service=search_service,
            ocr_search_service=ocr_search_service,
            detection_retriever=detection_search_service,
            trajectory_retriever=trajectory_search_service,
            event_retriever=event_search_service,
            result_aggregation_service=result_aggregation_service,
        )

        return cls(
            clip_service=clip_service,
            video_clip_service=video_clip_service,
            video_service=video_service,
            index_service=index_service,
            search_service=search_service,
            streetscene_import_service=streetscene_import_service,
            accident_import_service=accident_import_service,
            ocr_service=ocr_service,
            ocr_search_service=ocr_search_service,
            detection_service=detection_service,
            detection_search_service=detection_search_service,
            event_service=event_service,
            event_search_service=event_search_service,
            tracking_service=tracking_service,
            trajectory_search_service=trajectory_search_service,
            query_rewrite_service=query_rewrite_service,
            result_aggregation_service=result_aggregation_service,
            hybrid_search_service=hybrid_search_service,
            segment_service=segment_service,
            segment_build_service=segment_build_service,
        )

    def initialize_ocr_once(self, preload: bool = False) -> None:
        """Optionally preload the OCR model once during app startup.

        By default the OCR model remains lazily loaded, but it still lives inside
        the singleton OCRService instance and will only initialize once.
        """
        if preload:
            self.ocr_service.load_model()
