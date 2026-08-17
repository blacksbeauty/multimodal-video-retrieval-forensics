import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTask

from config import Settings
from core.app_state import AppState
from services.clip_service import (
    ClipError,
    ClipInvalidRangeError,
    ClipNotFoundError,
    ClipTimeoutError,
    ClipTooLongError,
)
from core.schemas import (
    AccidentDatasetIngestRequest,
    AccidentDatasetIngestResponse,
    BatchIngestRequest,
    BatchIngestResponse,
    DetectionIngestRequest,
    DetectionSearchRequest,
    DetectionSearchResponse,
    EventIngestRequest,
    EventIngestResponse,
    EventSearchRequest,
    EventSearchResponse,
    HealthResponse,
    HybridSearchRequest,
    HybridSearchResponse,
    IndexStatsResponse,
    IngestRequest,
    IngestResponse,
    OCRIngestRequest,
    OCRSearchRequest,
    OCRSearchResponse,
    SearchRequest,
    SearchResponse,
    StreetSceneIngestRequest,
    StreetSceneIngestResponse,
    TrackingIngestRequest,
    TrackingIngestResponse,
    TrajectorySearchRequest,
    TrajectorySearchResponse,
)


logger = logging.getLogger(__name__)
page_router = APIRouter()
router = APIRouter()
templates = Jinja2Templates(
    directory=str(Path(__file__).resolve().parent.parent / "web" / "templates")
)


def get_services(request: Request) -> AppState:
    """Resolve shared application services from FastAPI state."""
    return request.app.state.services


def get_settings_dependency(request: Request) -> Settings:
    """Resolve shared application settings from FastAPI state."""
    return request.app.state.settings


def _build_template_context(
    request: Request,
    query: str = "",
    top_k: int = 5,
    results=None,
    error_message: str = "",
):
    normalized_results = []
    for item in results or []:
        frame_path = str(item.get("frame_path", ""))
        frame_name = Path(frame_path).name if frame_path else ""
        normalized_results.append(
            {
                "video_name": item.get("video_name", ""),
                "timestamp": f"{float(item.get('timestamp', 0.0)):.1f}s",
                "score": f"{float(item.get('score', 0.0)):.4f}",
                "frame_path": frame_path,
                "frame_url": f"/frames/{frame_name}" if frame_name else "",
            }
        )

    return {
        "request": request,
        "page_title": "Traffic Retrieval Control Panel",
        "query": query,
        "selected_top_k": top_k,
        "top_k_options": [3, 5, 10, 20],
        "results": normalized_results,
        "error_message": error_message,
    }


@page_router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index_page() -> HTMLResponse:
    """主页：取证三帧快照检索界面（真实接入 /api/search/hybrid，同源访问）。"""
    page_path = Path(__file__).resolve().parent.parent / "web" / "key_snapshots_demo.html"
    if not page_path.exists():
        raise HTTPException(status_code=404, detail="key_snapshots page not found")
    return HTMLResponse(page_path.read_text(encoding="utf-8"))


@page_router.get("/search", response_class=HTMLResponse, include_in_schema=False)
async def search_page(request: Request) -> HTMLResponse:
    """兼容页：原检索主页（templates/search.html），迁移到 /search。"""
    return templates.TemplateResponse(
        request=request,
        name="search.html",
        context=_build_template_context(request),
    )


@page_router.get("/video-probe", response_class=HTMLResponse, include_in_schema=False)
def video_probe_page() -> HTMLResponse:
    """Diagnostic page for verifying clip playback in a real browser.

    Must be served from the same origin as /api/search/download_clip so the
    browser can stream the clip without CORS/network-isolation interference
    (e.g. the IDE's built-in preview webview cannot reach localhost:8000).
    """
    probe_path = Path(__file__).resolve().parent.parent / "video_probe.html"
    return HTMLResponse(probe_path.read_text(encoding="utf-8"))


@page_router.get("/key-snapshots", response_class=HTMLResponse, include_in_schema=False)
def key_snapshots_page() -> HTMLResponse:
    """取证三帧快照检索界面（真实接入 /api/search/hybrid，同源访问无 CORS 问题）。"""
    page_path = Path(__file__).resolve().parent.parent / "web" / "key_snapshots_demo.html"
    if not page_path.exists():
        raise HTTPException(status_code=404, detail="key_snapshots page not found")
    return HTMLResponse(page_path.read_text(encoding="utf-8"))


@router.get("/health", response_model=HealthResponse)
async def health(
    services: AppState = Depends(get_services),
    settings: Settings = Depends(get_settings_dependency),
) -> HealthResponse:
    """Report service liveness plus per-channel availability (P5 degradation
    observability). Note: CLIP is lazily loaded; before the first encode call
    ``clip_available`` reflects the last load attempt state."""
    faiss_metadata = services.index_service.faiss_service.metadata
    return HealthResponse(
        service=settings.project_name,
        clip_available=services.clip_service.is_available(),
        faiss_available=services.index_service.faiss_service.index is not None,
        indexed_frames=len(faiss_metadata or []),
    )


@router.get("/index/stats", response_model=IndexStatsResponse)
async def index_stats(services: AppState = Depends(get_services)) -> IndexStatsResponse:
    stats = services.index_service.get_stats()
    return IndexStatsResponse(**stats)


@router.post("/videos/ingest", response_model=IngestResponse)
async def ingest_video(
    payload: IngestRequest,
    services: AppState = Depends(get_services),
    settings: Settings = Depends(get_settings_dependency),
) -> IngestResponse:
    try:
        result = services.video_service.ingest_video(
            video_path=payload.video_path,
            frame_interval=payload.frame_interval or settings.default_frame_interval,
            clip_service=services.clip_service,
            index_service=services.index_service,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net for API layer
        logger.exception("Failed to ingest video")
        raise HTTPException(status_code=500, detail="Failed to ingest video.") from exc

    return IngestResponse(**result)


@router.post("/videos/ingest-directory", response_model=BatchIngestResponse)
async def ingest_directory(
    payload: BatchIngestRequest,
    services: AppState = Depends(get_services),
    settings: Settings = Depends(get_settings_dependency),
) -> BatchIngestResponse:
    try:
        result = services.video_service.ingest_directory(
            directory=payload.directory,
            frame_interval=payload.frame_interval or settings.default_frame_interval,
            clip_service=services.clip_service,
            index_service=services.index_service,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net for API layer
        logger.exception("Failed to ingest video directory")
        raise HTTPException(status_code=500, detail="Failed to ingest video directory.") from exc

    return BatchIngestResponse(
        total_videos=result["total_videos"],
        succeeded_videos=result["succeeded_videos"],
        failed_videos=result["failed_videos"],
        results=[IngestResponse(**item) for item in result["results"]],
        errors=result["errors"],
    )


@router.post("/datasets/streetscene/ingest", response_model=StreetSceneIngestResponse)
async def ingest_streetscene(
    payload: StreetSceneIngestRequest,
    services: AppState = Depends(get_services),
) -> StreetSceneIngestResponse:
    try:
        result = services.streetscene_import_service.ingest_directory(
            dataset_root=payload.dataset_root,
            split=payload.split,
            sequence_names=payload.sequence_names,
            max_sequences=payload.max_sequences,
            frame_step=payload.frame_step,
            clip_service=services.clip_service,
            index_service=services.index_service,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net for API layer
        logger.exception("Failed to ingest StreetScene dataset")
        raise HTTPException(status_code=500, detail="Failed to ingest StreetScene dataset.") from exc

    return StreetSceneIngestResponse(**result)


@router.post("/datasets/accident/ingest", response_model=AccidentDatasetIngestResponse)
async def ingest_accident_dataset(
    payload: AccidentDatasetIngestRequest,
    services: AppState = Depends(get_services),
) -> AccidentDatasetIngestResponse:
    try:
        result = services.accident_import_service.ingest_directory(
            dataset_root=payload.dataset_root,
            scenario_names=payload.scenario_names,
            max_scenarios=payload.max_scenarios,
            frame_step=payload.frame_step,
            clip_service=services.clip_service,
            index_service=services.index_service,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net for API layer
        logger.exception("Failed to ingest accident dataset")
        raise HTTPException(status_code=500, detail="Failed to ingest accident dataset.") from exc

    return AccidentDatasetIngestResponse(**result)


@router.post("/search", response_model=SearchResponse, deprecated=True)
async def search(
    payload: SearchRequest,
    services: AppState = Depends(get_services),
) -> SearchResponse:
    """[DEPRECATED] 纯 CLIP 语义检索，已收敛至 POST /api/search/hybrid。

    保留本端点仅为兼容既有调用方；新调用请统一走 /api/search/hybrid。
    """
    logger.warning("Deprecated endpoint called: POST /api/search (use /api/search/hybrid)")
    return await _search_impl(payload, services)


@page_router.post("/search", response_model=SearchResponse, include_in_schema=False, deprecated=True)
async def search_compat(
    payload: SearchRequest,
    services: AppState = Depends(get_services),
) -> SearchResponse:
    logger.warning("Deprecated endpoint called: POST /search (use /api/search/hybrid)")
    return await _search_impl(payload, services)


async def _search_impl(payload: SearchRequest, services: AppState) -> SearchResponse:
    try:
        results = services.search_service.search_text(
            query=payload.query,
            top_k=payload.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return SearchResponse(query=payload.query, results=results)


@router.get("/search/download_clip", response_class=FileResponse)
def download_clip(
    video_path: str,
    start_ts: float,
    end_ts: float,
    output_name: str | None = None,
    services: AppState = Depends(get_services),
) -> FileResponse:
    """Losslessly cut a playable MP4 clip from a source video via FFmpeg.

    * ``-ss`` is placed before ``-i`` (fast seek) and ``-c copy`` performs a
      stream copy (no re-encode).
    * Duration is capped at ``clip_max_duration_sec`` (60s default).
    * At most ``clip_max_concurrent`` ffmpeg cuts run at once; requests beyond
      that are answered with ``429`` instead of saturating the CPU.
    * The temporary clip lives under ``/dev/shm`` (or OS temp) and is removed
      asynchronously right after the response is sent.

    Sync ``def`` (not ``async``): ffmpeg is a blocking subprocess, so the
    endpoint runs in FastAPI's thread pool and never blocks the event loop.
    """
    if not services.video_clip_service.try_acquire_slot():
        raise HTTPException(
            status_code=429,
            detail="Too many concurrent clip requests; please retry later.",
        )
    try:
        clip_path = services.video_clip_service.cut_clip(
            video_path=video_path,
            start_ts=start_ts,
            end_ts=end_ts,
            output_name=output_name,
        )
    except ClipNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ClipInvalidRangeError, ClipTooLongError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ClipTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except ClipError as exc:
        logger.error("Clip generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        # The ffmpeg slot is only needed for the cut itself; streaming the
        # response does not consume one.
        services.video_clip_service.release_slot()

    download_name = Path(clip_path).name
    return FileResponse(
        path=str(clip_path),
        media_type="video/mp4",
        filename=download_name,
        # inline (not attachment) so the browser's <video> element can stream
        # the clip; attachment would force a download and break in-page playback.
        content_disposition_type="inline",
        background=BackgroundTask(services.video_clip_service.cleanup, clip_path),
    )


@router.post("/search/hybrid", response_model=HybridSearchResponse)
async def hybrid_search(
    payload: HybridSearchRequest,
    services: AppState = Depends(get_services),
) -> HybridSearchResponse:
    try:
        result = services.hybrid_search_service.search(
            query=payload.query,
            top_k=payload.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net for API layer
        logger.exception("Failed to run hybrid search")
        raise HTTPException(status_code=500, detail="Failed to run hybrid search.") from exc

    return HybridSearchResponse(**result)


@router.post("/detection/ingest-directory", response_model=BatchIngestResponse)
async def detection_ingest_directory(
    payload: DetectionIngestRequest,
    services: AppState = Depends(get_services),
    settings: Settings = Depends(get_settings_dependency),
) -> BatchIngestResponse:
    frames_dir = payload.frames_dir or str(settings.frames_dir)
    try:
        videos = services.detection_service.process_frames_directory(frames_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net for API layer
        logger.exception("Failed to ingest detection metadata from directory")
        raise HTTPException(status_code=500, detail="Failed to ingest detection metadata.") from exc

    results = [
        IngestResponse(
            video_name=video.video_name,
            video_path=video.video_path,
            total_frames=len(video.frames),
            extracted_frames=len(video.frames),
            indexed_frames=sum(len(frame.detections) for frame in video.frames),
        )
        for video in videos
    ]

    return BatchIngestResponse(
        total_videos=len(videos),
        succeeded_videos=len(videos),
        failed_videos=0,
        results=results,
        errors=[],
    )


@router.post("/detection/search", response_model=DetectionSearchResponse, deprecated=True)
async def detection_search(
    payload: DetectionSearchRequest,
    services: AppState = Depends(get_services),
) -> DetectionSearchResponse:
    """[DEPRECATED] 目标检测单通道检索，已收敛至 POST /api/search/hybrid。

    保留本端点仅为兼容既有调用方；新调用请统一走 /api/search/hybrid。
    """
    logger.warning("Deprecated endpoint called: POST /api/detection/search (use /api/search/hybrid)")
    try:
        results = services.detection_search_service.search_objects(
            query=payload.query,
            top_k=payload.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net for API layer
        logger.exception("Failed to search detection metadata")
        raise HTTPException(status_code=500, detail="Failed to search detection metadata.") from exc

    return DetectionSearchResponse(query=payload.query, results=results)


@router.post("/tracking/ingest-directory", response_model=TrackingIngestResponse)
async def tracking_ingest_directory(
    payload: TrackingIngestRequest,
    services: AppState = Depends(get_services),
    settings: Settings = Depends(get_settings_dependency),
) -> TrackingIngestResponse:
    metadata_dir = payload.metadata_dir or str(settings.detection_metadata_dir)
    try:
        videos = services.tracking_service.process_detection_metadata_directory(metadata_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net for API layer
        logger.exception("Failed to ingest trajectory metadata from directory")
        raise HTTPException(status_code=500, detail="Failed to ingest trajectory metadata.") from exc

    return TrackingIngestResponse(
        total_videos=len(videos),
        succeeded_videos=len(videos),
        failed_videos=0,
        results=[
            {
                "video_name": video.video_name,
                "video_path": video.video_path,
                "total_tracks": len(video.tracks),
                "total_points": sum(len(track.points) for track in video.tracks),
            }
            for video in videos
        ],
        errors=[],
    )


@router.post("/trajectory/search", response_model=TrajectorySearchResponse, deprecated=True)
async def trajectory_search(
    payload: TrajectorySearchRequest,
    services: AppState = Depends(get_services),
) -> TrajectorySearchResponse:
    """[DEPRECATED] 轨迹单通道检索，已收敛至 POST /api/search/hybrid。

    保留本端点仅为兼容既有调用方；新调用请统一走 /api/search/hybrid。
    """
    logger.warning("Deprecated endpoint called: POST /api/trajectory/search (use /api/search/hybrid)")
    try:
        results = services.trajectory_search_service.search_tracks(
            query=payload.query,
            top_k=payload.top_k,
            direction=payload.direction,
            min_duration_sec=payload.min_duration_sec,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net for API layer
        logger.exception("Failed to search trajectory metadata")
        raise HTTPException(status_code=500, detail="Failed to search trajectory metadata.") from exc

    return TrajectorySearchResponse(query=payload.query, results=results)


@router.post("/event/ingest-directory", response_model=EventIngestResponse)
async def event_ingest_directory(
    payload: EventIngestRequest,
    services: AppState = Depends(get_services),
    settings: Settings = Depends(get_settings_dependency),
) -> EventIngestResponse:
    detection_dir = payload.detection_metadata_dir or str(settings.detection_metadata_dir)
    trajectory_dir = payload.trajectory_metadata_dir or str(settings.trajectory_metadata_dir)
    try:
        videos = services.event_service.process_metadata_directories(
            detection_dir=detection_dir,
            trajectory_dir=trajectory_dir,
            plugin_names=payload.plugin_names or None,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net for API layer
        logger.exception("Failed to ingest event metadata from directory")
        raise HTTPException(status_code=500, detail="Failed to ingest event metadata.") from exc

    return EventIngestResponse(
        total_videos=len(videos),
        succeeded_videos=len(videos),
        failed_videos=0,
        results=[
            {
                "video_name": video.video_name,
                "video_path": video.video_path,
                "total_events": len(video.events),
                "event_types": sorted({event.event_type for event in video.events}),
            }
            for video in videos
        ],
        errors=[],
    )


@router.post("/event/search", response_model=EventSearchResponse, deprecated=True)
async def event_search(
    payload: EventSearchRequest,
    services: AppState = Depends(get_services),
) -> EventSearchResponse:
    """[DEPRECATED] 事件单通道检索，已收敛至 POST /api/search/hybrid。

    保留本端点仅为兼容既有调用方；新调用请统一走 /api/search/hybrid。
    """
    logger.warning("Deprecated endpoint called: POST /api/event/search (use /api/search/hybrid)")
    try:
        results = services.event_search_service.search(
            query=payload.query,
            top_k=payload.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net for API layer
        logger.exception("Failed to search event metadata")
        raise HTTPException(status_code=500, detail="Failed to search event metadata.") from exc

    return EventSearchResponse(query=payload.query, results=results)


@router.post("/ingest/segments/{video_id}")
async def ingest_segments(
    video_id: str,
    services: AppState = Depends(get_services),
) -> dict:
    """S7: 独立触发某视频的段级构建（评审 #4 异步 + 状态标记）。

    失败不影响帧级索引/检索；重试即重新调用。结果写 segment_built 标记。
    """
    try:
        result = services.segment_build_service.ingest_segment_pipeline(
            video_id=video_id,
            clip_service=services.clip_service,
            index_service=services.index_service,
        )
    except Exception as exc:  # pragma: no cover - safety net for API layer
        logger.exception("Failed to build segments for video_id=%s", video_id)
        raise HTTPException(status_code=500, detail=f"Failed to build segments: {exc}") from exc

    if not result.get("built"):
        raise HTTPException(
            status_code=422,
            detail=result.get("reason", "segment build skipped"),
        )
    return result


@router.post("/ocr/ingest-directory", response_model=BatchIngestResponse)
async def ocr_ingest_directory(
    payload: OCRIngestRequest,
    services: AppState = Depends(get_services),
    settings: Settings = Depends(get_settings_dependency),
) -> BatchIngestResponse:
    frames_dir = payload.frames_dir or str(settings.frames_dir)
    try:
        videos = services.ocr_service.process_directory(frames_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net for API layer
        logger.exception("Failed to ingest OCR metadata from directory")
        raise HTTPException(status_code=500, detail="Failed to ingest OCR metadata.") from exc

    results = [
        IngestResponse(
            video_name=video.video_name,
            video_path="",
            total_frames=len(video.frames),
            extracted_frames=len(video.frames),
            indexed_frames=sum(len(frame.ocr_results) for frame in video.frames),
        )
        for video in videos
    ]

    return BatchIngestResponse(
        total_videos=len(videos),
        succeeded_videos=len(videos),
        failed_videos=0,
        results=results,
        errors=[],
    )


@router.post("/ocr/search", response_model=OCRSearchResponse, deprecated=True)
async def ocr_search(
    payload: OCRSearchRequest,
    services: AppState = Depends(get_services),
) -> OCRSearchResponse:
    """[DEPRECATED] OCR 单通道检索，已收敛至 POST /api/search/hybrid。

    保留本端点仅为兼容既有调用方；新调用请统一走 /api/search/hybrid。
    """
    logger.warning("Deprecated endpoint called: POST /api/ocr/search (use /api/search/hybrid)")
    try:
        results = services.ocr_search_service.search(
            query=payload.query,
            top_k=payload.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net for API layer
        logger.exception("Failed to search OCR metadata")
        raise HTTPException(status_code=500, detail="Failed to search OCR metadata.") from exc

    return OCRSearchResponse(query=payload.query, results=results)
