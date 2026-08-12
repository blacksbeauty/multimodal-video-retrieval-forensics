from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response for the FastAPI service."""

    status: str = Field(default="ok", description="Health status of the service.")
    service: str = Field(..., description="Human-readable service name.")


class IngestRequest(BaseModel):
    """Request payload for ingesting a single video into the CLIP retrieval pipeline."""

    video_path: str = Field(
        ...,
        description="Absolute path or project-relative path to the source video file.",
    )
    frame_interval: Optional[int] = Field(
        default=None,
        ge=1,
        description="Extract one frame every N frames. If omitted, the system default is used.",
    )


class BatchIngestRequest(BaseModel):
    """Request payload for ingesting every supported video under a directory."""

    directory: Optional[str] = Field(
        default=None,
        description="Directory containing videos. Defaults to the project's videos directory when omitted.",
    )
    frame_interval: Optional[int] = Field(
        default=None,
        ge=1,
        description="Extract one frame every N frames. If omitted, the system default is used.",
    )


class SearchRequest(BaseModel):
    """Request payload for CLIP-based natural language video retrieval."""

    query: str = Field(..., min_length=1, description="Natural language query text.")
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum number of retrieval results to return.",
    )


class FrameSearchResult(BaseModel):
    """Single frame-level result returned by the CLIP retrieval pipeline."""

    video_name: str = Field(..., description="Name of the source video containing the matched frame.")
    timestamp: float = Field(..., description="Timestamp of the matched frame in seconds.")
    score: float = Field(..., description="Similarity score returned by the retrieval engine.")
    frame_path: str = Field(..., description="Absolute or project-relative path to the matched frame image.")


class SearchResponse(BaseModel):
    """Response payload for CLIP-based video retrieval requests."""

    query: str = Field(..., description="Original natural language query text.")
    results: List[FrameSearchResult] = Field(
        default_factory=list,
        description="Ordered list of matched frame results.",
    )


class QueryIntent(BaseModel):
    """Structured traffic query intent for multi-entity retrieval and future event reasoning."""

    query_type: str = Field(default="general", description="High-level query category such as object, motion, relational, composite, or semantic.")
    primary_entities: List[str] = Field(
        default_factory=list,
        description="Main retrieval targets that should be returned by the system.",
    )
    context_entities: List[str] = Field(
        default_factory=list,
        description="Context or constraint entities that refine the primary retrieval target.",
    )
    relations: List[str] = Field(
        default_factory=list,
        description="Relationships between primary entities and context entities.",
    )
    directions: List[str] = Field(
        default_factory=list,
        description="Explicit movement directions such as left_to_right.",
    )
    motions: List[str] = Field(
        default_factory=list,
        description="Action- or behavior-style intent labels such as turn_left or stop.",
    )
    event_types: List[str] = Field(
        default_factory=list,
        description="Explicit event types requested by the query, such as red_light_violation.",
    )
    event_confidence: float = Field(
        default=0.0,
        description="Confidence of the event type detection (0.0-1.0). Higher values indicate stronger event routing signal.",
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parsed attributes such as light_state or duration qualifiers.",
    )
    rewritten_queries: List[str] = Field(
        default_factory=list,
        description="Retrieval-friendly rewritten variants derived from the original query.",
    )
    normalized_query: str = Field(
        default="",
        description="Normalized text used internally by the parser.",
    )
    intent_candidates: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top-k intent candidates with confidence scores for confidence-based routing.",
    )

    @property
    def kind(self) -> str:
        """Backward-compatible alias for earlier intent payloads."""
        return self.query_type

    @property
    def label_candidates(self) -> List[str]:
        """Backward-compatible alias that maps to primary retrieval targets."""
        return list(self.primary_entities)

    @property
    def direction(self) -> str:
        """Backward-compatible single-direction accessor."""
        return self.directions[0] if self.directions else ""

    def __getitem__(self, key: str) -> Any:
        if key == "kind":
            return self.kind
        if key == "label_candidates":
            return self.label_candidates
        if key == "direction":
            return self.direction
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except AttributeError:
            return default


class HybridSearchRequest(BaseModel):
    """Request payload for hybrid multimodal video retrieval."""

    query: str = Field(..., min_length=1, description="Natural language query text.")
    top_k: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Maximum number of segment-level results to return.",
    )


class HybridSegmentResult(BaseModel):
    """Segment-level retrieval result after aggregation and fusion."""

    video_id: str = Field(..., description="Stable video identifier for the matched segment.")
    video_name: str = Field(..., description="Source video file name.")
    video_path: str = Field(..., description="Normalized source video path.")
    start_ts: float = Field(..., description="Segment start timestamp in seconds.")
    end_ts: float = Field(..., description="Segment end timestamp in seconds.")
    best_score: float = Field(..., description="Best fused score observed within the segment.")
    matched_by: List[str] = Field(
        default_factory=list,
        description="Modalities that contributed to the segment match, such as clip or ocr.",
    )
    detection_score: float = Field(
        default=0.0,
        description="Detection branch score when the segment is matched by traffic object retrieval.",
    )
    trajectory_score: float = Field(
        default=0.0,
        description="Trajectory branch score when the segment is matched by trajectory retrieval.",
    )
    event_score: float = Field(
        default=0.0,
        description="Event branch score when the segment is matched by event retrieval.",
    )
    matched_event_type: str = Field(
        default="",
        description="Matched event type when event retrieval contributes to the segment.",
    )
    thumbnail_frame: str = Field(..., description="Representative frame path for the segment.")
    frame_id: str = Field(..., description="Stable frame identifier for the representative frame.")


class HybridSearchResponse(BaseModel):
    """Response payload for hybrid multimodal search."""

    query: str = Field(..., description="Original query text.")
    rewritten_queries: List[str] = Field(
        default_factory=list,
        description="Expanded query variants used during retrieval.",
    )
    results: List[HybridSegmentResult] = Field(
        default_factory=list,
        description="Segment-level multimodal retrieval results.",
    )


class DetectionItem(BaseModel):
    """Single object detection result extracted from a frame."""

    label: str = Field(..., description="Normalized object class label.")
    confidence: float = Field(..., description="Detection confidence score.")
    bbox: List[float] = Field(
        default_factory=list,
        description="Bounding box in [x1, y1, x2, y2] format.",
    )
    class_id: int | None = Field(
        default=None,
        description="Optional detector-native class id for downstream tracking or analytics.",
    )


class DetectionFrameMetadata(BaseModel):
    """Detection metadata for a single video frame."""

    video_name: str = Field(..., description="Name of the source video for the frame.")
    frame_path: str = Field(..., description="Normalized path to the frame image.")
    timestamp: float = Field(..., description="Frame timestamp in seconds.")
    detections: List[DetectionItem] = Field(
        default_factory=list,
        description="Detected traffic-related objects on the frame.",
    )


class DetectionVideoMetadata(BaseModel):
    """Detection metadata bundle stored per video."""

    video_id: str = Field(..., description="Stable video identifier used for metadata persistence.")
    video_name: str = Field(..., description="Name of the source video file.")
    video_path: str = Field(..., description="Normalized path to the source video file.")
    frames: List[DetectionFrameMetadata] = Field(
        default_factory=list,
        description="Ordered detection metadata records for the video frames.",
    )


class DetectionIngestRequest(BaseModel):
    """Request payload for batch traffic object detection over extracted frames."""

    frames_dir: Optional[str] = Field(
        default=None,
        description="Optional directory containing frame images. Defaults to the project's frames directory.",
    )


class DetectionSearchRequest(BaseModel):
    """Request payload for detection-based object retrieval."""

    query: str = Field(..., min_length=1, description="Target traffic object label query.")
    top_k: int = Field(
        default=10,
        ge=1,
        le=200,
        description="Maximum number of detection matches to return.",
    )


class DetectionSearchResult(BaseModel):
    """Single detection retrieval result mapped to a video frame."""

    video_name: str = Field(..., description="Name of the matched source video.")
    timestamp: float = Field(..., description="Matched frame timestamp in seconds.")
    frame_path: str = Field(..., description="Normalized matched frame path.")
    matched_label: str = Field(..., description="Detection label that matched the query.")
    confidence: float = Field(..., description="Detection confidence for the matched label.")


class DetectionSearchResponse(BaseModel):
    """Response payload for detection-based retrieval."""

    query: str = Field(..., description="Original detection query text.")
    results: List[DetectionSearchResult] = Field(
        default_factory=list,
        description="Ordered list of matching detection results.",
    )


class TrajectoryPoint(BaseModel):
    """One tracked observation of an object at a specific frame timestamp."""

    timestamp: float = Field(..., description="Tracked point timestamp in seconds.")
    frame_path: str = Field(..., description="Normalized frame path for this tracked point.")
    bbox: List[float] = Field(
        default_factory=list,
        description="Tracked bounding box in [x1, y1, x2, y2] format.",
    )
    center_x: float = Field(..., description="Bounding box center x coordinate.")
    center_y: float = Field(..., description="Bounding box center y coordinate.")
    confidence: float = Field(..., description="Tracking point confidence score.")


class TrajectoryTrackMetadata(BaseModel):
    """Aggregated trajectory metadata for one tracked object instance."""

    track_id: str = Field(..., description="Stable track identifier within a video.")
    label: str = Field(..., description="Tracked object label.")
    start_ts: float = Field(..., description="Track start timestamp in seconds.")
    end_ts: float = Field(..., description="Track end timestamp in seconds.")
    duration_sec: float = Field(..., description="Track duration in seconds.")
    frame_count: int = Field(..., description="Number of tracked points in the trajectory.")
    avg_confidence: float = Field(..., description="Average confidence across trajectory points.")
    max_confidence: float = Field(..., description="Maximum confidence across trajectory points.")
    direction: str = Field(default="unknown", description="Coarse movement direction inferred from the trajectory.")
    representative_frame: str = Field(..., description="Frame path representing the trajectory.")
    points: List[TrajectoryPoint] = Field(
        default_factory=list,
        description="Ordered tracked points for this trajectory.",
    )


class TrajectoryVideoMetadata(BaseModel):
    """Trajectory metadata bundle stored per video."""

    video_id: str = Field(..., description="Stable video identifier for the trajectory bundle.")
    video_name: str = Field(..., description="Source video name for the trajectory bundle.")
    video_path: str = Field(..., description="Normalized source video path.")
    tracks: List[TrajectoryTrackMetadata] = Field(
        default_factory=list,
        description="Tracked object trajectories extracted from the video.",
    )


class TrackingIngestRequest(BaseModel):
    """Request payload for generating trajectory metadata from detection metadata."""

    metadata_dir: Optional[str] = Field(
        default=None,
        description="Optional detection metadata directory. Defaults to metadata/detections.",
    )


class TrackingIngestResult(BaseModel):
    """Per-video result for trajectory metadata generation."""

    video_name: str = Field(..., description="Source video name.")
    video_path: str = Field(..., description="Normalized source video path.")
    total_tracks: int = Field(..., description="Number of trajectory tracks generated for the video.")
    total_points: int = Field(..., description="Total tracked points across all trajectories.")


class TrackingIngestResponse(BaseModel):
    """Response payload for batch trajectory metadata generation."""

    total_videos: int = Field(..., description="Total number of detection metadata files processed.")
    succeeded_videos: int = Field(..., description="Number of videos successfully converted to trajectories.")
    failed_videos: int = Field(..., description="Number of videos that failed during trajectory generation.")
    results: List[TrackingIngestResult] = Field(
        default_factory=list,
        description="Detailed per-video trajectory generation results.",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Human-readable errors for failed videos.",
    )


class TrajectorySearchRequest(BaseModel):
    """Request payload for trajectory-based object retrieval."""

    query: str = Field(..., min_length=1, description="Trajectory object query.")
    top_k: int = Field(default=10, ge=1, le=200, description="Maximum number of trajectory results to return.")
    direction: Optional[str] = Field(
        default=None,
        description="Optional direction filter such as left_to_right or right_to_left.",
    )
    min_duration_sec: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Optional minimum trajectory duration filter.",
    )


class TrajectorySearchResult(BaseModel):
    """One trajectory retrieval result."""

    video_name: str = Field(..., description="Matched video name.")
    video_path: str = Field(..., description="Matched video path.")
    track_id: str = Field(..., description="Matched trajectory id.")
    label: str = Field(..., description="Tracked object label.")
    start_ts: float = Field(..., description="Trajectory start timestamp in seconds.")
    end_ts: float = Field(..., description="Trajectory end timestamp in seconds.")
    duration_sec: float = Field(..., description="Trajectory duration in seconds.")
    direction: str = Field(..., description="Inferred trajectory direction.")
    avg_confidence: float = Field(..., description="Average trajectory confidence.")
    representative_frame: str = Field(..., description="Representative frame path for the trajectory.")


class TrajectorySearchResponse(BaseModel):
    """Response payload for trajectory-based retrieval."""

    query: str = Field(..., description="Original trajectory query text.")
    results: List[TrajectorySearchResult] = Field(
        default_factory=list,
        description="Ordered list of matching trajectories.",
    )


class StreetSceneIngestRequest(BaseModel):
    """Request payload for importing StreetScene image sequences into ForeSea assets."""

    dataset_root: Optional[str] = Field(
        default=None,
        description="Optional StreetScene dataset root. Defaults to the configured dataset path.",
    )
    split: Optional[str] = Field(
        default="Test",
        description="Optional dataset split to import. Use Train, Test, or omit for both.",
    )
    sequence_names: List[str] = Field(
        default_factory=list,
        description="Optional explicit sequence names such as Test001 or Train001.",
    )
    max_sequences: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional cap on number of sequences to import.",
    )
    frame_step: int = Field(
        default=1,
        ge=1,
        description="Import one frame every N frames from the StreetScene sequence.",
    )


class StreetSceneIngestResult(BaseModel):
    """Per-sequence result for StreetScene import."""

    video_id: str = Field(..., description="Stable imported video identifier.")
    video_name: str = Field(..., description="Logical imported video name.")
    source_sequence_dir: str = Field(..., description="Original StreetScene sequence directory.")
    imported_frames: int = Field(..., description="Number of frames imported into ForeSea assets.")
    indexed_frames: int = Field(..., description="Number of frames embedded and indexed for CLIP retrieval.")


class StreetSceneIngestResponse(BaseModel):
    """Response payload for StreetScene dataset import."""

    total_sequences: int = Field(..., description="Total number of sequences selected for import.")
    succeeded_sequences: int = Field(..., description="Number of sequences imported successfully.")
    failed_sequences: int = Field(..., description="Number of sequences that failed during import.")
    results: List[StreetSceneIngestResult] = Field(
        default_factory=list,
        description="Detailed per-sequence import results.",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Human-readable per-sequence errors.",
    )


class AccidentDatasetIngestRequest(BaseModel):
    """Request payload for importing CARLA accident BEV sequences into ForeSea assets."""

    dataset_root: Optional[str] = Field(
        default=None,
        description="Optional accident dataset root. Defaults to the configured dataset path.",
    )
    scenario_names: List[str] = Field(
        default_factory=list,
        description="Optional explicit scenario directory names to import.",
    )
    max_scenarios: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional cap on number of scenarios to import.",
    )
    frame_step: int = Field(
        default=1,
        ge=1,
        description="Import one frame every N frames from the scenario sequence.",
    )


class AccidentDatasetIngestResult(BaseModel):
    """Per-scenario result for accident dataset import."""

    video_id: str = Field(..., description="Stable imported video identifier.")
    video_name: str = Field(..., description="Logical imported video name.")
    source_scenario_dir: str = Field(..., description="Original scenario directory path.")
    imported_frames: int = Field(..., description="Number of frames imported into ForeSea assets.")
    indexed_frames: int = Field(..., description="Number of frames embedded and indexed for CLIP retrieval.")
    scenario_meta: dict[str, Any] = Field(
        default_factory=dict,
        description="Parsed accident scenario metadata from the meta file.",
    )


class AccidentDatasetIngestResponse(BaseModel):
    """Response payload for accident dataset import."""

    total_scenarios: int = Field(..., description="Total number of scenarios selected for import.")
    succeeded_scenarios: int = Field(..., description="Number of scenarios imported successfully.")
    failed_scenarios: int = Field(..., description="Number of scenarios that failed during import.")
    results: List[AccidentDatasetIngestResult] = Field(
        default_factory=list,
        description="Detailed per-scenario import results.",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Human-readable per-scenario errors.",
    )


class EventMetadata(BaseModel):
    """One normalized traffic event record produced by an event plugin."""

    event_id: str = Field(..., description="Stable event identifier.")
    event_type: str = Field(..., description="Canonical event type name.")
    plugin_name: str = Field(..., description="Plugin that produced this event.")
    video_id: str = Field(..., description="Source video identifier.")
    video_name: str = Field(..., description="Source video name.")
    video_path: str = Field(..., description="Normalized source video path.")
    start_ts: float = Field(..., description="Event start timestamp in seconds.")
    end_ts: float = Field(..., description="Event end timestamp in seconds.")
    track_ids: List[str] = Field(
        default_factory=list,
        description="Trajectory track identifiers involved in this event.",
    )
    confidence: float = Field(..., description="Plugin-assigned event confidence.")
    representative_frame: str = Field(..., description="Representative frame path for the event.")
    evidence_frames: List[str] = Field(
        default_factory=list,
        description="Frame paths that support the event decision.",
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific structured attributes.",
    )
    description: str = Field(default="", description="Human-readable event summary.")


class EventVideoMetadata(BaseModel):
    """Per-video bundle of traffic events."""

    video_id: str = Field(..., description="Source video identifier.")
    video_name: str = Field(..., description="Source video name.")
    video_path: str = Field(..., description="Normalized source video path.")
    events: List[EventMetadata] = Field(
        default_factory=list,
        description="Traffic events generated for this video.",
    )


class EventIngestRequest(BaseModel):
    """Request payload for generating event metadata from detections and trajectories."""

    detection_metadata_dir: Optional[str] = Field(
        default=None,
        description="Optional detection metadata directory. Defaults to metadata/detections.",
    )
    trajectory_metadata_dir: Optional[str] = Field(
        default=None,
        description="Optional trajectory metadata directory. Defaults to metadata/trajectories.",
    )
    plugin_names: List[str] = Field(
        default_factory=list,
        description="Optional subset of event plugins to run. Defaults to configured plugins.",
    )


class EventIngestResult(BaseModel):
    """Per-video result for event generation."""

    video_name: str = Field(..., description="Source video name.")
    video_path: str = Field(..., description="Normalized source video path.")
    total_events: int = Field(..., description="Total generated events for the video.")
    event_types: List[str] = Field(
        default_factory=list,
        description="Distinct event types generated for the video.",
    )


class EventIngestResponse(BaseModel):
    """Response payload for event metadata generation."""

    total_videos: int = Field(..., description="Total number of videos processed for event generation.")
    succeeded_videos: int = Field(..., description="Number of videos successfully processed.")
    failed_videos: int = Field(..., description="Number of videos that failed event generation.")
    results: List[EventIngestResult] = Field(
        default_factory=list,
        description="Per-video event generation results.",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Human-readable per-video errors.",
    )


class EventSearchRequest(BaseModel):
    """Request payload for event retrieval."""

    query: str = Field(..., min_length=1, description="Event type query.")
    top_k: int = Field(default=10, ge=1, le=200, description="Maximum number of event results to return.")


class EventSearchResult(BaseModel):
    """One event retrieval result."""

    event_id: str = Field(..., description="Stable event identifier.")
    event_type: str = Field(..., description="Canonical event type.")
    plugin_name: str = Field(..., description="Plugin that produced the event.")
    video_name: str = Field(..., description="Source video name.")
    video_path: str = Field(..., description="Source video path.")
    start_ts: float = Field(..., description="Event start timestamp in seconds.")
    end_ts: float = Field(..., description="Event end timestamp in seconds.")
    track_ids: List[str] = Field(
        default_factory=list,
        description="Trajectory tracks involved in the event.",
    )
    confidence: float = Field(..., description="Event confidence.")
    representative_frame: str = Field(..., description="Representative event frame.")
    description: str = Field(default="", description="Human-readable event summary.")
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Plugin-specific structured attributes.",
    )


class EventSearchResponse(BaseModel):
    """Response payload for event retrieval."""

    query: str = Field(..., description="Original event query text.")
    results: List[EventSearchResult] = Field(
        default_factory=list,
        description="Ordered list of matching events.",
    )


class IngestResponse(BaseModel):
    """Response payload for ingesting a single video into the retrieval pipeline."""

    video_name: str = Field(..., description="Name of the ingested video file.")
    video_path: str = Field(..., description="Resolved absolute path of the ingested video.")
    total_frames: int = Field(..., description="Total frame count reported by the source video.")
    extracted_frames: int = Field(..., description="Number of frames extracted from the source video.")
    indexed_frames: int = Field(..., description="Number of extracted frames successfully indexed.")


class BatchIngestResponse(BaseModel):
    """Response payload for batch ingestion of a video directory."""

    total_videos: int = Field(..., description="Total number of videos discovered for ingestion.")
    succeeded_videos: int = Field(..., description="Number of videos ingested successfully.")
    failed_videos: int = Field(..., description="Number of videos that failed during ingestion.")
    results: List[IngestResponse] = Field(
        default_factory=list,
        description="Detailed per-video ingestion results for successful videos.",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Human-readable error messages for videos that failed ingestion.",
    )


class IndexStatsResponse(BaseModel):
    """Response payload describing the current CLIP vector index state."""

    indexed_frames: int = Field(..., description="Total number of frame vectors currently indexed.")
    indexed_videos: int = Field(..., description="Total number of videos represented in the current index.")
    index_path: str = Field(..., description="Filesystem path to the persisted FAISS index file.")
    metadata_path: str = Field(..., description="Filesystem path to the persisted frame metadata file.")


class OCRTextResult(BaseModel):
    """Single OCR text box result extracted from a frame."""

    text: str = Field(..., description="Recognized text content from the OCR engine.")
    score: float = Field(..., description="Confidence score returned by the OCR model for the text result.")
    bbox: List[List[int]] = Field(
        ...,
        description="Bounding polygon points of the detected text region, formatted as list[list[int]].",
    )


class OCRFrameMetadata(BaseModel):
    """Structured OCR output for a single video frame."""

    frame_path: str = Field(..., description="Path to the frame image on local disk.")
    timestamp_sec: float = Field(..., description="Precise frame timestamp in seconds for machine processing.")
    display_time: str = Field(..., description="Human-readable timestamp string for UI display.")
    ocr_results: List[OCRTextResult] = Field(
        default_factory=list,
        description="All OCR text regions detected in this frame.",
    )


class OCRVideoMetadata(BaseModel):
    """Structured OCR metadata bundle for an entire video."""

    video_name: str = Field(..., description="Name of the source video associated with the OCR results.")
    frames: List[OCRFrameMetadata] = Field(
        default_factory=list,
        description="Ordered OCR frame metadata records extracted from the video.",
    )


class OCRIngestRequest(BaseModel):
    """Request payload for OCR metadata extraction over one video or a video directory."""

    frames_dir: Optional[str] = Field(
        default=None,
        description="Optional frame directory to scan for OCR processing. Defaults to the frames directory when omitted.",
    )
    video_path: Optional[str] = Field(
        default=None,
        description="Optional single video path for OCR extraction. Use either video_path or directory.",
    )
    directory: Optional[str] = Field(
        default=None,
        description="Optional directory containing videos for OCR extraction. Defaults to the videos directory when omitted.",
    )
    frame_interval: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional frame sampling interval. If omitted, the system default is used.",
    )
    use_existing_frames: bool = Field(
        default=True,
        description="Whether to reuse already extracted frames from the frames directory before extracting again.",
    )


class OCRSearchRequest(BaseModel):
    """Request payload for OCR-based video text retrieval."""

    query: str = Field(..., min_length=1, description="Chinese-first text query to match against OCR output.")
    top_k: int = Field(
        default=10,
        ge=1,
        le=200,
        description="Maximum number of OCR retrieval results to return.",
    )


class OCRSearchResult(BaseModel):
    """Single OCR text retrieval result mapped back to a specific video frame."""

    video_name: str = Field(..., description="Name of the source video containing the matched OCR text.")
    frame_path: str = Field(..., description="Path to the matched frame image.")
    timestamp_sec: float = Field(..., description="Precise matched frame timestamp in seconds.")
    display_time: str = Field(..., description="Human-readable matched frame timestamp for display.")
    matched_text: str = Field(..., description="OCR text snippet that matched the user's query.")
    similarity_score: float = Field(
        ...,
        description="Matching score for ranking OCR search results. Can later support fuzzy matching or multimodal fusion.",
    )


class OCRSearchResponse(BaseModel):
    """Response payload for OCR-based video text retrieval."""

    query: str = Field(..., description="Original OCR text query submitted by the user.")
    results: List[OCRSearchResult] = Field(
        default_factory=list,
        description="Ordered OCR search results matched from video frames.",
    )
