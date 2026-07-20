from __future__ import annotations

import logging
import re
from typing import List

from rapidfuzz import fuzz

from config import Settings
from core.schemas import OCRSearchResult, OCRVideoMetadata
from services.ocr_metadata_store import OCRMetadataStore


logger = logging.getLogger(__name__)


class OCRSearchService:
    """Search OCR metadata using substring and fuzzy text matching."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the OCR search service and metadata store."""
        self.settings = settings
        self.metadata_store = OCRMetadataStore(settings)

    def search(self, query_text: str, top_k: int = 10) -> List[OCRSearchResult]:
        """Search OCR metadata and return ranked OCR search results."""
        normalized_query = self.normalize_text(query_text)
        if not normalized_query:
            raise ValueError("OCR search query cannot be empty.")
        if top_k < 1:
            raise ValueError("top_k must be greater than or equal to 1.")

        results: List[OCRSearchResult] = []
        metadata_files = self.metadata_store.list_metadata()
        if not metadata_files:
            logger.warning("No OCR metadata files available for OCR search.")
            return []

        logger.info("Starting OCR search query=%s top_k=%s files=%s", query_text, top_k, len(metadata_files))
        for metadata_path in metadata_files:
            metadata = self.metadata_store.load_metadata(metadata_path.stem)
            if metadata is None:
                continue

            results.extend(self._search_video_metadata(metadata, normalized_query))

        ranked = self.rank_results(results)
        logger.info("Completed OCR search query=%s results=%s", query_text, len(ranked))
        return ranked[:top_k]

    def search_as_frames(self, query_text: str, top_k: int = 10) -> List[dict]:
        """Search OCR metadata and return frame-like records for hybrid fusion."""
        raw_results = self.search(query_text=query_text, top_k=top_k)
        return [
            {
                "video_id": self._build_video_id(result.video_name),
                "video_name": result.video_name,
                "video_path": "",
                "frame_id": f"ocr::{result.video_name}::{result.timestamp_sec}::{result.frame_path}",
                "frame_path": result.frame_path,
                "thumbnail_frame": result.frame_path,
                "timestamp_seconds": result.timestamp_sec,
                "score": float(result.similarity_score) / 100.0,
                "matched_by": ["ocr"],
                "matched_text": result.matched_text,
            }
            for result in raw_results
        ]

    def normalize_text(self, text: str) -> str:
        """Normalize OCR text for Chinese/English matching with case-insensitive behavior."""
        lowered = text.casefold().strip()
        collapsed = re.sub(r"\s+", "", lowered)
        return collapsed

    def fuzzy_match(self, query_text: str, candidate_text: str) -> float:
        """Compute a fuzzy similarity score between normalized query and OCR text."""
        normalized_query = self.normalize_text(query_text)
        normalized_candidate = self.normalize_text(candidate_text)
        if not normalized_query or not normalized_candidate:
            return 0.0

        if normalized_query in normalized_candidate:
            return 100.0

        return float(
            max(
                fuzz.partial_ratio(normalized_query, normalized_candidate),
                fuzz.ratio(normalized_query, normalized_candidate),
                fuzz.token_set_ratio(normalized_query, normalized_candidate),
            )
        )

    def rank_results(self, results: List[OCRSearchResult]) -> List[OCRSearchResult]:
        """Sort OCR search results by similarity score and then by timestamp."""
        return sorted(
            results,
            key=lambda item: (-item.similarity_score, item.video_name, item.timestamp_sec),
        )

    def _search_video_metadata(
        self,
        metadata: OCRVideoMetadata,
        normalized_query: str,
    ) -> List[OCRSearchResult]:
        matches: List[OCRSearchResult] = []

        for frame in metadata.frames:
            try:
                for ocr_item in frame.ocr_results:
                    candidate_text = ocr_item.text
                    normalized_candidate = self.normalize_text(candidate_text)
                    if not normalized_candidate:
                        continue

                    score = self.fuzzy_match(normalized_query, normalized_candidate)
                    if normalized_query not in normalized_candidate and score < 60.0:
                        continue

                    matches.append(
                        OCRSearchResult(
                            video_name=metadata.video_name,
                            frame_path=frame.frame_path,
                            timestamp_sec=frame.timestamp_sec,
                            display_time=frame.display_time,
                            matched_text=candidate_text,
                            similarity_score=score,
                        )
                    )
            except Exception:
                logger.exception(
                    "Failed to search OCR frame metadata for video=%s frame=%s",
                    metadata.video_name,
                    frame.frame_path,
                )
                continue

        return matches

    def _build_video_id(self, video_name: str) -> str:
        """Build a stable synthetic video identifier for OCR-only metadata records."""
        return f"ocr::{self.normalize_text(video_name)}"
