import unittest

from config import Settings
from services.hybrid_search_service import HybridSearchService
from services.result_aggregation_service import ResultAggregationService


class _FakeQueryRewriteService:
    def rewrite_query(self, query: str):
        return [query]

    def parse_query_intent(self, query: str):
        if query == "vehicle_crosses_line":
            return {
                "kind": "event",
                "query_type": "event",
                "direction": "",
                "label_candidates": [],
                "primary_entities": [],
                "context_entities": [],
                "relations": [],
                "directions": [],
                "motions": [],
                "event_types": ["vehicle_crosses_line"],
                "event_confidence": 1.0,
                "attributes": {},
                "rewritten_queries": [query],
                "normalized_query": query,
                "intent_candidates": [],
            }
        return {
            "kind": "object",
            "query_type": "object",
            "direction": "",
            "label_candidates": [query],
            "primary_entities": [query],
            "context_entities": [],
            "relations": [],
            "directions": [],
            "motions": [],
            "event_types": [],
            "event_confidence": 0.0,
            "attributes": {},
            "rewritten_queries": [query],
            "normalized_query": query,
            "intent_candidates": [],
        }


class _FakeClipSearchService:
    def __init__(self) -> None:
        self.calls = 0

    def search_text_variants(self, queries, top_k=5):
        self.calls += 1
        return [
            {
                "video_id": "video-1",
                "video_name": "traffic.mp4",
                "video_path": "/tmp/traffic.mp4",
                "frame_id": "frame-1",
                "frame_path": "/tmp/frame-1.jpg",
                "timestamp_seconds": 10.0,
                "score": 0.55,
                "matched_by": ["clip"],
            }
        ]


class _FakeDetectionRetriever:
    def search_as_frames(self, query: str, top_k: int):
        return [
            {
                "video_id": "video-1",
                "video_name": "traffic.mp4",
                "video_path": "/tmp/traffic.mp4",
                "frame_id": "detection::frame-1",
                "frame_path": "/tmp/frame-1.jpg",
                "timestamp_seconds": 10.0,
                "score": 0.72,
                "detection_score": 0.72,
                "matched_by": ["detection"],
                "matched_label": "car",
            }
        ]


class _FakeTrajectoryRetriever:
    def search_as_frames(self, query: str, top_k: int):
        return [
            {
                "video_id": "video-1",
                "video_name": "traffic.mp4",
                "video_path": "/tmp/traffic.mp4",
                "frame_id": "trajectory::track-1",
                "frame_path": "/tmp/frame-1.jpg",
                "start_ts": 10.0,
                "end_ts": 15.0,
                "timestamp_seconds": 10.0,
                "score": 0.81,
                "trajectory_score": 0.81,
                "matched_by": ["trajectory"],
                "matched_label": "car",
                "matched_direction": "left_to_right",
                "track_id": "video-1:1",
            }
        ]


class _FakeEventRetriever:
    def search_as_frames(self, query: str, top_k: int):
        return [
            {
                "video_id": "video-1",
                "video_name": "traffic.mp4",
                "video_path": "/tmp/traffic.mp4",
                "frame_id": "event::event-1",
                "frame_path": "/tmp/frame-1.jpg",
                "start_ts": 10.0,
                "end_ts": 15.0,
                "timestamp_seconds": 10.0,
                "score": 0.88,
                "event_score": 0.88,
                "matched_by": ["event"],
                "matched_event_type": "vehicle_crosses_line",
                "matched_label": "car",
                "track_id": "video-1:1",
                "event_id": "video-1:event-1",
            }
        ]


class HybridSearchServiceTests(unittest.TestCase):
    def test_hybrid_search_fuses_clip_detection_and_trajectory(self) -> None:
        settings = Settings()
        settings.hybrid_score_threshold = 0.1
        settings.segment_window_seconds = 5.0
        service = HybridSearchService(
            settings=settings,
            query_rewrite_service=_FakeQueryRewriteService(),
            clip_search_service=_FakeClipSearchService(),
            ocr_search_service=None,
            detection_retriever=_FakeDetectionRetriever(),
            trajectory_retriever=_FakeTrajectoryRetriever(),
            event_retriever=_FakeEventRetriever(),
            result_aggregation_service=ResultAggregationService(settings),
        )

        response = service.search("car", top_k=5)

        self.assertEqual(response["query"], "car")
        self.assertEqual(len(response["results"]), 1)
        result = response["results"][0]
        self.assertEqual(sorted(result["matched_by"]), ["clip", "detection", "event", "trajectory"])
        self.assertEqual(result["trajectory_score"], 0.81)
        self.assertEqual(result["detection_score"], 0.72)
        self.assertEqual(result["event_score"], 0.88)
        self.assertEqual(result["matched_event_type"], "vehicle_crosses_line")
        self.assertEqual(result["start_ts"], 10.0)
        self.assertEqual(result["end_ts"], 15.0)

    def test_hybrid_search_prioritizes_event_queries(self) -> None:
        settings = Settings()
        settings.hybrid_score_threshold = 0.1
        settings.segment_window_seconds = 5.0
        clip_service = _FakeClipSearchService()
        service = HybridSearchService(
            settings=settings,
            query_rewrite_service=_FakeQueryRewriteService(),
            clip_search_service=clip_service,
            ocr_search_service=None,
            detection_retriever=_FakeDetectionRetriever(),
            trajectory_retriever=_FakeTrajectoryRetriever(),
            event_retriever=_FakeEventRetriever(),
            result_aggregation_service=ResultAggregationService(settings),
        )

        response = service.search("vehicle_crosses_line", top_k=5)

        self.assertEqual(len(response["results"]), 1)
        result = response["results"][0]
        # Phase 3: event queries always run CLIP (no skip_non_event)
        self.assertIn("event", result["matched_by"])
        self.assertIn("clip", result["matched_by"])
        self.assertEqual(result["matched_event_type"], "vehicle_crosses_line")
        self.assertEqual(result["event_score"], 0.88)
        # CLIP is always called now (skip_non_event removed)
        self.assertEqual(clip_service.calls, 1)


if __name__ == "__main__":
    unittest.main()
