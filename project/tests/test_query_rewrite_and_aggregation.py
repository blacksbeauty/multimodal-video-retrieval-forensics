import unittest

from config import Settings
from services.query_rewrite_service import QueryRewriteService
from services.result_aggregation_service import ResultAggregationService


class QueryRewriteServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = QueryRewriteService()

    def test_rewrite_exact_chinese_aliases(self) -> None:
        self.assertEqual(
            self.service.rewrite_query("\u767d\u8272\u6c7d\u8f66"),
            ["white car", "white sedan", "white vehicle", "car"],
        )

    def test_rewrite_traffic_aliases(self) -> None:
        self.assertEqual(self.service.rewrite_query("\u6c7d\u8f66"), ["car"])
        self.assertEqual(self.service.rewrite_query("\u7ea2\u7eff\u706f"), ["traffic light"])

    def test_rewrite_phrase_contains_alias(self) -> None:
        values = self.service.rewrite_query("\u6c7d\u8f66\u4ece\u5de6\u5230\u53f3")
        self.assertIn("car", values)

    def test_parse_motion_query(self) -> None:
        intent = self.service.parse_query_intent("\u4ece\u5de6\u5230\u53f3\u7684\u884c\u4eba")
        self.assertEqual(intent.query_type, "motion")
        self.assertEqual(intent.primary_entities, ["person"])
        self.assertEqual(intent.directions, ["left_to_right"])
        self.assertEqual(intent.context_entities, [])

    def test_parse_relational_query(self) -> None:
        intent = self.service.parse_query_intent("\u7ea2\u7eff\u706f\u9644\u8fd1\u7684\u8f66")
        self.assertEqual(intent.query_type, "relational")
        self.assertEqual(intent.primary_entities, ["car"])
        self.assertEqual(intent.context_entities, ["traffic_light"])
        self.assertEqual(intent.relations, ["near"])

    def test_parse_motion_context_query(self) -> None:
        intent = self.service.parse_query_intent("\u8def\u53e3\u5de6\u8f6c\u7684\u8d27\u8f66")
        self.assertEqual(intent.query_type, "composite")
        self.assertEqual(intent.primary_entities, ["truck"])
        self.assertEqual(intent.context_entities, ["intersection"])
        self.assertEqual(intent.motions, ["turn_left"])

    def test_parse_attribute_relation_query(self) -> None:
        intent = self.service.parse_query_intent("\u7ea2\u706f\u65f6\u7a7f\u8fc7\u505c\u6b62\u7ebf\u7684\u6c7d\u8f66")
        self.assertEqual(intent.query_type, "event")
        self.assertEqual(intent.primary_entities, ["car"])
        self.assertEqual(intent.context_entities, ["traffic_light", "stop_line"])
        self.assertEqual(intent.relations, ["cross"])
        self.assertEqual(intent.attributes.get("light_state"), "red")
        self.assertIn("red_light_violation", intent.event_types)

    def test_parse_generic_vehicle_query(self) -> None:
        intent = self.service.parse_query_intent("\u9760\u8fd1\u505c\u6b62\u7ebf\u7684\u8f66\u8f86")
        self.assertEqual(intent.primary_entities, ["car", "truck", "bus", "motorcycle"])
        self.assertEqual(intent.context_entities, ["stop_line"])
        self.assertEqual(intent.relations, ["near"])

    def test_parse_event_query(self) -> None:
        intent = self.service.parse_query_intent("vehicle_crosses_line")
        self.assertEqual(intent.query_type, "event")
        self.assertEqual(intent.event_types, ["vehicle_crosses_line"])

    def test_keep_english_query(self) -> None:
        self.assertEqual(self.service.rewrite_query("white car"), ["white car"])


class ResultAggregationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        settings = Settings()
        settings.segment_window_seconds = 5.0
        settings.hybrid_score_threshold = 0.2
        self.service = ResultAggregationService(settings)

    def test_aggregate_keeps_best_frame_per_video_window(self) -> None:
        results = [
            {
                "video_id": "video-1",
                "video_name": "a.mp4",
                "video_path": "/tmp/a.mp4",
                "frame_id": "frame-1",
                "frame_path": "/tmp/a_1.jpg",
                "timestamp_seconds": 1.0,
                "score": 0.31,
                "matched_by": ["clip"],
            },
            {
                "video_id": "video-1",
                "video_name": "a.mp4",
                "video_path": "/tmp/a.mp4",
                "frame_id": "frame-2",
                "frame_path": "/tmp/a_2.jpg",
                "timestamp_seconds": 4.0,
                "score": 0.45,
                "matched_by": ["clip"],
            },
            {
                "video_id": "video-1",
                "video_name": "a.mp4",
                "video_path": "/tmp/a.mp4",
                "frame_id": "frame-3",
                "frame_path": "/tmp/a_3.jpg",
                "timestamp_seconds": 8.0,
                "score": 0.5,
                "matched_by": ["ocr"],
            },
        ]

        aggregated = self.service.aggregate_results(results, top_k=10)

        self.assertEqual(len(aggregated), 2)
        self.assertEqual(aggregated[0]["frame_id"], "frame-3")
        self.assertEqual(aggregated[1]["frame_id"], "frame-2")

    def test_aggregate_filters_below_threshold(self) -> None:
        results = [
            {
                "video_id": "video-1",
                "video_name": "a.mp4",
                "video_path": "/tmp/a.mp4",
                "frame_id": "frame-1",
                "frame_path": "/tmp/a_1.jpg",
                "timestamp_seconds": 1.0,
                "score": 0.1,
                "matched_by": ["clip"],
            }
        ]

        aggregated = self.service.aggregate_results(results, top_k=10)
        self.assertEqual(aggregated, [])


if __name__ == "__main__":
    unittest.main()
