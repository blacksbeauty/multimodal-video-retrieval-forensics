"""Tests for AccidentDatasetImportService: meta parsing, frame extraction, and end-to-end import."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from config import Settings
from services.accident_dataset_import_service import AccidentDatasetImportService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATASET_ROOT = Path(__file__).resolve().parent.parent.parent / "type1_subtype1_accident"
BEV_DIR = DATASET_ROOT / "ego_vehicle" / "BEV_instance_camera"
META_DIR = DATASET_ROOT / "meta"

FIRST_SCENARIO = "Town01_type001_subtype0001_scenario00004"
SAMPLE_NPZ = BEV_DIR / FIRST_SCENARIO / f"{FIRST_SCENARIO}_001.npz"
SAMPLE_META = META_DIR / f"{FIRST_SCENARIO}.txt"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings with temp dirs to avoid polluting the real project."""
    s = Settings()
    s.frames_dir = tmp_path / "frames"
    s.frames_dir.mkdir(parents=True, exist_ok=True)
    s.dataset_metadata_dir = tmp_path / "metadata" / "datasets"
    s.dataset_metadata_dir.mkdir(parents=True, exist_ok=True)
    s.accident_frame_rate = 20.0
    return s


@pytest.fixture
def service(settings: Settings) -> AccidentDatasetImportService:
    return AccidentDatasetImportService(settings)


# ---------------------------------------------------------------------------
# Meta file parsing
# ---------------------------------------------------------------------------

class TestParseMetaFile:
    def test_parse_existing_meta(self, service: AccidentDatasetImportService):
        if not SAMPLE_META.exists():
            pytest.skip(f"Meta file not found: {SAMPLE_META}")
        meta = service.parse_meta_file(SAMPLE_META)
        assert "weather" in meta
        assert "road_type" in meta
        assert "ego_vehicle_direction" in meta
        assert "other_vehicle_direction" in meta
        assert meta["weather"] in {"ClearNoon", "ClearSunset", "HardRainNoon", "WetSunset", "WetCloudyNoon", "WetCloudySunset", "MidRainSunset", "HardRainSunset", "SoftRainSunset"}

    def test_parse_nonexistent_meta(self, service: AccidentDatasetImportService):
        meta = service.parse_meta_file(Path("/nonexistent/meta.txt"))
        assert meta == {}

    def test_parse_meta_content_structure(self, service: AccidentDatasetImportService):
        """Verify the parsed meta contains all expected keys."""
        if not SAMPLE_META.exists():
            pytest.skip(f"Meta file not found: {SAMPLE_META}")
        meta = service.parse_meta_file(SAMPLE_META)
        expected_keys = {
            "weather", "ego_agent_id", "ego_agent_type",
            "other_agent_id", "other_agent_type",
            "collision_distance", "spawn_side", "collision_position",
            "colliding_agents", "agents_id", "road_type",
            "another_vehicle_spawn_side",
            "ego_vehicle_direction", "other_vehicle_direction",
        }
        assert expected_keys.issubset(set(meta.keys())), f"Missing keys: {expected_keys - set(meta.keys())}"


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

class TestFrameExtraction:
    def test_npz_contains_valid_image(self):
        if not SAMPLE_NPZ.exists():
            pytest.skip(f"NPZ file not found: {SAMPLE_NPZ}")
        data = np.load(str(SAMPLE_NPZ), allow_pickle=True)
        assert "data" in data
        array = data["data"]
        assert array.ndim == 3
        assert array.shape[2] == 3
        assert array.dtype == np.uint8

    def test_extract_and_save_frame(self, service: AccidentDatasetImportService, tmp_path: Path):
        if not SAMPLE_NPZ.exists():
            pytest.skip(f"NPZ file not found: {SAMPLE_NPZ}")
        target = tmp_path / "test_frame.jpg"
        service._extract_and_save_frame(SAMPLE_NPZ, target)
        assert target.exists()
        img = Image.open(str(target))
        assert img.size == (1200, 1200)
        assert img.mode == "RGB"

    def test_build_frame_metadata(self, service: AccidentDatasetImportService):
        scenario_dir = BEV_DIR / FIRST_SCENARIO
        if not scenario_dir.exists():
            pytest.skip(f"Scenario dir not found: {scenario_dir}")
        metadata = service.build_frame_metadata(
            scenario_dir=scenario_dir,
            video_name=f"Accident_{FIRST_SCENARIO}.mp4",
            frame_step=10,
        )
        assert len(metadata) > 0
        first = metadata[0]
        assert "video_id" in first
        assert "video_name" in first
        assert "frame_path" in first
        assert "frame_index" in first
        assert "timestamp_seconds" in first
        assert first["video_name"] == f"Accident_{FIRST_SCENARIO}.mp4"
        assert Path(first["frame_path"]).exists()


# ---------------------------------------------------------------------------
# Scenario iteration
# ---------------------------------------------------------------------------

class TestIterScenarios:
    def test_iter_all_scenarios(self, service: AccidentDatasetImportService):
        if not BEV_DIR.exists():
            pytest.skip(f"BEV dir not found: {BEV_DIR}")
        scenarios = list(service.iter_scenarios(DATASET_ROOT, scenario_names=[]))
        assert len(scenarios) > 0
        assert all(s.is_dir() for s in scenarios)

    def test_iter_filtered_scenarios(self, service: AccidentDatasetImportService):
        if not BEV_DIR.exists():
            pytest.skip(f"BEV dir not found: {BEV_DIR}")
        scenarios = list(service.iter_scenarios(DATASET_ROOT, scenario_names=[FIRST_SCENARIO]))
        assert len(scenarios) == 1
        assert scenarios[0].name == FIRST_SCENARIO


# ---------------------------------------------------------------------------
# End-to-end import (mocked CLIP and index)
# ---------------------------------------------------------------------------

class TestImportScenario:
    def test_import_single_scenario_mocked(self, service: AccidentDatasetImportService):
        scenario_dir = BEV_DIR / FIRST_SCENARIO
        if not scenario_dir.exists():
            pytest.skip(f"Scenario dir not found: {scenario_dir}")

        mock_clip = MagicMock()
        mock_clip.encode_image_paths.return_value = np.zeros((1, 512), dtype=np.float32)
        mock_index = MagicMock()

        result = service.import_scenario(
            scenario_dir=scenario_dir,
            dataset_root=DATASET_ROOT,
            frame_step=10,
            clip_service=mock_clip,
            index_service=mock_index,
        )

        assert result["video_id"]
        assert result["video_name"] == f"Accident_{FIRST_SCENARIO}.mp4"
        assert result["imported_frames"] > 0
        assert result["indexed_frames"] == result["imported_frames"]
        assert "weather" in result["scenario_meta"]
        assert "road_type" in result["scenario_meta"]

        mock_clip.encode_image_paths.assert_called_once()
        mock_index.upsert_video_records.assert_called_once()


class TestIngestDirectory:
    def test_ingest_with_max_scenarios(self, service: AccidentDatasetImportService):
        if not BEV_DIR.exists():
            pytest.skip(f"BEV dir not found: {BEV_DIR}")

        mock_clip = MagicMock()
        mock_clip.encode_image_paths.return_value = np.zeros((1, 512), dtype=np.float32)
        mock_index = MagicMock()

        result = service.ingest_directory(
            dataset_root=str(DATASET_ROOT),
            scenario_names=[],
            max_scenarios=1,
            frame_step=10,
            clip_service=mock_clip,
            index_service=mock_index,
        )

        assert result["total_scenarios"] == 1
        assert result["succeeded_scenarios"] == 1
        assert result["failed_scenarios"] == 0
        assert len(result["results"]) == 1
        assert result["results"][0]["imported_frames"] > 0

    def test_source_map_saved(self, service: AccidentDatasetImportService):
        if not BEV_DIR.exists():
            pytest.skip(f"BEV dir not found: {BEV_DIR}")

        mock_clip = MagicMock()
        mock_clip.encode_image_paths.return_value = np.zeros((1, 512), dtype=np.float32)
        mock_index = MagicMock()

        service.ingest_directory(
            dataset_root=str(DATASET_ROOT),
            scenario_names=[FIRST_SCENARIO],
            max_scenarios=None,
            frame_step=10,
            clip_service=mock_clip,
            index_service=mock_index,
        )

        assert service.source_map_path.exists()
        source_map = json.loads(service.source_map_path.read_text(encoding="utf-8"))
        assert len(source_map) > 0
        first_entry = list(source_map.values())[0]
        assert "video_id" in first_entry
        assert "meta" in first_entry
        assert "weather" in first_entry["meta"]
