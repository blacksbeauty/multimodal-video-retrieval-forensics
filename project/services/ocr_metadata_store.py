from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from config import Settings
from core.schemas import OCRVideoMetadata


logger = logging.getLogger(__name__)


class OCRMetadataStore:
    """Persistence layer for OCR metadata stored as local UTF-8 JSON files."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the store and ensure the OCR metadata directory exists."""
        self.settings = settings
        self.base_dir = self.settings.ocr_metadata_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_metadata(
        self,
        metadata: OCRVideoMetadata,
        overwrite: bool = True,
    ) -> Path:
        """Save OCR metadata to metadata/ocr/<video_name>.json."""
        output_path = self.base_dir / f"{metadata.video_name}.json"
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"OCR metadata already exists: {output_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            metadata.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.info("Saved OCR metadata to %s", output_path)
        return output_path

    def load_metadata(self, video_name: str) -> Optional[OCRVideoMetadata]:
        """Load OCR metadata for one video. Return None when the file is missing or invalid."""
        input_path = self.base_dir / f"{video_name}.json"
        if not input_path.exists():
            logger.warning("OCR metadata file not found: %s", input_path)
            return None

        try:
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            metadata = OCRVideoMetadata.model_validate(payload)
            logger.info("Loaded OCR metadata from %s", input_path)
            return metadata
        except json.JSONDecodeError:
            logger.exception("Failed to decode OCR metadata JSON: %s", input_path)
            return None
        except Exception:
            logger.exception("Failed to validate OCR metadata: %s", input_path)
            return None

    def list_metadata(self) -> List[Path]:
        """List all persisted OCR metadata JSON files under metadata/ocr."""
        if not self.base_dir.exists():
            logger.warning("OCR metadata directory does not exist: %s", self.base_dir)
            return []

        files = sorted(self.base_dir.glob("*.json"))
        logger.info("Found %s OCR metadata files in %s", len(files), self.base_dir)
        return files
