from __future__ import annotations

import logging

from config import Settings


logger = logging.getLogger(__name__)

TRAFFIC_LABELS = frozenset({
    "car",
    "truck",
    "bus",
    "motorcycle",
    "person",
    "traffic light",
})


class TrafficActivityFilter:
    """Traffic-aware video preprocessing layer.

    Filters video frames based on the presence of traffic objects (not motion),
    so that static-but-valuable scenes like red-light waiting or congestion are
    retained while truly empty road segments are discarded.

    Strategy:
      1. Every ``sample_interval``-th sampled frame is run through YOLO.
      2. If traffic objects are found, a retain-window of ``retain_window``
         sampled frames is opened — all frames within the window are kept.
      3. Frames outside any retain window with no detected traffic are skipped.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._model_loaded = False
        self._remaining_retain_window = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset per-video state so retain windows do not leak across videos."""
        self._remaining_retain_window = 0

    def should_retain(
        self,
        frame,
        sampled_index: int,
    ) -> tuple[bool, bool, int]:
        """Decide whether a sampled frame should be kept.

        Args:
            frame: OpenCV BGR image (numpy array).
            sampled_index: Zero-based index of the current sampled frame
                (counts *all* sampled frames, including those previously
                filtered out).

        Returns:
            ``(should_retain, did_detect, target_count)``

            * ``should_retain`` — whether to write the frame to disk.
            * ``did_detect`` — whether a YOLO forward pass was executed.
            * ``target_count`` — number of traffic objects found (``-1`` when
              no detection was performed).
        """
        sample_interval = max(self.settings.traffic_sample_interval, 1)
        should_sample = (sampled_index % sample_interval) == 0

        if should_sample:
            return self._sample_and_decide(frame)

        # Non-sample frame: keep only if inside a retain window.
        if self._remaining_retain_window > 0:
            self._remaining_retain_window -= 1
            return True, False, -1

        return False, False, -1

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _sample_and_decide(self, frame) -> tuple[bool, bool, int]:
        """Run YOLO on *frame* and update the retain window."""
        if not self._ensure_model():
            # Model unavailable — conservatively keep the frame.
            return True, False, -1

        target_count = self._count_traffic_targets(frame)

        if target_count > 0:
            self._remaining_retain_window = self.settings.traffic_retain_window
            return True, True, target_count

        # No traffic objects detected.
        if self._remaining_retain_window > 0:
            self._remaining_retain_window -= 1
            return True, True, 0

        return False, True, 0

    def _ensure_model(self) -> bool:
        """Lazily load the YOLOv8 model (CPU)."""
        if self._model_loaded:
            return self._model is not None

        self._model_loaded = True
        try:
            from ultralytics import YOLO

            logger.info(
                "TrafficActivityFilter: loading YOLO model=%s on cpu",
                self.settings.detection_model_name,
            )
            self._model = YOLO(self.settings.detection_model_name)
            return True
        except Exception:
            logger.exception(
                "TrafficActivityFilter: failed to load YOLO model; "
                "filtering will be bypassed"
            )
            self._model = None
            return False

    def _count_traffic_targets(self, frame) -> int:
        """Run detection on *frame* and count traffic-relevant objects."""
        try:
            results = self._model.predict(
                source=frame,
                conf=self.settings.detection_score_threshold,
                device="cpu",
                verbose=False,
            )
        except Exception:
            logger.exception("TrafficActivityFilter: detection failed")
            return 0

        count = 0
        for result in results:
            names = result.names or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            cls_values = boxes.cls.tolist() if boxes.cls is not None else []
            for cls_idx in cls_values:
                label = str(names.get(int(cls_idx), "")).strip().casefold()
                if label in TRAFFIC_LABELS:
                    count += 1
        return count
