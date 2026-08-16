"""API-level tests for the unified search entry point (收敛至 /api/search/hybrid).

Verifies the single-entry convention agreed for the "all-seeing vision" project:
- 5 legacy single-channel search endpoints are marked ``deprecated`` in OpenAPI
  (still callable for backward compatibility, but flagged for removal).
- ``POST /api/search/hybrid`` is the only non-deprecated search entry point.

Uses a minimal FastAPI app that mounts the real ``api.routes.router`` with no
services, so no FAISS / CLIP / OCR models are loaded.
"""

import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import router


LEGACY_SEARCH_PATHS = [
    "/api/search",
    "/api/detection/search",
    "/api/trajectory/search",
    "/api/event/search",
    "/api/ocr/search",
]

HYBRID_ENTRY_PATH = "/api/search/hybrid"

# 全部检索类端点（旧单通道 + 唯一正式入口）
ALL_SEARCH_PATHS = LEGACY_SEARCH_PATHS + [HYBRID_ENTRY_PATH]


class UnifiedSearchEntryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app = FastAPI()
        app.include_router(router, prefix="/api")
        cls.client = TestClient(app)
        cls.spec = cls.client.get("/openapi.json").json()

    def test_legacy_search_endpoints_are_deprecated(self) -> None:
        paths = self.spec.get("paths", {})
        for path in LEGACY_SEARCH_PATHS:
            with self.subTest(path=path):
                op = paths.get(path, {}).get("post", {})
                self.assertTrue(
                    op.get("deprecated", False),
                    f"{path} should be marked deprecated=True in OpenAPI",
                )

    def test_hybrid_is_the_only_non_deprecated_search_entry(self) -> None:
        paths = self.spec.get("paths", {})
        hybrid_op = paths.get(HYBRID_ENTRY_PATH, {}).get("post", {})
        self.assertFalse(hybrid_op.get("deprecated", False))

        non_deprecated = [
            p for p in ALL_SEARCH_PATHS
            if not paths.get(p, {}).get("post", {}).get("deprecated", False)
        ]
        # 唯一非 deprecated 的检索端点只能是 hybrid。
        self.assertEqual(non_deprecated, [HYBRID_ENTRY_PATH])


if __name__ == "__main__":
    unittest.main()
