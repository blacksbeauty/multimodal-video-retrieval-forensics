"""Tests for CN-CLIP integration: query expansion and dual-backend config."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config import Settings
from services.query_rewrite_service import QueryRewriteService


class TestQueryExpansion:
    """Test Chinese query expansion for CN-CLIP mode."""

    def test_expand_query_wrong_way(self):
        service = QueryRewriteService(use_chinese_clip=True)
        expansions = service.expand_query("车辆逆行")
        assert "车辆逆行" in expansions
        assert "车辆逆向行驶" in expansions
        assert "机动车逆行" in expansions

    def test_expand_query_red_light(self):
        service = QueryRewriteService(use_chinese_clip=True)
        expansions = service.expand_query("闯红灯")
        assert "闯红灯" in expansions
        assert "车辆闯红灯" in expansions
        assert "冲红灯" in expansions

    def test_expand_query_crosses_line(self):
        service = QueryRewriteService(use_chinese_clip=True)
        expansions = service.expand_query("压线")
        assert "压线" in expansions
        assert "车辆压线" in expansions
        assert "压线行驶" in expansions

    def test_expand_query_no_event_keyword(self):
        service = QueryRewriteService(use_chinese_clip=True)
        expansions = service.expand_query("白色汽车")
        assert expansions == ["白色汽车"]

    def test_expand_query_dedup(self):
        service = QueryRewriteService(use_chinese_clip=True)
        expansions = service.expand_query("逆行")
        assert expansions[0] == "逆行"
        assert len(expansions) == len(set(expansions))


class TestChineseClipRewrite:
    """Test query rewriting in CN-CLIP mode vs OpenCLIP mode."""

    def test_event_query_includes_chinese_synonyms(self):
        service = QueryRewriteService(use_chinese_clip=True)
        intent = service.parse_query_intent("车辆逆行")
        assert intent.rewritten_queries[0] == "车辆逆行"
        assert any("逆向行驶" in q for q in intent.rewritten_queries)
        # Layer 5: CN-CLIP mode generates Chinese event names, not English type names
        assert "逆行" in intent.rewritten_queries
        assert intent.query_type == "event"

    def test_event_query_red_light_chinese_expansion(self):
        service = QueryRewriteService(use_chinese_clip=True)
        intent = service.parse_query_intent("车辆闯红灯")
        assert intent.rewritten_queries[0] == "车辆闯红灯"
        assert any("冲红灯" in q for q in intent.rewritten_queries)

    def test_object_query_chinese_entity_display(self):
        service = QueryRewriteService(use_chinese_clip=True)
        intent = service.parse_query_intent("汽车")
        assert "汽车" in intent.rewritten_queries

    def test_openclip_mode_keeps_english(self):
        service = QueryRewriteService(use_chinese_clip=False)
        intent = service.parse_query_intent("汽车")
        assert "car" in intent.rewritten_queries

    def test_openclip_mode_no_chinese_synonym_expansion(self):
        service = QueryRewriteService(use_chinese_clip=False)
        intent = service.parse_query_intent("车辆逆行")
        assert intent.rewritten_queries[0] == "车辆逆行"
        assert "wrong_way_driving" in intent.rewritten_queries
        assert "车辆逆向行驶" not in intent.rewritten_queries

    def test_composite_query_chinese_expansion(self):
        service = QueryRewriteService(use_chinese_clip=True)
        intent = service.parse_query_intent("白色汽车")
        assert intent.rewritten_queries[0] == "白色汽车"
        assert any("汽车" in q for q in intent.rewritten_queries)


class TestEmbeddingServiceBackend:
    """Test EmbeddingService backend configuration without loading models."""

    def test_default_backend_is_cnclip(self):
        settings = Settings()
        assert settings.clip_backend == "cnclip"

    def test_cnclip_config_defaults(self):
        settings = Settings()
        settings.clip_backend = "cnclip"
        assert settings.cnclip_model_name == "ViT-B-16"
        assert settings.cnclip_use_modelscope is True
        assert settings.cnclip_download_root.name == "ckpts"

    def test_backend_selection_at_load_time(self):
        """Backend selection happens at load_model() time, not __init__."""
        settings = Settings()
        settings.clip_backend = "cnclip"
        from services.embedding_service import EmbeddingService

        service = EmbeddingService(settings)
        assert service._model is None
        assert service._embedding_dim is None

    def test_openclip_backend_still_works(self):
        """OpenCLIP backend remains selectable and must not break."""
        settings = Settings()
        settings.clip_backend = "openclip"
        assert settings.clip_backend == "openclip"
        from services.embedding_service import EmbeddingService

        service = EmbeddingService(settings)
        assert service._model is None


class TestHybridSearchWeights:
    """Test that event-type CLIP weight was adjusted for CN-CLIP."""

    def test_event_clip_weight_increased(self):
        from services.hybrid_search_service import HybridSearchService

        # 权重通过 _weights_for_intent 静态分配，无需实例化完整服务
        service = object.__new__(HybridSearchService)
        weights = service._weights_for_intent("event")
        assert weights["clip"] == 0.10
        assert weights["event"] == 0.60
        # 权重总和应归一化
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01
