from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes import page_router, router
from config import get_settings
from core.app_state import AppState
from utils.logging import setup_logging


logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the OCR-enabled FastAPI application."""
    settings = get_settings()
    setup_logging(settings.index_dir, settings.log_level)

    for directory in (
        settings.videos_dir,
        settings.frames_dir,
        settings.embeddings_dir,
        settings.index_dir,
        settings.metadata_dir,
        settings.dataset_metadata_dir,
        settings.ocr_metadata_dir,
        settings.detection_metadata_dir,
        settings.trajectory_metadata_dir,
        settings.event_metadata_dir,
        settings.event_config_dir,
        settings.web_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Initialize application-scoped singletons and emit startup/shutdown logs."""
        app.state.settings = settings
        app.state.services = AppState.build(settings)
        app.state.services.initialize_ocr_once(preload=False)
        logger.info("Application started: %s", settings.project_name)
        logger.info("OCR frame directory: %s", settings.frames_dir)
        logger.info("OCR metadata directory: %s", settings.ocr_metadata_dir)
        logger.info("Dataset metadata directory: %s", settings.dataset_metadata_dir)
        logger.info("Detection metadata directory: %s", settings.detection_metadata_dir)
        logger.info("Trajectory metadata directory: %s", settings.trajectory_metadata_dir)
        logger.info("Event metadata directory: %s", settings.event_metadata_dir)
        logger.info("Event config directory: %s", settings.event_config_dir)
        logger.info("OCR routes ready: %s/ocr/ingest-directory, %s/ocr/search", settings.api_prefix, settings.api_prefix)
        try:
            yield
        finally:
            logger.info("Application shutting down.")

    app = FastAPI(
        title=settings.project_name,
        version="0.1.0",
        description="Natural language video retrieval system based on OpenCLIP and FAISS.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(page_router)
    app.include_router(router, prefix=settings.api_prefix)
    app.mount("/frames", StaticFiles(directory=settings.frames_dir), name="frames")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        reload_includes=["*.py", "*.html", "*.css", "*.js"],
        reload_excludes=[
            "frames/*",
            "embeddings/*",
            "index/*",
            "metadata/*",
            "videos/*",
            "**/__pycache__/*",
            "*.log",
            "*.index",
            "*.json",
            "*.npy",
            "*.jpg",
        ],
    )
