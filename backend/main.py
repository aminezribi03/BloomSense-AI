"""
Entry point for the ML API.

This module initialises the FastAPI application, loads the trained
model and scaler into memory, configures logging, creates database
tables if they do not exist, and registers API routers. On startup
the application prints its configuration and logs a message to
facilitate troubleshooting.
"""

import logging
from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .database import init_db
from .routers import history, metrics, predict


def create_app() -> FastAPI:
    """Factory to create and configure the FastAPI application."""
    app = FastAPI(
        title="Iris ML API",
        description="A production-ready API for predicting iris species.",
        version="2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialise the SQLite database (creates tables if they do not exist)
    init_db()

    # Register routers
    app.include_router(predict.router)
    app.include_router(metrics.router)
    app.include_router(history.router)

    @app.on_event("startup")
    async def load_model_and_scaler() -> None:
        """Load the trained model and scaler into memory on startup."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        logging.info("Starting Iris ML API version 2")
        logging.info("Configuration: %s", settings.to_dict())

        model_path: Path = settings.MODEL_PATH
        scaler_path: Path = settings.SCALER_PATH

        if not model_path.exists() or not scaler_path.exists():
            logging.error(
                "Model or scaler file not found. Have you run the pipeline?"
            )
            return

        app.state.model = joblib.load(model_path)
        app.state.scaler = joblib.load(scaler_path)

        logging.info(
            "Loaded model from %s and scaler from %s",
            model_path.name,
            scaler_path.name,
        )

    return app


app: FastAPI = create_app()
