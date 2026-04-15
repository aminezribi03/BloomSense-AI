"""
Metrics endpoint for the ML API.

This route exposes the `/metrics` endpoint, which returns the
evaluation metrics produced by the most recent pipeline run.  It reads
the JSON file defined in the application settings and responds with a
structured object.  If the metrics file is missing an error is
returned.
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..schemas import MetricsResponse


router = APIRouter(tags=["metrics"])


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    """Return evaluation metrics from the latest pipeline run."""
    metrics_path: Path = settings.METRICS_PATH
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrics file not found")
    try:
        with metrics_path.open() as f:
            metrics_data = json.load(f)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid metrics file format")
    return MetricsResponse(**metrics_data)
