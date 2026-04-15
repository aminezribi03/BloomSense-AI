"""
History endpoint for the ML API.

This route aggregates all stored predictions and returns a count of how
many times each class has been predicted.  The result can be used by
the front‑end to visualise the distribution of predictions.
"""

from typing import List

from fastapi import APIRouter, Depends
import sqlite3

from .. import crud, schemas
from ..database import get_db


router = APIRouter(tags=["history"])



@router.get("/history", response_model=schemas.HistoryResponse)
async def get_history(db: sqlite3.Connection = Depends(get_db)) -> schemas.HistoryResponse:
    """Return aggregated prediction counts by class."""
    history = crud.get_history(db)
    
    items: List[schemas.HistoryItem] = [
        schemas.HistoryItem(**record) for record in history
    ]
    return schemas.HistoryResponse(history=items)
