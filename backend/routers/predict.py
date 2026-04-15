"""
Prediction endpoints for the ML API.

These routes expose the `/predict` and `/predict/batch` endpoints, allowing
clients to send feature values and receive predicted class indices
together with probabilities.  Each call is logged and persisted to the
database.  The router depends on the FastAPI application storing the
loaded model and scaler on its `.state` attribute (configured in
`backend.main`).
"""

from typing import List

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
import sqlite3

from .. import crud, schemas
from ..database import get_db


router = APIRouter(tags=["predictions"])



@router.post("/predict", response_model=schemas.PredictionResponse)
async def predict(
    request: Request,
    payload: schemas.PredictionRequest,
    db: sqlite3.Connection = Depends(get_db),
) -> schemas.PredictionResponse:
    """Predict the class of a single iris sample.

    Accepts a JSON body with the four iris measurements, scales the
    features using the pre‑fitted scaler and returns the predicted
    class index and probability.  The prediction is stored in the
    database for later analysis.
    """
    model = getattr(request.app.state, "model", None)
    scaler = getattr(request.app.state, "scaler", None)
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    
    X = np.array(
        [
            [
                payload.sepal_length,
                payload.sepal_width,
                payload.petal_length,
                payload.petal_width,
            ]
        ]
    )
    X_scaled = scaler.transform(X)
    probas = model.predict_proba(X_scaled)[0]
    predicted_class = int(np.argmax(probas))
    probability = float(probas[predicted_class])

   
    crud.create_prediction(
        db,
        sepal_length=payload.sepal_length,
        sepal_width=payload.sepal_width,
        petal_length=payload.petal_length,
        petal_width=payload.petal_width,
        predicted_class=predicted_class,
        probability=probability,
    )

    return schemas.PredictionResponse(
        predicted_class=predicted_class, probability=probability
    )


@router.post("/predict/batch", response_model=List[schemas.PredictionResponse])
async def predict_batch(
    request: Request,
    payloads: List[schemas.PredictionRequest],
    db: sqlite3.Connection = Depends(get_db),
) -> List[schemas.PredictionResponse]:
    """Predict classes for multiple iris samples.

    Accepts a JSON array of feature objects.  Each prediction is stored in
    the database and the results are returned in the same order.
    """
    model = getattr(request.app.state, "model", None)
    scaler = getattr(request.app.state, "scaler", None)
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    responses: List[schemas.PredictionResponse] = []
   
    X = np.array([
        [
            p.sepal_length,
            p.sepal_width,
            p.petal_length,
            p.petal_width,
        ]
        for p in payloads
    ])
    X_scaled = scaler.transform(X)
    probas = model.predict_proba(X_scaled)
    for i, probs in enumerate(probas):
        predicted_class = int(np.argmax(probs))
        probability = float(probs[predicted_class])
        
        crud.create_prediction(
            db,
            sepal_length=payloads[i].sepal_length,
            sepal_width=payloads[i].sepal_width,
            petal_length=payloads[i].petal_length,
            petal_width=payloads[i].petal_width,
            predicted_class=predicted_class,
            probability=probability,
        )
        responses.append(
            schemas.PredictionResponse(
                predicted_class=predicted_class,
                probability=probability,
            )
        )
    return responses
