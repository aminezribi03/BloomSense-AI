# Iris ML System V2

A production-oriented end-to-end machine learning application for Iris species classification, featuring a reproducible training pipeline, a FastAPI backend, SQLite-based prediction history, and an interactive frontend dashboard with model metrics and visual analytics.

## Features
- End-to-end ML pipeline
- Model training and evaluation
- FastAPI inference backend
- SQLite prediction history
- Interactive frontend dashboard
- Prediction history chart
- Model metrics display

## Architecture
The project is divided into three main layers:
- Pipeline: data preparation, encryption, training, evaluation
- Backend: inference API, metrics endpoint, history endpoint
- Frontend: prediction form, chart visualization, metrics dashboard

## Tech Stack
- Python
- Scikit-learn
- FastAPI
- SQLite
- HTML / CSS / JavaScript
- Chart.js
- Docker

## Project Structure
```text
backend/
frontend/
pipeline/
data_v2/
README.md
requirements.txt
Dockerfile
docker-compose.yml
