"""
backend package for the ML API.

This package contains the FastAPI application, database models and
routers for handling prediction requests, serving evaluation metrics and
returning prediction history.  The application is designed as a
micro‑service: configuration and infrastructure concerns live in this
package while model training is performed separately by the pipeline
scripts.
"""
