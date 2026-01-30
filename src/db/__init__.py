"""Database layer for Prismata.

Provides SQLAlchemy ORM models and repository patterns for
videos, labels, models, detections, and jobs.
"""

from .models import (
    Base,
    Video,
    Label,
    Model,
    Detection,
    DetectionEvent,
    Job,
    JobStatus,
    VideoStatus,
    ReviewStatus,
)
from .repository import Repository, get_repository
from .config import DatabaseConfig, get_database_url

__all__ = [
    # Models
    "Base",
    "Video",
    "Label",
    "Model",
    "Detection",
    "DetectionEvent",
    "Job",
    # Enums
    "JobStatus",
    "VideoStatus",
    "ReviewStatus",
    # Repository
    "Repository",
    "get_repository",
    # Config
    "DatabaseConfig",
    "get_database_url",
]
