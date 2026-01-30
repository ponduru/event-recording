"""SQLAlchemy ORM models for Prismata."""

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    Index,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class VideoStatus(str, enum.Enum):
    """Video processing status."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class JobStatus(str, enum.Enum):
    """Async job status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReviewStatus(str, enum.Enum):
    """Detection event review status."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class Video(Base):
    """Video metadata and tracking."""

    __tablename__ = "videos"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    s3_key = Column(String(1024), nullable=False, unique=True)
    filename = Column(String(512), nullable=False)
    domain = Column(String(64), index=True)

    # Video properties
    width = Column(Integer)
    height = Column(Integer)
    fps = Column(Numeric(10, 4))
    total_frames = Column(Integer)
    duration_seconds = Column(Numeric(12, 4))
    file_size_bytes = Column(Integer)

    # Status tracking
    status = Column(Enum(VideoStatus), default=VideoStatus.UPLOADED)
    error_message = Column(Text)

    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)

    # Relationships
    labels = relationship("Label", back_populates="video", cascade="all, delete-orphan")
    detections = relationship("Detection", back_populates="video", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_videos_domain_status", "domain", "status"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "s3_key": self.s3_key,
            "filename": self.filename,
            "domain": self.domain,
            "width": self.width,
            "height": self.height,
            "fps": float(self.fps) if self.fps else None,
            "total_frames": self.total_frames,
            "duration_seconds": float(self.duration_seconds) if self.duration_seconds else None,
            "status": self.status.value if self.status else None,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None,
        }


class Label(Base):
    """Labeled events in videos."""

    __tablename__ = "labels"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    event_id = Column(String(64), nullable=False)

    # Frame range
    start_frame = Column(Integer, nullable=False)
    end_frame = Column(Integer, nullable=False)

    # Event type (for multi-event domains)
    event_type = Column(String(64), default="default")

    # Label metadata
    is_false_positive = Column(Boolean, default=False)
    source = Column(String(64), default="manual")  # manual, detection, import
    confidence = Column(Numeric(5, 4))  # For detection-sourced labels
    notes = Column(Text)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    video = relationship("Video", back_populates="labels")

    __table_args__ = (
        Index("idx_labels_video_id", "video_id"),
        Index("idx_labels_event_type", "event_type"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "video_id": str(self.video_id),
            "event_id": self.event_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "event_type": self.event_type,
            "is_false_positive": self.is_false_positive,
            "source": self.source,
            "confidence": float(self.confidence) if self.confidence else None,
        }


class Model(Base):
    """Trained model registry."""

    __tablename__ = "models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(256), nullable=False)
    domain = Column(String(64), nullable=False, index=True)
    version = Column(String(64))
    s3_key = Column(String(1024), nullable=False, unique=True)

    # Model configuration (stored as JSON)
    config = Column(JSONB)

    # Training metrics
    metrics = Column(JSONB)
    val_f1 = Column(Numeric(6, 4))
    val_accuracy = Column(Numeric(6, 4))
    val_precision = Column(Numeric(6, 4))
    val_recall = Column(Numeric(6, 4))

    # Training info
    training_job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id"))
    num_epochs = Column(Integer)
    training_samples = Column(Integer)
    validation_samples = Column(Integer)

    # Deployment status
    is_production = Column(Boolean, default=False)
    is_archived = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    published_at = Column(DateTime)

    # Relationships
    detections = relationship("Detection", back_populates="model")
    training_job = relationship("Job", foreign_keys=[training_job_id])

    __table_args__ = (
        Index("idx_models_domain_production", "domain", "is_production"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "domain": self.domain,
            "version": self.version,
            "s3_key": self.s3_key,
            "config": self.config,
            "metrics": self.metrics,
            "val_f1": float(self.val_f1) if self.val_f1 else None,
            "val_accuracy": float(self.val_accuracy) if self.val_accuracy else None,
            "is_production": self.is_production,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Detection(Base):
    """Inference runs on videos."""

    __tablename__ = "detections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    model_id = Column(UUID(as_uuid=True), ForeignKey("models.id"), nullable=False)

    # Detection parameters
    threshold = Column(Numeric(5, 4), default=0.5)
    merge_distance = Column(Numeric(10, 4))  # seconds

    # Results summary
    num_events = Column(Integer, default=0)
    s3_results_key = Column(String(1024))

    # Status
    status = Column(Enum(JobStatus), default=JobStatus.PENDING)
    job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id"))
    error_message = Column(Text)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

    # Relationships
    video = relationship("Video", back_populates="detections")
    model = relationship("Model", back_populates="detections")
    events = relationship("DetectionEvent", back_populates="detection", cascade="all, delete-orphan")
    job = relationship("Job", foreign_keys=[job_id])

    __table_args__ = (
        Index("idx_detections_video_model", "video_id", "model_id"),
        Index("idx_detections_status", "status"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "video_id": str(self.video_id),
            "model_id": str(self.model_id),
            "threshold": float(self.threshold) if self.threshold else None,
            "num_events": self.num_events,
            "status": self.status.value if self.status else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class DetectionEvent(Base):
    """Individual detected events."""

    __tablename__ = "detection_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    detection_id = Column(UUID(as_uuid=True), ForeignKey("detections.id", ondelete="CASCADE"), nullable=False)

    # Event timing
    start_time = Column(Numeric(12, 4), nullable=False)  # seconds
    end_time = Column(Numeric(12, 4), nullable=False)
    start_frame = Column(Integer)
    end_frame = Column(Integer)

    # Detection confidence
    confidence = Column(Numeric(5, 4), nullable=False)

    # Review status
    review_status = Column(Enum(ReviewStatus), default=ReviewStatus.PENDING)
    reviewed_at = Column(DateTime)
    reviewed_by = Column(String(256))

    # Link to created label (if approved)
    label_id = Column(UUID(as_uuid=True), ForeignKey("labels.id"))

    # Clip storage
    clip_s3_key = Column(String(1024))

    # Relationships
    detection = relationship("Detection", back_populates="events")
    label = relationship("Label")

    __table_args__ = (
        Index("idx_detection_events_detection_id", "detection_id"),
        Index("idx_detection_events_review_status", "review_status"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "detection_id": str(self.detection_id),
            "start_time": float(self.start_time),
            "end_time": float(self.end_time),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "confidence": float(self.confidence),
            "review_status": self.review_status.value if self.review_status else None,
        }


class Job(Base):
    """Async job tracking."""

    __tablename__ = "jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_type = Column(String(64), nullable=False)  # training, inference, video_processing

    # AWS integration
    aws_job_id = Column(String(256))  # SageMaker job ARN or EC2 instance ID
    aws_job_name = Column(String(256))

    # Status tracking
    status = Column(Enum(JobStatus), default=JobStatus.PENDING)
    progress = Column(Integer, default=0)  # 0-100
    status_message = Column(Text)
    error_message = Column(Text)

    # Job configuration
    config = Column(JSONB)
    result = Column(JSONB)

    # Resource tracking
    instance_type = Column(String(64))
    runtime_seconds = Column(Integer)
    estimated_cost = Column(Numeric(10, 4))

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    __table_args__ = (
        Index("idx_jobs_type_status", "job_type", "status"),
        Index("idx_jobs_aws_job_id", "aws_job_id"),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "job_type": self.job_type,
            "aws_job_id": self.aws_job_id,
            "status": self.status.value if self.status else None,
            "progress": self.progress,
            "status_message": self.status_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
