"""Repository pattern for database access."""

import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

from sqlalchemy import create_engine, select, update, delete
from sqlalchemy.orm import Session, sessionmaker

from .config import DatabaseConfig, get_database_url
from .models import (
    Base,
    Video,
    Label,
    Model,
    Detection,
    DetectionEvent,
    Job,
    VideoStatus,
    JobStatus,
    ReviewStatus,
)


class Repository:
    """Data access layer for Prismata database."""

    def __init__(self, database_url: str | None = None, config: DatabaseConfig | None = None):
        """Initialize repository with database connection.

        Args:
            database_url: SQLAlchemy database URL. If None, uses config or environment.
            config: Database configuration for connection pool settings.
        """
        if database_url is None:
            database_url = get_database_url(config)

        if config is None:
            config = DatabaseConfig.from_env()

        # Create engine with connection pool
        engine_kwargs = {
            "pool_pre_ping": True,  # Test connections before use
        }

        # PostgreSQL-specific pool settings
        if "postgresql" in database_url:
            engine_kwargs.update({
                "pool_size": config.pool_size,
                "max_overflow": config.max_overflow,
                "pool_timeout": config.pool_timeout,
                "pool_recycle": config.pool_recycle,
            })

        self.engine = create_engine(database_url, **engine_kwargs)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

    def create_tables(self) -> None:
        """Create all tables in the database."""
        Base.metadata.create_all(self.engine)

    def drop_tables(self) -> None:
        """Drop all tables (use with caution)."""
        Base.metadata.drop_all(self.engine)

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Context manager for database sessions."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # Video operations

    def create_video(
        self,
        s3_key: str,
        filename: str,
        domain: str | None = None,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        total_frames: int | None = None,
        duration_seconds: float | None = None,
        file_size_bytes: int | None = None,
    ) -> Video:
        """Create a new video record."""
        with self.session() as session:
            video = Video(
                s3_key=s3_key,
                filename=filename,
                domain=domain,
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                duration_seconds=duration_seconds,
                file_size_bytes=file_size_bytes,
            )
            session.add(video)
            session.flush()
            return video

    def get_video(self, video_id: uuid.UUID | str) -> Video | None:
        """Get video by ID."""
        if isinstance(video_id, str):
            video_id = uuid.UUID(video_id)
        with self.session() as session:
            return session.get(Video, video_id)

    def get_video_by_s3_key(self, s3_key: str) -> Video | None:
        """Get video by S3 key."""
        with self.session() as session:
            stmt = select(Video).where(Video.s3_key == s3_key)
            return session.scalar(stmt)

    def list_videos(
        self,
        domain: str | None = None,
        status: VideoStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Video]:
        """List videos with optional filters."""
        with self.session() as session:
            stmt = select(Video).order_by(Video.uploaded_at.desc())

            if domain:
                stmt = stmt.where(Video.domain == domain)
            if status:
                stmt = stmt.where(Video.status == status)

            stmt = stmt.limit(limit).offset(offset)
            return list(session.scalars(stmt).all())

    def update_video_status(
        self,
        video_id: uuid.UUID | str,
        status: VideoStatus,
        error_message: str | None = None,
    ) -> None:
        """Update video processing status."""
        if isinstance(video_id, str):
            video_id = uuid.UUID(video_id)
        with self.session() as session:
            stmt = (
                update(Video)
                .where(Video.id == video_id)
                .values(
                    status=status,
                    error_message=error_message,
                    processed_at=datetime.utcnow() if status == VideoStatus.READY else None,
                )
            )
            session.execute(stmt)

    # Label operations

    def create_label(
        self,
        video_id: uuid.UUID | str,
        event_id: str,
        start_frame: int,
        end_frame: int,
        event_type: str = "default",
        is_false_positive: bool = False,
        source: str = "manual",
        confidence: float | None = None,
    ) -> Label:
        """Create a new label."""
        if isinstance(video_id, str):
            video_id = uuid.UUID(video_id)
        with self.session() as session:
            label = Label(
                video_id=video_id,
                event_id=event_id,
                start_frame=start_frame,
                end_frame=end_frame,
                event_type=event_type,
                is_false_positive=is_false_positive,
                source=source,
                confidence=confidence,
            )
            session.add(label)
            session.flush()
            return label

    def get_labels_for_video(self, video_id: uuid.UUID | str) -> list[Label]:
        """Get all labels for a video."""
        if isinstance(video_id, str):
            video_id = uuid.UUID(video_id)
        with self.session() as session:
            stmt = (
                select(Label)
                .where(Label.video_id == video_id)
                .order_by(Label.start_frame)
            )
            return list(session.scalars(stmt).all())

    def delete_label(self, label_id: uuid.UUID | str) -> bool:
        """Delete a label."""
        if isinstance(label_id, str):
            label_id = uuid.UUID(label_id)
        with self.session() as session:
            stmt = delete(Label).where(Label.id == label_id)
            result = session.execute(stmt)
            return result.rowcount > 0

    # Model operations

    def create_model(
        self,
        name: str,
        domain: str,
        s3_key: str,
        config: dict | None = None,
        metrics: dict | None = None,
        val_f1: float | None = None,
        val_accuracy: float | None = None,
        training_job_id: uuid.UUID | str | None = None,
    ) -> Model:
        """Register a new trained model."""
        if isinstance(training_job_id, str):
            training_job_id = uuid.UUID(training_job_id)
        with self.session() as session:
            model = Model(
                name=name,
                domain=domain,
                s3_key=s3_key,
                config=config,
                metrics=metrics,
                val_f1=val_f1,
                val_accuracy=val_accuracy,
                training_job_id=training_job_id,
            )
            session.add(model)
            session.flush()
            return model

    def get_model(self, model_id: uuid.UUID | str) -> Model | None:
        """Get model by ID."""
        if isinstance(model_id, str):
            model_id = uuid.UUID(model_id)
        with self.session() as session:
            return session.get(Model, model_id)

    def get_production_model(self, domain: str) -> Model | None:
        """Get the production model for a domain."""
        with self.session() as session:
            stmt = (
                select(Model)
                .where(Model.domain == domain)
                .where(Model.is_production == True)
                .order_by(Model.created_at.desc())
            )
            return session.scalar(stmt)

    def list_models(
        self,
        domain: str | None = None,
        include_archived: bool = False,
        limit: int = 100,
    ) -> list[Model]:
        """List models with optional filters."""
        with self.session() as session:
            stmt = select(Model).order_by(Model.created_at.desc())

            if domain:
                stmt = stmt.where(Model.domain == domain)
            if not include_archived:
                stmt = stmt.where(Model.is_archived == False)

            stmt = stmt.limit(limit)
            return list(session.scalars(stmt).all())

    def set_production_model(self, model_id: uuid.UUID | str) -> None:
        """Set a model as the production model for its domain."""
        if isinstance(model_id, str):
            model_id = uuid.UUID(model_id)
        with self.session() as session:
            model = session.get(Model, model_id)
            if not model:
                raise ValueError(f"Model not found: {model_id}")

            # Unset current production model for this domain
            stmt = (
                update(Model)
                .where(Model.domain == model.domain)
                .where(Model.is_production == True)
                .values(is_production=False)
            )
            session.execute(stmt)

            # Set new production model
            model.is_production = True
            model.published_at = datetime.utcnow()

    # Detection operations

    def create_detection(
        self,
        video_id: uuid.UUID | str,
        model_id: uuid.UUID | str,
        threshold: float = 0.5,
        job_id: uuid.UUID | str | None = None,
    ) -> Detection:
        """Create a new detection run."""
        if isinstance(video_id, str):
            video_id = uuid.UUID(video_id)
        if isinstance(model_id, str):
            model_id = uuid.UUID(model_id)
        if isinstance(job_id, str):
            job_id = uuid.UUID(job_id)

        with self.session() as session:
            detection = Detection(
                video_id=video_id,
                model_id=model_id,
                threshold=threshold,
                job_id=job_id,
            )
            session.add(detection)
            session.flush()
            return detection

    def get_detection(self, detection_id: uuid.UUID | str) -> Detection | None:
        """Get detection by ID."""
        if isinstance(detection_id, str):
            detection_id = uuid.UUID(detection_id)
        with self.session() as session:
            return session.get(Detection, detection_id)

    def update_detection_results(
        self,
        detection_id: uuid.UUID | str,
        num_events: int,
        s3_results_key: str | None = None,
        status: JobStatus = JobStatus.COMPLETED,
    ) -> None:
        """Update detection with results."""
        if isinstance(detection_id, str):
            detection_id = uuid.UUID(detection_id)
        with self.session() as session:
            stmt = (
                update(Detection)
                .where(Detection.id == detection_id)
                .values(
                    num_events=num_events,
                    s3_results_key=s3_results_key,
                    status=status,
                    completed_at=datetime.utcnow(),
                )
            )
            session.execute(stmt)

    def add_detection_event(
        self,
        detection_id: uuid.UUID | str,
        start_time: float,
        end_time: float,
        confidence: float,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> DetectionEvent:
        """Add a detected event."""
        if isinstance(detection_id, str):
            detection_id = uuid.UUID(detection_id)
        with self.session() as session:
            event = DetectionEvent(
                detection_id=detection_id,
                start_time=start_time,
                end_time=end_time,
                confidence=confidence,
                start_frame=start_frame,
                end_frame=end_frame,
            )
            session.add(event)
            session.flush()
            return event

    def get_detection_events(
        self,
        detection_id: uuid.UUID | str,
        review_status: ReviewStatus | None = None,
    ) -> list[DetectionEvent]:
        """Get events for a detection run."""
        if isinstance(detection_id, str):
            detection_id = uuid.UUID(detection_id)
        with self.session() as session:
            stmt = (
                select(DetectionEvent)
                .where(DetectionEvent.detection_id == detection_id)
                .order_by(DetectionEvent.start_time)
            )
            if review_status:
                stmt = stmt.where(DetectionEvent.review_status == review_status)
            return list(session.scalars(stmt).all())

    def update_event_review(
        self,
        event_id: uuid.UUID | str,
        review_status: ReviewStatus,
        reviewed_by: str | None = None,
        label_id: uuid.UUID | str | None = None,
    ) -> None:
        """Update event review status."""
        if isinstance(event_id, str):
            event_id = uuid.UUID(event_id)
        if isinstance(label_id, str):
            label_id = uuid.UUID(label_id)
        with self.session() as session:
            stmt = (
                update(DetectionEvent)
                .where(DetectionEvent.id == event_id)
                .values(
                    review_status=review_status,
                    reviewed_at=datetime.utcnow(),
                    reviewed_by=reviewed_by,
                    label_id=label_id,
                )
            )
            session.execute(stmt)

    # Job operations

    def create_job(
        self,
        job_type: str,
        config: dict | None = None,
        aws_job_id: str | None = None,
        instance_type: str | None = None,
    ) -> Job:
        """Create a new async job."""
        with self.session() as session:
            job = Job(
                job_type=job_type,
                config=config,
                aws_job_id=aws_job_id,
                instance_type=instance_type,
            )
            session.add(job)
            session.flush()
            return job

    def get_job(self, job_id: uuid.UUID | str) -> Job | None:
        """Get job by ID."""
        if isinstance(job_id, str):
            job_id = uuid.UUID(job_id)
        with self.session() as session:
            return session.get(Job, job_id)

    def get_job_by_aws_id(self, aws_job_id: str) -> Job | None:
        """Get job by AWS job ID."""
        with self.session() as session:
            stmt = select(Job).where(Job.aws_job_id == aws_job_id)
            return session.scalar(stmt)

    def update_job_status(
        self,
        job_id: uuid.UUID | str,
        status: JobStatus,
        progress: int | None = None,
        status_message: str | None = None,
        error_message: str | None = None,
        result: dict | None = None,
    ) -> None:
        """Update job status."""
        if isinstance(job_id, str):
            job_id = uuid.UUID(job_id)
        with self.session() as session:
            values: dict[str, Any] = {"status": status}

            if progress is not None:
                values["progress"] = progress
            if status_message is not None:
                values["status_message"] = status_message
            if error_message is not None:
                values["error_message"] = error_message
            if result is not None:
                values["result"] = result

            if status == JobStatus.RUNNING:
                values["started_at"] = datetime.utcnow()
            elif status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                values["completed_at"] = datetime.utcnow()

            stmt = update(Job).where(Job.id == job_id).values(**values)
            session.execute(stmt)

    def list_jobs(
        self,
        job_type: str | None = None,
        status: JobStatus | None = None,
        limit: int = 100,
    ) -> list[Job]:
        """List jobs with optional filters."""
        with self.session() as session:
            stmt = select(Job).order_by(Job.created_at.desc())

            if job_type:
                stmt = stmt.where(Job.job_type == job_type)
            if status:
                stmt = stmt.where(Job.status == status)

            stmt = stmt.limit(limit)
            return list(session.scalars(stmt).all())

    def get_pending_jobs(self, job_type: str | None = None) -> list[Job]:
        """Get all pending jobs."""
        return self.list_jobs(job_type=job_type, status=JobStatus.PENDING)


# Singleton repository instance
_repository: Repository | None = None


def get_repository(database_url: str | None = None) -> Repository:
    """Get or create the repository singleton."""
    global _repository
    if _repository is None:
        _repository = Repository(database_url)
    return _repository
