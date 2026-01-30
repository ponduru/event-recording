"""Cloud-aware inference module with storage and database integration."""

import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from .predictor import EventPredictor, DetectionResult, Event


@dataclass
class CloudInferenceConfig:
    """Configuration for cloud inference."""

    # Model
    model_s3_key: str = ""
    model_local_path: str = ""

    # Video
    video_s3_key: str = ""
    video_local_path: str = ""

    # Output
    output_s3_key: str = ""
    save_to_database: bool = True

    # Detection parameters
    threshold: float = 0.5
    buffer_seconds: float = 3.0
    min_gap_seconds: float = 2.0

    # Predictor settings
    window_size: int = 8
    stride: int = 4
    frame_size: tuple[int, int] = (224, 224)
    target_fps: float = 10.0
    batch_size: int = 8
    num_workers: int = 4

    # Storage
    use_cloud_storage: bool = False

    # Job tracking
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class CloudPredictor:
    """Predictor with cloud storage and database integration.

    This wraps the core EventPredictor class to add:
    - S3 storage for videos and models
    - Database tracking for detections and events
    - Progress reporting for async job monitoring
    """

    def __init__(self, config: CloudInferenceConfig):
        """Initialize cloud predictor.

        Args:
            config: Cloud inference configuration.
        """
        self.config = config
        self._storage = None
        self._repository = None
        self._predictor = None
        self._detection_db_id = None

    @property
    def storage(self):
        """Lazy-load storage backend."""
        if self._storage is None and self.config.use_cloud_storage:
            from src.storage import get_storage_backend

            self._storage = get_storage_backend()
        return self._storage

    @property
    def repository(self):
        """Lazy-load database repository."""
        if self._repository is None and self.config.save_to_database:
            from src.db import get_repository

            self._repository = get_repository()
        return self._repository

    def _load_predictor(self, model_path: Path) -> EventPredictor:
        """Load predictor from model path.

        Args:
            model_path: Path to model checkpoint.

        Returns:
            Configured EventPredictor.
        """
        return EventPredictor.from_checkpoint(
            model_path,
            window_size=self.config.window_size,
            stride=self.config.stride,
            frame_size=self.config.frame_size,
            target_fps=self.config.target_fps,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

    def _get_model_path(self) -> Path:
        """Get local path to model, downloading if needed.

        Returns:
            Local path to model checkpoint.
        """
        if self.config.model_local_path:
            return Path(self.config.model_local_path)

        if self.storage and self.config.model_s3_key:
            return self.storage.read_model(self.config.model_s3_key)

        raise ValueError("No model path or S3 key provided")

    def _get_video_path(self) -> Path:
        """Get local path to video, downloading if needed.

        Returns:
            Local path to video file.
        """
        if self.config.video_local_path:
            return Path(self.config.video_local_path)

        if self.storage and self.config.video_s3_key:
            return self.storage.read_video(self.config.video_s3_key)

        raise ValueError("No video path or S3 key provided")

    def _get_video_db_id(self) -> Optional[str]:
        """Get or create video record in database.

        Returns:
            Video database ID or None.
        """
        if not self.repository:
            return None

        s3_key = self.config.video_s3_key or self.config.video_local_path

        # Try to find existing video
        video = self.repository.get_video_by_s3_key(s3_key)
        if video:
            return str(video.id)

        # Create new video record
        video = self.repository.create_video(
            s3_key=s3_key,
            filename=Path(s3_key).name,
        )
        return str(video.id)

    def _get_model_db_id(self) -> Optional[str]:
        """Get model database ID if available.

        Returns:
            Model database ID or None.
        """
        if not self.repository:
            return None

        s3_key = self.config.model_s3_key
        if not s3_key:
            return None

        # Try to find model by S3 key
        models = self.repository.list_models()
        for model in models:
            if model.s3_key == s3_key:
                return str(model.id)

        return None

    def _create_detection_record(self, video_id: str, model_id: Optional[str]) -> None:
        """Create detection record in database."""
        if not self.repository or not video_id:
            return

        from src.db import JobStatus

        detection = self.repository.create_detection(
            video_id=video_id,
            model_id=model_id or "00000000-0000-0000-0000-000000000000",
            threshold=self.config.threshold,
        )
        self._detection_db_id = detection.id

    def _save_detection_events(self, events: list[Event]) -> None:
        """Save detected events to database."""
        if not self.repository or not self._detection_db_id:
            return

        for event in events:
            self.repository.add_detection_event(
                detection_id=self._detection_db_id,
                start_time=event.start_time,
                end_time=event.end_time,
                confidence=event.confidence,
                start_frame=event.start_frame,
                end_frame=event.end_frame,
            )

    def _complete_detection(
        self, result: DetectionResult, s3_key: str = ""
    ) -> None:
        """Mark detection as completed in database."""
        if not self.repository or not self._detection_db_id:
            return

        from src.db import JobStatus

        self.repository.update_detection_results(
            detection_id=self._detection_db_id,
            num_events=len(result.events),
            s3_results_key=s3_key,
            status=JobStatus.COMPLETED,
        )

    def _upload_results(self, result: DetectionResult, temp_dir: Path) -> str:
        """Upload detection results to S3.

        Args:
            result: Detection results.
            temp_dir: Temporary directory for JSON file.

        Returns:
            S3 key where results were uploaded.
        """
        if not self.storage:
            return ""

        # Save to temp file
        temp_file = temp_dir / "detections.json"
        result.save(temp_file)

        # Upload
        s3_key = self.config.output_s3_key or f"detections/{self.config.job_id}/results.json"
        self.storage.write_detection(result.to_dict(), s3_key)

        return s3_key

    def predict(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        event_callback: Optional[Callable[[Event], None]] = None,
    ) -> dict[str, Any]:
        """Run inference with cloud integration.

        Args:
            progress_callback: Optional callback(current, total, status) for progress.
            event_callback: Optional callback(event) when event is detected.

        Returns:
            Dict with detection results and S3 key.
        """
        temp_dir = Path(tempfile.mkdtemp(prefix="prismata_infer_"))

        try:
            # Load model
            model_path = self._get_model_path()
            self._predictor = self._load_predictor(model_path)

            # Get video
            video_path = self._get_video_path()

            # Database setup
            video_db_id = self._get_video_db_id()
            model_db_id = self._get_model_db_id()
            self._create_detection_record(video_db_id, model_db_id)

            # Run detection
            result = self._predictor.predict_video(
                video_path,
                threshold=self.config.threshold,
                buffer_seconds=self.config.buffer_seconds,
                min_gap_seconds=self.config.min_gap_seconds,
                progress_callback=progress_callback,
                event_callback=event_callback,
            )

            # Save events to database
            self._save_detection_events(result.events)

            # Upload results
            s3_key = ""
            if self.config.use_cloud_storage and self.storage:
                s3_key = self._upload_results(result, temp_dir)

            # Complete detection record
            self._complete_detection(result, s3_key)

            return {
                "success": True,
                "num_events": len(result.events),
                "events": [e.to_dict() for e in result.events],
                "s3_key": s3_key,
                "job_id": self.config.job_id,
                "detection_id": str(self._detection_db_id) if self._detection_db_id else None,
            }

        except Exception as e:
            if self.repository and self._detection_db_id:
                from src.db import JobStatus

                self.repository.update_job_status(
                    self._detection_db_id,
                    status=JobStatus.FAILED,
                    error_message=str(e),
                )
            raise

        finally:
            # Cleanup temp directory
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)


def predict_cloud(
    config: CloudInferenceConfig,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    event_callback: Optional[Callable[[Event], None]] = None,
) -> dict[str, Any]:
    """Run cloud inference.

    Args:
        config: Cloud inference configuration.
        progress_callback: Optional progress callback.
        event_callback: Optional event callback.

    Returns:
        Detection results with S3 key and event details.
    """
    predictor = CloudPredictor(config)
    return predictor.predict(
        progress_callback=progress_callback,
        event_callback=event_callback,
    )


class CloudEventPredictor(EventPredictor):
    """Extended EventPredictor with cloud storage support.

    This class extends EventPredictor to load models from S3
    and optionally save results to S3.
    """

    @classmethod
    def from_s3(
        cls,
        s3_key: str,
        **kwargs,
    ) -> "CloudEventPredictor":
        """Create predictor from S3 model.

        Args:
            s3_key: S3 key for model checkpoint.
            **kwargs: Additional predictor arguments.

        Returns:
            CloudEventPredictor instance.
        """
        from src.storage import get_storage_backend

        storage = get_storage_backend()
        local_path = storage.read_model(s3_key)

        # Use parent class method with local path
        return cls.from_checkpoint(local_path, **kwargs)

    def predict_from_s3(
        self,
        video_s3_key: str,
        output_s3_key: Optional[str] = None,
        **kwargs,
    ) -> DetectionResult:
        """Run detection on S3 video.

        Args:
            video_s3_key: S3 key for video.
            output_s3_key: Optional S3 key for results.
            **kwargs: Additional predict_video arguments.

        Returns:
            DetectionResult.
        """
        from src.storage import get_storage_backend

        storage = get_storage_backend()

        # Download video
        local_video = storage.read_video(video_s3_key)

        # Run detection
        result = self.predict_video(local_video, **kwargs)

        # Upload results
        if output_s3_key:
            storage.write_detection(result.to_dict(), output_s3_key)

        return result
