"""Base storage abstraction interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import os


@dataclass
class StorageConfig:
    """Configuration for storage backends."""

    backend_type: str = "local"  # "local" or "s3"

    # Local storage paths
    base_dir: Path = field(default_factory=lambda: Path.cwd())
    videos_dir: str = "data/raw"
    labels_dir: str = "data/labels"
    models_dir: str = "models"
    detections_dir: str = "data/detections"
    cache_dir: str = ".cache/prismata"

    # S3 configuration
    # Single bucket mode: set s3_bucket to use one bucket with prefixed keys
    # Multi bucket mode: set individual s3_bucket_* for separate buckets
    s3_bucket: str = ""  # Single bucket (keys prefixed by category)
    s3_bucket_videos: str = ""
    s3_bucket_labels: str = ""
    s3_bucket_models: str = ""
    s3_bucket_detections: str = ""
    s3_region: str = "us-east-1"

    # AWS credentials (optional - uses default chain if not set)
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None

    # Custom endpoint URL (for LocalStack / MinIO local testing)
    s3_endpoint_url: str | None = None

    # Presigned URL expiry (seconds)
    presigned_url_expiry: int = 3600

    @classmethod
    def from_env(cls) -> "StorageConfig":
        """Create configuration from environment variables."""
        backend_type = os.getenv("PRISMATA_STORAGE_BACKEND", "local")

        return cls(
            backend_type=backend_type,
            base_dir=Path(os.getenv("PRISMATA_BASE_DIR", str(Path.cwd()))),
            videos_dir=os.getenv("PRISMATA_VIDEOS_DIR", "data/raw"),
            labels_dir=os.getenv("PRISMATA_LABELS_DIR", "data/labels"),
            models_dir=os.getenv("PRISMATA_MODELS_DIR", "models"),
            detections_dir=os.getenv("PRISMATA_DETECTIONS_DIR", "data/detections"),
            cache_dir=os.getenv("PRISMATA_CACHE_DIR", ".cache/prismata"),
            s3_bucket=os.getenv("PRISMATA_S3_BUCKET", ""),
            s3_bucket_videos=os.getenv("PRISMATA_S3_BUCKET_VIDEOS", ""),
            s3_bucket_labels=os.getenv("PRISMATA_S3_BUCKET_LABELS", ""),
            s3_bucket_models=os.getenv("PRISMATA_S3_BUCKET_MODELS", ""),
            s3_bucket_detections=os.getenv("PRISMATA_S3_BUCKET_DETECTIONS", ""),
            s3_region=os.getenv("AWS_REGION", "us-east-1"),
            s3_endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
            presigned_url_expiry=int(os.getenv("PRISMATA_PRESIGNED_URL_EXPIRY", "3600")),
        )

    def get_local_path(self, category: str) -> Path:
        """Get local path for a category (videos, labels, models, detections)."""
        paths = {
            "videos": self.videos_dir,
            "labels": self.labels_dir,
            "models": self.models_dir,
            "detections": self.detections_dir,
            "cache": self.cache_dir,
        }
        return self.base_dir / paths.get(category, category)

    @property
    def is_single_bucket(self) -> bool:
        """True if using a single shared bucket with prefix-based separation."""
        if self.s3_bucket:
            return True
        # Also detect when all per-category buckets are the same non-empty value
        per_category = [
            self.s3_bucket_videos,
            self.s3_bucket_labels,
            self.s3_bucket_models,
            self.s3_bucket_detections,
        ]
        non_empty = [b for b in per_category if b]
        return len(non_empty) > 1 and len(set(non_empty)) == 1

    def get_s3_bucket(self, category: str) -> str:
        """Get S3 bucket for a category.

        In single-bucket mode, always returns the shared bucket.
        In multi-bucket mode, returns the category-specific bucket.
        """
        if self.s3_bucket:
            return self.s3_bucket

        buckets = {
            "videos": self.s3_bucket_videos,
            "labels": self.s3_bucket_labels,
            "models": self.s3_bucket_models,
            "detections": self.s3_bucket_detections,
        }
        return buckets.get(category, "")

    def get_s3_prefix(self, category: str) -> str:
        """Get the S3 key prefix for a category.

        In single-bucket mode, returns 'videos/', 'labels/', etc.
        In multi-bucket mode, returns '' (no prefix needed).
        """
        if self.is_single_bucket:
            return f"{category}/"
        return ""


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    def __init__(self, config: StorageConfig):
        self.config = config

    # Video operations

    @abstractmethod
    def read_video(self, path: str, domain: str | None = None) -> Path:
        """Download/get video to local cache and return local path.

        For local storage, returns the path directly.
        For S3, downloads to cache and returns cached path.

        Args:
            path: Video path/key (relative to storage root)
            domain: Optional domain for organization

        Returns:
            Local filesystem path to the video file.
        """
        pass

    @abstractmethod
    def write_video(self, local_path: Path, dest: str, domain: str | None = None) -> str:
        """Upload/copy video to storage.

        Args:
            local_path: Local path to video file
            dest: Destination path/key
            domain: Optional domain for organization

        Returns:
            Storage key/path where video was saved.
        """
        pass

    @abstractmethod
    def get_video_url(self, path: str, domain: str | None = None) -> str:
        """Get URL for video playback.

        For local storage, returns file:// URL.
        For S3, returns presigned URL.

        Args:
            path: Video path/key
            domain: Optional domain for organization

        Returns:
            URL for video playback.
        """
        pass

    @abstractmethod
    def list_videos(self, domain: str | None = None, pattern: str = "*.mp4") -> list[str]:
        """List available videos.

        Args:
            domain: Optional domain filter
            pattern: Glob pattern for matching

        Returns:
            List of video paths/keys.
        """
        pass

    # Label operations

    @abstractmethod
    def read_labels(self, path: str) -> dict[str, Any]:
        """Read labels JSON file.

        Args:
            path: Label file path/key

        Returns:
            Parsed JSON data as dictionary.
        """
        pass

    @abstractmethod
    def write_labels(self, data: dict[str, Any], path: str) -> str:
        """Write labels JSON file.

        Args:
            data: Label data to write
            path: Destination path/key

        Returns:
            Storage key/path where labels were saved.
        """
        pass

    @abstractmethod
    def list_labels(self, domain: str | None = None, pattern: str = "*.json") -> list[str]:
        """List available label files.

        Args:
            domain: Optional domain filter
            pattern: Glob pattern for matching

        Returns:
            List of label file paths/keys.
        """
        pass

    @abstractmethod
    def labels_exist(self, path: str) -> bool:
        """Check if labels file exists.

        Args:
            path: Label file path/key

        Returns:
            True if file exists.
        """
        pass

    # Model operations

    @abstractmethod
    def read_model(self, path: str) -> Path:
        """Get model checkpoint file.

        Args:
            path: Model path/key

        Returns:
            Local filesystem path to model file.
        """
        pass

    @abstractmethod
    def write_model(self, local_path: Path, dest: str, metadata: dict | None = None) -> str:
        """Upload/save model checkpoint.

        Args:
            local_path: Local path to model file
            dest: Destination path/key
            metadata: Optional metadata to store with model

        Returns:
            Storage key/path where model was saved.
        """
        pass

    @abstractmethod
    def list_models(self, domain: str | None = None, pattern: str = "*.pt") -> list[str]:
        """List available models.

        Args:
            domain: Optional domain filter
            pattern: Glob pattern for matching

        Returns:
            List of model paths/keys.
        """
        pass

    # Detection operations

    @abstractmethod
    def read_detection(self, path: str) -> dict[str, Any]:
        """Read detection results JSON.

        Args:
            path: Detection file path/key

        Returns:
            Parsed JSON data.
        """
        pass

    @abstractmethod
    def write_detection(self, data: dict[str, Any], path: str) -> str:
        """Write detection results JSON.

        Args:
            data: Detection data to write
            path: Destination path/key

        Returns:
            Storage key/path where detection was saved.
        """
        pass

    @abstractmethod
    def list_detections(self, video_id: str | None = None, pattern: str = "*.json") -> list[str]:
        """List detection results.

        Args:
            video_id: Optional video ID filter
            pattern: Glob pattern for matching

        Returns:
            List of detection file paths/keys.
        """
        pass

    # Generic operations

    @abstractmethod
    def read_json(self, path: str, category: str = "labels") -> dict[str, Any]:
        """Read any JSON file.

        Args:
            path: File path/key
            category: Storage category (labels, detections, etc.)

        Returns:
            Parsed JSON data.
        """
        pass

    @abstractmethod
    def write_json(self, data: dict[str, Any], path: str, category: str = "labels") -> str:
        """Write any JSON file.

        Args:
            data: Data to write
            path: Destination path/key
            category: Storage category

        Returns:
            Storage key/path.
        """
        pass

    @abstractmethod
    def exists(self, path: str, category: str) -> bool:
        """Check if file exists.

        Args:
            path: File path/key
            category: Storage category

        Returns:
            True if file exists.
        """
        pass

    @abstractmethod
    def delete(self, path: str, category: str) -> bool:
        """Delete a file.

        Args:
            path: File path/key
            category: Storage category

        Returns:
            True if deleted successfully.
        """
        pass

    # Cache management

    @abstractmethod
    def get_cache_path(self, key: str) -> Path:
        """Get local cache path for a key.

        Args:
            key: Cache key

        Returns:
            Local filesystem path for cached item.
        """
        pass

    @abstractmethod
    def clear_cache(self, max_age_hours: int | None = None) -> int:
        """Clear cached files.

        Args:
            max_age_hours: Only clear files older than this (None = all)

        Returns:
            Number of files cleared.
        """
        pass
