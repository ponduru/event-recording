"""Local filesystem storage backend."""

import json
import shutil
import time
from pathlib import Path
from typing import Any
import fnmatch

from .base import StorageBackend, StorageConfig


class LocalStorageBackend(StorageBackend):
    """Storage backend for local filesystem.

    This is the default backend that maintains backwards compatibility
    with the existing file-based storage structure.
    """

    def __init__(self, config: StorageConfig):
        super().__init__(config)
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create storage directories if they don't exist."""
        for category in ["videos", "labels", "models", "detections", "cache"]:
            path = self.config.get_local_path(category)
            path.mkdir(parents=True, exist_ok=True)

    def _get_full_path(self, path: str, category: str) -> Path:
        """Get full filesystem path for a relative path."""
        base = self.config.get_local_path(category)
        return base / path

    # Video operations

    def read_video(self, path: str, domain: str | None = None) -> Path:
        """Get video path (already local)."""
        if domain:
            full_path = self._get_full_path(f"{domain}/{path}", "videos")
            if full_path.exists():
                return full_path
        # Try without domain prefix
        full_path = self._get_full_path(path, "videos")
        if full_path.exists():
            return full_path
        # Try as absolute path
        if Path(path).exists():
            return Path(path)
        raise FileNotFoundError(f"Video not found: {path}")

    def write_video(self, local_path: Path, dest: str, domain: str | None = None) -> str:
        """Copy video to storage location."""
        if domain:
            dest_path = self._get_full_path(f"{domain}/{dest}", "videos")
        else:
            dest_path = self._get_full_path(dest, "videos")

        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path != dest_path:
            shutil.copy2(local_path, dest_path)

        return str(dest_path.relative_to(self.config.get_local_path("videos")))

    def get_video_url(self, path: str, domain: str | None = None) -> str:
        """Get file:// URL for local video."""
        video_path = self.read_video(path, domain)
        return f"file://{video_path.absolute()}"

    def list_videos(self, domain: str | None = None, pattern: str = "*.mp4") -> list[str]:
        """List videos in storage."""
        base = self.config.get_local_path("videos")
        if domain:
            base = base / domain

        if not base.exists():
            return []

        # Support multiple patterns like "*.mp4,*.mov"
        patterns = pattern.split(",")
        results = []
        for p in patterns:
            p = p.strip()
            results.extend(base.rglob(p))

        # Return relative paths
        videos_root = self.config.get_local_path("videos")
        return sorted([str(f.relative_to(videos_root)) for f in results])

    # Label operations

    def read_labels(self, path: str) -> dict[str, Any]:
        """Read labels JSON file."""
        full_path = self._get_full_path(path, "labels")
        if not full_path.exists():
            raise FileNotFoundError(f"Labels not found: {path}")
        with open(full_path, "r") as f:
            return json.load(f)

    def write_labels(self, data: dict[str, Any], path: str) -> str:
        """Write labels JSON file."""
        full_path = self._get_full_path(path, "labels")
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def list_labels(self, domain: str | None = None, pattern: str = "*.json") -> list[str]:
        """List label files."""
        base = self.config.get_local_path("labels")
        if domain:
            base = base / domain

        if not base.exists():
            return []

        labels_root = self.config.get_local_path("labels")
        return sorted([str(f.relative_to(labels_root)) for f in base.rglob(pattern)])

    def labels_exist(self, path: str) -> bool:
        """Check if labels file exists."""
        full_path = self._get_full_path(path, "labels")
        return full_path.exists()

    # Model operations

    def read_model(self, path: str) -> Path:
        """Get model file path."""
        full_path = self._get_full_path(path, "models")
        if not full_path.exists():
            # Try as absolute path
            if Path(path).exists():
                return Path(path)
            raise FileNotFoundError(f"Model not found: {path}")
        return full_path

    def write_model(self, local_path: Path, dest: str, metadata: dict | None = None) -> str:
        """Copy model to storage location."""
        dest_path = self._get_full_path(dest, "models")
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path != dest_path:
            shutil.copy2(local_path, dest_path)

        # Optionally save metadata alongside model
        if metadata:
            meta_path = dest_path.with_suffix(".meta.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

        return dest

    def list_models(self, domain: str | None = None, pattern: str = "*.pt") -> list[str]:
        """List model files."""
        base = self.config.get_local_path("models")
        if domain:
            base = base / domain

        if not base.exists():
            return []

        models_root = self.config.get_local_path("models")
        return sorted([str(f.relative_to(models_root)) for f in base.rglob(pattern)])

    # Detection operations

    def read_detection(self, path: str) -> dict[str, Any]:
        """Read detection results."""
        full_path = self._get_full_path(path, "detections")
        if not full_path.exists():
            raise FileNotFoundError(f"Detection not found: {path}")
        with open(full_path, "r") as f:
            return json.load(f)

    def write_detection(self, data: dict[str, Any], path: str) -> str:
        """Write detection results."""
        full_path = self._get_full_path(path, "detections")
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def list_detections(self, video_id: str | None = None, pattern: str = "*.json") -> list[str]:
        """List detection files."""
        base = self.config.get_local_path("detections")
        if video_id:
            base = base / video_id

        if not base.exists():
            return []

        detections_root = self.config.get_local_path("detections")
        return sorted([str(f.relative_to(detections_root)) for f in base.rglob(pattern)])

    # Generic operations

    def read_json(self, path: str, category: str = "labels") -> dict[str, Any]:
        """Read any JSON file."""
        full_path = self._get_full_path(path, category)
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with open(full_path, "r") as f:
            return json.load(f)

    def write_json(self, data: dict[str, Any], path: str, category: str = "labels") -> str:
        """Write any JSON file."""
        full_path = self._get_full_path(path, category)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def exists(self, path: str, category: str) -> bool:
        """Check if file exists."""
        full_path = self._get_full_path(path, category)
        return full_path.exists()

    def delete(self, path: str, category: str) -> bool:
        """Delete a file."""
        full_path = self._get_full_path(path, category)
        if full_path.exists():
            full_path.unlink()
            return True
        return False

    # Cache management (no-op for local storage)

    def get_cache_path(self, key: str) -> Path:
        """Get cache path."""
        cache_dir = self.config.get_local_path("cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / key

    def clear_cache(self, max_age_hours: int | None = None) -> int:
        """Clear cached files."""
        cache_dir = self.config.get_local_path("cache")
        if not cache_dir.exists():
            return 0

        count = 0
        now = time.time()
        max_age_seconds = max_age_hours * 3600 if max_age_hours else None

        for f in cache_dir.rglob("*"):
            if f.is_file():
                if max_age_seconds:
                    age = now - f.stat().st_mtime
                    if age > max_age_seconds:
                        f.unlink()
                        count += 1
                else:
                    f.unlink()
                    count += 1

        return count
