"""Storage abstraction layer for Prismata.

Provides a unified interface for accessing videos, labels, models, and detections
across different storage backends (local filesystem, S3, etc.).
"""

from .base import StorageBackend, StorageConfig
from .local import LocalStorageBackend
from .s3 import S3StorageBackend

__all__ = [
    "StorageBackend",
    "StorageConfig",
    "LocalStorageBackend",
    "S3StorageBackend",
    "get_storage_backend",
]


def get_storage_backend(config: StorageConfig | None = None) -> StorageBackend:
    """Factory function to get the appropriate storage backend.

    Args:
        config: Storage configuration. If None, uses environment variables
                or defaults to local storage.

    Returns:
        Configured storage backend instance.
    """
    import os

    if config is None:
        config = StorageConfig.from_env()

    if config.backend_type == "s3":
        return S3StorageBackend(config)
    else:
        return LocalStorageBackend(config)
