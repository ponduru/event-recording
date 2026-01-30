"""AWS S3 storage backend."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any
import fnmatch

from .base import StorageBackend, StorageConfig


class S3StorageBackend(StorageBackend):
    """Storage backend for AWS S3.

    Provides cloud storage with local caching for videos and models.
    Uses presigned URLs for secure video streaming.
    """

    def __init__(self, config: StorageConfig):
        super().__init__(config)
        self._s3_client = None
        self._ensure_cache_dir()

    def _ensure_cache_dir(self) -> None:
        """Create local cache directory."""
        cache_dir = self.config.get_local_path("cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def s3_client(self):
        """Lazy initialization of S3 client."""
        if self._s3_client is None:
            try:
                import boto3
                from botocore.config import Config

                boto_config = Config(
                    region_name=self.config.s3_region,
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "standard"},
                )

                session_kwargs = {}
                if self.config.aws_access_key_id:
                    session_kwargs["aws_access_key_id"] = self.config.aws_access_key_id
                if self.config.aws_secret_access_key:
                    session_kwargs["aws_secret_access_key"] = self.config.aws_secret_access_key
                if self.config.aws_session_token:
                    session_kwargs["aws_session_token"] = self.config.aws_session_token

                client_kwargs = {"config": boto_config}
                if self.config.s3_endpoint_url:
                    client_kwargs["endpoint_url"] = self.config.s3_endpoint_url

                if session_kwargs:
                    session = boto3.Session(**session_kwargs)
                    self._s3_client = session.client("s3", **client_kwargs)
                else:
                    self._s3_client = boto3.client("s3", **client_kwargs)
            except ImportError:
                raise ImportError(
                    "boto3 is required for S3 storage. Install with: pip install boto3"
                )

        return self._s3_client

    def _get_cache_key(self, bucket: str, key: str) -> str:
        """Generate cache filename from bucket and key."""
        hash_input = f"{bucket}/{key}"
        hash_val = hashlib.md5(hash_input.encode()).hexdigest()[:12]
        # Preserve file extension
        ext = Path(key).suffix
        return f"{hash_val}{ext}"

    def _get_s3_key(self, path: str, category: str, domain: str | None = None) -> str:
        """Build S3 key from path, category, and optional domain.

        In single-bucket mode, prepends the category prefix (e.g. 'videos/').
        In multi-bucket mode, no prefix is added.
        """
        prefix = self.config.get_s3_prefix(category)
        if domain:
            return f"{prefix}{domain}/{path}"
        return f"{prefix}{path}"

    def _strip_category_prefix(self, key: str, category: str) -> str:
        """Strip the category prefix from a key for external-facing paths.

        Returns keys without the internal 'videos/', 'labels/' etc. prefix
        so callers don't need to know about single-bucket internals.
        """
        prefix = self.config.get_s3_prefix(category)
        if prefix and key.startswith(prefix):
            return key[len(prefix):]
        return key

    def _download_to_cache(self, bucket: str, key: str) -> Path:
        """Download S3 object to local cache."""
        cache_key = self._get_cache_key(bucket, key)
        cache_path = self.get_cache_path(cache_key)

        # Check if already cached
        if cache_path.exists():
            # Could add TTL check here
            return cache_path

        # Download from S3
        self.s3_client.download_file(bucket, key, str(cache_path))
        return cache_path

    def _upload_from_local(self, local_path: Path, bucket: str, key: str) -> None:
        """Upload local file to S3."""
        self.s3_client.upload_file(str(local_path), bucket, key)

    def _list_s3_objects(self, bucket: str, prefix: str = "", pattern: str = "*") -> list[str]:
        """List S3 objects matching pattern."""
        results = []
        paginator = self.s3_client.get_paginator("list_objects_v2")

        try:
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    filename = key.split("/")[-1]
                    if fnmatch.fnmatch(filename, pattern):
                        results.append(key)
        except self.s3_client.exceptions.NoSuchBucket:
            return []

        return results

    def _s3_object_exists(self, bucket: str, key: str) -> bool:
        """Check if S3 object exists."""
        try:
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except self.s3_client.exceptions.ClientError:
            return False

    # Video operations

    def read_video(self, path: str, domain: str | None = None) -> Path:
        """Download video from S3 to cache."""
        bucket = self.config.get_s3_bucket("videos")
        if not bucket:
            raise ValueError("S3 videos bucket not configured")

        key = self._get_s3_key(path, "videos", domain)
        return self._download_to_cache(bucket, key)

    def write_video(self, local_path: Path, dest: str, domain: str | None = None) -> str:
        """Upload video to S3."""
        bucket = self.config.get_s3_bucket("videos")
        if not bucket:
            raise ValueError("S3 videos bucket not configured")

        key = self._get_s3_key(dest, "videos", domain)
        self._upload_from_local(local_path, bucket, key)
        return self._strip_category_prefix(key, "videos")

    def get_video_url(self, path: str, domain: str | None = None) -> str:
        """Get presigned URL for video streaming."""
        bucket = self.config.get_s3_bucket("videos")
        if not bucket:
            raise ValueError("S3 videos bucket not configured")

        key = self._get_s3_key(path, "videos", domain)
        return self.s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=self.config.presigned_url_expiry,
        )

    def list_videos(self, domain: str | None = None, pattern: str = "*.mp4") -> list[str]:
        """List videos in S3 bucket."""
        bucket = self.config.get_s3_bucket("videos")
        if not bucket:
            return []

        cat_prefix = self.config.get_s3_prefix("videos")
        prefix = f"{cat_prefix}{domain}/" if domain else cat_prefix
        patterns = pattern.split(",")
        results = []
        for p in patterns:
            raw = self._list_s3_objects(bucket, prefix, p.strip())
            results.extend(self._strip_category_prefix(k, "videos") for k in raw)
        return sorted(set(results))

    # Label operations

    def read_labels(self, path: str) -> dict[str, Any]:
        """Read labels JSON from S3."""
        bucket = self.config.get_s3_bucket("labels")
        if not bucket:
            raise ValueError("S3 labels bucket not configured")

        key = self._get_s3_key(path, "labels")
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        return json.loads(response["Body"].read().decode("utf-8"))

    def write_labels(self, data: dict[str, Any], path: str) -> str:
        """Write labels JSON to S3."""
        bucket = self.config.get_s3_bucket("labels")
        if not bucket:
            raise ValueError("S3 labels bucket not configured")

        key = self._get_s3_key(path, "labels")
        body = json.dumps(data, indent=2, default=str)
        self.s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body.encode("utf-8"),
            ContentType="application/json",
        )
        return self._strip_category_prefix(key, "labels")

    def list_labels(self, domain: str | None = None, pattern: str = "*.json") -> list[str]:
        """List label files in S3."""
        bucket = self.config.get_s3_bucket("labels")
        if not bucket:
            return []

        cat_prefix = self.config.get_s3_prefix("labels")
        prefix = f"{cat_prefix}{domain}/" if domain else cat_prefix
        raw = self._list_s3_objects(bucket, prefix, pattern)
        return sorted(self._strip_category_prefix(k, "labels") for k in raw)

    def labels_exist(self, path: str) -> bool:
        """Check if labels file exists in S3."""
        bucket = self.config.get_s3_bucket("labels")
        if not bucket:
            return False
        key = self._get_s3_key(path, "labels")
        return self._s3_object_exists(bucket, key)

    # Model operations

    def read_model(self, path: str) -> Path:
        """Download model from S3 to cache."""
        bucket = self.config.get_s3_bucket("models")
        if not bucket:
            raise ValueError("S3 models bucket not configured")

        key = self._get_s3_key(path, "models")
        return self._download_to_cache(bucket, key)

    def write_model(self, local_path: Path, dest: str, metadata: dict | None = None) -> str:
        """Upload model to S3 with optional metadata."""
        bucket = self.config.get_s3_bucket("models")
        if not bucket:
            raise ValueError("S3 models bucket not configured")

        key = self._get_s3_key(dest, "models")

        # Upload model file
        extra_args = {}
        if metadata:
            # Store metadata as S3 object metadata (limited to 2KB)
            s3_metadata = {k: str(v)[:256] for k, v in metadata.items()}
            extra_args["Metadata"] = s3_metadata

        self.s3_client.upload_file(
            str(local_path), bucket, key, ExtraArgs=extra_args if extra_args else None
        )

        # Also upload full metadata as separate JSON if provided
        if metadata:
            meta_key = key.rsplit(".", 1)[0] + ".meta.json"
            meta_body = json.dumps(metadata, indent=2, default=str)
            self.s3_client.put_object(
                Bucket=bucket,
                Key=meta_key,
                Body=meta_body.encode("utf-8"),
                ContentType="application/json",
            )

        return self._strip_category_prefix(key, "models")

    def list_models(self, domain: str | None = None, pattern: str = "*.pt") -> list[str]:
        """List models in S3."""
        bucket = self.config.get_s3_bucket("models")
        if not bucket:
            return []

        cat_prefix = self.config.get_s3_prefix("models")
        prefix = f"{cat_prefix}{domain}/" if domain else cat_prefix
        raw = self._list_s3_objects(bucket, prefix, pattern)
        return sorted(self._strip_category_prefix(k, "models") for k in raw)

    # Detection operations

    def read_detection(self, path: str) -> dict[str, Any]:
        """Read detection results from S3."""
        bucket = self.config.get_s3_bucket("detections")
        if not bucket:
            raise ValueError("S3 detections bucket not configured")

        key = self._get_s3_key(path, "detections")
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        return json.loads(response["Body"].read().decode("utf-8"))

    def write_detection(self, data: dict[str, Any], path: str) -> str:
        """Write detection results to S3."""
        bucket = self.config.get_s3_bucket("detections")
        if not bucket:
            raise ValueError("S3 detections bucket not configured")

        key = self._get_s3_key(path, "detections")
        body = json.dumps(data, indent=2, default=str)
        self.s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body.encode("utf-8"),
            ContentType="application/json",
        )
        return self._strip_category_prefix(key, "detections")

    def list_detections(self, video_id: str | None = None, pattern: str = "*.json") -> list[str]:
        """List detection results in S3."""
        bucket = self.config.get_s3_bucket("detections")
        if not bucket:
            return []

        cat_prefix = self.config.get_s3_prefix("detections")
        prefix = f"{cat_prefix}{video_id}/" if video_id else cat_prefix
        raw = self._list_s3_objects(bucket, prefix, pattern)
        return sorted(self._strip_category_prefix(k, "detections") for k in raw)

    # Generic operations

    def read_json(self, path: str, category: str = "labels") -> dict[str, Any]:
        """Read JSON file from S3."""
        bucket = self.config.get_s3_bucket(category)
        if not bucket:
            raise ValueError(f"S3 bucket not configured for category: {category}")

        key = self._get_s3_key(path, category)
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        return json.loads(response["Body"].read().decode("utf-8"))

    def write_json(self, data: dict[str, Any], path: str, category: str = "labels") -> str:
        """Write JSON file to S3."""
        bucket = self.config.get_s3_bucket(category)
        if not bucket:
            raise ValueError(f"S3 bucket not configured for category: {category}")

        key = self._get_s3_key(path, category)
        body = json.dumps(data, indent=2, default=str)
        self.s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body.encode("utf-8"),
            ContentType="application/json",
        )
        return self._strip_category_prefix(key, category)

    def exists(self, path: str, category: str) -> bool:
        """Check if file exists in S3."""
        bucket = self.config.get_s3_bucket(category)
        if not bucket:
            return False
        key = self._get_s3_key(path, category)
        return self._s3_object_exists(bucket, key)

    def delete(self, path: str, category: str) -> bool:
        """Delete file from S3."""
        bucket = self.config.get_s3_bucket(category)
        if not bucket:
            return False

        try:
            key = self._get_s3_key(path, category)
            self.s3_client.delete_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False

    # Cache management

    def get_cache_path(self, key: str) -> Path:
        """Get local cache path."""
        cache_dir = self.config.get_local_path("cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / key

    def clear_cache(self, max_age_hours: int | None = None) -> int:
        """Clear local cache."""
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
