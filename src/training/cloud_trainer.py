"""Cloud-aware training module with storage and database integration."""

import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from .trainer import TrainingConfig, Trainer, TrainingMetrics
from src.core.domain import DomainRegistry
from src.data.dataset import create_train_val_split
from torch.utils.data import DataLoader


@dataclass
class CloudTrainingConfig(TrainingConfig):
    """Extended training config for cloud execution."""

    # Storage
    use_cloud_storage: bool = False
    s3_labels_prefix: str = ""
    s3_videos_prefix: str = ""
    s3_output_key: str = ""

    # Database
    use_database: bool = False
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Cloud-specific settings
    cache_videos_locally: bool = True
    upload_checkpoints: bool = True


class CloudTrainer:
    """Trainer with cloud storage and database integration.

    This wraps the core Trainer class to add:
    - S3 storage for labels, videos, and model outputs
    - Database tracking for jobs and models
    - Progress reporting for async job monitoring
    """

    def __init__(self, config: CloudTrainingConfig):
        """Initialize cloud trainer.

        Args:
            config: Cloud training configuration.
        """
        self.config = config
        self._storage = None
        self._repository = None
        self._job_db_id = None

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
        if self._repository is None and self.config.use_database:
            from src.db import get_repository

            self._repository = get_repository()
        return self._repository

    def _download_data(self, temp_dir: Path) -> tuple[Path, Path]:
        """Download training data from cloud storage.

        Args:
            temp_dir: Temporary directory for downloaded files.

        Returns:
            Tuple of (labels_dir, videos_dir) paths.
        """
        labels_dir = temp_dir / "labels"
        videos_dir = temp_dir / "videos"
        labels_dir.mkdir(parents=True)
        videos_dir.mkdir(parents=True)

        if self.storage and self.config.s3_labels_prefix:
            # Download labels
            label_files = self.storage.list_labels(pattern="*.json")
            for label_file in label_files:
                if label_file.startswith(self.config.s3_labels_prefix):
                    data = self.storage.read_labels(label_file)
                    local_path = labels_dir / Path(label_file).name
                    import json

                    with open(local_path, "w") as f:
                        json.dump(data, f, indent=2)

        if self.storage and self.config.s3_videos_prefix:
            # Download videos (or just get paths for lazy loading)
            video_files = self.storage.list_videos(pattern="*.mp4,*.mov")
            for video_file in video_files:
                if video_file.startswith(self.config.s3_videos_prefix):
                    if self.config.cache_videos_locally:
                        local_path = self.storage.read_video(video_file)
                        # Symlink to temp dir
                        target = videos_dir / Path(video_file).name
                        if not target.exists():
                            import shutil

                            shutil.copy2(local_path, target)

        return labels_dir, videos_dir

    def _upload_model(self, model_path: Path) -> str:
        """Upload trained model to cloud storage.

        Args:
            model_path: Local path to model checkpoint.

        Returns:
            S3 key where model was uploaded.
        """
        if not self.storage:
            return str(model_path)

        s3_key = self.config.s3_output_key or f"models/{self.config.domain}/{model_path.name}"

        metadata = {
            "domain": self.config.domain,
            "experiment_name": self.config.experiment_name,
            "job_id": self.config.job_id,
            "trained_at": datetime.utcnow().isoformat(),
        }

        return self.storage.write_model(model_path, s3_key, metadata=metadata)

    def _create_db_job(self) -> None:
        """Create job record in database."""
        if not self.repository:
            return

        job = self.repository.create_job(
            job_type="training",
            config={
                "domain": self.config.domain,
                "experiment_name": self.config.experiment_name,
                "num_epochs": self.config.num_epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
            },
        )
        self._job_db_id = job.id

    def _update_db_job_progress(self, epoch: int, metrics: dict) -> None:
        """Update job progress in database."""
        if not self.repository or not self._job_db_id:
            return

        from src.db import JobStatus

        progress = int((epoch / self.config.num_epochs) * 100)
        self.repository.update_job_status(
            self._job_db_id,
            status=JobStatus.RUNNING,
            progress=progress,
            status_message=f"Epoch {epoch}/{self.config.num_epochs}",
        )

    def _complete_db_job(self, success: bool, metrics: dict, s3_key: str = "") -> None:
        """Mark job as completed in database."""
        if not self.repository or not self._job_db_id:
            return

        from src.db import JobStatus

        self.repository.update_job_status(
            self._job_db_id,
            status=JobStatus.COMPLETED if success else JobStatus.FAILED,
            progress=100 if success else 0,
            result={
                "metrics": metrics,
                "s3_key": s3_key,
            },
        )

    def _register_model_in_db(self, s3_key: str, metrics: dict) -> None:
        """Register trained model in database."""
        if not self.repository:
            return

        self.repository.create_model(
            name=self.config.experiment_name,
            domain=self.config.domain,
            s3_key=s3_key,
            config=self.config.to_model_config(),
            metrics=metrics,
            val_f1=metrics.get("val_f1"),
            val_accuracy=metrics.get("val_accuracy"),
            training_job_id=self._job_db_id,
        )

    def train(
        self,
        progress_callback: Optional[Callable[[dict], bool]] = None,
    ) -> dict[str, Any]:
        """Run training with cloud integration.

        Args:
            progress_callback: Optional callback for progress updates.
                Receives dict with training metrics, returns True to continue.

        Returns:
            Dict with training results including model S3 key.
        """
        # Create job record
        self._create_db_job()

        use_temp_dir = self.config.use_cloud_storage and self.storage
        temp_dir = None

        try:
            # Download or use local data
            if use_temp_dir:
                temp_dir = Path(tempfile.mkdtemp(prefix="prismata_train_"))
                labels_dir, videos_dir = self._download_data(temp_dir)
            else:
                labels_dir = Path(self.config.labels_dir)
                videos_dir = Path(self.config.videos_dir)

            # Get domain
            import src.domains  # noqa: F401

            domain = DomainRegistry.get(self.config.domain)

            # Create datasets
            train_dataset, val_dataset = create_train_val_split(
                domain=domain,
                labels_dir=str(labels_dir),
                videos_dir=str(videos_dir),
                val_ratio=self.config.val_ratio,
                window_size=self.config.window_size,
                frame_size=self.config.frame_size,
                target_fps=self.config.target_fps,
                negative_ratio=self.config.negative_ratio,
            )

            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )

            # Create output directory
            if use_temp_dir:
                output_dir = temp_dir / "output"
            else:
                output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Update config with resolved paths
            train_config = TrainingConfig(
                domain=self.config.domain,
                labels_dir=str(labels_dir),
                videos_dir=str(videos_dir),
                output_dir=str(output_dir),
                window_size=self.config.window_size,
                frame_size=self.config.frame_size,
                target_fps=self.config.target_fps,
                negative_ratio=self.config.negative_ratio,
                backbone=self.config.backbone,
                pretrained=self.config.pretrained,
                freeze_backbone=self.config.freeze_backbone,
                feature_dim=self.config.feature_dim,
                hidden_dim=self.config.hidden_dim,
                num_lstm_layers=self.config.num_lstm_layers,
                dropout=self.config.dropout,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                num_epochs=self.config.num_epochs,
                early_stopping_patience=self.config.early_stopping_patience,
                val_ratio=self.config.val_ratio,
                experiment_name=self.config.experiment_name,
                use_wandb=self.config.use_wandb,
                wandb_project=self.config.wandb_project,
            )

            # Create and run trainer
            trainer = Trainer(train_config, train_loader, val_loader)

            def wrapped_callback(metrics_dict: dict) -> bool:
                epoch = metrics_dict.get("epoch", 0)
                self._update_db_job_progress(epoch, metrics_dict)

                if progress_callback:
                    return progress_callback(metrics_dict)
                return True

            trainer.train(progress_callback=wrapped_callback)

            # Get best metrics
            best_metrics = {}
            if trainer.history:
                best_epoch = min(trainer.history, key=lambda m: m.val_loss)
                best_metrics = best_epoch.to_dict()

            # Find and upload model
            model_path = output_dir / f"{self.config.experiment_name}_best.pt"
            s3_key = ""

            if model_path.exists():
                if self.config.upload_checkpoints and self.storage:
                    s3_key = self._upload_model(model_path)
                    print(f"Model uploaded to S3: {s3_key}")
                else:
                    s3_key = str(model_path)

                # Register in database
                self._register_model_in_db(s3_key, best_metrics)

            # Mark job complete
            self._complete_db_job(True, best_metrics, s3_key)

            return {
                "success": True,
                "model_s3_key": s3_key,
                "metrics": best_metrics,
                "job_id": self.config.job_id,
            }

        except Exception as e:
            self._complete_db_job(False, {"error": str(e)})
            raise

        finally:
            # Cleanup temp directory
            if temp_dir and temp_dir.exists():
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)


def train_model_cloud(
    config: CloudTrainingConfig,
    progress_callback: Optional[Callable[[dict], bool]] = None,
) -> dict[str, Any]:
    """Train a model with cloud storage and database integration.

    Args:
        config: Cloud training configuration.
        progress_callback: Optional callback for progress updates.

    Returns:
        Training result dict with model S3 key and metrics.
    """
    trainer = CloudTrainer(config)
    return trainer.train(progress_callback=progress_callback)
