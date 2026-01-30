"""Training module for Prismata event detection."""

from .trainer import (
    TrainingConfig,
    TrainingMetrics,
    Trainer,
    train_model,
    train_model_with_progress,
    compute_metrics,
)

# Cloud training (lazy import to avoid boto3 requirement)
def get_cloud_trainer():
    from .cloud_trainer import CloudTrainer, CloudTrainingConfig, train_model_cloud
    return CloudTrainer, CloudTrainingConfig, train_model_cloud

__all__ = [
    "TrainingConfig",
    "TrainingMetrics",
    "Trainer",
    "train_model",
    "train_model_with_progress",
    "compute_metrics",
    "get_cloud_trainer",
]
