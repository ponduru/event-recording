"""Base detector classes for event detection.

This module provides domain-agnostic base classes for event detection models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class BaseDetectorConfig:
    """Base configuration for event detectors."""

    backbone: str = "resnet18"
    pretrained: bool = True
    freeze_backbone: bool = False
    feature_dim: int = 512
    hidden_dim: int = 256
    num_lstm_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "backbone": self.backbone,
            "pretrained": self.pretrained,
            "freeze_backbone": self.freeze_backbone,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "num_lstm_layers": self.num_lstm_layers,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BaseDetectorConfig":
        """Create config from dictionary."""
        return cls(**data)


class BaseEventDetector(nn.Module, ABC):
    """Abstract base class for event detection models.

    This class defines the interface that all event detectors should implement,
    regardless of the specific domain (cricket, soccer, warehouse, etc.).
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (B, T, C, H, W) where:
                B = batch size
                T = temporal sequence length (number of frames)
                C = channels
                H, W = height, width

        Returns:
            Logits of shape (B, num_classes) or (B, 1) for binary classification.
        """
        pass

    @abstractmethod
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions.

        Args:
            x: Input tensor of shape (B, T, C, H, W).

        Returns:
            Probabilities of shape (B, num_classes) or (B, 1).
        """
        pass


def load_checkpoint(
    checkpoint_path: str | Path,
    model_class: type[nn.Module],
    config_class: type[BaseDetectorConfig],
    device: Optional[torch.device] = None,
) -> tuple[nn.Module, BaseDetectorConfig]:
    """Load a model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        model_class: Model class to instantiate.
        config_class: Config class to use.
        device: Device to load model to.

    Returns:
        (model, config) tuple.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle both old and new checkpoint formats
    if "config" in checkpoint:
        config = config_class.from_dict(checkpoint["config"])
    else:
        # Fallback for old checkpoints
        config = config_class()

    # Create model
    model = model_class(**config.to_dict())
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, config


def save_checkpoint(
    model: nn.Module,
    config: BaseDetectorConfig,
    checkpoint_path: str | Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[dict] = None,
    domain: Optional[str] = None,
) -> None:
    """Save model checkpoint.

    Args:
        model: Model to save.
        config: Model configuration.
        checkpoint_path: Output path.
        optimizer: Optional optimizer state.
        epoch: Optional epoch number.
        metrics: Optional metrics dict.
        domain: Optional domain name for metadata.
    """
    checkpoint = {
        "config": config.to_dict(),
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if metrics is not None:
        checkpoint["metrics"] = metrics
    if domain is not None:
        checkpoint["domain"] = domain

    # Ensure parent directory exists
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
