"""Soccer-specific detector model."""

import torch
import torch.nn as nn

from src.core.base_detector import BaseEventDetector
from src.models.detector import FrameEncoder, TemporalModel


class SoccerDetector(BaseEventDetector):
    """Soccer event detection model.

    Uses CNN+LSTM architecture similar to cricket detector but optimized
    for soccer match events.
    """

    def __init__(
        self,
        backbone: str = "resnet34",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        """Initialize the soccer detector.

        Args:
            backbone: CNN backbone for frame encoding.
            pretrained: Use pretrained weights.
            freeze_backbone: Freeze CNN weights.
            feature_dim: Frame feature dimension.
            hidden_dim: LSTM hidden dimension.
            num_lstm_layers: Number of LSTM layers.
            dropout: Dropout probability.
            bidirectional: Use bidirectional LSTM.
        """
        super().__init__()

        self.frame_encoder = FrameEncoder(
            backbone=backbone,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            feature_dim=feature_dim,
        )

        self.temporal_model = TemporalModel(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # Classification head
        classifier_input_dim = self.temporal_model.output_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Video tensor of shape (B, T, C, H, W)

        Returns:
            Logits of shape (B, 1)
        """
        frame_features = self.frame_encoder(x)
        temporal_features = self.temporal_model(frame_features)
        logits = self.classifier(temporal_features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability prediction.

        Args:
            x: Video tensor of shape (B, T, C, H, W)

        Returns:
            Probabilities of shape (B, 1)
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)
