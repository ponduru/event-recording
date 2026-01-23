"""CNN + LSTM model for cricket delivery detection."""

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models


class FrameEncoder(nn.Module):
    """CNN encoder for individual frames.

    Uses a pretrained ResNet backbone to extract features from each frame.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        feature_dim: int = 512,
    ):
        """Initialize the frame encoder.

        Args:
            backbone: ResNet variant to use ('resnet18', 'resnet34', 'resnet50').
            pretrained: Whether to use pretrained ImageNet weights.
            freeze_backbone: Whether to freeze backbone weights.
            feature_dim: Output feature dimension.
        """
        super().__init__()

        # Load backbone
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            backbone_dim = 512
        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            backbone_dim = 512
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove classification head
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Freeze if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Project to feature_dim
        self.projection = nn.Linear(backbone_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of frames.

        Args:
            x: Tensor of shape (B, C, H, W) or (B, T, C, H, W)

        Returns:
            Features of shape (B, feature_dim) or (B, T, feature_dim)
        """
        if x.dim() == 5:
            # Video input: (B, T, C, H, W)
            batch_size, seq_len = x.shape[:2]
            x = x.view(-1, *x.shape[2:])  # (B*T, C, H, W)
            features = self.backbone(x)  # (B*T, backbone_dim, 1, 1)
            features = features.flatten(1)  # (B*T, backbone_dim)
            features = self.projection(features)  # (B*T, feature_dim)
            features = features.view(batch_size, seq_len, -1)  # (B, T, feature_dim)
        else:
            # Single frame: (B, C, H, W)
            features = self.backbone(x)  # (B, backbone_dim, 1, 1)
            features = features.flatten(1)  # (B, backbone_dim)
            features = self.projection(features)  # (B, feature_dim)

        return features


class TemporalModel(nn.Module):
    """LSTM-based temporal modeling."""

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        """Initialize the temporal model.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: LSTM hidden dimension.
            num_layers: Number of LSTM layers.
            dropout: Dropout probability.
            bidirectional: Whether to use bidirectional LSTM.
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process temporal sequence.

        Args:
            x: Tensor of shape (B, T, input_dim)

        Returns:
            Output of shape (B, output_dim)
        """
        output, (hidden, cell) = self.lstm(x)

        # Use last hidden state
        if self.lstm.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]

        return hidden


class DeliveryDetector(nn.Module):
    """Full model for cricket delivery detection.

    Combines CNN frame encoder with LSTM temporal modeling
    for binary classification of video windows.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        """Initialize the delivery detector.

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
        # Encode frames
        frame_features = self.frame_encoder(x)  # (B, T, feature_dim)

        # Temporal modeling
        temporal_features = self.temporal_model(frame_features)  # (B, output_dim)

        # Classification
        logits = self.classifier(temporal_features)  # (B, 1)

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


class DeliveryDetectorConfig:
    """Configuration for DeliveryDetector."""

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        self.backbone = backbone
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

    def to_dict(self) -> dict:
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
    def from_dict(cls, data: dict) -> "DeliveryDetectorConfig":
        return cls(**data)

    def create_model(self) -> DeliveryDetector:
        return DeliveryDetector(**self.to_dict())


def load_model(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> tuple[DeliveryDetector, DeliveryDetectorConfig]:
    """Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to load model to.

    Returns:
        (model, config) tuple
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = DeliveryDetectorConfig.from_dict(checkpoint["config"])
    model = config.create_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, config


def save_model(
    model: DeliveryDetector,
    config: DeliveryDetectorConfig,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[dict] = None,
) -> None:
    """Save model checkpoint.

    Args:
        model: Model to save.
        config: Model configuration.
        checkpoint_path: Output path.
        optimizer: Optional optimizer state.
        epoch: Optional epoch number.
        metrics: Optional metrics dict.
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

    torch.save(checkpoint, checkpoint_path)
