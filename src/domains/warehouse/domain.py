"""Warehouse domain implementation for event detection."""

from typing import Any, Dict, List

from torch.utils.data import Dataset

from src.core.domain import Domain, EventType, register_domain
from src.domains.warehouse.detector import WarehouseDetector
from src.domains.warehouse.dataset import WarehouseDataset, WarehouseInferenceDataset


@register_domain
class WarehouseDomain(Domain):
    """Warehouse domain for detecting operational events."""

    @property
    def name(self) -> str:
        return "warehouse"

    @property
    def event_types(self) -> List[EventType]:
        return [
            EventType.WAREHOUSE_PACKAGE_PICKUP,
            EventType.WAREHOUSE_FORKLIFT_MOVEMENT,
            EventType.WAREHOUSE_SAFETY_VIOLATION,
        ]

    @property
    def description(self) -> str:
        return "Detect warehouse operational events including package pickups, forklift movements, and safety violations"

    def create_model(self, config: Dict[str, Any]) -> WarehouseDetector:
        """Create a warehouse event detection model.

        Args:
            config: Model configuration parameters.

        Returns:
            WarehouseDetector instance.
        """
        return WarehouseDetector(**config)

    def create_dataset(
        self,
        labels_dir: str,
        videos_dir: str,
        **kwargs,
    ) -> Dataset:
        """Create a warehouse training dataset.

        Args:
            labels_dir: Directory containing label JSON files.
            videos_dir: Directory containing video files.
            **kwargs: Additional dataset parameters.

        Returns:
            WarehouseDataset instance.
        """
        return WarehouseDataset(labels_dir, videos_dir, **kwargs)

    def create_inference_dataset(
        self,
        video_path: str,
        **kwargs,
    ) -> Dataset:
        """Create a warehouse inference dataset.

        Args:
            video_path: Path to video file.
            **kwargs: Additional dataset parameters.

        Returns:
            WarehouseInferenceDataset instance.
        """
        return WarehouseInferenceDataset(video_path, **kwargs)

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for warehouse domain.

        Returns:
            Dictionary of default parameters optimized for warehouse surveillance.
        """
        return {
            "backbone": "resnet18",
            "pretrained": True,
            "freeze_backbone": False,
            "feature_dim": 512,
            "hidden_dim": 256,
            "num_lstm_layers": 2,
            "dropout": 0.3,
            "bidirectional": True,
            # Dataset parameters
            "window_size": 16,  # Longer window for warehouse activities
            "frame_size": (224, 224),
            "target_fps": 5.0,  # Lower FPS for surveillance cameras
            "negative_ratio": 2.0,  # More negatives for rare events
        }
