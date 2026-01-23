"""Cricket domain implementation for event detection."""

from typing import Any, Dict, List

from torch.utils.data import Dataset

from src.core.domain import Domain, EventType, register_domain
from src.domains.cricket.detector import CricketDetector
from src.domains.cricket.dataset import CricketDataset, CricketInferenceDataset


@register_domain
class CricketDomain(Domain):
    """Cricket domain for detecting bowling deliveries."""

    @property
    def name(self) -> str:
        return "cricket"

    @property
    def event_types(self) -> List[EventType]:
        return [EventType.CRICKET_DELIVERY]

    @property
    def description(self) -> str:
        return "Detect cricket bowling deliveries in match videos"

    def create_model(self, config: Dict[str, Any]) -> CricketDetector:
        """Create a cricket delivery detection model.

        Args:
            config: Model configuration parameters.

        Returns:
            CricketDetector instance.
        """
        return CricketDetector(**config)

    def create_dataset(
        self,
        labels_dir: str,
        videos_dir: str,
        **kwargs,
    ) -> Dataset:
        """Create a cricket training dataset.

        Args:
            labels_dir: Directory containing label JSON files.
            videos_dir: Directory containing video files.
            **kwargs: Additional dataset parameters.

        Returns:
            CricketDataset instance.
        """
        return CricketDataset(labels_dir, videos_dir, **kwargs)

    def create_inference_dataset(
        self,
        video_path: str,
        **kwargs,
    ) -> Dataset:
        """Create a cricket inference dataset.

        Args:
            video_path: Path to video file.
            **kwargs: Additional dataset parameters.

        Returns:
            CricketInferenceDataset instance.
        """
        return CricketInferenceDataset(video_path, **kwargs)

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for cricket domain.

        Returns:
            Dictionary of default parameters optimized for cricket.
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
            "window_size": 8,
            "frame_size": (224, 224),
            "target_fps": 10.0,
            "negative_ratio": 1.0,
        }
