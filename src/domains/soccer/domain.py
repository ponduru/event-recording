"""Soccer domain implementation for event detection."""

from typing import Any, Dict, List

from torch.utils.data import Dataset

from src.core.domain import Domain, EventType, register_domain
from src.domains.soccer.detector import SoccerDetector
from src.domains.soccer.dataset import SoccerDataset, SoccerInferenceDataset


@register_domain
class SoccerDomain(Domain):
    """Soccer domain for detecting match events (goals, penalties, etc.)."""

    @property
    def name(self) -> str:
        return "soccer"

    @property
    def event_types(self) -> List[EventType]:
        return [
            EventType.SOCCER_GOAL,
            EventType.SOCCER_PENALTY,
            EventType.SOCCER_CORNER_KICK,
            EventType.SOCCER_FREE_KICK,
        ]

    @property
    def description(self) -> str:
        return "Detect soccer match events including goals, penalties, corner kicks, and free kicks"

    def create_model(self, config: Dict[str, Any]) -> SoccerDetector:
        """Create a soccer event detection model.

        Args:
            config: Model configuration parameters.

        Returns:
            SoccerDetector instance.
        """
        return SoccerDetector(**config)

    def create_dataset(
        self,
        labels_dir: str,
        videos_dir: str,
        **kwargs,
    ) -> Dataset:
        """Create a soccer training dataset.

        Args:
            labels_dir: Directory containing label JSON files.
            videos_dir: Directory containing video files.
            **kwargs: Additional dataset parameters.

        Returns:
            SoccerDataset instance.
        """
        return SoccerDataset(labels_dir, videos_dir, **kwargs)

    def create_inference_dataset(
        self,
        video_path: str,
        **kwargs,
    ) -> Dataset:
        """Create a soccer inference dataset.

        Args:
            video_path: Path to video file.
            **kwargs: Additional dataset parameters.

        Returns:
            SoccerInferenceDataset instance.
        """
        return SoccerInferenceDataset(video_path, **kwargs)

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for soccer domain.

        Returns:
            Dictionary of default parameters optimized for soccer.
        """
        return {
            "backbone": "resnet34",  # Slightly larger backbone for more complex events
            "pretrained": True,
            "freeze_backbone": False,
            "feature_dim": 512,
            "hidden_dim": 256,
            "num_lstm_layers": 2,
            "dropout": 0.3,
            "bidirectional": True,
            # Dataset parameters
            "window_size": 12,  # Longer window for soccer events
            "frame_size": (224, 224),
            "target_fps": 10.0,
            "negative_ratio": 1.5,  # More negatives for harder task
        }
