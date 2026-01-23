"""Soccer domain module."""

from src.domains.soccer.dataset import SoccerDataset, SoccerInferenceDataset
from src.domains.soccer.detector import SoccerDetector
from src.domains.soccer.domain import SoccerDomain

__all__ = [
    "SoccerDomain",
    "SoccerDetector",
    "SoccerDataset",
    "SoccerInferenceDataset",
]
