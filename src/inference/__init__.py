"""Inference module for Prismata event detection."""

from .predictor import (
    Event,
    DetectionResult,
    EventPredictor,
    extract_event_clips,
    detect_and_extract,
)

# Cloud inference (lazy import to avoid boto3 requirement)
def get_cloud_predictor():
    from .cloud_predictor import (
        CloudPredictor,
        CloudInferenceConfig,
        CloudEventPredictor,
        predict_cloud,
    )
    return CloudPredictor, CloudInferenceConfig, CloudEventPredictor, predict_cloud

__all__ = [
    "Event",
    "DetectionResult",
    "EventPredictor",
    "extract_event_clips",
    "detect_and_extract",
    "get_cloud_predictor",
]
