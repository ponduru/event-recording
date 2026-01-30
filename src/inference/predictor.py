"""Inference pipeline for event detection across multiple domains."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.core.base_detector import load_checkpoint, BaseDetectorConfig
from src.core.domain import DomainRegistry
from src.utils.video import extract_clip, get_video_metadata

# Import domains to trigger registration
import src.domains  # noqa: F401


@dataclass
class Event:
    """A detected event."""

    id: int
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    confidence: float
    duration: float

    def to_dict(self) -> dict:
        """Convert to dict with JSON-serializable types."""
        return {
            "id": int(self.id),
            "start_time": float(self.start_time),
            "end_time": float(self.end_time),
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "confidence": float(self.confidence),
            "duration": float(self.duration),
        }


@dataclass
class DetectionResult:
    """Result of detection on a video."""

    video_path: str
    fps: float
    total_frames: int
    duration: float
    events: list[Event]
    threshold: float
    buffer_seconds: float
    domain: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "video_path": self.video_path,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration": self.duration,
            "threshold": self.threshold,
            "buffer_seconds": self.buffer_seconds,
            "domain": self.domain,
            "num_events": len(self.events),
            "events": [e.to_dict() for e in self.events],
        }

    def save(self, output_path: str | Path) -> None:
        """Save detection result to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "DetectionResult":
        """Load detection result from JSON."""
        with open(path) as f:
            data = json.load(f)

        events = [Event(**e) for e in data["events"]]
        return cls(
            video_path=data["video_path"],
            fps=data["fps"],
            total_frames=data["total_frames"],
            duration=data["duration"],
            events=events,
            threshold=data["threshold"],
            buffer_seconds=data["buffer_seconds"],
            domain=data.get("domain"),
        )


class EventPredictor:
    """Predictor for detecting events in videos across domains."""

    def __init__(
        self,
        model,
        domain: Optional[str] = None,
        device: Optional[torch.device] = None,
        window_size: int = 8,
        stride: int = 4,
        frame_size: tuple[int, int] = (224, 224),
        target_fps: float = 10.0,
        batch_size: int = 8,
        num_workers: int = 4,
    ):
        """Initialize predictor.

        Args:
            model: Trained detection model.
            domain: Domain name (optional, for metadata).
            device: Device to run inference on.
            window_size: Number of frames per window.
            stride: Stride between windows.
            frame_size: Frame resize dimensions.
            target_fps: Target FPS for extraction.
            batch_size: Batch size for inference.
            num_workers: Number of data loading workers.
        """
        self.model = model
        self.domain = domain
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

        self.window_size = window_size
        self.stride = stride
        self.frame_size = frame_size
        self.target_fps = target_fps
        self.batch_size = batch_size
        self.num_workers = num_workers

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        **kwargs,
    ) -> "EventPredictor":
        """Create predictor from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint.
            **kwargs: Additional arguments for predictor.

        Returns:
            EventPredictor instance.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        domain_name = checkpoint.get("domain", "cricket")  # Default to cricket for old checkpoints
        
        # Get domain and create model
        domain = DomainRegistry.get(domain_name)
        model_config = checkpoint.get("config", {})
        model = domain.create_model(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return cls(model=model, domain=domain_name, **kwargs)

    @torch.no_grad()
    def predict_video(
        self,
        video_path: str | Path,
        threshold: float = 0.5,
        buffer_seconds: float = 3.0,
        min_gap_seconds: float = 2.0,
        progress_callback: Optional[callable] = None,
        event_callback: Optional[callable] = None,
    ) -> DetectionResult:
        """Detect events in a video.

        Args:
            video_path: Path to video file.
            threshold: Confidence threshold for detection.
            buffer_seconds: Seconds to add before/after detection.
            min_gap_seconds: Minimum gap between detections to merge.
            progress_callback: Optional callback(current, total, status) for progress updates.
            event_callback: Optional callback(event) called when an event is finalized.

        Returns:
            DetectionResult with detected events.
        """
        video_path = Path(video_path)
        metadata = get_video_metadata(video_path)

        if progress_callback:
            progress_callback(0, 100, "Loading video...")

        # Get domain to create inference dataset
        if self.domain:
            domain = DomainRegistry.get(self.domain)
            dataset = domain.create_inference_dataset(
                video_path=video_path,
                window_size=self.window_size,
                stride=self.stride,
                frame_size=self.frame_size,
                target_fps=self.target_fps,
            )
        else:
            # Fallback for old checkpoints - use base inference dataset
            from src.core.base_dataset import BaseInferenceDataset
            dataset = BaseInferenceDataset(
                video_path=video_path,
                window_size=self.window_size,
                stride=self.stride,
                frame_size=self.frame_size,
                target_fps=self.target_fps,
            )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        total_batches = len(loader)

        # STATE for streaming detection
        events = []
        current_event_group = []  # List of (timestamp, confidence)
        last_positive_time = -float("inf")
        event_counter = 1

        for batch_idx, (frames, info) in enumerate(tqdm(loader, desc="Detecting events")):
            frames = frames.to(self.device)

            probs = self.model.predict_proba(frames)
            probs = probs.cpu().numpy().flatten()
            batch_timestamps = info["timestamp"].numpy()

            for i, prob in enumerate(probs):
                timestamp = batch_timestamps[i]
                
                # Check for gap to finalize previous event
                if current_event_group and (timestamp - last_positive_time > min_gap_seconds):
                    # Finalize current group
                    events.append(self._finalize_event(
                        current_event_group, 
                        event_counter, 
                        metadata.fps, 
                        buffer_seconds
                    ))
                    if event_callback:
                        event_callback(events[-1])
                    
                    event_counter += 1
                    current_event_group = []

                if prob >= threshold:
                    current_event_group.append((timestamp, prob))
                    last_positive_time = timestamp

            # Update progress
            if progress_callback:
                progress = int((batch_idx + 1) / total_batches * 100)
                progress_callback(progress, 100, f"Processing batch {batch_idx + 1}/{total_batches}")

        # Finalize any remaining group
        if current_event_group:
            events.append(self._finalize_event(
                current_event_group, 
                event_counter, 
                metadata.fps, 
                buffer_seconds
            ))
            if event_callback:
                event_callback(events[-1])

        return DetectionResult(
            video_path=str(video_path),
            fps=metadata.fps,
            total_frames=metadata.total_frames,
            duration=metadata.duration_seconds,
            events=events,
            threshold=threshold,
            buffer_seconds=buffer_seconds,
            domain=self.domain,
        )

    def _finalize_event(
        self, 
        group: list[tuple[float, float]], 
        event_id: int, 
        fps: float, 
        buffer_seconds: float
    ) -> Event:
        """Create an Event object from a group of detections."""
        timestamps = [t for t, _ in group]
        confidences = [c for _, c in group]
        
        start_time = max(0, min(timestamps) - buffer_seconds)
        end_time = max(timestamps) + buffer_seconds
        
        return Event(
            id=event_id,
            start_time=start_time,
            end_time=end_time,
            start_frame=int(start_time * fps),
            end_frame=int(end_time * fps),
            confidence=max(confidences),
            duration=end_time - start_time,
        )
            
    # Kept for backward compatibility if needed, but not used in streaming
    def _group_detections(self, *args, **kwargs):
        pass


def extract_event_clips(
    detection_result: DetectionResult,
    output_dir: str | Path,
    codec: str = "libx264",
    crf: int = 23,
) -> list[Path]:
    """Extract video clips for detected events.

    Args:
        detection_result: Detection result with events.
        output_dir: Directory to save clips.
        codec: Video codec.
        crf: Quality factor.

    Returns:
        List of paths to extracted clips.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_name = Path(detection_result.video_path).stem
    clip_paths = []

    for event in tqdm(detection_result.events, desc="Extracting clips"):
        clip_name = f"{video_name}_event_{event.id:03d}"
        output_path = output_dir / f"{clip_name}.mp4"

        extract_clip(
            detection_result.video_path,
            output_path,
            event.start_time,
            event.end_time,
            codec=codec,
            crf=crf,
        )

        clip_paths.append(output_path)

    return clip_paths


def detect_and_extract(
    video_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    threshold: float = 0.5,
    buffer_seconds: float = 3.0,
    save_detections: bool = True,
    extract_clips: bool = True,
    **predictor_kwargs,
) -> tuple[DetectionResult, list[Path]]:
    """Detect events and extract clips in one call.
    
    Uses streaming extraction to generate clips while detection is running.

    Args:
        video_path: Path to video.
        checkpoint_path: Path to model checkpoint.
        output_dir: Output directory.
        threshold: Detection threshold.
        buffer_seconds: Buffer around detections.
        save_detections: Save detection JSON.
        extract_clips: Extract video clips.
        **predictor_kwargs: Additional predictor arguments.

    Returns:
        (DetectionResult, list of clip paths)
    """
    import concurrent.futures
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "clips"
    
    if extract_clips:
        clips_dir.mkdir(parents=True, exist_ok=True)

    # Create predictor
    predictor = EventPredictor.from_checkpoint(checkpoint_path, **predictor_kwargs)
    
    video_name = Path(video_path).stem
    clip_paths = []
    futures = []
    
    # Thread pool for async extraction
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def _on_event(event: Event):
        """Callback to extract clip immediately when event is detected."""
        if extract_clips:
            clip_name = f"{video_name}_event_{event.id:03d}"
            output_path = clips_dir / f"{clip_name}.mp4"
            clip_paths.append(output_path)
            
            # Submit to thread pool
            futures.append(executor.submit(
                extract_clip,
                video_path,
                output_path,
                event.start_time,
                event.end_time,
            ))

    try:
        # Run detection with callback
        result = predictor.predict_video(
            video_path,
            threshold=threshold,
            buffer_seconds=buffer_seconds,
            event_callback=_on_event,
        )
        
        # Wait for all extractions to finish
        if futures:
            concurrent.futures.wait(futures)
            
    finally:
        executor.shutdown(wait=True)

    # Save detections
    if save_detections:
        result.save(output_dir / f"{video_name}_detections.json")

    return result, clip_paths
