"""ONNX Runtime predictor for event detection — no PyTorch dependency.

This module provides lightweight inference using only onnxruntime, numpy, and cv2.
It replicates the windowing/detection logic from EventPredictor but avoids
importing torch entirely, saving ~500MB of memory.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None


# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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
    events: list
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


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Normalize a single RGB frame to ImageNet stats.

    Args:
        frame: (H, W, 3) uint8 RGB array.

    Returns:
        (3, H, W) float32 normalized array.
    """
    img = frame.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)  # HWC → CHW


class OnnxEventPredictor:
    """Event predictor using ONNX Runtime — no PyTorch required."""

    def __init__(
        self,
        onnx_path: str,
        domain: Optional[str] = None,
        window_size: int = 8,
        stride: int = 4,
        frame_size: tuple[int, int] = (224, 224),
        target_fps: float = 10.0,
        batch_size: int = 4,
    ):
        if ort is None:
            raise ImportError("onnxruntime is required: pip install onnxruntime")

        self.domain = domain
        self.window_size = window_size
        self.stride = stride
        self.frame_size = frame_size
        self.target_fps = target_fps
        self.batch_size = batch_size

        # Create session with minimal memory settings
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _load_frames(self, cap: cv2.VideoCapture, start_frame: int, frame_step: int) -> np.ndarray:
        """Load and preprocess a window of frames.

        Returns:
            (window_size, 3, H, W) float32 array, or None if frames can't be read.
        """
        frames = []
        w, h = self.frame_size

        for i in range(self.window_size):
            frame_idx = start_frame + i * frame_step
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (w, h))

            frames.append(_normalize_frame(frame))

        return np.stack(frames)  # (T, C, H, W)

    def predict_video(
        self,
        video_path: str,
        threshold: float = 0.5,
        buffer_seconds: float = 3.0,
        min_gap_seconds: float = 2.0,
        progress_callback=None,
    ) -> DetectionResult:
        """Detect events in a video using ONNX Runtime.

        Args:
            video_path: Path to video file.
            threshold: Confidence threshold.
            buffer_seconds: Seconds to add before/after detection.
            min_gap_seconds: Minimum gap to merge detections.
            progress_callback: Optional callback(current, total, status).

        Returns:
            DetectionResult with detected events.
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0

        frame_step = max(1, int(fps / self.target_fps))
        window_frames = self.window_size * frame_step
        stride_frames = self.stride * frame_step

        # Build windows
        windows = []
        for start_frame in range(0, total_frames - window_frames, stride_frames):
            center_frame = start_frame + window_frames // 2
            windows.append({
                "start_frame": start_frame,
                "center_frame": center_frame,
                "timestamp": center_frame / fps,
            })

        if progress_callback:
            progress_callback(0, 100, "Starting detection...")

        # Process in batches
        events = []
        current_event_group = []
        last_positive_time = -float("inf")
        event_counter = 1
        total_batches = (len(windows) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(total_batches):
            start_i = batch_idx * self.batch_size
            end_i = min(start_i + self.batch_size, len(windows))
            batch_windows = windows[start_i:end_i]

            # Load frames for this batch
            batch_frames = []
            for w in batch_windows:
                batch_frames.append(self._load_frames(cap, w["start_frame"], frame_step))

            batch_input = np.stack(batch_frames).astype(np.float32)  # (B, T, C, H, W)

            # Run inference
            logits = self.session.run([self.output_name], {self.input_name: batch_input})[0]
            probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
            probs = probs.flatten()

            for i, prob in enumerate(probs):
                timestamp = batch_windows[i]["timestamp"]

                if current_event_group and (timestamp - last_positive_time > min_gap_seconds):
                    events.append(self._finalize_event(
                        current_event_group, event_counter, fps, buffer_seconds
                    ))
                    event_counter += 1
                    current_event_group = []

                if prob >= threshold:
                    current_event_group.append((timestamp, float(prob)))
                    last_positive_time = timestamp

            if progress_callback:
                progress = int((batch_idx + 1) / total_batches * 100)
                progress_callback(progress, 100, f"Batch {batch_idx + 1}/{total_batches}")

        cap.release()

        # Finalize remaining group
        if current_event_group:
            events.append(self._finalize_event(
                current_event_group, event_counter, fps, buffer_seconds
            ))

        return DetectionResult(
            video_path=str(video_path),
            fps=fps,
            total_frames=total_frames,
            duration=duration,
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
        buffer_seconds: float,
    ) -> Event:
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
