"""Cricket-specific dataset implementation."""

import json
import random
from pathlib import Path
from typing import Tuple

import torch

from src.core.base_dataset import BaseEventDataset, BaseInferenceDataset
from src.data.labeler import VideoLabels


class CricketDataset(BaseEventDataset):
    """Dataset for cricket delivery detection.

    Creates sliding window samples from labeled videos.
    Positive samples: windows containing a delivery.
    Negative samples: windows not containing any delivery.
    """

    def __init__(
        self,
        labels_dir: str | Path,
        videos_dir: str | Path,
        window_size: int = 8,
        frame_size: Tuple[int, int] = (224, 224),
        target_fps: float = 10.0,
        augment: bool = True,
        negative_ratio: float = 1.0,
        overlap_threshold: float = 0.5,
    ):
        """Initialize the cricket dataset.

        Args:
            labels_dir: Directory containing label JSON files.
            videos_dir: Directory containing video files.
            window_size: Number of frames per sample.
            frame_size: (width, height) to resize frames to.
            target_fps: Target FPS for frame extraction.
            augment: Whether to apply data augmentation.
            negative_ratio: Ratio of negative to positive samples.
            overlap_threshold: Minimum overlap with delivery to be positive.
        """
        self.negative_ratio = negative_ratio
        self.overlap_threshold = overlap_threshold
        self.video_labels = []

        super().__init__(labels_dir, videos_dir, window_size, frame_size, target_fps, augment)

    def _load_labels(self) -> None:
        """Load all cricket label files."""
        for label_file in self.labels_dir.glob("*.json"):
            labels = VideoLabels.load(label_file)

            # Verify video exists
            video_path = self.videos_dir / Path(labels.video_path).name
            if not video_path.exists():
                print(f"Warning: Video not found: {video_path}")
                continue

            labels.video_path = str(video_path)
            self.video_labels.append(labels)

    def _create_samples(self) -> None:
        """Create positive and negative samples from all videos."""
        for labels in self.video_labels:
            # Calculate frame step based on target FPS
            frame_step = max(1, int(labels.fps / self.target_fps))

            # Create positive samples (windows containing deliveries)
            positive_samples = []
            for delivery in labels.deliveries:
                # Create multiple windows around each delivery
                delivery_center = (delivery.start_frame + delivery.end_frame) // 2
                window_frames = self.window_size * frame_step

                # Slide window to cover the delivery
                for offset in range(-window_frames // 2, window_frames // 2, frame_step):
                    start_frame = delivery_center + offset - window_frames // 2
                    end_frame = start_frame + window_frames

                    if start_frame < 0 or end_frame >= labels.total_frames:
                        continue

                    # Check overlap with delivery
                    overlap = self._compute_overlap(
                        start_frame, end_frame, delivery.start_frame, delivery.end_frame
                    )
                    if overlap >= self.overlap_threshold:
                        positive_samples.append(
                            {
                                "video_path": labels.video_path,
                                "start_frame": start_frame,
                                "frame_step": frame_step,
                                "fps": labels.fps,
                                "label": 1,
                            }
                        )

            # Create negative samples (windows not containing deliveries)
            negative_samples = []
            num_negatives = int(len(positive_samples) * self.negative_ratio)
            delivery_ranges = [
                (d.start_frame, d.end_frame) for d in labels.deliveries
            ]

            window_frames = self.window_size * frame_step
            attempts = 0
            max_attempts = num_negatives * 10

            while len(negative_samples) < num_negatives and attempts < max_attempts:
                # Random start frame
                start_frame = random.randint(0, labels.total_frames - window_frames - 1)
                end_frame = start_frame + window_frames

                # Check if it overlaps with any delivery
                is_negative = True
                for d_start, d_end in delivery_ranges:
                    overlap = self._compute_overlap(start_frame, end_frame, d_start, d_end)
                    if overlap > 0.1:  # Allow small overlap for hard negatives
                        is_negative = False
                        break

                if is_negative:
                    negative_samples.append(
                        {
                            "video_path": labels.video_path,
                            "start_frame": start_frame,
                            "frame_step": frame_step,
                            "fps": labels.fps,
                            "label": 0,
                        }
                    )
                attempts += 1

            # Add hard negatives from false positives
            hard_negatives = []
            for fp in getattr(labels, 'false_positives', []):
                fp_center = (fp.start_frame + fp.end_frame) // 2
                start_frame = fp_center - window_frames // 2
                end_frame = start_frame + window_frames

                if start_frame >= 0 and end_frame < labels.total_frames:
                    hard_negatives.append(
                        {
                            "video_path": labels.video_path,
                            "start_frame": start_frame,
                            "frame_step": frame_step,
                            "fps": labels.fps,
                            "label": 0,
                        }
                    )

            self.samples.extend(positive_samples)
            self.samples.extend(negative_samples)
            self.samples.extend(hard_negatives)

            if hard_negatives:
                print(f"Added {len(hard_negatives)} hard negative(s) from false positives")

        # Shuffle samples
        random.shuffle(self.samples)

    def _compute_overlap(
        self, win_start: int, win_end: int, del_start: int, del_end: int
    ) -> float:
        """Compute overlap ratio between window and delivery."""
        overlap_start = max(win_start, del_start)
        overlap_end = min(win_end, del_end)
        overlap = max(0, overlap_end - overlap_start)

        delivery_length = del_end - del_start
        if delivery_length == 0:
            return 0.0

        return overlap / delivery_length


class CricketInferenceDataset(BaseInferenceDataset):
    """Inference dataset for cricket videos.

    Creates overlapping windows for continuous prediction.
    """

    pass  # Inherits all functionality from BaseInferenceDataset
