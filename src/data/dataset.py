"""PyTorch dataset for cricket delivery detection."""

import json
import random
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from src.data.labeler import VideoLabels


class DeliveryDataset(Dataset):
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
        frame_size: tuple[int, int] = (224, 224),
        target_fps: float = 10.0,
        augment: bool = True,
        negative_ratio: float = 1.0,
        overlap_threshold: float = 0.5,
    ):
        """Initialize the dataset.

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
        self.labels_dir = Path(labels_dir)
        self.videos_dir = Path(videos_dir)
        self.window_size = window_size
        self.frame_size = frame_size
        self.target_fps = target_fps
        self.augment = augment
        self.negative_ratio = negative_ratio
        self.overlap_threshold = overlap_threshold

        # Load all labels
        self.video_labels: list[VideoLabels] = []
        self.samples: list[dict] = []

        self._load_labels()
        self._create_samples()

        # Setup augmentation
        self.transform = self._get_transform()

    def _load_labels(self) -> None:
        """Load all label files."""
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

    def _get_transform(self) -> A.Compose:
        """Get augmentation pipeline."""
        if self.augment:
            return A.Compose(
                [
                    A.RandomBrightnessContrast(p=0.5),
                    A.HueSaturationValue(p=0.3),
                    A.GaussNoise(p=0.2),
                    A.MotionBlur(blur_limit=3, p=0.2),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                    ToTensorV2(),
                ],
                additional_targets={f"image{i}": "image" for i in range(1, self.window_size)},
            )
        else:
            return A.Compose(
                [
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                    ToTensorV2(),
                ]
            )

    def _load_frames(self, video_path: str, start_frame: int, frame_step: int) -> list[np.ndarray]:
        """Load a window of frames from video."""
        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            for i in range(self.window_size):
                frame_idx = start_frame + i * frame_step
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    # Pad with black frame if we can't read
                    frame = np.zeros(
                        (self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8
                    )
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, self.frame_size)

                frames.append(frame)
        finally:
            cap.release()

        return frames

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sample.

        Returns:
            frames: Tensor of shape (window_size, 3, H, W)
            label: Tensor of shape (1,) with 0 or 1
        """
        sample = self.samples[idx]

        # Load frames
        frames = self._load_frames(
            sample["video_path"], sample["start_frame"], sample["frame_step"]
        )

        # Apply augmentation
        if self.augment:
            # Prepare targets for albumentations
            targets = {"image": frames[0]}
            for i in range(1, len(frames)):
                targets[f"image{i}"] = frames[i]

            transformed = self.transform(**targets)

            # Collect transformed frames
            tensor_frames = [transformed["image"]]
            for i in range(1, len(frames)):
                tensor_frames.append(transformed[f"image{i}"])
        else:
            tensor_frames = []
            for frame in frames:
                transformed = self.transform(image=frame)
                tensor_frames.append(transformed["image"])

        # Stack frames: (T, C, H, W)
        frames_tensor = torch.stack(tensor_frames)

        # Label
        label = torch.tensor([sample["label"]], dtype=torch.float32)

        return frames_tensor, label


class InferenceDataset(Dataset):
    """Dataset for inference on a single video.

    Creates overlapping windows for continuous prediction.
    """

    def __init__(
        self,
        video_path: str | Path,
        window_size: int = 8,
        stride: int = 4,
        frame_size: tuple[int, int] = (224, 224),
        target_fps: float = 10.0,
    ):
        """Initialize inference dataset.

        Args:
            video_path: Path to video file.
            window_size: Number of frames per window.
            stride: Number of frames to move between windows.
            frame_size: (width, height) to resize frames.
            target_fps: Target FPS for frame extraction.
        """
        self.video_path = str(video_path)
        self.window_size = window_size
        self.stride = stride
        self.frame_size = frame_size
        self.target_fps = target_fps

        # Get video info
        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        self.frame_step = max(1, int(self.fps / self.target_fps))

        # Create windows
        self.windows: list[dict] = []
        window_frames = self.window_size * self.frame_step
        stride_frames = self.stride * self.frame_step

        for start_frame in range(0, self.total_frames - window_frames, stride_frames):
            center_frame = start_frame + window_frames // 2
            self.windows.append(
                {
                    "start_frame": start_frame,
                    "center_frame": center_frame,
                    "timestamp": center_frame / self.fps,
                }
            )

        # Transform (no augmentation)
        self.transform = A.Compose(
            [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    def _load_frames(self, start_frame: int) -> list[np.ndarray]:
        """Load a window of frames."""
        cap = cv2.VideoCapture(self.video_path)
        frames = []

        try:
            for i in range(self.window_size):
                frame_idx = start_frame + i * self.frame_step
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    frame = np.zeros(
                        (self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8
                    )
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, self.frame_size)

                frames.append(frame)
        finally:
            cap.release()

        return frames

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        """Get a window.

        Returns:
            frames: Tensor of shape (window_size, 3, H, W)
            info: Dict with center_frame and timestamp
        """
        window = self.windows[idx]
        frames = self._load_frames(window["start_frame"])

        # Transform
        tensor_frames = []
        for frame in frames:
            transformed = self.transform(image=frame)
            tensor_frames.append(transformed["image"])

        frames_tensor = torch.stack(tensor_frames)

        return frames_tensor, {
            "center_frame": window["center_frame"],
            "timestamp": window["timestamp"],
        }


def create_train_val_split(
    domain,
    labels_dir: str | Path,
    videos_dir: str | Path,
    val_ratio: float = 0.2,
    **dataset_kwargs,
) -> tuple[Dataset, Dataset]:
    """Create train and validation datasets with video-level split.

    Args:
        domain: Domain instance to use for dataset creation.
        labels_dir: Directory containing label files.
        videos_dir: Directory containing videos.
        val_ratio: Fraction of videos for validation.
        **dataset_kwargs: Additional arguments for dataset.

    Returns:
        (train_dataset, val_dataset)
    """
    labels_dir = Path(labels_dir)
    label_files = list(labels_dir.glob("*.json"))

    return create_train_val_split_from_files(
        domain=domain,
        label_files=label_files,
        videos_dir=videos_dir,
        val_ratio=val_ratio,
        **dataset_kwargs,
    )


def create_train_val_split_from_files(
    domain,
    label_files: list[Path],
    videos_dir: str | Path,
    val_ratio: float = 0.2,
    **dataset_kwargs,
) -> tuple[Dataset, Dataset]:
    """Create train and validation datasets from specific label files.

    This function allows selective training on specific videos rather than
    using all videos in a directory.

    Args:
        domain: Domain instance to use for dataset creation.
        label_files: List of specific label file paths to use.
        videos_dir: Directory containing videos.
        val_ratio: Fraction of videos for validation.
        **dataset_kwargs: Additional arguments for dataset.

    Returns:
        (train_dataset, val_dataset)
    """
    import tempfile
    import shutil

    # Ensure all paths are Path objects
    label_files = [Path(f) for f in label_files]
    videos_dir = Path(videos_dir)

    # Filter to only existing files
    existing_files = [f for f in label_files if f.exists()]
    if not existing_files:
        raise ValueError("No valid label files provided")

    # Shuffle and split
    random.shuffle(existing_files)

    # Handle case with only 1 label file - use same file for train and val
    if len(existing_files) == 1:
        print("Warning: Only 1 label file found. Using same data for train and validation.")
        train_files = existing_files
        val_files = existing_files
    else:
        n_val = max(1, int(len(existing_files) * val_ratio))
        val_files = existing_files[:n_val]
        train_files = existing_files[n_val:]

    print(f"Training on {len(train_files)} videos, validating on {len(val_files)} videos")

    # Create temporary directories for split
    train_labels_dir = Path(tempfile.mkdtemp())
    val_labels_dir = Path(tempfile.mkdtemp())

    for f in train_files:
        shutil.copy(f, train_labels_dir / f.name)
    for f in val_files:
        shutil.copy(f, val_labels_dir / f.name)

    # Use domain to create datasets
    train_dataset = domain.create_dataset(
        train_labels_dir, videos_dir, augment=True, **dataset_kwargs
    )
    val_dataset = domain.create_dataset(
        val_labels_dir, videos_dir, augment=False, **dataset_kwargs
    )

    return train_dataset, val_dataset
