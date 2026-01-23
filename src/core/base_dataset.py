"""Base dataset classes for event detection.

This module provides domain-agnostic base classes for training and inference datasets.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class BaseEventDataset(Dataset, ABC):
    """Abstract base class for event detection training datasets.

    Subclasses should implement domain-specific logic for loading labels
    and creating positive/negative samples.
    """

    def __init__(
        self,
        labels_dir: str | Path,
        videos_dir: str | Path,
        window_size: int = 8,
        frame_size: Tuple[int, int] = (224, 224),
        target_fps: float = 10.0,
        augment: bool = True,
    ):
        """Initialize base dataset.

        Args:
            labels_dir: Directory containing label files.
            videos_dir: Directory containing video files.
            window_size: Number of frames per sample.
            frame_size: (width, height) to resize frames to.
            target_fps: Target FPS for frame extraction.
            augment: Whether to apply data augmentation.
        """
        self.labels_dir = Path(labels_dir)
        self.videos_dir = Path(videos_dir)
        self.window_size = window_size
        self.frame_size = frame_size
        self.target_fps = target_fps
        self.augment = augment

        self.samples = []
        self.transform = self._get_transform()

    @abstractmethod
    def _load_labels(self) -> None:
        """Load label files. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _create_samples(self) -> None:
        """Create training samples. Must be implemented by subclasses."""
        pass

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

    def _load_frames(
        self, video_path: str, start_frame: int, frame_step: int
    ) -> list[np.ndarray]:
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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


class BaseInferenceDataset(Dataset):
    """Base class for inference datasets.

    Creates overlapping windows for continuous prediction on a single video.
    """

    def __init__(
        self,
        video_path: str | Path,
        window_size: int = 8,
        stride: int = 4,
        frame_size: Tuple[int, int] = (224, 224),
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
        self.windows = []
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
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
