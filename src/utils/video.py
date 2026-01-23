"""Video processing utilities for cricket delivery detection."""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import cv2
import ffmpeg
import numpy as np


@dataclass
class VideoMetadata:
    """Metadata about a video file."""

    path: Path
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float

    @property
    def duration_str(self) -> str:
        """Return duration as HH:MM:SS string."""
        total_seconds = int(self.duration_seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_video_metadata(video_path: str | Path) -> VideoMetadata:
    """Extract metadata from a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        VideoMetadata object with video properties.

    Raises:
        FileNotFoundError: If video file doesn't exist.
        ValueError: If video cannot be opened.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / fps if fps > 0 else 0

        return VideoMetadata(
            path=video_path,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration_seconds,
        )
    finally:
        cap.release()


def extract_frames(
    video_path: str | Path,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    step: int = 1,
    resize: Optional[tuple[int, int]] = None,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Extract frames from a video file.

    Args:
        video_path: Path to the video file.
        start_frame: First frame to extract (0-indexed).
        end_frame: Last frame to extract (exclusive). None for end of video.
        step: Extract every Nth frame.
        resize: Optional (width, height) to resize frames.

    Yields:
        Tuples of (frame_number, frame_array) where frame_array is RGB.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if end_frame is None:
            end_frame = total_frames

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame

        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (current_frame - start_frame) % step == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if resize:
                    frame_rgb = cv2.resize(frame_rgb, resize)

                yield current_frame, frame_rgb

            current_frame += 1
    finally:
        cap.release()


def extract_frames_at_fps(
    video_path: str | Path,
    target_fps: float,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    resize: Optional[tuple[int, int]] = None,
) -> Generator[tuple[float, np.ndarray], None, None]:
    """Extract frames at a specific FPS rate.

    Args:
        video_path: Path to the video file.
        target_fps: Target frames per second to extract.
        start_time: Start time in seconds.
        end_time: End time in seconds. None for end of video.
        resize: Optional (width, height) to resize frames.

    Yields:
        Tuples of (timestamp_seconds, frame_array) where frame_array is RGB.
    """
    metadata = get_video_metadata(video_path)
    video_fps = metadata.fps

    if end_time is None:
        end_time = metadata.duration_seconds

    # Calculate frame step to achieve target FPS
    frame_step = max(1, int(video_fps / target_fps))

    start_frame = int(start_time * video_fps)
    end_frame = int(end_time * video_fps)

    for frame_num, frame in extract_frames(
        video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        step=frame_step,
        resize=resize,
    ):
        timestamp = frame_num / video_fps
        yield timestamp, frame


def load_frame_batch(
    video_path: str | Path,
    frame_indices: list[int],
    resize: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Load a batch of specific frames from a video.

    Args:
        video_path: Path to the video file.
        frame_indices: List of frame indices to load.
        resize: Optional (width, height) to resize frames.

    Returns:
        Numpy array of shape (N, H, W, 3) with RGB frames.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames = []
    try:
        for frame_idx in sorted(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                raise ValueError(f"Cannot read frame {frame_idx}")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize:
                frame_rgb = cv2.resize(frame_rgb, resize)

            frames.append(frame_rgb)
    finally:
        cap.release()

    # Reorder to match original frame_indices order
    sorted_indices = sorted(range(len(frame_indices)), key=lambda i: frame_indices[i])
    reordered = [frames[sorted_indices.index(i)] for i in range(len(frames))]

    return np.stack(reordered)


def extract_clip(
    video_path: str | Path,
    output_path: str | Path,
    start_time: float,
    end_time: float,
    codec: str = "libx264",
    crf: int = 23,
) -> Path:
    """Extract a clip from a video file using ffmpeg.

    Args:
        video_path: Path to source video.
        output_path: Path for output clip.
        start_time: Start time in seconds.
        end_time: End time in seconds.
        codec: Video codec to use.
        crf: Constant rate factor (quality, lower = better).

    Returns:
        Path to the output clip.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = end_time - start_time

    (
        ffmpeg.input(str(video_path), ss=start_time, t=duration)
        .output(str(output_path), vcodec=codec, crf=crf, acodec="aac")
        .overwrite_output()
        .run(quiet=True)
    )

    return output_path


def extract_clips_batch(
    video_path: str | Path,
    clips: list[dict],
    output_dir: str | Path,
    codec: str = "libx264",
    crf: int = 23,
) -> list[Path]:
    """Extract multiple clips from a video file.

    Args:
        video_path: Path to source video.
        clips: List of dicts with 'start_time', 'end_time', and optional 'name'.
        output_dir: Directory for output clips.
        codec: Video codec to use.
        crf: Constant rate factor.

    Returns:
        List of paths to output clips.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []
    for i, clip in enumerate(clips):
        name = clip.get("name", f"clip_{i:04d}")
        output_path = output_dir / f"{name}.mp4"

        extract_clip(
            video_path,
            output_path,
            clip["start_time"],
            clip["end_time"],
            codec=codec,
            crf=crf,
        )
        output_paths.append(output_path)

    return output_paths


def frame_to_time(frame: int, fps: float) -> float:
    """Convert frame number to time in seconds."""
    return frame / fps


def time_to_frame(time_seconds: float, fps: float) -> int:
    """Convert time in seconds to frame number."""
    return int(time_seconds * fps)
