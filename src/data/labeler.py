"""Streamlit-based video labeling tool for cricket delivery detection."""

import json
import re
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import argparse
import sys

from src.utils.video import VideoMetadata, get_video_metadata
from src.data.ui_theme import inject_theme, COLORS, status_badge, domain_badge, styled_header, training_stat_card, video_status_icon
from src.storage import get_storage_backend, StorageBackend


@dataclass
class Delivery:
    """A single labeled delivery event."""

    id: str
    start_frame: int
    end_frame: int
    delivery_type: Optional[str] = None  # Future: fast, spin, etc.
    outcome: Optional[str] = None  # Future: runs, wicket, etc.
    notes: str = ""

    @classmethod
    def create(cls, start_frame: int, end_frame: int) -> "Delivery":
        """Create a new delivery with auto-generated ID."""
        return cls(
            id=str(uuid.uuid4())[:8],
            start_frame=start_frame,
            end_frame=end_frame,
        )

    def duration_frames(self) -> int:
        """Return duration in frames."""
        return self.end_frame - self.start_frame


@dataclass
class FalsePositive:
    """A false positive detection to use as hard negative during training."""

    id: str
    start_frame: int
    end_frame: int
    notes: str = ""


@dataclass
class VideoLabels:
    """Labels for a single video file."""

    video_path: str
    fps: float
    total_frames: int
    width: int
    height: int
    deliveries: list[Delivery] = field(default_factory=list)
    false_positives: list[FalsePositive] = field(default_factory=list)

    @classmethod
    def from_metadata(cls, metadata: VideoMetadata) -> "VideoLabels":
        """Create empty labels from video metadata."""
        return cls(
            video_path=str(metadata.path),
            fps=metadata.fps,
            total_frames=metadata.total_frames,
            width=metadata.width,
            height=metadata.height,
        )

    def add_delivery(self, start_frame: int, end_frame: int) -> Delivery:
        """Add a new delivery and return it."""
        delivery = Delivery.create(start_frame, end_frame)
        self.deliveries.append(delivery)
        self.deliveries.sort(key=lambda d: d.start_frame)
        return delivery

    def remove_delivery(self, delivery_id: str) -> bool:
        """Remove a delivery by ID. Returns True if found and removed."""
        for i, d in enumerate(self.deliveries):
            if d.id == delivery_id:
                self.deliveries.pop(i)
                return True
        return False

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "video_path": self.video_path,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "deliveries": [asdict(d) for d in self.deliveries],
        }
        if self.false_positives:
            result["false_positives"] = [asdict(fp) for fp in self.false_positives]
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "VideoLabels":
        """Load from dictionary."""
        deliveries = [Delivery(**d) for d in data.get("deliveries", [])]
        false_positives = [FalsePositive(**fp) for fp in data.get("false_positives", [])]
        return cls(
            video_path=data["video_path"],
            fps=data["fps"],
            total_frames=data["total_frames"],
            width=data["width"],
            height=data["height"],
            deliveries=deliveries,
            false_positives=false_positives,
        )

    def save(self, output_path: str | Path) -> None:
        """Save labels to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "VideoLabels":
        """Load labels from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


def get_frame(cap: cv2.VideoCapture, frame_num: int) -> tuple[Optional[np.ndarray], int]:
    """Get a specific frame from video capture as numpy array.

    Returns:
        Tuple of (frame_data, actual_frame_position)
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    ret, frame = cap.read()
    if not ret:
        return None, actual_pos
    # Convert BGR to RGB for display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb, actual_pos


def format_time(frame: int, fps: float) -> str:
    """Format frame number as timestamp string."""
    total_seconds = frame / fps
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    ms = int((total_seconds % 1) * 1000)
    return f"{minutes:02d}:{seconds:02d}.{ms:03d}"


def parse_timestamp(ts: str) -> Optional[int]:
    """Parse timestamp string (MM:SS or HH:MM:SS) to seconds."""
    if not ts or not ts.strip():
        return None

    ts = ts.strip()
    parts = ts.split(":")

    try:
        if len(parts) == 2:
            # MM:SS
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            # HH:MM:SS
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        else:
            return int(ts)  # Just seconds
    except ValueError:
        return None


def is_youtube_url(url: str) -> bool:
    """Check if URL is a valid YouTube URL."""
    youtube_patterns = [
        r'(https?://)?(www\.)?youtube\.com/watch\?v=[\w-]+',
        r'(https?://)?(www\.)?youtu\.be/[\w-]+',
        r'(https?://)?(www\.)?youtube\.com/embed/[\w-]+',
        r'(https?://)?(www\.)?youtube\.com/v/[\w-]+',
    ]
    return any(re.match(pattern, url) for pattern in youtube_patterns)


def get_video_info(url: str) -> Optional[dict]:
    """Get video info from YouTube URL using yt-dlp."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--dump-json", "--no-download", url],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        pass
    return None


def download_youtube_video(
    url: str,
    output_dir: Path,
    filename: Optional[str] = None,
    quality: str = "720",
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    progress_callback=None,
) -> Optional[Path]:
    """Download a YouTube video using yt-dlp.

    Args:
        url: YouTube URL
        output_dir: Directory to save video
        filename: Custom filename (without extension)
        quality: Max video height (360, 480, 720, 1080)
        start_time: Start time (MM:SS or HH:MM:SS)
        end_time: End time (MM:SS or HH:MM:SS)
        progress_callback: Function to call with progress updates

    Returns:
        Path to downloaded video or None if failed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build output template
    if filename:
        # Sanitize filename
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        output_template = str(output_dir / f"{filename}.%(ext)s")
    else:
        output_template = str(output_dir / "%(title)s.%(ext)s")

    # Build command
    cmd = [
        "yt-dlp",
        "-f", f"best[height<={quality}][ext=mp4]/best[height<={quality}]/best",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-playlist",
        "--progress",
    ]

    # Add time range if specified
    if start_time or end_time:
        section = "*"
        if start_time:
            section += start_time
        section += "-"
        if end_time:
            section += end_time
        cmd.extend(["--download-sections", section])

    cmd.append(url)

    try:
        if progress_callback:
            progress_callback("Starting download...")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode == 0:
            # Find the downloaded file
            if filename:
                expected_path = output_dir / f"{filename}.mp4"
                if expected_path.exists():
                    return expected_path

            # Search for most recent mp4 file
            mp4_files = sorted(output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
            if mp4_files:
                return mp4_files[0]
        else:
            if progress_callback:
                progress_callback(f"Download failed: {result.stderr[:200]}")

    except subprocess.TimeoutExpired:
        if progress_callback:
            progress_callback("Download timed out (10 min limit)")
    except FileNotFoundError:
        if progress_callback:
            progress_callback("yt-dlp not found. Install with: pip install yt-dlp")

    return None


def split_video(
    video_path: Path,
    chunk_duration_minutes: int = 10,
    output_dir: Optional[Path] = None,
    progress_callback=None,
) -> list[Path]:
    """Split a video into smaller chunks using ffmpeg.

    Args:
        video_path: Path to the video file to split.
        chunk_duration_minutes: Duration of each chunk in minutes.
        output_dir: Directory to save chunks. Defaults to same directory as video.
        progress_callback: Function to call with progress updates.

    Returns:
        List of paths to the created chunk files.
    """
    video_path = Path(video_path)
    if output_dir is None:
        output_dir = video_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get video duration using ffprobe
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        total_duration = float(result.stdout.strip())
    except Exception as e:
        if progress_callback:
            progress_callback(f"Error getting video duration: {e}")
        return []

    chunk_duration_seconds = chunk_duration_minutes * 60
    num_chunks = int(total_duration // chunk_duration_seconds) + (1 if total_duration % chunk_duration_seconds > 0 else 0)

    if progress_callback:
        progress_callback(f"Splitting into {num_chunks} chunks of {chunk_duration_minutes} min each...")

    base_name = video_path.stem
    chunks = []

    for i in range(num_chunks):
        start_time = i * chunk_duration_seconds
        chunk_filename = f"{base_name}_part{i+1:03d}.mp4"
        chunk_path = output_dir / chunk_filename

        if progress_callback:
            progress_callback(f"Creating chunk {i+1}/{num_chunks}: {chunk_filename}")

        try:
            # Use stream copy (-c copy) for fast splitting without re-encoding
            # Put -ss before -i for fast seeking
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-i", str(video_path),
                "-t", str(chunk_duration_seconds),
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                "-loglevel", "error",
                str(chunk_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
            chunks.append(chunk_path)
        except subprocess.CalledProcessError as e:
            if progress_callback:
                progress_callback(f"Error creating chunk {i+1}: {e}")
        except subprocess.TimeoutExpired:
            if progress_callback:
                progress_callback(f"Timeout creating chunk {i+1}")

    if progress_callback:
        progress_callback(f"Split complete! Created {len(chunks)} chunks.")

    return chunks


def keyboard_listener():
    """Inject JavaScript to capture keyboard events for navigation and labeling."""
    js_code = """
    <script>
    const streamlitDoc = window.parent.document;

    // Prevent duplicate listeners
    if (!streamlitDoc.keyboardListenerAdded) {
        streamlitDoc.keyboardListenerAdded = true;

        streamlitDoc.addEventListener('keydown', function(e) {
            // Ignore if typing in input field
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                return;
            }

            let action = null;

            // Navigation keys
            if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') {
                if (e.ctrlKey || e.metaKey) {
                    action = 'prev_100';
                } else if (e.shiftKey) {
                    action = 'prev_10';
                } else {
                    action = 'prev_1';
                }
                e.preventDefault();
            } else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') {
                if (e.ctrlKey || e.metaKey) {
                    action = 'next_100';
                } else if (e.shiftKey) {
                    action = 'next_10';
                } else {
                    action = 'next_1';
                }
                e.preventDefault();
            }
            // Marking keys
            else if (e.key === 's' && !e.ctrlKey && !e.metaKey) {
                action = 'mark_start';
                e.preventDefault();
            } else if (e.key === 'e' || e.key === 'E') {
                action = 'mark_end';
                e.preventDefault();
            } else if (e.key === 'Escape' || e.key === 'c' || e.key === 'C') {
                action = 'cancel';
                e.preventDefault();
            }
            // Save (Ctrl+S)
            else if ((e.ctrlKey || e.metaKey) && e.key === 's') {
                action = 'save';
                e.preventDefault();
            }

            if (action) {
                // Send action to Streamlit via query params hack
                const url = new URL(window.parent.location);
                url.searchParams.set('kb_action', action);
                url.searchParams.set('kb_ts', Date.now());
                window.parent.history.replaceState({}, '', url);

                // Trigger Streamlit rerun
                const buttons = streamlitDoc.querySelectorAll('button');
                for (let btn of buttons) {
                    if (btn.innerText.includes('REFRESH')) {
                        btn.click();
                        break;
                    }
                }
            }
        });
    }
    </script>
    """
    components.html(js_code, height=0)


def load_detections(storage: StorageBackend, video_name: str) -> Optional[dict]:
    """Load saved detection results for a video.

    Args:
        storage: Storage backend instance.
        video_name: Name of the video file.

    Returns:
        Detection dict if found, None otherwise.
    """
    detection_key = f"{Path(video_name).stem}_detections.json"
    try:
        return storage.read_detection(detection_key)
    except (FileNotFoundError, ValueError):
        return None
    except json.JSONDecodeError:
        # Delete corrupted file so user can re-run detection
        print(f"Warning: Corrupted detection file deleted: {detection_key}")
        storage.delete(detection_key, "detections")
    except Exception as e:
        # Handle S3 ClientError (NoSuchKey) and other "not found" errors
        err_str = str(e)
        if "NoSuchKey" in err_str or "Not Found" in err_str or "404" in err_str:
            return None
        print(f"Error loading detections: {e}")
    return None


def save_detections(storage: StorageBackend, video_name: str, detections: dict) -> None:
    """Save detection results for a video.

    Args:
        storage: Storage backend instance.
        video_name: Name of the video file.
        detections: Detection results dict.
    """
    detection_key = f"{Path(video_name).stem}_detections.json"
    storage.write_detection(detections, detection_key)


def get_detection_status(storage: StorageBackend, video_name: str) -> Optional[dict]:
    """Get quick status of detections for a video without loading full data.

    Args:
        storage: Storage backend instance.
        video_name: Name of the video file.

    Returns:
        Dict with count and status summary, or None if no detections.
    """
    detections = load_detections(storage, video_name)
    if detections and "deliveries" in detections:
        deliveries = detections["deliveries"]
        return {
            "total": len(deliveries),
            "pending": sum(1 for d in deliveries if d.get("status") == "pending"),
            "approved": sum(1 for d in deliveries if d.get("status") == "approved"),
            "rejected": sum(1 for d in deliveries if d.get("status") == "rejected"),
        }
    return None


def run_analysis_tab(domain: str = "cricket"):
    """Run the analysis tab for reviewing model detections."""
    st.markdown(f"## Analysis {domain_badge(domain)}", unsafe_allow_html=True)

    storage: StorageBackend = st.session_state.storage

    # Settings in a card-like container
    with st.container():
        col1, col2 = st.columns([2, 1])

        with col1:
            # Video selection with detection status indicators
            video_names = storage.list_videos(pattern="*.mp4,*.mov")

            # Build video options with detection status
            video_options = ["Select a video..."]
            for vname in video_names:
                status = get_detection_status(storage, vname)
                if status:
                    video_options.append(f"{vname} [{status['total']} detections]")
                else:
                    video_options.append(vname)

            selected_option = st.selectbox("Select Video", video_options, key="analysis_video")

            if selected_option == "Select a video...":
                st.info("Select a video to analyze")
                return

            # Extract actual video name (remove detection count suffix if present)
            if " [" in selected_option:
                selected_video = selected_option.rsplit(" [", 1)[0]
            else:
                selected_video = selected_option

        with col2:
            threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)

            # Check if model exists
            model_name = "delivery_detector_best.pt"
            if not storage.exists(model_name, "models"):
                st.warning("No model found. Train a model first to run new detections.")

    # Advanced settings in expander
    with st.expander("Speed Settings", expanded=False):
        speed_cols = st.columns(3)
        with speed_cols[0]:
            target_fps = st.select_slider(
                "Target FPS",
                options=[2.0, 5.0, 10.0, 15.0],
                value=5.0,
                help="Lower = faster but may miss short events"
            )
        with speed_cols[1]:
            stride = st.select_slider(
                "Stride",
                options=[2, 4, 8, 16],
                value=8,
                help="Higher = faster but less precise"
            )
        with speed_cols[2]:
            batch_size = st.select_slider(
                "Batch Size",
                options=[4, 8, 16, 32],
                value=16,
                help="Higher = faster if GPU has memory"
            )

    # Detection state
    detection_key = f"detections_{selected_video}_{threshold}"

    # Try to load existing detections from storage if not in session state
    if detection_key not in st.session_state:
        existing_detections = load_detections(storage, selected_video)
        if existing_detections:
            st.session_state[detection_key] = existing_detections

    # Show existing detections status and options
    existing_status = get_detection_status(storage, selected_video)

    # Track detection in progress
    if "detection_running" not in st.session_state:
        st.session_state.detection_running = False

    col_btn1, col_btn2, col_info = st.columns([1, 1, 2])

    with col_btn1:
        if st.session_state.detection_running:
            st.button("Detecting...", type="primary", use_container_width=True, disabled=True)
        else:
            run_new = st.button("Run Detection", type="primary", use_container_width=True)

    with col_btn2:
        if existing_status:
            load_existing = st.button("Load Saved", type="secondary", use_container_width=True)
            if load_existing:
                saved_detections = load_detections(storage, selected_video)
                if saved_detections:
                    st.session_state[detection_key] = saved_detections
                    st.success(f"Loaded {existing_status['total']} saved detections")
                    st.rerun()

    with col_info:
        if existing_status:
            st.caption(
                f"Saved: {existing_status['total']} detections "
                f"({existing_status['approved']} approved, "
                f"{existing_status['pending']} pending, "
                f"{existing_status['rejected']} rejected)"
            )

    # Run detection button with styled appearance
    if not st.session_state.detection_running:
        pass  # run_new already set by button above
    else:
        run_new = False

    if run_new:
        st.session_state.detection_running = True
        if not storage.exists(model_name, "models"):
            st.session_state.detection_running = False
            st.error("No model found. Train a model first in the TRAINING tab.")
            return

        try:
            import time
            import tempfile
            import os

            progress_info = st.empty()
            progress_info.markdown("**Downloading model and video...**")

            # Download model and video to local cache
            local_model_path = storage.read_model(model_name)
            local_video_path = storage.read_video(selected_video)

            progress_info.markdown("**Running detection in background process...**")

            # Run detection in a subprocess to isolate memory usage.
            # If PyTorch OOMs, the subprocess dies but Streamlit survives.
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                output_path = tmp.name

            detect_script = f"""
import json, sys, os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import time
start = time.time()
from src.inference.predictor import EventPredictor
predictor = EventPredictor.from_checkpoint(
    "{local_model_path}",
    target_fps={target_fps},
    stride={stride},
    batch_size={batch_size},
    num_workers=0,
)
result = predictor.predict_video(
    "{local_video_path}",
    threshold={threshold},
    buffer_seconds=2.0,
)
elapsed = time.time() - start
data = {{
    "deliveries": [
        {{
            "id": int(d.id),
            "start_time": float(d.start_time),
            "end_time": float(d.end_time),
            "confidence": float(d.confidence),
            "status": "pending",
        }}
        for d in result.events
    ],
    "video_path": "{selected_video}",
    "fps": float(result.fps),
    "threshold": float({threshold}),
    "detected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "elapsed": elapsed,
}}
with open("{output_path}", "w") as f:
    json.dump(data, f)
"""
            proc = subprocess.run(
                [sys.executable, "-c", detect_script],
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min timeout
            )

            if proc.returncode != 0:
                st.session_state.detection_running = False
                stderr = proc.stderr[-500:] if proc.stderr else "No error output"
                if proc.returncode == -9 or "Killed" in stderr or "oom" in stderr.lower():
                    st.error("Detection ran out of memory. Try reducing batch size or video length.")
                else:
                    st.error(f"Detection failed (exit code {proc.returncode})")
                with st.expander("Error details"):
                    st.code(stderr)
                try:
                    os.unlink(output_path)
                except OSError:
                    pass
                return

            # Load results from subprocess output
            with open(output_path) as f:
                detection_data = json.load(f)
            os.unlink(output_path)

            elapsed = detection_data.pop("elapsed", 0)
            st.session_state[detection_key] = detection_data

            # Save to storage for persistence
            save_detections(storage, selected_video, detection_data)

            st.session_state.detection_running = False
            num_events = len(detection_data["deliveries"])
            st.success(f"Found {num_events} events in {elapsed:.1f}s! Results saved.")
            st.rerun()
        except subprocess.TimeoutExpired:
            st.session_state.detection_running = False
            st.error("Detection timed out (30 min limit).")
            return
        except Exception as e:
            st.session_state.detection_running = False
            import traceback
            st.error(f"Detection failed: {e}")
            with st.expander("Error details"):
                st.code(traceback.format_exc())
            return

    # Show detections if available
    if detection_key not in st.session_state:
        st.info("Click 'Run Detection' to analyze the video")
        return

    detections = st.session_state[detection_key]
    deliveries = detections["deliveries"]
    fps = detections["fps"]

    if not deliveries:
        st.warning("No deliveries detected. Try lowering the threshold.")
        return

    st.divider()

    # Summary stats with custom metric cards
    pending = sum(1 for d in deliveries if d["status"] == "pending")
    approved = sum(1 for d in deliveries if d["status"] == "approved")
    rejected = sum(1 for d in deliveries if d["status"] == "rejected")

    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.metric("Total", len(deliveries))
    with stat_cols[1]:
        st.metric("Pending", pending)
    with stat_cols[2]:
        st.metric("Approved", approved)
    with stat_cols[3]:
        st.metric("Rejected", rejected)

    st.divider()

    # Delivery list and viewer
    list_col, viewer_col = st.columns([1, 2])

    with list_col:
        st.markdown("### Detections")

        # Filter options
        filter_status = st.radio(
            "Filter",
            ["All", "Pending", "Approved", "Rejected"],
            horizontal=True,
            key="analysis_filter",
        )

        # Filter deliveries
        filtered = deliveries
        if filter_status == "Pending":
            filtered = [d for d in deliveries if d["status"] == "pending"]
        elif filter_status == "Approved":
            filtered = [d for d in deliveries if d["status"] == "approved"]
        elif filter_status == "Rejected":
            filtered = [d for d in deliveries if d["status"] == "rejected"]

        # Initialize selected delivery
        if "selected_delivery_idx" not in st.session_state:
            st.session_state.selected_delivery_idx = 0

        # Delivery list as styled buttons
        for i, delivery in enumerate(filtered):
            # Find actual index in original list
            actual_idx = deliveries.index(delivery)

            status_icon = {
                "pending": "PENDING",
                "approved": "APPROVED",
                "rejected": "REJECTED",
            }.get(delivery["status"], "PENDING")

            start_min = int(delivery["start_time"] // 60)
            start_sec = int(delivery["start_time"] % 60)
            conf_pct = int(delivery["confidence"] * 100)

            btn_label = f"#{i+1} | {start_min}:{start_sec:02d} | {conf_pct}%"

            is_selected = actual_idx == st.session_state.selected_delivery_idx

            if st.button(
                btn_label,
                key=f"sel_{actual_idx}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
            ):
                st.session_state.selected_delivery_idx = actual_idx
                st.rerun()

    with viewer_col:
        st.markdown(f"### {domain.title()} Viewer")

        if st.session_state.selected_delivery_idx < len(deliveries):
            selected = deliveries[st.session_state.selected_delivery_idx]

            # Display info with metrics
            info_cols = st.columns(3)
            with info_cols[0]:
                st.metric("Time", f"{selected['start_time']:.1f}s - {selected['end_time']:.1f}s")
            with info_cols[1]:
                st.metric("Confidence", f"{selected['confidence']*100:.1f}%")
            with info_cols[2]:
                # Status badge
                st.markdown(f"**Status:** {status_badge(selected['status'])}", unsafe_allow_html=True)

            # Video clip player
            start_time = max(0, selected["start_time"] - 2)

            # Use ffmpeg to extract clip to temp file
            import tempfile

            clip_duration = (selected["end_time"] - selected["start_time"]) + 4

            # Download video to local cache for ffmpeg processing
            local_video_path = storage.read_video(selected_video)

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                subprocess.run([
                    "ffmpeg", "-y", "-ss", str(start_time), "-i", str(local_video_path),
                    "-t", str(clip_duration), "-c:v", "libx264", "-c:a", "aac",
                    "-loglevel", "error", tmp_path
                ], check=True, capture_output=True)

                st.video(tmp_path)
            except Exception as e:
                st.error(f"Could not extract clip: {e}")
            finally:
                # Clean up temp file after a delay (Streamlit needs time to read it)
                pass

            st.divider()

            # Action buttons with custom styling
            action_cols = st.columns(3)

            with action_cols[0]:
                if st.button("Approve", type="primary", use_container_width=True,
                           disabled=selected["status"] == "approved", key="approve_btn"):
                    # Mark as approved and add to training labels
                    selected["status"] = "approved"

                    # Save detection status to storage
                    save_detections(storage, selected_video, detections)

                    # Add to labels file
                    labels_key = f"{Path(selected_video).stem}.json"
                    if storage.labels_exist(labels_key):
                        labels = VideoLabels.from_dict(storage.read_labels(labels_key))
                    else:
                        local_video_path = storage.read_video(selected_video)
                        from src.utils.video import get_video_metadata
                        metadata = get_video_metadata(str(local_video_path))
                        labels = VideoLabels.from_metadata(metadata)

                    # Check if already exists (by time overlap)
                    start_frame = int(selected["start_time"] * fps)
                    end_frame = int(selected["end_time"] * fps)

                    already_exists = any(
                        abs(d.start_frame - start_frame) < fps * 2  # Within 2 seconds
                        for d in labels.deliveries
                    )

                    if not already_exists:
                        labels.add_delivery(start_frame, end_frame)
                        storage.write_labels(labels.to_dict(), labels_key)
                        st.success("Added to training set!")
                    else:
                        st.info("Already in training set")

                    st.rerun()

            with action_cols[1]:
                if st.button("Reject", type="secondary", use_container_width=True,
                           disabled=selected["status"] == "rejected", key="reject_btn"):
                    # Mark as rejected and add as false positive
                    selected["status"] = "rejected"

                    # Save detection status to storage
                    save_detections(storage, selected_video, detections)

                    # Add to labels file as false positive
                    labels_key = f"{Path(selected_video).stem}.json"
                    if storage.labels_exist(labels_key):
                        labels = VideoLabels.from_dict(storage.read_labels(labels_key))
                    else:
                        local_video_path = storage.read_video(selected_video)
                        from src.utils.video import get_video_metadata
                        metadata = get_video_metadata(str(local_video_path))
                        labels = VideoLabels.from_metadata(metadata)

                    start_frame = int(selected["start_time"] * fps)
                    end_frame = int(selected["end_time"] * fps)

                    # Check if already exists
                    already_exists = any(
                        abs(fp.start_frame - start_frame) < fps * 2
                        for fp in labels.false_positives
                    )

                    if not already_exists:
                        labels.false_positives.append(FalsePositive(
                            id=f"fp_{uuid.uuid4().hex[:8]}",
                            start_frame=start_frame,
                            end_frame=end_frame,
                            notes="Rejected from analysis UI",
                        ))
                        storage.write_labels(labels.to_dict(), labels_key)
                        st.success("Added as false positive for training!")
                    else:
                        st.info("Already marked as false positive")

                    st.rerun()

            with action_cols[2]:
                if st.button("Next", use_container_width=True, key="next_detection"):
                    if st.session_state.selected_delivery_idx < len(deliveries) - 1:
                        st.session_state.selected_delivery_idx += 1
                        st.rerun()

            # Navigation hints
            st.caption("Approve = add to training positives | Reject = add to training negatives")

    # Save progress button
    st.divider()
    save_cols = st.columns([3, 1])
    with save_cols[1]:
        if st.button("Save Progress", use_container_width=True, key="save_progress"):
            # Save current detection state to storage
            save_detections(storage, selected_video, detections)
            st.success("Progress saved!")


def get_labeled_videos(storage: StorageBackend) -> list[dict]:
    """Get all labeled videos with stats.

    Args:
        storage: Storage backend instance.

    Returns:
        List of dicts with video info and stats.
    """
    videos = []
    available_videos = set(storage.list_videos(pattern="*.mp4,*.mov"))

    for label_name in storage.list_labels():
        try:
            label_data = storage.read_labels(label_name)
            labels = VideoLabels.from_dict(label_data)

            # Determine video name from label filename
            video_stem = Path(label_name).stem
            video_name = f"{video_stem}.mp4"
            # Check if the video exists in storage
            video_exists = video_name in available_videos

            # Calculate duration
            duration_seconds = labels.total_frames / labels.fps if labels.fps > 0 else 0
            duration_min = int(duration_seconds // 60)
            duration_sec = int(duration_seconds % 60)

            videos.append({
                "label_file": label_name,
                "video_name": video_name,
                "video_exists": video_exists,
                "num_events": len(labels.deliveries),
                "num_false_positives": len(labels.false_positives) if labels.false_positives else 0,
                "duration": f"{duration_min}:{duration_sec:02d}",
                "duration_seconds": duration_seconds,
                "fps": labels.fps,
                "total_frames": labels.total_frames,
            })
        except Exception as e:
            print(f"Error loading {label_name}: {e}")
            continue

    return sorted(videos, key=lambda v: v["video_name"])


def get_previous_models(storage: StorageBackend) -> list[dict]:
    """Get list of previous training runs from models directory.

    Args:
        storage: Storage backend instance.

    Returns:
        List of dicts with model info.
    """
    models = []

    for model_name in storage.list_models():
        try:
            # Download to cache and load checkpoint to get metadata
            import torch
            local_model_path = storage.read_model(model_name)
            checkpoint = torch.load(local_model_path, map_location="cpu", weights_only=False)
            metrics = checkpoint.get("metrics", {})
            domain = checkpoint.get("domain", "unknown")
            epoch = checkpoint.get("epoch", 0)

            models.append({
                "name": Path(model_name).stem,
                "storage_key": model_name,
                "domain": domain,
                "epoch": epoch,
                "val_f1": metrics.get("val_f1", 0),
                "val_accuracy": metrics.get("val_accuracy", 0),
                "modified": local_model_path.stat().st_mtime,
            })
        except Exception as e:
            # If we can't load it, just add basic info
            models.append({
                "name": Path(model_name).stem,
                "storage_key": model_name,
                "domain": "unknown",
                "epoch": 0,
                "val_f1": 0,
                "val_accuracy": 0,
                "modified": 0,
            })

    return sorted(models, key=lambda m: m["modified"], reverse=True)


def run_training_tab(domain: str = "cricket"):
    """Run the training tab for model training and management."""
    st.markdown(f"## Training {domain_badge(domain)}", unsafe_allow_html=True)

    storage: StorageBackend = st.session_state.storage

    # Initialize session state
    if "training_selected_videos" not in st.session_state:
        st.session_state.training_selected_videos = set()
    if "training_in_progress" not in st.session_state:
        st.session_state.training_in_progress = False
    if "training_progress" not in st.session_state:
        st.session_state.training_progress = {}
    if "training_stop_requested" not in st.session_state:
        st.session_state.training_stop_requested = False

    # ========== VIDEO SELECTION SECTION ==========
    st.markdown("### Training Videos")

    # Get labeled videos
    labeled_videos = get_labeled_videos(storage)

    if not labeled_videos:
        st.warning("No labeled videos found. Label some videos first in the LABELING tab.")
        return

    # Video selection with stats
    col_select, col_stats = st.columns([3, 1])

    with col_select:
        # Select/deselect all buttons
        btn_cols = st.columns([1, 1, 4])
        with btn_cols[0]:
            if st.button("Select All", use_container_width=True, key="select_all_videos"):
                for v in labeled_videos:
                    if v["video_exists"]:
                        st.session_state.training_selected_videos.add(str(v["label_file"]))
                st.rerun()
        with btn_cols[1]:
            if st.button("Clear All", use_container_width=True, key="clear_all_videos"):
                st.session_state.training_selected_videos.clear()
                st.rerun()

        # Video list with checkboxes
        for video in labeled_videos:
            label_key = str(video["label_file"])
            col1, col2, col3, col4 = st.columns([0.5, 3, 1.5, 0.5])

            with col1:
                # Checkbox
                is_selected = label_key in st.session_state.training_selected_videos
                disabled = not video["video_exists"]
                new_selected = st.checkbox(
                    "select",
                    value=is_selected,
                    key=f"video_select_{label_key}",
                    label_visibility="collapsed",
                    disabled=disabled,
                )
                if new_selected and not is_selected:
                    st.session_state.training_selected_videos.add(label_key)
                elif not new_selected and is_selected:
                    st.session_state.training_selected_videos.discard(label_key)

            with col2:
                # Video name
                name = video["video_name"]
                if len(name) > 40:
                    name = name[:37] + "..."
                st.markdown(f"**{name}**")

            with col3:
                # Stats
                st.caption(f"{video['duration']} min | {video['num_events']} events")

            with col4:
                # Status icon
                if video["video_exists"]:
                    st.markdown("✓", help="Video file exists")
                else:
                    st.markdown("⚠", help="Video file not found")

    with col_stats:
        # Selection summary
        selected_count = len(st.session_state.training_selected_videos)
        total_events = sum(
            v["num_events"]
            for v in labeled_videos
            if str(v["label_file"]) in st.session_state.training_selected_videos
        )
        total_fp = sum(
            v["num_false_positives"]
            for v in labeled_videos
            if str(v["label_file"]) in st.session_state.training_selected_videos
        )
        total_duration = sum(
            v["duration_seconds"]
            for v in labeled_videos
            if str(v["label_file"]) in st.session_state.training_selected_videos
        )

        st.metric("Videos Selected", selected_count)
        st.metric("Total Events", total_events)
        if total_fp > 0:
            st.metric("False Positives", total_fp)
        st.metric("Total Duration", f"{int(total_duration // 60)}m")

    st.divider()

    # ========== TRAINING CONFIGURATION SECTION ==========
    with st.expander("Training Settings", expanded=True):
        config_col1, config_col2, config_col3 = st.columns(3)

        with config_col1:
            backbone = st.selectbox(
                "Backbone",
                ["resnet18", "resnet34", "resnet50"],
                index=0,
                help="CNN backbone for feature extraction"
            )
            num_epochs = st.slider(
                "Epochs",
                min_value=10,
                max_value=100,
                value=50,
                step=5,
                help="Number of training epochs"
            )

        with config_col2:
            batch_size = st.select_slider(
                "Batch Size",
                options=[4, 8, 16, 32],
                value=8,
                help="Samples per batch"
            )
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                value=1e-4,
                format_func=lambda x: f"{x:.0e}",
                help="Initial learning rate"
            )

        with config_col3:
            early_stopping = st.slider(
                "Early Stopping Patience",
                min_value=5,
                max_value=20,
                value=10,
                help="Stop if no improvement for N epochs"
            )
            window_size = st.select_slider(
                "Window Size",
                options=[4, 8, 12, 16],
                value=8,
                help="Number of frames per window"
            )

        # Additional settings row
        extra_col1, extra_col2 = st.columns(2)
        with extra_col1:
            target_fps = st.select_slider(
                "Target FPS",
                options=[5.0, 10.0, 15.0],
                value=10.0,
                help="Frame extraction rate"
            )
        with extra_col2:
            experiment_name = st.text_input(
                "Experiment Name",
                value=f"{domain}_detector",
                help="Name for saving the model"
            )

    st.divider()

    # ========== TRAINING CONTROLS SECTION ==========
    st.markdown("### Training Controls")

    control_col1, control_col2 = st.columns([1, 3])

    with control_col1:
        # Start/Stop buttons
        can_start = selected_count > 0 and not st.session_state.training_in_progress

        if st.button(
            "Start Training",
            type="primary",
            use_container_width=True,
            disabled=not can_start,
            key="start_training_btn"
        ):
            # Start training
            st.session_state.training_in_progress = True
            st.session_state.training_stop_requested = False
            st.session_state.training_progress = {
                "epoch": 0,
                "total_epochs": num_epochs,
                "train_loss": 0,
                "val_loss": 0,
                "train_acc": 0,
                "val_acc": 0,
                "val_f1": 0,
                "status": "initializing",
            }
            st.rerun()

        if st.session_state.training_in_progress:
            if st.button(
                "Stop Training",
                type="secondary",
                use_container_width=True,
                key="stop_training_btn"
            ):
                st.session_state.training_stop_requested = True
                st.session_state.training_progress["status"] = "stopping"
                st.rerun()

    with control_col2:
        if not can_start and not st.session_state.training_in_progress:
            st.info("Select at least one video to start training")

    # ========== TRAINING PROGRESS SECTION ==========
    if st.session_state.training_in_progress:
        st.markdown("### Training Progress")

        progress = st.session_state.training_progress

        # Progress bar
        progress_pct = progress["epoch"] / progress["total_epochs"] if progress["total_epochs"] > 0 else 0
        st.progress(progress_pct)
        st.markdown(f"**Epoch {progress['epoch']}/{progress['total_epochs']}** - {progress.get('status', 'training')}")

        # Metrics display
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Train Loss", f"{progress.get('train_loss', 0):.4f}")
        with metric_cols[1]:
            st.metric("Val Loss", f"{progress.get('val_loss', 0):.4f}")
        with metric_cols[2]:
            st.metric("Val Accuracy", f"{progress.get('val_acc', 0)*100:.1f}%")
        with metric_cols[3]:
            st.metric("Val F1", f"{progress.get('val_f1', 0):.3f}")

        # Run training in background
        if progress.get("status") == "initializing":
            with st.spinner("Starting training..."):
                try:
                    # Import training modules
                    from src.training.trainer import TrainingConfig, train_model_with_progress
                    from src.data.dataset import create_train_val_split_from_files

                    # Get selected label files
                    selected_files = [Path(f) for f in st.session_state.training_selected_videos]

                    # Create training config
                    config = TrainingConfig(
                        domain=domain,
                        labels_dir=str(storage.config.get_local_path("labels")),
                        videos_dir=str(storage.config.get_local_path("videos")),
                        backbone=backbone,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        early_stopping_patience=early_stopping,
                        window_size=window_size,
                        target_fps=target_fps,
                        experiment_name=experiment_name,
                    )

                    # Progress callback
                    def update_progress(metrics: dict):
                        st.session_state.training_progress.update(metrics)
                        st.session_state.training_progress["status"] = "training"
                        # Check for stop request
                        return not st.session_state.training_stop_requested

                    # Run training
                    st.session_state.training_progress["status"] = "training"
                    result = train_model_with_progress(
                        config,
                        label_files=selected_files,
                        progress_callback=update_progress,
                    )

                    if result:
                        st.session_state.training_progress["status"] = "completed"
                        st.success(f"Training completed! Model saved as {experiment_name}_best.pt")
                    else:
                        st.session_state.training_progress["status"] = "stopped"
                        st.warning("Training stopped by user")

                except ImportError:
                    st.session_state.training_progress["status"] = "error"
                    st.error("Training modules not fully implemented yet. See training progress callback implementation.")
                except Exception as e:
                    st.session_state.training_progress["status"] = "error"
                    st.error(f"Training failed: {e}")
                finally:
                    st.session_state.training_in_progress = False

    st.divider()

    # ========== TRAINING HISTORY SECTION ==========
    st.markdown("### Previous Models")

    previous_models = get_previous_models(storage)

    if not previous_models:
        st.info("No trained models found. Train a model to see it here.")
    else:
        for model in previous_models[:5]:  # Show last 5 models
            with st.expander(f"{model['name']}", expanded=False):
                info_cols = st.columns(4)
                with info_cols[0]:
                    st.caption("Domain")
                    st.markdown(f"**{model['domain']}**")
                with info_cols[1]:
                    st.caption("Epoch")
                    st.markdown(f"**{model['epoch']}**")
                with info_cols[2]:
                    st.caption("Val F1")
                    st.markdown(f"**{model['val_f1']:.3f}**")
                with info_cols[3]:
                    st.caption("Val Accuracy")
                    st.markdown(f"**{model['val_accuracy']*100:.1f}%**")

                # Load model button
                if st.button("Use this model", key=f"load_model_{model['name']}", use_container_width=False):
                    # Copy to the default model key
                    local_model_path = storage.read_model(model["storage_key"])
                    storage.write_model(local_model_path, "delivery_detector_best.pt")
                    st.success(f"Model loaded! Now available for detection in Analysis tab.")


def run_labeler():
    """Run the Streamlit labeling application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="cricket")
    # Streamlit passes args after --
    args = parser.parse_args(sys.argv[1:]) if "--" not in sys.argv else parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    domain = args.domain

    # Initialize storage backend (shared across tabs via session state)
    if "storage" not in st.session_state:
        st.session_state.storage = get_storage_backend()

    st.set_page_config(
        page_title=f"Prismata | {domain.title()}",
        page_icon="P",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject custom theme CSS
    inject_theme()

    # Header with gradient title and domain badge
    styled_header("Prismata", domain)
    st.caption("Multi-domain video event detection")

    # Main tabs with cleaner styling
    tab1, tab2, tab3 = st.tabs(["LABELING", "ANALYSIS", "TRAINING"])

    with tab2:
        run_analysis_tab(domain)

    with tab3:
        run_training_tab(domain)

    with tab1:
        run_labeling_tab(domain)


def run_labeling_tab(domain: str = "cricket"):
    """Run the labeling tab content."""
    # Check for keyboard action from URL params
    query_params = st.query_params
    kb_action = query_params.get("kb_action", None)

    # Clear the action from URL to prevent repeated triggers
    if kb_action:
        st.query_params.clear()

    storage: StorageBackend = st.session_state.storage

    # Sidebar for file selection and shortcuts help
    with st.sidebar:
        st.markdown("### Video Source")

        # Tab selection for video source with styled radio
        source_tab = st.radio(
            "Source",
            ["Library", "YouTube"],
            horizontal=True,
            label_visibility="collapsed",
        )

        video_path = None  # Local filesystem path for cv2/ffmpeg
        selected_video_name = None  # Storage key/name for the video

        st.divider()

        if source_tab == "YouTube":
            st.markdown("### YouTube Download")

            youtube_url = st.text_input(
                "YouTube URL",
                placeholder="https://youtube.com/watch?v=...",
                help="Paste a YouTube video URL",
                label_visibility="collapsed",
            )

            if youtube_url and is_youtube_url(youtube_url):
                # Video info
                with st.spinner("Fetching video info..."):
                    info = get_video_info(youtube_url)

                if info:
                    st.success(f"**{info.get('title', 'Video')}**")
                    duration = info.get('duration', 0)
                    st.caption(f"Duration: {duration // 60}:{duration % 60:02d}")

                    # Download options
                    col1, col2 = st.columns(2)
                    with col1:
                        quality = st.selectbox(
                            "Quality",
                            ["720", "480", "360", "1080"],
                            index=0,
                            help="Video height in pixels"
                        )
                    with col2:
                        custom_name = st.text_input(
                            "Filename",
                            value="",
                            placeholder="Auto from title",
                            help="Custom filename (optional)"
                        )

                    # Time range (optional)
                    st.caption("Time range (optional)")
                    col1, col2 = st.columns(2)
                    with col1:
                        start_time = st.text_input(
                            "Start",
                            placeholder="MM:SS",
                            help="e.g., 1:30 or 01:30:00"
                        )
                    with col2:
                        end_time = st.text_input(
                            "End",
                            placeholder="MM:SS",
                            help="e.g., 5:00 or 00:05:00"
                        )

                    # Split options
                    st.caption("Split into chunks (optional)")
                    split_enabled = st.checkbox(
                        "Split video into smaller files",
                        value=False,
                        help="Split the downloaded video into smaller chunks for easier labeling"
                    )
                    if split_enabled:
                        chunk_duration = st.select_slider(
                            "Chunk duration (minutes)",
                            options=[5, 10, 15, 20, 30],
                            value=10,
                            help="Duration of each chunk"
                        )
                        delete_original = st.checkbox(
                            "Delete original after splitting",
                            value=False,
                            help="Remove the full video after creating chunks"
                        )

                    # Download button
                    if st.button("Download Video", type="primary", use_container_width=True):
                        progress_placeholder = st.empty()

                        def update_progress(msg):
                            progress_placeholder.info(msg)

                        # Download to a temp local dir first, then upload to storage
                        import tempfile
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            with st.spinner("Downloading..."):
                                downloaded_path = download_youtube_video(
                                    url=youtube_url,
                                    output_dir=Path(tmp_dir),
                                    filename=custom_name if custom_name else None,
                                    quality=quality,
                                    start_time=start_time if start_time else None,
                                    end_time=end_time if end_time else None,
                                    progress_callback=update_progress,
                                )

                            if downloaded_path:
                                # Split if enabled
                                if split_enabled:
                                    with st.spinner("Splitting video..."):
                                        chunks = split_video(
                                            video_path=downloaded_path,
                                            chunk_duration_minutes=chunk_duration,
                                            output_dir=Path(tmp_dir),
                                            progress_callback=update_progress,
                                        )
                                    if chunks:
                                        # Upload all chunks to storage
                                        for chunk_path in chunks:
                                            storage.write_video(chunk_path, chunk_path.name)
                                        st.success(f"Uploaded {len(chunks)} chunks")
                                        if not delete_original:
                                            storage.write_video(downloaded_path, downloaded_path.name)
                                        st.session_state.downloaded_video = chunks[0].name
                                    else:
                                        st.warning("Splitting failed, uploading original file")
                                        storage.write_video(downloaded_path, downloaded_path.name)
                                        st.session_state.downloaded_video = downloaded_path.name
                                else:
                                    storage.write_video(downloaded_path, downloaded_path.name)
                                    st.session_state.downloaded_video = downloaded_path.name
                                st.rerun()
                            else:
                                st.error("Download failed. Check the URL and try again.")
                else:
                    st.error("Could not fetch video info. Check the URL.")
            elif youtube_url:
                st.warning("Please enter a valid YouTube URL")

            # Show recently downloaded
            if "downloaded_video" in st.session_state:
                st.divider()
                st.caption("Recently downloaded:")
                selected_video_name = st.session_state.downloaded_video
                st.code(selected_video_name, language=None)
                if st.button("Load this video", use_container_width=True):
                    pass  # selected_video_name is already set

        else:  # Library
            st.markdown("### Video Library")

            # List available videos from storage
            video_names = storage.list_videos(pattern="*.mp4,*.mov")
            if video_names:
                selected = st.selectbox(
                    "Select video",
                    [""] + video_names,
                    format_func=lambda x: "Choose..." if x == "" else x,
                    label_visibility="collapsed",
                )
                if selected:
                    selected_video_name = selected
            else:
                st.info("No videos found in storage.")

        # Validate and load video metadata
        metadata = None
        if selected_video_name:
            try:
                video_path = str(storage.read_video(selected_video_name))
                metadata = get_video_metadata(video_path)
                st.divider()
                st.markdown("### Video Info")
                st.markdown(f"**{selected_video_name}**")
                st.caption(f"{metadata.width}x{metadata.height} | {metadata.fps:.1f} fps")
                st.caption(f"{metadata.duration_str} | {metadata.total_frames} frames")
            except FileNotFoundError:
                st.warning("Video file not found in storage")
                video_path = None
                selected_video_name = None
            except Exception as e:
                st.error(f"Error loading video: {e}")
                video_path = None
                selected_video_name = None

        st.divider()

        # Keyboard shortcuts help in collapsible card
        with st.expander("Keyboard Shortcuts", expanded=False):
            st.caption("NAVIGATION")
            st.text("Left/A      Previous frame")
            st.text("Right/D     Next frame")
            st.text("Shift+Arrow Skip 10 frames")
            st.text("Ctrl+Arrow  Skip 100 frames")
            st.caption("LABELING")
            st.text("S           Mark start")
            st.text("E           Mark end")
            st.text("Esc/C       Cancel marking")
            st.caption("SAVE")
            st.text("Ctrl+S      Save labels")

    # Show instructions if no video loaded
    if not video_path or not metadata:
        st.info("Select a video source from the sidebar to begin labeling.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### Video Library
            Select from videos in storage
            """)
        with col2:
            st.markdown("""
            ### YouTube
            1. Paste a YouTube URL
            2. Set quality and time range (optional)
            3. Click Download
            4. Start labeling!
            """)
        return

    # Initialize session state
    labels_key = f"{Path(selected_video_name).stem}.json"
    if "labels" not in st.session_state or st.session_state.get("current_video") != selected_video_name:
        if storage.labels_exist(labels_key):
            st.session_state.labels = VideoLabels.from_dict(storage.read_labels(labels_key))
            st.sidebar.success("Loaded existing labels")
        else:
            st.session_state.labels = VideoLabels.from_metadata(metadata)
        st.session_state.current_video = selected_video_name
        st.session_state.current_frame = 0
        st.session_state.marking_start = None
        st.session_state.status_message = None

    labels: VideoLabels = st.session_state.labels
    current_frame = st.session_state.current_frame

    def auto_save():
        """Auto-save labels after modifications."""
        storage.write_labels(labels.to_dict(), labels_key)
        st.session_state.status_message = f"Auto-saved labels"

    # Process keyboard action
    if kb_action:
        max_frame = labels.total_frames - 1
        if kb_action == "prev_1":
            st.session_state.current_frame = max(0, current_frame - 1)
        elif kb_action == "prev_10":
            st.session_state.current_frame = max(0, current_frame - 10)
        elif kb_action == "prev_100":
            st.session_state.current_frame = max(0, current_frame - 100)
        elif kb_action == "next_1":
            st.session_state.current_frame = min(max_frame, current_frame + 1)
        elif kb_action == "next_10":
            st.session_state.current_frame = min(max_frame, current_frame + 10)
        elif kb_action == "next_100":
            st.session_state.current_frame = min(max_frame, current_frame + 100)
        elif kb_action == "mark_start":
            st.session_state.marking_start = current_frame
            st.session_state.status_message = f"Start marked at frame {current_frame}"
        elif kb_action == "mark_end":
            if st.session_state.marking_start is not None:
                if current_frame > st.session_state.marking_start:
                    labels.add_delivery(st.session_state.marking_start, current_frame)
                    auto_save()
                    st.session_state.status_message = f"Delivery added & saved: {st.session_state.marking_start} -> {current_frame}"
                    st.session_state.marking_start = None
                else:
                    st.session_state.status_message = "End frame must be after start frame"
            else:
                st.session_state.status_message = "Mark start first (press S)"
        elif kb_action == "cancel":
            if st.session_state.marking_start is not None:
                st.session_state.marking_start = None
                st.session_state.status_message = "Marking cancelled"
        elif kb_action == "save":
            storage.write_labels(labels.to_dict(), labels_key)
            st.session_state.status_message = "Labels saved"
        # Rerun to apply changes
        st.rerun()

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Cannot open video file")
        return

    # Inject keyboard listener for navigation shortcuts
    keyboard_listener()

    # Hidden refresh button for keyboard events (styled to blend in)
    st.button("REFRESH", key="refresh_btn", help="Refresh (used by keyboard shortcuts)", type="secondary")

    # Status message bar
    if st.session_state.get("status_message"):
        msg = st.session_state.status_message
        if "added" in msg.lower() or "saved" in msg.lower():
            st.success(msg)
        elif "must" in msg.lower() or "first" in msg.lower():
            st.error(msg)
        elif "cancelled" in msg.lower():
            st.warning(msg)
        else:
            st.info(msg)
        st.session_state.status_message = None

    # Navigation callback functions
    def go_to_frame(frame: int):
        st.session_state.current_frame = max(0, min(labels.total_frames - 1, frame))

    def on_slider_change():
        st.session_state.current_frame = st.session_state.frame_slider

    def nav_prev_100():
        go_to_frame(st.session_state.current_frame - 100)

    def nav_prev_10():
        go_to_frame(st.session_state.current_frame - 10)

    def nav_prev_1():
        go_to_frame(st.session_state.current_frame - 1)

    def nav_next_1():
        go_to_frame(st.session_state.current_frame + 1)

    def nav_next_10():
        go_to_frame(st.session_state.current_frame + 10)

    def nav_next_100():
        go_to_frame(st.session_state.current_frame + 100)

    # Main layout
    col1, col2 = st.columns([3, 1])

    # Sync slider state with current_frame before rendering
    if "frame_slider" not in st.session_state:
        st.session_state.frame_slider = current_frame
    elif st.session_state.frame_slider != current_frame:
        st.session_state.frame_slider = current_frame

    with col1:
        # View mode tabs
        view_mode = st.radio(
            "View Mode",
            ["Frame View", "Video Playback"],
            horizontal=True,
            help="Use Frame View for precise labeling, Video Playback to watch the video",
        )

        if view_mode == "Video Playback":
            # Native HTML5 video player for smooth playback
            # Use presigned URL for S3 streaming, local path for local storage
            video_url = storage.get_video_url(selected_video_name)
            st.video(video_url)
            st.caption("Use the video controls above to play. Switch to Frame View to label specific frames.")
        else:
            # Frame navigation slider with purple accent
            st.slider(
                "Frame",
                0,
                labels.total_frames - 1,
                key="frame_slider",
                on_change=on_slider_change,
            )

            # Display current frame in container
            frame_data, actual_pos = get_frame(cap, current_frame)
            if frame_data is not None and frame_data.size > 0:
                # Add glow effect if marking
                if st.session_state.marking_start is not None:
                    st.markdown(
                        f'<div style="border: 2px solid {COLORS["emerald"]}; border-radius: 12px; '
                        f'box-shadow: 0 0 20px rgba(16, 185, 129, 0.4); padding: 4px;">',
                        unsafe_allow_html=True
                    )
                st.image(frame_data, use_container_width=True)
                if st.session_state.marking_start is not None:
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error(f"Cannot read frame {current_frame} (actual pos: {actual_pos})")

        # Time and marking status display
        status_parts = [
            f"**Frame:** {current_frame}",
            f"**Time:** {format_time(current_frame, labels.fps)}",
        ]
        if st.session_state.marking_start is not None:
            status_parts.append(f"**Marking from:** {st.session_state.marking_start}")

        st.markdown(" | ".join(status_parts))

        # Navigation buttons in a clean row
        st.caption("Frame Navigation")
        nav_cols = st.columns(6)
        with nav_cols[0]:
            st.button("-100", help="Ctrl+Left", on_click=nav_prev_100, use_container_width=True)
        with nav_cols[1]:
            st.button("-10", help="Shift+Left", on_click=nav_prev_10, use_container_width=True)
        with nav_cols[2]:
            st.button("-1", help="Left or A", on_click=nav_prev_1, use_container_width=True)
        with nav_cols[3]:
            st.button("+1", help="Right or D", on_click=nav_next_1, use_container_width=True)
        with nav_cols[4]:
            st.button("+10", help="Shift+Right", on_click=nav_next_10, use_container_width=True)
        with nav_cols[5]:
            st.button("+100", help="Ctrl+Right", on_click=nav_next_100, use_container_width=True)

        st.divider()

        # Marking controls with prominent colored buttons
        st.markdown(f"### Mark {domain.title()} Event")
        mark_cols = st.columns(4)

        with mark_cols[0]:
            if st.button("Mark Start", type="primary", use_container_width=True, help="Press S"):
                st.session_state.marking_start = current_frame
                st.session_state.status_message = f"Start marked at frame {current_frame}"
                st.rerun()

        with mark_cols[1]:
            end_disabled = st.session_state.marking_start is None
            if st.button("Mark End", disabled=end_disabled, use_container_width=True, help="Press E"):
                if current_frame > st.session_state.marking_start:
                    labels.add_delivery(st.session_state.marking_start, current_frame)
                    auto_save()
                    st.session_state.status_message = f"Delivery added & saved!"
                    st.session_state.marking_start = None
                    st.rerun()
                else:
                    st.error("End frame must be after start frame")

        with mark_cols[2]:
            cancel_disabled = st.session_state.marking_start is None
            if st.button("Cancel", disabled=cancel_disabled, use_container_width=True, help="Press Esc or C"):
                st.session_state.marking_start = None
                st.rerun()

        with mark_cols[3]:
            if st.button("Save", type="secondary", use_container_width=True, help="Ctrl+S"):
                storage.write_labels(labels.to_dict(), labels_key)
                st.session_state.status_message = "Labels saved"
                st.rerun()

    with col2:
        # Deliveries list with styled cards
        st.markdown(f"### Events ({len(labels.deliveries)})")

        if not labels.deliveries:
            st.info("No events labeled yet.\nPress S to start marking.")

        for delivery in labels.deliveries:
            with st.expander(
                f"#{delivery.id} | {format_time(delivery.start_frame, labels.fps)}"
            ):
                st.caption(f"Start: {delivery.start_frame}")
                st.caption(f"End: {delivery.end_frame}")
                st.caption(f"Duration: {delivery.duration_frames()} frames ({delivery.duration_frames() / labels.fps:.2f}s)")

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Go to", key=f"goto_{delivery.id}", use_container_width=True):
                        st.session_state.current_frame = delivery.start_frame
                        st.rerun()
                with col_b:
                    if st.button("Delete", key=f"del_{delivery.id}", use_container_width=True):
                        labels.remove_delivery(delivery.id)
                        auto_save()
                        st.rerun()

        # Quick stats
        if labels.deliveries:
            st.divider()
            st.markdown("### Stats")
            total_duration = sum(d.duration_frames() for d in labels.deliveries) / labels.fps
            avg_duration = total_duration / len(labels.deliveries)

            st.metric("Total Labeled", f"{total_duration:.1f}s")
            st.metric("Avg Duration", f"{avg_duration:.2f}s")

    cap.release()


if __name__ == "__main__":
    run_labeler()
