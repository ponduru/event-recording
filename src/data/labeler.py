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

from src.utils.video import VideoMetadata, get_video_metadata


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
                    if (btn.innerText.includes('🔄')) {
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


def run_analysis_tab():
    """Run the analysis tab for reviewing model detections."""
    st.header("🔍 Video Analysis")

    # Settings
    col1, col2 = st.columns([2, 1])

    with col1:
        videos_dir = Path("data/raw")
        model_path = Path("models/delivery_detector_best.pt")
        labels_dir = Path("data/labels")

        # Video selection
        video_files = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.mov")) if videos_dir.exists() else []
        video_options = ["Select a video..."] + [f.name for f in video_files]
        selected_video = st.selectbox("Select Video", video_options)

        if selected_video == "Select a video...":
            st.info("Select a video to analyze")
            return

        video_path = videos_dir / selected_video

    with col2:
        threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)

        # Check if model exists
        if not model_path.exists():
            st.error("Model not found. Train a model first.")
            return

    # Detection state
    detection_key = f"detections_{selected_video}_{threshold}"

    # Run detection button
    if st.button("🚀 Run Detection", type="primary"):
        with st.spinner("Running detection..."):
            try:
                from src.inference.predictor import DeliveryPredictor

                predictor = DeliveryPredictor.from_checkpoint(str(model_path))
                result = predictor.predict_video(
                    str(video_path),
                    threshold=threshold,
                    buffer_seconds=2.0,
                )

                # Store in session state
                st.session_state[detection_key] = {
                    "deliveries": [
                        {
                            "id": d.id,
                            "start_time": d.start_time,
                            "end_time": d.end_time,
                            "confidence": d.confidence,
                            "status": "pending",  # pending, approved, rejected
                        }
                        for d in result.deliveries
                    ],
                    "video_path": str(video_path),
                    "fps": result.fps,
                }
                st.success(f"Found {len(result.deliveries)} deliveries!")
                st.rerun()
            except Exception as e:
                st.error(f"Detection failed: {e}")
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

    # Summary stats
    pending = sum(1 for d in deliveries if d["status"] == "pending")
    approved = sum(1 for d in deliveries if d["status"] == "approved")
    rejected = sum(1 for d in deliveries if d["status"] == "rejected")

    stat_cols = st.columns(4)
    stat_cols[0].metric("Total", len(deliveries))
    stat_cols[1].metric("Pending", pending)
    stat_cols[2].metric("Approved", approved, delta=None)
    stat_cols[3].metric("Rejected", rejected, delta=None)

    st.divider()

    # Delivery list and viewer
    list_col, viewer_col = st.columns([1, 2])

    with list_col:
        st.subheader("Detections")

        # Filter options
        filter_status = st.radio(
            "Filter",
            ["All", "Pending", "Approved", "Rejected"],
            horizontal=True,
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

        # Delivery list
        for i, delivery in enumerate(filtered):
            # Find actual index in original list
            actual_idx = deliveries.index(delivery)

            status_icon = {
                "pending": "⏳",
                "approved": "✅",
                "rejected": "❌",
            }.get(delivery["status"], "⏳")

            start_min = int(delivery["start_time"] // 60)
            start_sec = int(delivery["start_time"] % 60)
            conf_pct = int(delivery["confidence"] * 100)

            btn_label = f"{status_icon} #{i+1} | {start_min}:{start_sec:02d} | {conf_pct}%"

            if st.button(
                btn_label,
                key=f"sel_{actual_idx}",
                use_container_width=True,
                type="primary" if actual_idx == st.session_state.selected_delivery_idx else "secondary",
            ):
                st.session_state.selected_delivery_idx = actual_idx
                st.rerun()

    with viewer_col:
        st.subheader("Delivery Viewer")

        if st.session_state.selected_delivery_idx < len(deliveries):
            selected = deliveries[st.session_state.selected_delivery_idx]

            # Display info
            info_cols = st.columns(3)
            info_cols[0].metric("Time", f"{selected['start_time']:.1f}s - {selected['end_time']:.1f}s")
            info_cols[1].metric("Confidence", f"{selected['confidence']*100:.1f}%")
            info_cols[2].metric("Status", selected["status"].title())

            # Video clip player
            start_time = max(0, selected["start_time"] - 2)

            # Use ffmpeg to extract clip to temp file
            import tempfile
            import subprocess

            clip_duration = (selected["end_time"] - selected["start_time"]) + 4

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                subprocess.run([
                    "ffmpeg", "-y", "-ss", str(start_time), "-i", str(video_path),
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

            # Action buttons
            action_cols = st.columns(3)

            with action_cols[0]:
                if st.button("✅ Approve", type="primary", use_container_width=True,
                           disabled=selected["status"] == "approved"):
                    # Mark as approved and add to training labels
                    selected["status"] = "approved"

                    # Add to labels file
                    labels_path = labels_dir / f"{video_path.stem}.json"
                    if labels_path.exists():
                        labels = VideoLabels.load(labels_path)
                    else:
                        from src.utils.video import get_video_metadata
                        metadata = get_video_metadata(str(video_path))
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
                        labels.save(labels_path)
                        st.success("Added to training set!")
                    else:
                        st.info("Already in training set")

                    st.rerun()

            with action_cols[1]:
                if st.button("❌ Reject", type="secondary", use_container_width=True,
                           disabled=selected["status"] == "rejected"):
                    # Mark as rejected and add as false positive
                    selected["status"] = "rejected"

                    # Add to labels file as false positive
                    labels_path = labels_dir / f"{video_path.stem}.json"
                    if labels_path.exists():
                        labels = VideoLabels.load(labels_path)
                    else:
                        from src.utils.video import get_video_metadata
                        metadata = get_video_metadata(str(video_path))
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
                        labels.save(labels_path)
                        st.success("Added as false positive for training!")
                    else:
                        st.info("Already marked as false positive")

                    st.rerun()

            with action_cols[2]:
                if st.button("⏭️ Next", use_container_width=True):
                    if st.session_state.selected_delivery_idx < len(deliveries) - 1:
                        st.session_state.selected_delivery_idx += 1
                        st.rerun()

            # Navigation hints
            st.caption("Approve = add to training positives | Reject = add to training negatives")

    # Save progress button
    st.divider()
    save_cols = st.columns([3, 1])
    with save_cols[1]:
        if st.button("💾 Save Progress", use_container_width=True):
            st.success("Progress saved in session!")


def run_labeler():
    """Run the Streamlit labeling application."""
    st.set_page_config(
        page_title="Cricket Delivery Labeler",
        page_icon="🏏",
        layout="wide",
    )

    st.title("🏏 Cricket Delivery Detector")

    # Main tabs
    tab1, tab2 = st.tabs(["📝 Labeling", "🔍 Analysis"])

    with tab2:
        run_analysis_tab()

    with tab1:
        run_labeling_tab()


def run_labeling_tab():
    """Run the labeling tab content."""
    # Check for keyboard action from URL params
    query_params = st.query_params
    kb_action = query_params.get("kb_action", None)

    # Clear the action from URL to prevent repeated triggers
    if kb_action:
        st.query_params.clear()

    # Sidebar for file selection and shortcuts help
    with st.sidebar:
        # Tab selection for video source
        source_tab = st.radio(
            "Video Source",
            ["Local File", "YouTube"],
            horizontal=True,
        )

        video_path = None
        labels_dir = st.text_input("Labels Directory", value="data/labels")
        videos_dir = st.text_input("Videos Directory", value="data/raw")

        if source_tab == "YouTube":
            st.subheader("📺 YouTube Download")

            youtube_url = st.text_input(
                "YouTube URL",
                placeholder="https://youtube.com/watch?v=...",
                help="Paste a YouTube video URL"
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

                    # Download button
                    if st.button("⬇️ Download Video", type="primary"):
                        progress_placeholder = st.empty()

                        def update_progress(msg):
                            progress_placeholder.info(msg)

                        with st.spinner("Downloading..."):
                            downloaded_path = download_youtube_video(
                                url=youtube_url,
                                output_dir=Path(videos_dir),
                                filename=custom_name if custom_name else None,
                                quality=quality,
                                start_time=start_time if start_time else None,
                                end_time=end_time if end_time else None,
                                progress_callback=update_progress,
                            )

                        if downloaded_path:
                            st.success(f"Downloaded: {downloaded_path.name}")
                            st.session_state.downloaded_video = str(downloaded_path)
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
                video_path = st.session_state.downloaded_video
                st.code(video_path, language=None)
                if st.button("📂 Load this video"):
                    pass  # video_path is already set

        else:  # Local File
            st.subheader("📁 Local Video")
            video_path = st.text_input("Video Path", placeholder="/path/to/video.mp4")

            # List available videos in videos_dir
            videos_path = Path(videos_dir)
            if videos_path.exists():
                video_files = list(videos_path.glob("*.mp4")) + list(videos_path.glob("*.mov"))
                if video_files:
                    st.caption(f"Videos in {videos_dir}:")
                    selected = st.selectbox(
                        "Select video",
                        [""] + [f.name for f in video_files],
                        format_func=lambda x: "Choose..." if x == "" else x,
                    )
                    if selected:
                        video_path = str(videos_path / selected)

        # Validate video path
        metadata = None
        if video_path and Path(video_path).exists():
            try:
                metadata = get_video_metadata(video_path)
                st.success(f"Loaded: {metadata.path.name}")
                st.text(f"Resolution: {metadata.width}x{metadata.height}")
                st.text(f"FPS: {metadata.fps:.2f}")
                st.text(f"Duration: {metadata.duration_str}")
                st.text(f"Frames: {metadata.total_frames}")
            except Exception as e:
                st.error(f"Error loading video: {e}")
                video_path = None
        elif video_path:
            st.warning("Video file not found")
            video_path = None

        # Keyboard shortcuts help
        with st.expander("⌨️ Keyboard Shortcuts"):
            st.markdown("""
            **Navigation (in Frame View):**
            - `←` / `A` : Previous frame
            - `→` / `D` : Next frame
            - `Shift + ←/→` : Skip 10 frames
            - `Ctrl + ←/→` : Skip 100 frames

            **Labeling:**
            - `S` : Mark start
            - `E` : Mark end
            - `Esc` / `C` : Cancel marking

            **Save:**
            - `Ctrl + S` : Save labels
            """)

    # Show instructions if no video loaded
    if not video_path or not metadata:
        st.info("👈 Select a video source from the sidebar to begin labeling.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### Local File
            1. Enter a video path, or
            2. Select from videos in your data directory
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
    if "labels" not in st.session_state or st.session_state.get("current_video") != video_path:
        labels_path = Path(labels_dir) / f"{Path(video_path).stem}.json"
        if labels_path.exists():
            st.session_state.labels = VideoLabels.load(labels_path)
            st.sidebar.success("Loaded existing labels")
        else:
            st.session_state.labels = VideoLabels.from_metadata(metadata)
        st.session_state.current_video = video_path
        st.session_state.current_frame = 0
        st.session_state.marking_start = None
        st.session_state.status_message = None

    labels: VideoLabels = st.session_state.labels
    current_frame = st.session_state.current_frame
    labels_path = Path(labels_dir) / f"{Path(video_path).stem}.json"

    def auto_save():
        """Auto-save labels after modifications."""
        labels.save(labels_path)
        st.session_state.status_message = f"💾 Auto-saved to {labels_path}"

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
            st.session_state.status_message = f"✅ Start marked at frame {current_frame}"
        elif kb_action == "mark_end":
            if st.session_state.marking_start is not None:
                if current_frame > st.session_state.marking_start:
                    labels.add_delivery(st.session_state.marking_start, current_frame)
                    auto_save()
                    st.session_state.status_message = f"✅ Delivery added & saved: {st.session_state.marking_start} → {current_frame}"
                    st.session_state.marking_start = None
                else:
                    st.session_state.status_message = "❌ End frame must be after start frame"
            else:
                st.session_state.status_message = "❌ Mark start first (press S)"
        elif kb_action == "cancel":
            if st.session_state.marking_start is not None:
                st.session_state.marking_start = None
                st.session_state.status_message = "🚫 Marking cancelled"
        elif kb_action == "save":
            labels_path = Path(labels_dir) / f"{Path(video_path).stem}.json"
            labels.save(labels_path)
            st.session_state.status_message = f"💾 Saved to {labels_path}"
        # Rerun to apply changes
        st.rerun()

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Cannot open video file")
        return

    # Inject keyboard listener for navigation shortcuts
    keyboard_listener()

    # Hidden refresh button for keyboard events
    col_hidden = st.columns([1])[0]
    with col_hidden:
        st.button("🔄", key="refresh_btn", help="Refresh (used by keyboard shortcuts)")

    # Status message bar
    if st.session_state.get("status_message"):
        if "✅" in st.session_state.status_message or "💾" in st.session_state.status_message:
            st.success(st.session_state.status_message)
        elif "❌" in st.session_state.status_message:
            st.error(st.session_state.status_message)
        else:
            st.info(st.session_state.status_message)
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
            ["Frame View (for labeling)", "Video Playback"],
            horizontal=True,
            help="Use Frame View for precise labeling, Video Playback to watch the video",
        )

        if view_mode == "Video Playback":
            # Native HTML5 video player for smooth playback
            st.video(video_path)
            st.caption("Use the video controls above to play. Switch to Frame View to label specific frames.")
        else:
            # Frame navigation slider
            st.slider(
                "Frame",
                0,
                labels.total_frames - 1,
                key="frame_slider",
                on_change=on_slider_change,
            )

            # Display current frame
            frame_data, actual_pos = get_frame(cap, current_frame)
            if frame_data is not None and frame_data.size > 0:
                st.image(frame_data, use_container_width=True)
            else:
                st.error(f"Cannot read frame {current_frame} (actual pos: {actual_pos})")

        # Time and marking status display
        status_text = f"Frame: {current_frame} | Time: {format_time(current_frame, labels.fps)}"
        if st.session_state.marking_start is not None:
            status_text += f" | 🟢 Marking from: {st.session_state.marking_start}"
        st.markdown(f"**{status_text}**")

        # Navigation buttons
        st.caption("Frame Navigation")
        nav_cols = st.columns(6)
        with nav_cols[0]:
            st.button("⏮️ -100", help="Ctrl+←", on_click=nav_prev_100)
        with nav_cols[1]:
            st.button("⏪ -10", help="Shift+←", on_click=nav_prev_10)
        with nav_cols[2]:
            st.button("◀️ -1", help="← or A", on_click=nav_prev_1)
        with nav_cols[3]:
            st.button("▶️ +1", help="→ or D", on_click=nav_next_1)
        with nav_cols[4]:
            st.button("⏩ +10", help="Shift+→", on_click=nav_next_10)
        with nav_cols[5]:
            st.button("⏭️ +100", help="Ctrl+→", on_click=nav_next_100)

        # Marking controls
        st.subheader("Mark Delivery")
        mark_cols = st.columns(4)

        with mark_cols[0]:
            if st.button("🟢 Mark Start (S)", type="primary"):
                st.session_state.marking_start = current_frame
                st.session_state.status_message = f"✅ Start marked at frame {current_frame}"
                st.rerun()

        with mark_cols[1]:
            end_disabled = st.session_state.marking_start is None
            if st.button("🔴 Mark End (E)", disabled=end_disabled):
                if current_frame > st.session_state.marking_start:
                    labels.add_delivery(st.session_state.marking_start, current_frame)
                    auto_save()
                    st.session_state.status_message = f"✅ Delivery added & saved!"
                    st.session_state.marking_start = None
                    st.rerun()
                else:
                    st.error("End frame must be after start frame")

        with mark_cols[2]:
            cancel_disabled = st.session_state.marking_start is None
            if st.button("❌ Cancel (Esc)", disabled=cancel_disabled):
                st.session_state.marking_start = None
                st.rerun()

        with mark_cols[3]:
            if st.button("💾 Save (Ctrl+S)", type="secondary"):
                labels_path = Path(labels_dir) / f"{Path(video_path).stem}.json"
                labels.save(labels_path)
                st.session_state.status_message = f"💾 Saved to {labels_path}"
                st.rerun()

    with col2:
        # Deliveries list
        st.subheader(f"Deliveries ({len(labels.deliveries)})")

        if not labels.deliveries:
            st.info("No deliveries labeled yet.\nPress S to start marking.")

        for delivery in labels.deliveries:
            with st.expander(
                f"#{delivery.id} | {format_time(delivery.start_frame, labels.fps)}"
            ):
                st.text(f"Start: {delivery.start_frame}")
                st.text(f"End: {delivery.end_frame}")
                st.text(f"Duration: {delivery.duration_frames()} frames")
                st.text(f"Duration: {delivery.duration_frames() / labels.fps:.2f}s")

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Go to", key=f"goto_{delivery.id}"):
                        st.session_state.current_frame = delivery.start_frame
                        st.rerun()
                with col_b:
                    if st.button("🗑️", key=f"del_{delivery.id}", help="Delete"):
                        labels.remove_delivery(delivery.id)
                        auto_save()
                        st.rerun()

        # Quick stats
        if labels.deliveries:
            st.divider()
            st.caption("Stats")
            total_duration = sum(d.duration_frames() for d in labels.deliveries) / labels.fps
            st.text(f"Total labeled: {total_duration:.1f}s")
            avg_duration = total_duration / len(labels.deliveries)
            st.text(f"Avg duration: {avg_duration:.2f}s")

    cap.release()


if __name__ == "__main__":
    run_labeler()
