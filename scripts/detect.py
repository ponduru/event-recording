"""CLI script for detecting events in videos."""

import click
from pathlib import Path

from src.inference.predictor import (
    EventPredictor,
    detect_and_extract,
    extract_event_clips,
)


@click.command()
@click.argument("video", type=click.Path(exists=True))
@click.option(
    "--checkpoint",
    "-m",
    type=click.Path(exists=True),
    required=True,
    help="Path to model checkpoint",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="outputs",
    help="Output directory for results",
)
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=0.5,
    help="Detection confidence threshold (0-1)",
)
@click.option(
    "--buffer",
    "-b",
    type=float,
    default=3.0,
    help="Seconds to add before/after detected delivery",
)
@click.option(
    "--extract-clips/--no-extract-clips",
    default=True,
    help="Extract video clips for detections",
)
@click.option(
    "--batch-size",
    type=int,
    default=8,
    help="Inference batch size",
)
@click.option(
    "--window-size",
    "-w",
    type=int,
    default=8,
    help="Number of frames per window",
)
@click.option(
    "--stride",
    "-s",
    type=int,
    default=4,
    help="Stride between windows",
)
@click.option(
    "--target-fps",
    type=float,
    default=10.0,
    help="Target FPS for frame extraction",
)
def main(
    video: str,
    checkpoint: str,
    output: str,
    threshold: float,
    buffer: float,
    extract_clips: bool,
    batch_size: int,
    window_size: int,
    stride: int,
    target_fps: float,
):
    """Detect events in a video.

    VIDEO is the path to the input video file.

    Example usage:

        python -m scripts.detect video.mp4 -m models/event_detector_best.pt

        python -m scripts.detect match.mp4 -m model.pt -o results/ --threshold 0.6
    """
    video_path = Path(video)
    output_dir = Path(output)

    click.echo(f"Video: {video_path}")
    click.echo(f"Model: {checkpoint}")
    click.echo(f"Threshold: {threshold}")
    click.echo(f"Buffer: {buffer}s")
    click.echo()

    # Run detection and extraction
    result, clip_paths = detect_and_extract(
        video_path=video_path,
        checkpoint_path=checkpoint,
        output_dir=output_dir,
        threshold=threshold,
        buffer_seconds=buffer,
        save_detections=True,
        extract_clips=extract_clips,
        batch_size=batch_size,
        window_size=window_size,
        stride=stride,
        target_fps=target_fps,
    )

    # Print results
    click.echo(f"\nDetected {len(result.events)} event(s)")
    click.echo()

    for event in result.events:
        click.echo(
            f"  Event #{event.id}: "
            f"{event.start_time:.2f}s - {event.end_time:.2f}s "
            f"(confidence: {event.confidence:.2f})"
        )

    click.echo()
    click.echo(f"Results saved to: {output_dir}")

    if extract_clips and clip_paths:
        click.echo(f"Clips saved to: {output_dir}/clips/")
        for path in clip_paths:
            click.echo(f"  - {path.name}")


@click.command("batch")
@click.argument("videos_dir", type=click.Path(exists=True))
@click.option(
    "--checkpoint",
    "-m",
    type=click.Path(exists=True),
    required=True,
    help="Path to model checkpoint",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="outputs",
    help="Output directory for results",
)
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=0.5,
    help="Detection confidence threshold",
)
@click.option(
    "--buffer",
    "-b",
    type=float,
    default=3.0,
    help="Seconds to add before/after detection",
)
@click.option(
    "--extract-clips/--no-extract-clips",
    default=True,
    help="Extract video clips",
)
@click.option(
    "--pattern",
    "-p",
    default="*.mp4",
    help="Glob pattern for video files",
)
def batch(
    videos_dir: str,
    checkpoint: str,
    output: str,
    threshold: float,
    buffer: float,
    extract_clips: bool,
    pattern: str,
):
    """Detect events in multiple videos.

    VIDEOS_DIR is the directory containing video files.

    Example:

        python -m scripts.detect batch videos/ -m model.pt -o results/
    """
    videos_dir = Path(videos_dir)
    output_dir = Path(output)

    video_files = list(videos_dir.glob(pattern))
    if not video_files:
        raise click.ClickException(f"No video files found matching {pattern}")

    click.echo(f"Found {len(video_files)} video(s)")
    click.echo()

    # Create predictor once
    predictor = EventPredictor.from_checkpoint(checkpoint)

    total_deliveries = 0

    for video_path in video_files:
        click.echo(f"Processing: {video_path.name}")

        # Create video-specific output directory
        video_output_dir = output_dir / video_path.stem

        # Run detection
        result = predictor.predict_video(
            video_path,
            threshold=threshold,
            buffer_seconds=buffer,
        )

        # Save detections
        result.save(video_output_dir / "detections.json")

        # Extract clips
        if extract_clips and result.events:
            extract_event_clips(result, video_output_dir / "clips")

        click.echo(f"  Found {len(result.events)} event(s)")
        total_deliveries += len(result.events)

    click.echo()
    click.echo(f"Total: {total_deliveries} events in {len(video_files)} videos")
    click.echo(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
