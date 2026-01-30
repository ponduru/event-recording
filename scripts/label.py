"""CLI script to launch the video labeling tool."""

import click
import subprocess
import sys
from pathlib import Path


@click.command()
@click.option(
    "--domain",
    "-d",
    default="cricket",
    help="Domain name (cricket, soccer, warehouse)",
)
@click.option(
    "--video",
    "-v",
    type=click.Path(exists=True),
    help="Video file to label (optional, can also enter in UI)",
)
@click.option(
    "--labels-dir",
    "-l",
    type=click.Path(),
    default="data/labels",
    help="Directory to save label files",
)
@click.option(
    "--port",
    "-p",
    type=int,
    default=8501,
    help="Port to run Streamlit on",
)
def main(domain: str, video: str, labels_dir: str, port: int):
    """Launch the video labeling tool.

    This starts a Streamlit web application for labeling video events
    across multiple domains.

    Example usage:

        prismata-label --domain cricket

        prismata-label --domain soccer --video match.mp4

        python -m scripts.label --port 8502
    """
    # Ensure labels directory exists
    labels_path = Path(labels_dir)
    labels_path.mkdir(parents=True, exist_ok=True)

    click.echo(f"Starting Prismata Labeling Tool ({domain})...")
    click.echo(f"Labels will be saved to: {labels_path.absolute()}")
    click.echo()

    # Build streamlit command
    labeler_path = Path(__file__).parent.parent / "src" / "data" / "labeler.py"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(labeler_path),
        "--server.port",
        str(port),
        "--",
        "--domain",
        domain,
    ]

    if video:
        click.echo(f"Pre-loading video: {video}")

    click.echo(f"Open http://localhost:{port} in your browser")
    click.echo("Press Ctrl+C to stop")
    click.echo()

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        click.echo("\nLabeling tool stopped")


if __name__ == "__main__":
    main()
