"""Job runner for GPU instances.

This script is executed on EC2 GPU instances to run training or inference jobs.
It reads job configuration from a JSON file and executes the appropriate task.
"""

import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path


def setup_environment():
    """Set up environment for job execution."""
    # Ensure GPU is available
    import torch

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available, running on CPU")
    else:
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def download_s3_data(s3_uri: str, local_path: Path):
    """Download data from S3."""
    import boto3

    s3 = boto3.client("s3")

    # Parse S3 URI
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")

    parts = s3_uri[5:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    local_path.mkdir(parents=True, exist_ok=True)

    # List and download objects
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative_path = key[len(prefix):].lstrip("/")
            if relative_path:
                local_file = local_path / relative_path
                local_file.parent.mkdir(parents=True, exist_ok=True)
                print(f"Downloading s3://{bucket}/{key} -> {local_file}")
                s3.download_file(bucket, key, str(local_file))


def upload_to_s3(local_path: Path, s3_uri: str):
    """Upload file or directory to S3."""
    import boto3

    s3 = boto3.client("s3")

    # Parse S3 URI
    parts = s3_uri[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""

    if local_path.is_file():
        print(f"Uploading {local_path} -> {s3_uri}")
        s3.upload_file(str(local_path), bucket, key)
    else:
        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                relative = file_path.relative_to(local_path)
                s3_key = f"{key}/{relative}" if key else str(relative)
                print(f"Uploading {file_path} -> s3://{bucket}/{s3_key}")
                s3.upload_file(str(file_path), bucket, s3_key)


def run_training_job(config: dict):
    """Run a training job."""
    from src.training.trainer import Trainer, TrainingConfig
    from src.core.domain import DomainRegistry

    # Import domains to register them
    import src.domains

    print(f"Starting training job for domain: {config['domain']}")

    # Create temp directories for data
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        labels_dir = tmpdir / "labels"
        videos_dir = tmpdir / "videos"
        output_dir = tmpdir / "output"
        output_dir.mkdir()

        # Download data from S3
        if config.get("labels_s3_prefix"):
            s3_uri = f"s3://{os.environ.get('PRISMATA_S3_BUCKET_LABELS', '')}/{config['labels_s3_prefix']}"
            download_s3_data(s3_uri, labels_dir)

        if config.get("videos_s3_prefix"):
            s3_uri = f"s3://{os.environ.get('PRISMATA_S3_BUCKET_VIDEOS', '')}/{config['videos_s3_prefix']}"
            download_s3_data(s3_uri, videos_dir)

        # Build training config
        training_config_params = config.get("training_config", {})
        training_config = TrainingConfig(
            domain=config["domain"],
            labels_dir=str(labels_dir),
            videos_dir=str(videos_dir),
            output_dir=str(output_dir),
            experiment_name=f"job_{config['job_id'][:8]}",
            **training_config_params,
        )

        # Get domain
        domain = DomainRegistry.get(config["domain"])

        # Create trainer and run
        trainer = Trainer(training_config, domain)

        def progress_callback(epoch: int, metrics: dict):
            print(f"Epoch {epoch}: {metrics}")

        best_metrics = trainer.train(progress_callback=progress_callback)

        print(f"Training completed. Best metrics: {best_metrics}")

        # Find and upload the model
        model_files = list(output_dir.glob("*.pt"))
        if model_files:
            model_path = model_files[0]
            output_s3_key = config.get("output_model_s3_key", f"models/{config['domain']}/{model_path.name}")
            s3_uri = f"s3://{os.environ.get('PRISMATA_S3_BUCKET_MODELS', '')}/{output_s3_key}"
            upload_to_s3(model_path, s3_uri)
            print(f"Model uploaded to {s3_uri}")

        return {"metrics": best_metrics}


def run_inference_job(config: dict):
    """Run an inference job."""
    from src.inference.predictor import EventPredictor

    print(f"Starting inference job")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        model_path = tmpdir / "model.pt"
        video_path = tmpdir / "video.mp4"
        output_dir = tmpdir / "output"
        output_dir.mkdir()

        # Download model
        model_s3_key = config.get("model_s3_key", "")
        if model_s3_key:
            s3_uri = f"s3://{os.environ.get('PRISMATA_S3_BUCKET_MODELS', '')}/{model_s3_key}"
            download_s3_data(s3_uri, model_path.parent)
            # Rename if needed
            downloaded_models = list(model_path.parent.glob("*.pt"))
            if downloaded_models and downloaded_models[0] != model_path:
                downloaded_models[0].rename(model_path)

        # Download video
        video_s3_key = config.get("video_s3_key", "")
        if video_s3_key:
            s3_uri = f"s3://{os.environ.get('PRISMATA_S3_BUCKET_VIDEOS', '')}/{video_s3_key}"
            download_s3_data(s3_uri, video_path.parent)
            # Rename if needed
            downloaded_videos = list(video_path.parent.glob("*.mp4")) + list(video_path.parent.glob("*.mov"))
            if downloaded_videos and downloaded_videos[0] != video_path:
                downloaded_videos[0].rename(video_path)

        # Load predictor
        predictor = EventPredictor.from_checkpoint(str(model_path))

        # Run detection
        threshold = config.get("detection_threshold", 0.5)

        def progress_callback(progress: float, message: str):
            print(f"Progress: {progress:.1%} - {message}")

        result = predictor.predict_video(
            str(video_path),
            threshold=threshold,
            progress_callback=progress_callback,
        )

        print(f"Detection completed. Found {len(result.events)} events")

        # Save results
        output_file = output_dir / "detections.json"
        result.save(str(output_file))

        # Upload results
        output_s3_key = config.get("output_detection_s3_key", "")
        if output_s3_key:
            s3_uri = f"s3://{os.environ.get('PRISMATA_S3_BUCKET_DETECTIONS', '')}/{output_s3_key}"
            upload_to_s3(output_file, s3_uri)
            print(f"Results uploaded to {s3_uri}")

        return {
            "num_events": len(result.events),
            "output_s3_key": output_s3_key,
        }


def main():
    parser = argparse.ArgumentParser(description="Prismata GPU Job Runner")
    parser.add_argument("config_file", type=str, help="Path to job configuration JSON file")
    args = parser.parse_args()

    # Load configuration
    with open(args.config_file, "r") as f:
        config = json.load(f)

    print(f"Job ID: {config.get('job_id', 'unknown')}")
    print(f"Job Type: {config.get('job_type', 'unknown')}")

    try:
        setup_environment()

        job_type = config.get("job_type", "")

        if job_type == "training":
            result = run_training_job(config)
        elif job_type == "inference":
            result = run_inference_job(config)
        else:
            raise ValueError(f"Unknown job type: {job_type}")

        print(f"Job completed successfully: {result}")
        sys.exit(0)

    except Exception as e:
        print(f"Job failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
